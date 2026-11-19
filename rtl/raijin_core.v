// Raijin attention core
// Single-head MXINT8 attention with two dataflows on the same blocks:
//   unfused : S = Q K^T into the score buffer, batch softmax per row, then O = P V
//   fused   : keys stream past P query rows with base-2 online softmax; S is never stored
//
// Bus: byte-wide, 16-bit address, writes take effect on the clock edge,
// reads return rdata one cycle after re.
//   0x0rcc  Q[r][c]   (write)          0x4000+r  e_q[r]  (r/w)
//   0x1rcc  K[r][c]   (write)          0x4100+r  e_k[r]  (r/w)
//   0x2rcc  V[r][c]   (write)          0x4200+r  e_v[r]  (r/w)
//   0x3rcc  O[r][c]   (read)           0x4300+r  e_o[r]  (read)
//   (offset = r * D + c)
//   0x5000  CTRL   w: bit0 start, bit1 fused, bits3:2 mode (0 exact, 1 MF1, 2 MF2)
//   0x5001  OFFSET signed score requant offset
//   0x5002  STATUS bit0 busy, bit1 done
//   0x5004..7 CYCLES (little endian)   0x5008..B SKIPS
//   0x500C N, 0x500D D, 0x500E P, 0x500F version
`default_nettype none
module raijin_core #(
    parameter N = 16,       // sequence length (multiple of P, power of two)
    parameter D = 16,       // head dimension (power of two)
    parameter P = 4         // DPUs / softmax lanes / accumulator rows
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [15:0] addr,
    input  wire [7:0]  wdata,
    input  wire        we,
    input  wire        re,
    output reg  [7:0]  rdata,
    output wire        busy,
    output wire        done
);
    localparam NG  = N / P;
    localparam LN  = (N  > 1) ? $clog2(N)  : 1;
    localparam LD  = (D  > 1) ? $clog2(D)  : 1;
    localparam LG  = (NG > 1) ? $clog2(NG) : 1;
    localparam LP  = (P  > 1) ? $clog2(P)  : 1;
    localparam TW  = LG + LN;
    localparam VERSION = 8'h01;

    // ------------------------------------------------------------------
    // Storage (flip-flops)
    // ------------------------------------------------------------------
    reg [D*8-1:0] qm [0:N-1];
    reg [D*8-1:0] km [0:N-1];
    reg [D*8-1:0] vm [0:N-1];
    reg [D*8-1:0] om [0:N-1];
    reg [7:0]     eqm [0:N-1];
    reg [7:0]     ekm [0:N-1];
    reg [7:0]     evm [0:N-1];
    reg [7:0]     eom [0:N-1];
    reg [11:0]    sbuf [0:N*N-1];
    reg [12:0]    rinv [0:N-1];

    // ------------------------------------------------------------------
    // Control registers
    // ------------------------------------------------------------------
    reg               busy_r, done_r;
    reg               fused;
    reg [1:0]         mode;
    reg signed [7:0]  offset;
    reg [31:0]        cycles, skips;
    assign busy = busy_r;
    assign done = done_r;
    wire mf = (mode != 2'd0);

    // ------------------------------------------------------------------
    // Bus decode
    // ------------------------------------------------------------------
    wire [3:0]    region = addr[15:12];
    wire [11:0]   boff   = addr[11:0];
    wire [LN-1:0] brow   = boff[LD +: LN];
    wire [LD-1:0] bcol   = boff[LD-1:0];
    wire [LN-1:0] erow   = boff[LN-1:0];
    wire [1:0]    esel   = boff[9:8];
    wire          start_w = we && region == 4'h5 && boff[3:0] == 4'h0 && wdata[0] && !busy_r;

    always @(posedge clk) begin
        if (we && !busy_r) begin
            case (region)
                4'h0: qm[brow][bcol*8 +: 8] <= wdata;
                4'h1: km[brow][bcol*8 +: 8] <= wdata;
                4'h2: vm[brow][bcol*8 +: 8] <= wdata;
                4'h4: case (esel)
                          2'd0: eqm[erow] <= wdata;
                          2'd1: ekm[erow] <= wdata;
                          2'd2: evm[erow] <= wdata;
                          default: ;
                      endcase
                default: ;
            endcase
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rdata <= 8'd0;
        end else if (re) begin
            case (region)
                4'h3: rdata <= om[brow][bcol*8 +: 8];
                4'h4: case (esel)
                          2'd0: rdata <= eqm[erow];
                          2'd1: rdata <= ekm[erow];
                          2'd2: rdata <= evm[erow];
                          default: rdata <= eom[erow];
                      endcase
                4'h5: case (boff[3:0])
                          4'h0: rdata <= {4'd0, mode, fused, 1'b0};
                          4'h1: rdata <= offset;
                          4'h2: rdata <= {6'd0, done_r, busy_r};
                          4'h4: rdata <= cycles[7:0];
                          4'h5: rdata <= cycles[15:8];
                          4'h6: rdata <= cycles[23:16];
                          4'h7: rdata <= cycles[31:24];
                          4'h8: rdata <= skips[7:0];
                          4'h9: rdata <= skips[15:8];
                          4'hA: rdata <= skips[23:16];
                          4'hB: rdata <= skips[31:24];
                          4'hC: rdata <= N;
                          4'hD: rdata <= D;
                          4'hE: rdata <= P;
                          default: rdata <= VERSION;
                      endcase
                default: rdata <= 8'd0;
            endcase
        end
    end

    // ------------------------------------------------------------------
    // V exponent handling: e_vmax and per-row shift dv = e_vmax - e_v[j].
    // Both are registered: e_v only changes through the bus while idle, and a
    // pass first needs them several cycles after start, so the 16-way max tree
    // stays off the accumulate path.
    // ------------------------------------------------------------------
    reg [7:0] emax_c, emax;
    reg [4:0] dvr [0:N-1];
    integer ei, ew, di;
    reg [N*8-1:0] mt;     // pairwise max tree, log2(N) levels
    always @(*) begin
        for (ei = 0; ei < N; ei = ei + 1)
            mt[ei*8 +: 8] = evm[ei];
        for (ew = N / 2; ew >= 1; ew = ew / 2)
            for (ei = 0; ei < ew; ei = ei + 1)
                mt[ei*8 +: 8] = (mt[(2*ei)*8 +: 8] > mt[(2*ei+1)*8 +: 8]) ?
                                mt[(2*ei)*8 +: 8] : mt[(2*ei+1)*8 +: 8];
        emax_c = mt[7:0];
    end
    reg [7:0] dtmp;
    always @(posedge clk) begin
        emax <= emax_c;
        for (di = 0; di < N; di = di + 1) begin
            dtmp    = emax - evm[di];
            dvr[di] <= (dtmp > 8'd31) ? 5'd31 : dtmp[4:0];
        end
    end

    // ------------------------------------------------------------------
    // Controller state
    // ------------------------------------------------------------------
    localparam [4:0]
        S_IDLE     = 5'd0,
        U_QK       = 5'd1,
        U_QK_DRAIN = 5'd2,
        U_SM_CLR   = 5'd3,
        U_SM_P1    = 5'd4,
        U_SM_P2    = 5'd5,
        U_SM_DRAIN = 5'd6,
        RCP_START  = 5'd7,
        RCP_WAIT   = 5'd8,
        U_PV_CLR   = 5'd9,
        U_PV       = 5'd10,
        U_PV_DRAIN = 5'd11,
        OUT_ISSUE  = 5'd12,
        OUT_DRAIN  = 5'd13,
        F_CLR      = 5'd14,
        F_STREAM   = 5'd15,
        F_DRAIN    = 5'd16,
        S_DONE     = 5'd17;

    reg [4:0]    state;
    reg [LG-1:0] g;
    reg [LN-1:0] j;
    reg [LP-1:0] orow;
    reg [3:0]    dcnt;

    // ------------------------------------------------------------------
    // DPU array: P DPUs, query rows g*P + r, broadcast key row j
    // ------------------------------------------------------------------
    wire dpu_issue = (state == U_QK) || (state == F_STREAM);
    wire [TW-1:0] issue_tag = {g, j};

    wire [P-1:0]        dpu_valid;
    wire [P*TW-1:0]     dpu_tag;
    wire [P*20-1:0]     dpu_dot;
    wire [P*9-1:0]      dpu_esum;
    wire [P*12-1:0]     score;

    genvar gr;
    generate
        for (gr = 0; gr < P; gr = gr + 1) begin : g_dpu
            wire [LN-1:0] qrow = g * P + gr;
            raijin_dpu #(.D(D), .TW(TW)) u_dpu (
                .clk(clk), .rst_n(rst_n),
                .in_valid(dpu_issue), .in_tag(issue_tag),
                .a(qm[qrow]), .b(km[j]), .ea(eqm[qrow]), .eb(ekm[j]),
                .out_valid(dpu_valid[gr]), .out_tag(dpu_tag[gr*TW +: TW]),
                .dot(dpu_dot[gr*20 +: 20]), .esum(dpu_esum[gr*9 +: 9])
            );
            raijin_requant u_rq (
                .dot(dpu_dot[gr*20 +: 20]), .esum(dpu_esum[gr*9 +: 9]),
                .offset(offset), .s(score[gr*12 +: 12])
            );
        end
    endgenerate

    // ------------------------------------------------------------------
    // Softmax unit: P lanes
    // ------------------------------------------------------------------
    wire sm_clear   = (state == U_SM_CLR) || (state == F_CLR);
    wire sm_batch   = (state == U_SM_P1) || (state == U_SM_P2);
    wire sm_valid   = fused ? dpu_valid[0] : sm_batch;
    wire sm_mupd    = fused ? 1'b1 : (state == U_SM_P1);
    wire sm_emit    = fused ? 1'b1 : (state == U_SM_P2);
    wire [TW-1:0] sm_tag_in = fused ? dpu_tag[0 +: TW] : issue_tag;
    wire rcp_go     = (state == RCP_START);

    wire [P-1:0]    sm_ovalid;
    wire [P*TW-1:0] sm_otag;
    wire [P*9-1:0]  sm_p;
    wire [P*4-1:0]  sm_sa, sm_sb;
    wire [P*5-1:0]  sm_delta;
    wire [P-1:0]    sm_rvalid;
    wire [P*13-1:0] sm_r;

    generate
        for (gr = 0; gr < P; gr = gr + 1) begin : g_sm
            wire [LN-1:0] srow = g * P + gr;
            wire signed [11:0] s_in = fused ? score[gr*12 +: 12] : sbuf[srow * N + j];
            wire signed [8:0] m_unused;
            wire [12:0]       l_unused;
            raijin_softmax_lane #(.TW(TW)) u_lane (
                .clk(clk), .rst_n(rst_n), .clear(sm_clear), .mode(mode),
                .s_valid(sm_valid), .s(s_in), .tag(sm_tag_in),
                .m_update(sm_mupd), .emit(sm_emit),
                .out_valid(sm_ovalid[gr]), .out_tag(sm_otag[gr*TW +: TW]),
                .out_p(sm_p[gr*9 +: 9]), .out_sa(sm_sa[gr*4 +: 4]), .out_sb(sm_sb[gr*4 +: 4]),
                .out_delta(sm_delta[gr*5 +: 5]),
                .m(m_unused), .l(l_unused),
                .recip_start(rcp_go), .r_valid(sm_rvalid[gr]), .r(sm_r[gr*13 +: 13])
            );
        end
    endgenerate

    // ------------------------------------------------------------------
    // Score buffer writes: S from the DPU array, then p codes from softmax
    // ------------------------------------------------------------------
    integer wi;
    always @(posedge clk) begin
        if (!fused) begin
            for (wi = 0; wi < P; wi = wi + 1) begin
                if (dpu_valid[wi])
                    sbuf[(dpu_tag[wi*TW + LN +: LG] * P + wi) * N + dpu_tag[wi*TW +: LN]] <= score[wi*12 +: 12];
                if (sm_ovalid[wi])
                    sbuf[(sm_otag[wi*TW + LN +: LG] * P + wi) * N + sm_otag[wi*TW +: LN]] <=
                        mf ? {4'd0, sm_sb[wi*4 +: 4], sm_sa[wi*4 +: 4]} : {3'd0, sm_p[wi*9 +: 9]};
            end
        end
    end

    // ------------------------------------------------------------------
    // P V operand registers (unfused): read p codes back from the buffer
    // ------------------------------------------------------------------
    reg            pv_valid;
    reg [LN-1:0]   pv_j;
    reg [P*12-1:0] pv_code;
    integer pi;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) pv_valid <= 1'b0;
        else        pv_valid <= (state == U_PV);
    end
    always @(posedge clk) begin
        pv_j <= j;
        for (pi = 0; pi < P; pi = pi + 1)
            pv_code[pi*12 +: 12] <= sbuf[(g * P + pi) * N + j];
    end

    function [8:0] term;
        input [3:0] sh;
        begin
            term = (sh > 4'd8) ? 9'd0 : (9'd256 >> sh);
        end
    endfunction

    // ------------------------------------------------------------------
    // Accumulate array inputs
    // ------------------------------------------------------------------
    reg            acc_valid;
    reg [LN-1:0]   acc_j;
    reg [P*9-1:0]  acc_p;
    reg [P*4-1:0]  acc_sa, acc_sb;
    reg [P*5-1:0]  acc_delta;
    reg [P-1:0]    acc_en;
    integer ai;
    always @(*) begin
        if (fused) begin
            acc_valid = sm_ovalid[0];
            acc_j     = sm_otag[0 +: LN];
            acc_p     = sm_p;
            acc_sa    = sm_sa;
            acc_sb    = sm_sb;
            acc_delta = sm_delta;
        end else begin
            acc_valid = pv_valid;
            acc_j     = pv_j;
            acc_delta = {P*5{1'b0}};
            for (ai = 0; ai < P; ai = ai + 1) begin
                if (mf) begin
                    acc_sa[ai*4 +: 4] = pv_code[ai*12 +: 4];
                    acc_sb[ai*4 +: 4] = pv_code[ai*12 + 4 +: 4];
                    acc_p[ai*9 +: 9]  = term(pv_code[ai*12 +: 4]) + term(pv_code[ai*12 + 4 +: 4]);
                end else begin
                    acc_sa[ai*4 +: 4] = 4'd15;
                    acc_sb[ai*4 +: 4] = 4'd15;
                    acc_p[ai*9 +: 9]  = pv_code[ai*12 +: 9];
                end
            end
        end
        for (ai = 0; ai < P; ai = ai + 1)
            acc_en[ai] = acc_valid && (acc_p[ai*9 +: 9] != 9'd0);
    end

    wire acc_clear = (state == U_PV_CLR) || (state == F_CLR);
    wire [4:0] acc_dv  = dvr[acc_j];
    wire [P*D*24-1:0] acc;
    raijin_acc_array #(.R(P), .D(D)) u_acc (
        .clk(clk), .rst_n(rst_n), .clear(acc_clear), .mf(mf),
        .v(vm[acc_j]), .dv(acc_dv),
        .en(acc_en), .p(acc_p), .sa(acc_sa), .sb(acc_sb), .delta(acc_delta),
        .acc(acc)
    );

    // number of rows skipped this cycle
    reg [LP:0] nskip;
    integer ki;
    always @(*) begin
        nskip = 0;
        for (ki = 0; ki < P; ki = ki + 1)
            if (acc_valid && !acc_en[ki]) nskip = nskip + 1'b1;
    end

    // ------------------------------------------------------------------
    // Output stage
    // ------------------------------------------------------------------
    wire              out_issue = (state == OUT_ISSUE);
    wire [LN-1:0]     out_row_i = g * P + orow;
    wire              o_valid;
    wire [LN-1:0]     o_tag;
    wire [D*8-1:0]    o_row;
    wire [7:0]        o_e;
    raijin_out #(.D(D), .TW(LN)) u_out (
        .clk(clk), .rst_n(rst_n),
        .in_valid(out_issue), .in_tag(out_row_i),
        .acc(acc[orow*D*24 +: D*24]), .r(rinv[out_row_i]), .emax(emax),
        .out_valid(o_valid), .out_tag(o_tag), .o(o_row), .eo(o_e)
    );
    always @(posedge clk) begin
        if (o_valid) begin
            om[o_tag]  <= o_row;
            eom[o_tag] <= o_e;
        end
    end

    // reciprocal results
    integer ri;
    always @(posedge clk) begin
        if (state == RCP_WAIT && sm_rvalid[0])
            for (ri = 0; ri < P; ri = ri + 1)
                rinv[g * P + ri] <= sm_r[ri*13 +: 13];
    end

    // ------------------------------------------------------------------
    // Controller FSM
    // ------------------------------------------------------------------
    wire last_j = (j == N - 1);
    wire last_g = (g == NG - 1);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state  <= S_IDLE;
            busy_r <= 1'b0;
            done_r <= 1'b0;
            fused  <= 1'b0;
            mode   <= 2'd0;
            offset <= 8'sd0;
            cycles <= 32'd0;
            skips  <= 32'd0;
            g      <= {LG{1'b0}};
            j      <= {LN{1'b0}};
            orow   <= {LP{1'b0}};
            dcnt   <= 4'd0;
        end else begin
            if (we && region == 4'h5 && boff[3:0] == 4'h1 && !busy_r)
                offset <= wdata;
            if (busy_r) begin
                cycles <= cycles + 32'd1;
                skips  <= skips + nskip;
            end
            case (state)
                S_IDLE: begin
                    if (start_w) begin
                        fused  <= wdata[1];
                        mode   <= wdata[3:2];
                        busy_r <= 1'b1;
                        done_r <= 1'b0;
                        cycles <= 32'd0;
                        skips  <= 32'd0;
                        g      <= 0;
                        j      <= 0;
                        state  <= wdata[1] ? F_CLR : U_QK;
                    end
                end
                // ---------------- unfused ----------------
                U_QK: begin
                    j <= j + 1'b1;
                    if (last_j) begin
                        g <= g + 1'b1;
                        if (last_g) begin
                            dcnt  <= 4'd3;
                            state <= U_QK_DRAIN;
                        end
                    end
                end
                U_QK_DRAIN: begin
                    dcnt <= dcnt - 1'b1;
                    if (dcnt == 0) begin
                        g <= 0; j <= 0;
                        state <= U_SM_CLR;
                    end
                end
                U_SM_CLR: begin
                    j <= 0;
                    state <= U_SM_P1;
                end
                U_SM_P1: begin
                    j <= j + 1'b1;
                    if (last_j) state <= U_SM_P2;
                end
                U_SM_P2: begin
                    j <= j + 1'b1;
                    if (last_j) begin
                        dcnt  <= 4'd2;
                        state <= U_SM_DRAIN;
                    end
                end
                U_SM_DRAIN: begin
                    dcnt <= dcnt - 1'b1;
                    if (dcnt == 0) state <= RCP_START;
                end
                RCP_START: state <= RCP_WAIT;
                RCP_WAIT: begin
                    if (sm_rvalid[0]) begin
                        if (fused) begin
                            orow  <= 0;
                            state <= OUT_ISSUE;
                        end else if (last_g) begin
                            g <= 0;
                            state <= U_PV_CLR;
                        end else begin
                            g <= g + 1'b1;
                            state <= U_SM_CLR;
                        end
                    end
                end
                U_PV_CLR: begin
                    j <= 0;
                    state <= U_PV;
                end
                U_PV: begin
                    j <= j + 1'b1;
                    if (last_j) begin
                        dcnt  <= 4'd1;
                        state <= U_PV_DRAIN;
                    end
                end
                U_PV_DRAIN: begin
                    dcnt <= dcnt - 1'b1;
                    if (dcnt == 0) begin
                        orow  <= 0;
                        state <= OUT_ISSUE;
                    end
                end
                OUT_ISSUE: begin
                    orow <= orow + 1'b1;
                    if (orow == P - 1) begin
                        if (last_g) begin
                            dcnt  <= 4'd3;
                            state <= OUT_DRAIN;
                        end else begin
                            g     <= g + 1'b1;
                            state <= fused ? F_CLR : U_PV_CLR;
                        end
                    end
                end
                OUT_DRAIN: begin
                    dcnt <= dcnt - 1'b1;
                    if (dcnt == 0) state <= S_DONE;
                end
                // ---------------- fused ----------------
                F_CLR: begin
                    j <= 0;
                    state <= F_STREAM;
                end
                F_STREAM: begin
                    j <= j + 1'b1;
                    if (last_j) begin
                        dcnt  <= 4'd4;
                        state <= F_DRAIN;
                    end
                end
                F_DRAIN: begin
                    dcnt <= dcnt - 1'b1;
                    if (dcnt == 0) state <= RCP_START;
                end
                S_DONE: begin
                    busy_r <= 1'b0;
                    done_r <= 1'b1;
                    state  <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
`default_nettype wire
