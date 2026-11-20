// Raijin attention core
// Single-head MXINT8 attention with three dataflows on the same blocks:
//   unfused : S = Q K^T into the score buffer, batch softmax per row, then O = P V
//   fused   : keys stream past P query rows with base-2 online softmax; S is never stored
//   decode  : KV-cache step. One query row attends to keys 0..LEN-1 already
//             resident in SRAM (fused dataflow, only that row's group runs)
//
// Storage: Q, K, V, O and the score buffer live in four 1 KB SRAM macros
// (sky130_sram_1kbyte_1rw1r_32x256_8, one RW and one R port each). Bank b holds
// bytes 4b..4b+3 of every row, so the four banks together return one full
// 128-bit row per port per cycle: a K row on port 1 and a V row on port 0.
//   word  0 ..  N-1   Q rows          word 2N .. 3N-1  V rows
//   word  N .. 2N-1   K rows          word 3N .. 4N-1  O rows
//   word 4N + {g, j}  score buffer, lanes 2b and 2b+1 in bank b (12 bits each)
// Q rows for the current group are latched in P row registers before the keys
// stream. SRAM read data is registered once before use, so a read issued in
// cycle t is consumed in cycle t + 2.
//
// Bus: byte-wide, 16-bit address, all four inputs registered on entry.
// A write takes effect two rising edges after it is presented. A read returns
// rdata after the third rising edge (RD_LATENCY = 3).
// The accumulators and the output stage each carry one extra pipeline register
// for timing; the controller's drain counts cover them. Q/K/V/O and the exponent
// registers are accessible only while the core is idle.
//   0x0rcc  Q[r][c]   (write)          0x4000+r  e_q[r]  (r/w)
//   0x1rcc  K[r][c]   (write)          0x4100+r  e_k[r]  (r/w)
//   0x2rcc  V[r][c]   (write)          0x4200+r  e_v[r]  (r/w)
//   0x3rcc  O[r][c]   (read)           0x4300+r  e_o[r]  (read)
//   (offset = r * D + c)
//   0x5000  CTRL   w: bit0 start, bit1 fused, bits3:2 mode (0 exact, 1 MF1, 2 MF2),
//                     bit4 decode (implies the fused dataflow)
//   0x5001  OFFSET signed score requant offset
//   0x5002  STATUS bit0 busy, bit1 done
//   0x5003  RD_LATENCY (3)
//   0x5004..7 CYCLES (little endian)   0x5008..B SKIPS
//   0x500C N, 0x500D D, 0x500E P, 0x500F version
//   0x5010  LEN    decode: number of cached keys, 1..N
//   0x5011  QROW   decode: query row (its output lands in O[QROW])
`default_nettype none
`ifdef RAIJIN_SKY130_SRAM
`define RAIJIN_RAM sky130_sram_1kbyte_1rw1r_32x256_8
`else
`define RAIJIN_RAM raijin_mem
`endif
module raijin_core #(
    parameter N = 16,       // sequence length (multiple of P, power of two)
    parameter D = 16,       // head dimension (four SRAM banks of 4 bytes)
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
    localparam VERSION = 8'h02;
    localparam NB  = 4;               // SRAM banks
    localparam NSB = (P + 1) / 2;     // banks holding the score buffer
    localparam [7:0] AQ = 0, AK = N, AV = 2 * N, AO = 3 * N, AS = 4 * N;

`ifndef SYNTHESIS
    initial begin
        if (D != 16 || 4 * N + NG * N > 256 || P > 8) begin
            $display("raijin_core: unsupported N=%0d D=%0d P=%0d for the 4 x 1 KB SRAM map", N, D, P);
            $finish;
        end
    end
`endif

    // ------------------------------------------------------------------
    // Registered bus inputs
    // ------------------------------------------------------------------
    reg [15:0] addr_r;
    reg [7:0]  wdata_r;
    reg        we_r, re_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            we_r <= 1'b0;
            re_r <= 1'b0;
        end else begin
            we_r <= we;
            re_r <= re;
        end
    end
    always @(posedge clk) begin
        addr_r  <= addr;
        wdata_r <= wdata;
    end

    // ------------------------------------------------------------------
    // Small storage that stays in flip-flops: row exponents and 1/l
    // ------------------------------------------------------------------
    reg [7:0]     eqm [0:N-1];
    reg [7:0]     ekm [0:N-1];
    reg [7:0]     evm [0:N-1];
    reg [7:0]     eom [0:N-1];
    reg [12:0]    rinv [0:N-1];
    reg [D*8-1:0] qreg [0:P-1];       // Q rows of the current group

    // ------------------------------------------------------------------
    // Control registers
    // ------------------------------------------------------------------
    reg               busy_r, done_r;
    reg               fused, decode;
    reg [1:0]         mode;
    reg signed [7:0]  offset;
    reg [LN:0]        len;
    reg [LN-1:0]      qrow;
    reg [31:0]        cycles, skips;
    assign busy = busy_r;
    assign done = done_r;
    wire mf = (mode != 2'd0);
    wire [LP-1:0] qlane = qrow % P;

    // ------------------------------------------------------------------
    // Bus decode
    // ------------------------------------------------------------------
    wire [3:0]    region = addr_r[15:12];
    wire [11:0]   boff   = addr_r[11:0];
    wire [LN-1:0] brow   = boff[LD +: LN];
    wire [LD-1:0] bcol   = boff[LD-1:0];
    wire [1:0]    bbank  = bcol[3:2];
    wire [1:0]    bbyte  = bcol[1:0];
    wire [LN-1:0] erow   = boff[LN-1:0];
    wire [1:0]    esel   = boff[9:8];
    wire          creg   = (region == 4'h5);
    wire [4:0]    cidx   = boff[4:0];
    wire          start_w    = we_r && creg && cidx == 5'h00 && wdata_r[0] && !busy_r;
    wire          bus_mem_we = we_r && !busy_r && region < 4'h3;
    wire          bus_mem_re = re_r && !busy_r && region == 4'h3;

    always @(posedge clk) begin
        if (we_r && !busy_r && region == 4'h4) begin
            case (esel)
                2'd0: eqm[erow] <= wdata_r;
                2'd1: ekm[erow] <= wdata_r;
                2'd2: evm[erow] <= wdata_r;
                default: ;
            endcase
        end
    end

    // ------------------------------------------------------------------
    // Controller state (declared early, the SRAM ports depend on it)
    // ------------------------------------------------------------------
    localparam [4:0]
        S_IDLE     = 5'd0,
        Q_LOAD     = 5'd1,
        U_QK       = 5'd2,
        U_QK_DRAIN = 5'd3,
        U_SM_CLR   = 5'd4,
        U_SM_P1    = 5'd5,
        U_SM_P2    = 5'd6,
        U_SM_DRAIN = 5'd7,
        RCP_START  = 5'd8,
        RCP_WAIT   = 5'd9,
        U_PV_CLR   = 5'd10,
        U_PV       = 5'd11,
        U_PV_DRAIN = 5'd12,
        OUT_ISSUE  = 5'd13,
        OUT_DRAIN  = 5'd14,
        F_CLR      = 5'd15,
        F_STREAM   = 5'd16,
        F_DRAIN    = 5'd17,
        S_DONE     = 5'd18,
        OUT_WAIT   = 5'd19;

    reg [4:0]    state;
    reg [LG-1:0] g;
    reg [LN-1:0] j;
    reg [LP-1:0] orow;
    reg [LP-1:0] qcnt;
    reg [3:0]    dcnt;
    wire [TW-1:0] issue_tag = {g, j};

    // ------------------------------------------------------------------
    // SRAM banks
    // ------------------------------------------------------------------
    reg  [NB-1:0]    p0_csb, p0_web, p1_csb;
    reg  [NB*4-1:0]  p0_mask;
    reg  [7:0]       p0_addr, p1_addr;
    reg  [NB*32-1:0] p0_din;
    wire [NB*32-1:0] p0_dout, p1_dout;
    reg  [NB*32-1:0] d0_r, d1_r;          // registered read data

    `RAIJIN_RAM sram0 (.clk0(clk), .csb0(p0_csb[0]), .web0(p0_web[0]), .wmask0(p0_mask[3:0]),
        .addr0(p0_addr), .din0(p0_din[31:0]), .dout0(p0_dout[31:0]),
        .clk1(clk), .csb1(p1_csb[0]), .addr1(p1_addr), .dout1(p1_dout[31:0]));
    `RAIJIN_RAM sram1 (.clk0(clk), .csb0(p0_csb[1]), .web0(p0_web[1]), .wmask0(p0_mask[7:4]),
        .addr0(p0_addr), .din0(p0_din[63:32]), .dout0(p0_dout[63:32]),
        .clk1(clk), .csb1(p1_csb[1]), .addr1(p1_addr), .dout1(p1_dout[63:32]));
    `RAIJIN_RAM sram2 (.clk0(clk), .csb0(p0_csb[2]), .web0(p0_web[2]), .wmask0(p0_mask[11:8]),
        .addr0(p0_addr), .din0(p0_din[95:64]), .dout0(p0_dout[95:64]),
        .clk1(clk), .csb1(p1_csb[2]), .addr1(p1_addr), .dout1(p1_dout[95:64]));
    `RAIJIN_RAM sram3 (.clk0(clk), .csb0(p0_csb[3]), .web0(p0_web[3]), .wmask0(p0_mask[15:12]),
        .addr0(p0_addr), .din0(p0_din[127:96]), .dout0(p0_dout[127:96]),
        .clk1(clk), .csb1(p1_csb[3]), .addr1(p1_addr), .dout1(p1_dout[127:96]));

    always @(posedge clk) begin
        d0_r <= p0_dout;
        d1_r <= p1_dout;
    end

    // ---- port 1 (read only): K rows, Q rows, score buffer reads ----
    wire k_rd = (state == U_QK) || (state == F_STREAM);
    wire q_rd = (state == Q_LOAD);
    wire s_rd = (state == U_SM_P1) || (state == U_SM_P2) || (state == U_PV);
    integer pb;
    always @(*) begin
        p1_csb  = {NB{1'b1}};
        p1_addr = 8'd0;
        if (k_rd) begin
            p1_csb  = {NB{1'b0}};
            p1_addr = AK + j;
        end else if (q_rd) begin
            p1_csb  = {NB{1'b0}};
            p1_addr = AQ + g * P + qcnt;
        end else if (s_rd) begin
            for (pb = 0; pb < NSB; pb = pb + 1) p1_csb[pb] = 1'b0;
            p1_addr = AS + issue_tag;
        end
    end

    // read-side pipeline: a port-1 read in cycle t is used in cycle t + 2
    reg          k_d1, k_d2, q_d1, q_d2, b_d1, b_d2, pv_d1, pv_d2;
    reg          mu_d1, mu_d2, em_d1, em_d2;
    reg [TW-1:0] tag_d1, tag_d2;
    reg [LP-1:0] qc_d1, qc_d2;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            {k_d1, k_d2, q_d1, q_d2, b_d1, b_d2, pv_d1, pv_d2} <= 8'd0;
        end else begin
            k_d1  <= k_rd;                    k_d2  <= k_d1;
            q_d1  <= q_rd;                    q_d2  <= q_d1;
            b_d1  <= (state == U_SM_P1) || (state == U_SM_P2);
            b_d2  <= b_d1;
            pv_d1 <= (state == U_PV);         pv_d2 <= pv_d1;
        end
    end
    always @(posedge clk) begin
        tag_d1 <= issue_tag;           tag_d2 <= tag_d1;
        qc_d1  <= qcnt;                qc_d2  <= qc_d1;
        mu_d1  <= (state == U_SM_P1);  mu_d2  <= mu_d1;
        em_d1  <= (state == U_SM_P2);  em_d2  <= em_d1;
    end
    wire [LG-1:0] g_d2 = tag_d2[LN +: LG];
    wire [LN-1:0] j_d2 = tag_d2[LN-1:0];

    integer qi;
    always @(posedge clk) begin
        if (q_d2)
            for (qi = 0; qi < P; qi = qi + 1)
                if (qc_d2 == qi) qreg[qi] <= d1_r;
    end

    // score buffer read data, lane r
    wire [P*12-1:0] sb_rd;
    genvar gr;
    generate
        for (gr = 0; gr < P; gr = gr + 1) begin : g_sbrd
            assign sb_rd[gr*12 +: 12] = d1_r[(gr/2)*32 + (gr%2)*12 +: 12];
        end
    endgenerate

    // ------------------------------------------------------------------
    // V exponent handling: e_vmax and per-row shift dv = e_vmax - e_v[j].
    // In decode mode only the LEN cached rows count towards the maximum.
    // ------------------------------------------------------------------
    reg [7:0] emax_c, emax;
    reg [4:0] dvr [0:N-1];
    integer ei, ew, di;
    reg [N*8-1:0] mt;     // pairwise max tree, log2(N) levels
    always @(*) begin
        for (ei = 0; ei < N; ei = ei + 1)
            mt[ei*8 +: 8] = (!decode || ei < len) ? evm[ei] : 8'd0;
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
    // DPU array: P DPUs, Q rows from qreg, K row straight from the banks
    // ------------------------------------------------------------------
    wire [P-1:0]        dpu_valid;
    wire [P*TW-1:0]     dpu_tag;
    wire [P*20-1:0]     dpu_dot;
    wire [P*9-1:0]      dpu_esum;
    wire [P*12-1:0]     score;

    generate
        for (gr = 0; gr < P; gr = gr + 1) begin : g_dpu
            wire [LN-1:0] qr = g_d2 * P + gr;
            raijin_dpu #(.D(D), .TW(TW)) u_dpu (
                .clk(clk), .rst_n(rst_n),
                .in_valid(k_d2), .in_tag(tag_d2),
                .a(qreg[gr]), .b(d1_r), .ea(eqm[qr]), .eb(ekm[j_d2]),
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
    wire sm_valid   = fused ? dpu_valid[0] : b_d2;
    wire sm_mupd    = fused ? 1'b1 : mu_d2;
    wire sm_emit    = fused ? 1'b1 : em_d2;
    wire [TW-1:0] sm_tag_in = fused ? dpu_tag[0 +: TW] : tag_d2;
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
            wire signed [11:0] s_in = fused ? score[gr*12 +: 12] : sb_rd[gr*12 +: 12];
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

    // score buffer writes (unfused): scores from the DPUs, then p codes
    wire            sb_we   = !fused && (dpu_valid[0] || sm_ovalid[0]);
    wire [TW-1:0]   sb_wtag = dpu_valid[0] ? dpu_tag[0 +: TW] : sm_otag[0 +: TW];
    reg  [NB*32-1:0] sb_wdata;
    integer wi;
    always @(*) begin
        sb_wdata = {NB*32{1'b0}};
        for (wi = 0; wi < P; wi = wi + 1)
            sb_wdata[(wi/2)*32 + (wi%2)*12 +: 12] = dpu_valid[0] ? score[wi*12 +: 12] :
                (mf ? {4'd0, sm_sb[wi*4 +: 4], sm_sa[wi*4 +: 4]} : {3'd0, sm_p[wi*9 +: 9]});
    end

    function [8:0] term;
        input [3:0] sh;
        begin
            term = (sh > 4'd8) ? 9'd0 : (9'd256 >> sh);
        end
    endfunction

    // ------------------------------------------------------------------
    // Accumulate inputs. Stage c picks the source and issues the V read;
    // two register stages later the V row arrives and the array updates.
    // ------------------------------------------------------------------
    reg            acc_valid;
    reg [LN-1:0]   acc_j;
    reg [P*9-1:0]  acc_p;
    reg [P*4-1:0]  acc_sa, acc_sb;
    reg [P*5-1:0]  acc_delta;
    reg [P-1:0]    acc_en, acc_lane;
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
            acc_valid = pv_d2;
            acc_j     = j_d2;
            acc_delta = {P*5{1'b0}};
            for (ai = 0; ai < P; ai = ai + 1) begin
                if (mf) begin
                    acc_sa[ai*4 +: 4] = sb_rd[ai*12 +: 4];
                    acc_sb[ai*4 +: 4] = sb_rd[ai*12 + 4 +: 4];
                    acc_p[ai*9 +: 9]  = term(sb_rd[ai*12 +: 4]) + term(sb_rd[ai*12 + 4 +: 4]);
                end else begin
                    acc_sa[ai*4 +: 4] = 4'd15;
                    acc_sb[ai*4 +: 4] = 4'd15;
                    acc_p[ai*9 +: 9]  = sb_rd[ai*12 +: 9];
                end
            end
        end
        for (ai = 0; ai < P; ai = ai + 1) begin
            acc_lane[ai] = !decode || (qlane == ai);
            acc_en[ai]   = acc_valid && acc_lane[ai] && (acc_p[ai*9 +: 9] != 9'd0);
        end
    end
    wire v_rd = acc_valid;

    reg            av1, av2;
    reg [LN-1:0]   aj1, aj2;
    reg [P*9-1:0]  ap1, ap2;
    reg [P*4-1:0]  asa1, asa2, asb1, asb2;
    reg [P*5-1:0]  ad1, ad2;
    reg [P-1:0]    aen1, aen2, aln1, aln2;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            av1 <= 1'b0; av2 <= 1'b0; aen1 <= {P{1'b0}}; aen2 <= {P{1'b0}};
        end else begin
            av1 <= acc_valid; av2 <= av1;
            aen1 <= acc_en;   aen2 <= aen1;
        end
    end
    always @(posedge clk) begin
        aj1 <= acc_j;      aj2 <= aj1;
        ap1 <= acc_p;      ap2 <= ap1;
        asa1 <= acc_sa;    asa2 <= asa1;
        asb1 <= acc_sb;    asb2 <= asb1;
        ad1 <= acc_delta;  ad2 <= ad1;
        aln1 <= acc_lane;  aln2 <= aln1;
    end

    wire acc_clear = (state == U_PV_CLR) || (state == F_CLR);
    wire [P*D*24-1:0] acc;
    raijin_acc_array #(.R(P), .D(D)) u_acc (
        .clk(clk), .rst_n(rst_n), .clear(acc_clear), .mf(mf),
        .v(d0_r), .dv(dvr[aj2]),
        .en(aen2), .p(ap2), .sa(asa2), .sb(asb2), .delta(ad2),
        .acc(acc)
    );

    // number of rows skipped this cycle
    reg [LP:0] nskip;
    integer ki;
    always @(*) begin
        nskip = 0;
        for (ki = 0; ki < P; ki = ki + 1)
            if (av2 && aln2[ki] && !aen2[ki]) nskip = nskip + 1'b1;
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
    // stage 0: register the selected accumulator row and its 1/l, so the row
    // select does not sit in front of the 24 x 13 multipliers
    reg              o0_v;
    reg [LN-1:0]     o0_tag;
    reg [D*24-1:0]   o0_acc;
    reg [12:0]       o0_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) o0_v <= 1'b0;
        else        o0_v <= out_issue;
    end
    always @(posedge clk) begin
        o0_tag <= out_row_i;
        o0_acc <= acc[orow*D*24 +: D*24];
        o0_r   <= rinv[out_row_i];
    end
    raijin_out #(.D(D), .TW(LN)) u_out (
        .clk(clk), .rst_n(rst_n),
        .in_valid(o0_v), .in_tag(o0_tag),
        .acc(o0_acc), .r(o0_r), .emax(emax),
        .out_valid(o_valid), .out_tag(o_tag), .o(o_row), .eo(o_e)
    );
    always @(posedge clk) begin
        if (o_valid) eom[o_tag] <= o_e;
    end

    // ---- port 0 (read/write): O writes, V reads, score writes, bus ----
    integer mb;
    always @(*) begin
        p0_csb  = {NB{1'b1}};
        p0_web  = {NB{1'b1}};
        p0_mask = {NB*4{1'b0}};
        p0_addr = 8'd0;
        p0_din  = {NB*32{1'b0}};
        if (o_valid) begin
            p0_csb  = {NB{1'b0}};
            p0_web  = {NB{1'b0}};
            p0_mask = {NB*4{1'b1}};
            p0_addr = AO + o_tag;
            p0_din  = o_row;
        end else if (v_rd) begin
            p0_csb  = {NB{1'b0}};
            p0_addr = AV + acc_j;
        end else if (sb_we) begin
            for (mb = 0; mb < NSB; mb = mb + 1) begin
                p0_csb[mb] = 1'b0;
                p0_web[mb] = 1'b0;
                p0_mask[mb*4 +: 4] = 4'hF;
            end
            p0_addr = AS + sb_wtag;
            p0_din  = sb_wdata;
        end else if (bus_mem_we) begin
            p0_csb[bbank] = 1'b0;
            p0_web        = {NB{1'b0}};
            p0_mask[bbank*4 + bbyte] = 1'b1;
            p0_addr = region[1:0] * N + brow;
            p0_din  = {NB*4{wdata_r}};
        end else if (bus_mem_re) begin
            p0_csb[bbank] = 1'b0;
            p0_addr = AO + brow;
        end
    end

`ifndef SYNTHESIS
    always @(posedge clk)
        if (rst_n && (o_valid + v_rd + sb_we) > 1)
            $display("%t raijin_core ERROR: SRAM port 0 conflict o=%b v=%b sb=%b", $time, o_valid, v_rd, sb_we);
`endif

    // ------------------------------------------------------------------
    // Bus read path: stage 1 picks a register value, stage 2 muxes in SRAM data
    // ------------------------------------------------------------------
    reg       rd_v, rd_mem;
    reg [1:0] rd_bank, rd_byte;
    reg [7:0] rd_reg;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_v   <= 1'b0;
            rd_mem <= 1'b0;
            rdata  <= 8'd0;
        end else begin
            rd_v    <= re_r;
            rd_mem  <= (region == 4'h3);
            rd_bank <= bbank;
            rd_byte <= bbyte;
            if (re_r) begin
                case (region)
                    4'h4: case (esel)
                              2'd0: rd_reg <= eqm[erow];
                              2'd1: rd_reg <= ekm[erow];
                              2'd2: rd_reg <= evm[erow];
                              default: rd_reg <= eom[erow];
                          endcase
                    4'h5: case (cidx)
                              5'h00: rd_reg <= {3'd0, decode, mode, fused, 1'b0};
                              5'h01: rd_reg <= offset;
                              5'h02: rd_reg <= {6'd0, done_r, busy_r};
                              5'h03: rd_reg <= 8'd3;
                              5'h04: rd_reg <= cycles[7:0];
                              5'h05: rd_reg <= cycles[15:8];
                              5'h06: rd_reg <= cycles[23:16];
                              5'h07: rd_reg <= cycles[31:24];
                              5'h08: rd_reg <= skips[7:0];
                              5'h09: rd_reg <= skips[15:8];
                              5'h0A: rd_reg <= skips[23:16];
                              5'h0B: rd_reg <= skips[31:24];
                              5'h0C: rd_reg <= N;
                              5'h0D: rd_reg <= D;
                              5'h0E: rd_reg <= P;
                              5'h0F: rd_reg <= VERSION;
                              5'h10: rd_reg <= len;
                              5'h11: rd_reg <= qrow;
                              default: rd_reg <= 8'd0;
                          endcase
                    default: rd_reg <= 8'd0;
                endcase
            end
            if (rd_v)
                rdata <= rd_mem ? p0_dout[rd_bank*32 + rd_byte*8 +: 8] : rd_reg;
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
    wire last_j  = (j == N - 1);
    wire last_jf = decode ? ({1'b0, j} == len - 1'b1) : last_j;
    wire last_g  = (g == NG - 1);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state  <= S_IDLE;
            busy_r <= 1'b0;
            done_r <= 1'b0;
            fused  <= 1'b0;
            decode <= 1'b0;
            mode   <= 2'd0;
            offset <= 8'sd0;
            len    <= N;
            qrow   <= {LN{1'b0}};
            cycles <= 32'd0;
            skips  <= 32'd0;
            g      <= {LG{1'b0}};
            j      <= {LN{1'b0}};
            orow   <= {LP{1'b0}};
            qcnt   <= {LP{1'b0}};
            dcnt   <= 4'd0;
        end else begin
            if (we_r && creg && !busy_r) begin
                if (cidx == 5'h01) offset <= wdata_r;
                if (cidx == 5'h10) len    <= wdata_r[LN:0];
                if (cidx == 5'h11) qrow   <= wdata_r[LN-1:0];
            end
            if (busy_r) begin
                cycles <= cycles + 32'd1;
                skips  <= skips + nskip;
            end
            case (state)
                S_IDLE: begin
                    if (start_w) begin
                        fused  <= wdata_r[1] | wdata_r[4];
                        decode <= wdata_r[4];
                        mode   <= wdata_r[3:2];
                        busy_r <= 1'b1;
                        done_r <= 1'b0;
                        cycles <= 32'd0;
                        skips  <= 32'd0;
                        g      <= wdata_r[4] ? qrow / P : {LG{1'b0}};
                        j      <= 0;
                        qcnt   <= 0;
                        state  <= (wdata_r[1] | wdata_r[4]) ? F_CLR : Q_LOAD;
                    end
                end
                // load the P query rows of group g (both dataflows)
                Q_LOAD: begin
                    qcnt <= qcnt + 1'b1;
                    if (qcnt == P - 1) begin
                        j     <= 0;
                        state <= fused ? F_STREAM : U_QK;
                    end
                end
                // ---------------- unfused ----------------
                U_QK: begin
                    j <= j + 1'b1;
                    if (last_j) begin
                        g <= g + 1'b1;
                        if (last_g) begin
                            dcnt  <= 4'd5;
                            state <= U_QK_DRAIN;
                        end else begin
                            qcnt  <= 0;
                            state <= Q_LOAD;
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
                        dcnt  <= 4'd4;
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
                            orow  <= decode ? qlane : {LP{1'b0}};
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
                        dcnt  <= 4'd4;
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
                    if (decode || orow == P - 1) begin
                        if (decode || last_g) begin
                            dcnt  <= 4'd4;
                            state <= OUT_DRAIN;
                        end else begin
                            g     <= g + 1'b1;
                            state <= OUT_WAIT;
                        end
                    end
                end
                // one cycle so the last O write (port 0) lands before the next V read
                OUT_WAIT: state <= fused ? F_CLR : U_PV_CLR;
                OUT_DRAIN: begin
                    dcnt <= dcnt - 1'b1;
                    if (dcnt == 0) state <= S_DONE;
                end
                // ---------------- fused / decode ----------------
                F_CLR: begin
                    qcnt  <= 0;
                    state <= Q_LOAD;
                end
                F_STREAM: begin
                    j <= j + 1'b1;
                    if (last_jf) begin
                        dcnt  <= 4'd6;
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
