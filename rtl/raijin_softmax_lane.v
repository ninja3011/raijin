// Raijin softmax lane: integer running max, base-2 exponent, denominator,
// and a sequential reciprocal. One lane serves one query row.
//
//   Stage A (registered): m' = m_update ? max(m, ceil(s)) : m,  delta = m' - m
//   Stage B (registered): p = exp2(s - m'),  l <= (l >> delta) + p
//
// Batch mode, pass 1 : s_valid, m_update, !emit   (max only)
// Batch mode, pass 2 : s_valid, !m_update, emit   (exp and sum, delta = 0)
// Online mode        : s_valid, m_update, emit    (max, exp, rescale every key)
//
// recip_start launches floor(2^19 / l), done 13 cycles later (r_valid).
//
// Group pipelining (fused dataflow): first marks the first key of a new query
// group and restarts m and l without a clear cycle, so the next group can
// stream right behind the previous one. last marks a group's final key: the
// lane latches the final l and starts the reciprocal on its own, so the next
// group's keys can update l while the division runs. Both are 0 elsewhere.
`default_nettype none
module raijin_softmax_lane #(
    parameter TW = 8
) (
    input  wire              clk,
    input  wire              rst_n,
    input  wire              clear,
    input  wire [1:0]        mode,
    input  wire              s_valid,
    input  wire signed [11:0] s,
    input  wire [TW-1:0]     tag,
    input  wire              m_update,
    input  wire              emit,
    input  wire              first,      // first key of a group (restart m, l)
    input  wire              last,       // last key of a group (auto reciprocal)
    // stage B outputs
    output reg               out_valid,
    output reg  [TW-1:0]     out_tag,
    output reg  [8:0]        out_p,
    output reg  [3:0]        out_sa,
    output reg  [3:0]        out_sb,
    output wire [4:0]        out_delta,
    output reg               out_first,
    output reg               out_last,
    // state
    output reg  signed [8:0] m,
    output reg  [12:0]       l,
    // reciprocal
    input  wire              recip_start,
    output reg               r_valid,
    output reg  [12:0]       r
);
    // ---------------- stage A ----------------
    wire signed [12:0] s_p15 = $signed(s) + 13'sd15;
    wire signed [8:0]  mc    = s_p15 >>> 4;
    wire signed [8:0]  m_b   = first ? -9'sd256 : m;    // m before this key
    wire signed [8:0]  mn    = (m_update && (mc > m_b)) ? mc : m_b;
    wire        [9:0]  dlt   = $signed({mn[8], mn}) - $signed({m_b[8], m_b});

    reg               a_valid;
    reg  [TW-1:0]     a_tag;
    reg signed [11:0] a_s;
    reg signed [8:0]  a_m;
    reg        [9:0]  a_delta;
    reg               a_first, a_last;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m       <= -9'sd256;
            a_valid <= 1'b0;
        end else if (clear) begin
            m       <= -9'sd256;
            a_valid <= 1'b0;
        end else begin
            a_valid <= s_valid & emit;
            if (s_valid) m <= mn;
        end
    end

    always @(posedge clk) begin
        a_tag   <= tag;
        a_s     <= s;
        a_m     <= mn;
        a_delta <= dlt;
        a_first <= first;
        a_last  <= last;
    end

    // ---------------- stage B ----------------
    wire [8:0] p;
    wire [3:0] sa, sb;
    raijin_exp2 u_exp (.s(a_s), .m(a_m), .mode(mode), .p(p), .sa(sa), .sb(sb));

    wire [4:0]  dsh = (a_delta > 10'd31) ? 5'd31 : a_delta[4:0];
    wire [12:0] l_b  = a_first ? 13'd0 : l;              // l before this key
    wire [12:0] l_sh = l_b >> dsh;
    wire [12:0] l_nx = l_sh + {4'd0, p};
    reg         auto_go;                                  // reciprocal of l_fin
    reg  [12:0] l_fin;
    reg  [4:0]  b_delta;
    assign out_delta = b_delta;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            l         <= 13'd0;
            auto_go   <= 1'b0;
        end else if (clear) begin
            out_valid <= 1'b0;
            l         <= 13'd0;
            auto_go   <= 1'b0;
        end else begin
            out_valid <= a_valid;
            auto_go   <= a_valid && a_last;
            if (a_valid) l <= l_nx;
        end
    end
    always @(posedge clk) begin
        if (a_valid && a_last) l_fin <= l_nx;
    end

    always @(posedge clk) begin
        out_tag <= a_tag;
        out_p   <= p;
        out_sa  <= sa;
        out_sb  <= sb;
        b_delta <= dsh;
        out_first <= a_valid && a_first;
        out_last  <= a_valid && a_last;
    end

    // ---------------- reciprocal: floor(2^19 / l), l >= 128 ----------------
    reg [13:0] rem;
    reg [3:0]  cnt;
    reg        busy;
    reg [12:0] l_div;                 // divisor, latched at the start
    wire [14:0] rem2 = {rem, 1'b0};
    wire        ge   = rem2 >= {2'b00, l_div};
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy    <= 1'b0;
            r_valid <= 1'b0;
            r       <= 13'd0;
            rem     <= 14'd0;
            cnt     <= 4'd0;
        end else if (recip_start || auto_go) begin
            l_div   <= auto_go ? l_fin : l;
            busy    <= 1'b1;
            r_valid <= 1'b0;
            rem     <= 14'd64;
            cnt     <= 4'd12;
            r       <= 13'd0;
        end else if (busy) begin
            rem <= ge ? (rem2 - {2'b00, l_div}) : rem2[13:0];
            r   <= {r[11:0], ge};
            if (cnt == 4'd0) begin
                busy    <= 1'b0;
                r_valid <= 1'b1;
            end
            cnt <= cnt - 4'd1;
        end
    end
endmodule
`default_nettype wire
