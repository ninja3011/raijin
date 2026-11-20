// Raijin accumulate array: R rows x D lanes of 24-bit accumulators in
// outer-product form. All rows share one V row; each row has its own p.
//
//   acc[r][c] <= (acc[r][c] >>> delta[r]) + ((p[r] * v[c]) >>> dv)
//
// Exact mode uses a 9x8 multiplier per lane. Multiplier-free mode (mf = 1)
// replaces it with two shifts and an add driven by the codes sa/sb, which give
// the same value because p = 2^(8-sa) + 2^(8-sb) there.
// A row whose en is low keeps its value (underflow skip, delta is 0 then).
// The product is registered, so an update lands two edges after its inputs.
`default_nettype none
module raijin_acc_array #(
    parameter R = 4,
    parameter D = 16
) (
    input  wire              clk,
    input  wire              rst_n,
    input  wire              clear,
    input  wire              mf,
    input  wire [D*8-1:0]    v,
    input  wire [4:0]        dv,
    input  wire [R-1:0]      en,
    input  wire [R*9-1:0]    p,
    input  wire [R*4-1:0]    sa,
    input  wire [R*4-1:0]    sb,
    input  wire [R*5-1:0]    delta,
    output wire [R*D*24-1:0] acc
);
    genvar gr, gc;
    generate
        for (gr = 0; gr < R; gr = gr + 1) begin : g_row
            wire [8:0] pr  = p[gr*9 +: 9];
            wire [3:0] sar = sa[gr*4 +: 4];
            wire [3:0] sbr = sb[gr*4 +: 4];
            wire [4:0] dr  = delta[gr*5 +: 5];
            reg  [4:0] dr_r;
            reg        en_r;
            always @(posedge clk) dr_r <= dr;
            always @(posedge clk or negedge rst_n)
                if (!rst_n) en_r <= 1'b0;
                else        en_r <= en[gr] && !clear;
            for (gc = 0; gc < D; gc = gc + 1) begin : g_lane
                wire signed [7:0]  vc = v[gc*8 +: 8];
                // exact: 9x8 multiply
                wire signed [17:0] prod_x = $signed({1'b0, pr}) * vc;
                // multiplier-free: v << (8 - sa) + v << (8 - sb)
                wire signed [17:0] vx = {{10{vc[7]}}, vc};
                wire signed [17:0] ta = (sar > 4'd8) ? 18'sd0 : (vx <<< (4'd8 - sar));
                wire signed [17:0] tb = (sbr > 4'd8) ? 18'sd0 : (vx <<< (4'd8 - sbr));
                wire signed [17:0] prod_m = ta + tb;
                wire signed [17:0] prod = mf ? prod_m : prod_x;
                wire signed [17:0] t_sh = prod >>> dv;
                reg  signed [17:0] t_r;
                reg  signed [23:0] a;
                always @(posedge clk) t_r <= t_sh;
                always @(posedge clk or negedge rst_n) begin
                    if (!rst_n)                a <= 24'sd0;
                    else if (clear)            a <= 24'sd0;
                    else if (en_r)             a <= (a >>> dr_r) + $signed({{6{t_r[17]}}, t_r});
                end
                assign acc[(gr*D + gc)*24 +: 24] = a;
            end
        end
    endgenerate
endmodule
`default_nettype wire
