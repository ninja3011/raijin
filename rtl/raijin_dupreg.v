// Raijin: C identical copies of a W-bit register.
//
// Used for signals that fan out to lanes spread across the die (1/l and the
// output shift amount). Each copy drives a local net instead of one die-wide
// buffer tree. Yosys merges identical flip-flops even when they carry
// (* keep *), so the sky130 build (RAIJIN_SKY130_CELLS) instantiates library
// flops directly: opt_merge never merges library cell instances.
`default_nettype none
module raijin_dupreg #(
    parameter W = 8,
    parameter C = 4
) (
    input  wire           clk,
    input  wire [W-1:0]   d,
    output wire [C*W-1:0] q
);
    genvar i, b;
    generate
        for (i = 0; i < C; i = i + 1) begin : g_copy
`ifdef RAIJIN_SKY130_CELLS
            for (b = 0; b < W; b = b + 1) begin : g_bit
                (* keep *) sky130_fd_sc_hd__dfxtp_1 u_ff (.CLK(clk), .D(d[b]), .Q(q[i*W + b]));
            end
`else
            reg [W-1:0] r;
            always @(posedge clk) r <= d;
            assign q[i*W +: W] = r;
`endif
        end
    endgenerate
endmodule
`default_nettype wire
