// Raijin dot-product unit (DPU)
// D signed INT8 multipliers and an adder tree, registered in two stages.
// Stage 1 registers D/4 partial sums of four products each; stage 2 registers
// the 20-bit dot product. A tag and the summed row exponent ride alongside.
`default_nettype none
module raijin_dpu #(
    parameter D  = 16,
    parameter TW = 8          // tag width
) (
    input  wire              clk,
    input  wire              rst_n,
    input  wire              in_valid,
    input  wire [TW-1:0]     in_tag,
    input  wire [D*8-1:0]    a,        // row of signed INT8, element c at [c*8 +: 8]
    input  wire [D*8-1:0]    b,
    input  wire [7:0]        ea,       // E8M0 row exponents
    input  wire [7:0]        eb,
    output reg               out_valid,
    output reg  [TW-1:0]     out_tag,
    output reg  signed [19:0] dot,
    output reg  [8:0]        esum
);
    localparam G = D / 4;

    // ---- stage 1: products and groups of four ----
    reg signed [17:0] part_r [0:G-1];
    reg               v1;
    reg [TW-1:0]      tag1;
    reg [8:0]         es1;

    integer gi, ci;
    reg signed [17:0] acc4;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v1 <= 1'b0;
        end else begin
            v1 <= in_valid;
        end
    end

    always @(posedge clk) begin
        tag1 <= in_tag;
        es1  <= {1'b0, ea} + {1'b0, eb};
        for (gi = 0; gi < G; gi = gi + 1) begin
            acc4 = 18'sd0;
            for (ci = 0; ci < 4; ci = ci + 1)
                acc4 = acc4 + $signed(a[(gi*4+ci)*8 +: 8]) * $signed(b[(gi*4+ci)*8 +: 8]);
            part_r[gi] <= acc4;
        end
    end

    // ---- stage 2: final sum ----
    integer si;
    reg signed [19:0] tot;
    always @(*) begin
        tot = 20'sd0;
        for (si = 0; si < G; si = si + 1)
            tot = tot + part_r[si];
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) out_valid <= 1'b0;
        else        out_valid <= v1;
    end

    always @(posedge clk) begin
        out_tag <= tag1;
        esum    <= es1;
        dot     <= tot;
    end
endmodule
`default_nettype wire
