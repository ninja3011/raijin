// Raijin output stage: per-row requant by 1/l and MXINT8 packing.
//   y[c]  = acc[c] * r                         (stage 1)
//   sh    = max(0, bitlen(OR_c |y[c]|') - 7)   (stage 2, leading-one detect)
//   o[c]  = sat8((y[c] + 2^(sh-1)) >>> sh)     (stage 3)
//   e_o   = clamp(e_vmax + sh - 19, 0, 255)
`default_nettype none
module raijin_out #(
    parameter D  = 16,
    parameter TW = 8
) (
    input  wire              clk,
    input  wire              rst_n,
    input  wire              in_valid,
    input  wire [TW-1:0]     in_tag,
    input  wire [D*24-1:0]   acc,
    input  wire [12:0]       r,
    input  wire [7:0]        emax,
    output reg               out_valid,
    output reg  [TW-1:0]     out_tag,
    output reg  [D*8-1:0]    o,
    output reg  [7:0]        eo
);
    // ---------------- stage 1 ----------------
    reg signed [37:0] y1 [0:D-1];
    reg               v1;
    reg [TW-1:0]      tag1;
    reg [7:0]         em1, em2;
    // one loop variable per always block, so synthesis never infers a register for it
    integer c1, c2, c3, c4;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) v1 <= 1'b0;
        else        v1 <= in_valid;
    end
    always @(posedge clk) begin
        tag1 <= in_tag;
        em1  <= emax;
        for (c1 = 0; c1 < D; c1 = c1 + 1)
            y1[c1] <= $signed(acc[c1*24 +: 24]) * $signed({1'b0, r});
    end

    // ---------------- stage 2 ----------------
    reg [37:0] orv;
    reg [5:0]  nb;
    integer b;
    always @(*) begin
        orv = 38'd0;
        for (c2 = 0; c2 < D; c2 = c2 + 1)
            orv = orv | (y1[c2] ^ {38{y1[c2][37]}});
        nb = 6'd0;
        for (b = 0; b < 38; b = b + 1)
            if (orv[b]) nb = b + 1;
    end

    reg signed [37:0] y2 [0:D-1];
    reg               v2;
    reg [TW-1:0]      tag2;
    reg [5:0]         sh2;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) v2 <= 1'b0;
        else        v2 <= v1;
    end
    always @(posedge clk) begin
        tag2 <= tag1;
        em2  <= em1;
        sh2  <= (nb > 6'd7) ? (nb - 6'd7) : 6'd0;
        for (c3 = 0; c3 < D; c3 = c3 + 1) y2[c3] <= y1[c3];
    end

    // ---------------- stage 3 ----------------
    reg signed [38:0] rnd;
    reg [D*8-1:0]     o_n;
    wire [9:0]        eo_w = {2'b00, em2} + {4'd0, sh2};
    always @(*) begin
        for (c4 = 0; c4 < D; c4 = c4 + 1) begin
            if (sh2 == 6'd0) rnd = {y2[c4][37], y2[c4]};
            else             rnd = ($signed({y2[c4][37], y2[c4]}) + (39'sd1 <<< (sh2 - 6'd1))) >>> sh2;
            if (rnd > 39'sd127)       o_n[c4*8 +: 8] = 8'sd127;
            else if (rnd < -39'sd128) o_n[c4*8 +: 8] = -8'sd128;
            else                      o_n[c4*8 +: 8] = rnd[7:0];
        end
    end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) out_valid <= 1'b0;
        else        out_valid <= v2;
    end
    always @(posedge clk) begin
        out_tag <= tag2;
        o       <= o_n;
        eo      <= (eo_w < 10'd19) ? 8'd0 : ((eo_w - 10'd19) > 10'd255 ? 8'd255 : (eo_w - 10'd19));
    end
endmodule
`default_nettype wire
