// Raijin score requantiser (combinational)
// s = sat12(dot << sh) or sat12(dot >>> -sh), with sh = e_q + e_k - 262 + offset.
// The left shift is clamped to 15 and the right shift to 31.
`default_nettype none
module raijin_requant (
    input  wire signed [19:0] dot,
    input  wire        [8:0]  esum,
    input  wire signed [7:0]  offset,
    output reg  signed [11:0] s
);
    wire signed [11:0] sh = $signed({3'b000, esum}) - 12'sd262 + offset;
    reg  signed [35:0] wide;
    reg  [4:0]         amt;
    always @(*) begin
        if (sh >= 0) begin
            amt  = (sh > 12'sd15) ? 5'd15 : sh[4:0];
            wide = $signed({{16{dot[19]}}, dot}) <<< amt;
        end else begin
            amt  = (-sh > 12'sd31) ? 5'd31 : (-sh);
            wide = $signed({{16{dot[19]}}, dot}) >>> amt;
        end
        if (wide > 36'sd2047)       s = 12'sd2047;
        else if (wide < -36'sd2048) s = -12'sd2048;
        else                        s = wide[11:0];
    end
endmodule
`default_nettype wire
