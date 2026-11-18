// Raijin base-2 exponent unit (combinational)
// t = 16*m - s (>= 0), k = t >> 4, f = t & 15.
//   exact mode : p = LUT[f] >> k            (p = 0 when k >= 9)
//   MF1 mode   : p = 2^-(k + s1(f))          (one power of two)
//   MF2 mode   : p = 2^-(k + s1(f)) + 2^-(k + s2(f))
// p is unsigned Q1.8. sa/sb are the shift codes (15 = no term) that the
// multiplier-free AV path uses instead of p.
`default_nettype none
module raijin_exp2 (
    input  wire signed [11:0] s,
    input  wire signed [8:0]  m,
    input  wire        [1:0]  mode,     // 0 exact, 1 MF1, 2 MF2
    output reg         [8:0]  p,
    output reg         [3:0]  sa,
    output reg         [3:0]  sb
);
    wire signed [14:0] t = ($signed(m) <<< 4) - $signed(s);  // 0 .. 6144
    wire [10:0] k = t[14:4];
    wire [3:0]  f = t[3:0];

    reg [8:0] lut;
    reg [3:0] s1, s2;
    always @(*) begin
        case (f)
            4'd0:  lut = 9'd256; 4'd1:  lut = 9'd245; 4'd2:  lut = 9'd235; 4'd3:  lut = 9'd225;
            4'd4:  lut = 9'd215; 4'd5:  lut = 9'd206; 4'd6:  lut = 9'd197; 4'd7:  lut = 9'd189;
            4'd8:  lut = 9'd181; 4'd9:  lut = 9'd173; 4'd10: lut = 9'd166; 4'd11: lut = 9'd159;
            4'd12: lut = 9'd152; 4'd13: lut = 9'd146; 4'd14: lut = 9'd140; default: lut = 9'd134;
        endcase
        if (mode == 2'd1) begin
            s1 = (f >= 4'd7) ? 4'd1 : 4'd0;
            s2 = 4'd15;
        end else begin
            s1 = (f >= 4'd4) ? 4'd1 : 4'd0;
            case (f)
                4'd0, 4'd1, 4'd2, 4'd3:           s2 = 4'd15;
                4'd4, 4'd5, 4'd6, 4'd7, 4'd8:     s2 = 4'd2;
                4'd9, 4'd10, 4'd11, 4'd12:        s2 = 4'd3;
                4'd13:                            s2 = 4'd4;
                4'd14:                            s2 = 4'd5;
                default:                          s2 = 4'd6;
            endcase
        end
    end

    function [8:0] term;
        input [3:0] sh;
        begin
            term = (sh > 4'd8) ? 9'd0 : (9'd256 >> sh);
        end
    endfunction

    wire [11:0] ka = {1'b0, k} + s1;
    wire [11:0] kb = {1'b0, k} + s2;
    always @(*) begin
        if (mode == 2'd0) begin
            p  = (k >= 11'd9) ? 9'd0 : (lut >> k[3:0]);
            sa = 4'd15;
            sb = 4'd15;
        end else begin
            sa = (ka > 12'd15) ? 4'd15 : ka[3:0];
            sb = (s2 == 4'd15 || kb > 12'd15) ? 4'd15 : kb[3:0];
            p  = term(sa) + term(sb);
        end
    end
endmodule
`default_nettype wire
