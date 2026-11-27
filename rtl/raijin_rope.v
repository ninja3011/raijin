// Raijin RoPE-on-write rotator (feature 9).
//
// With ROPE on, the host writes Q and K rows as usual, one byte at a time,
// and each element pair (2i, 2i+1) is turned by pos * theta_i on its way
// into SRAM, theta_i = 10000^(-2i/16). Real models cache K after RoPE, so
// rotating on write is the natural place, and it keeps the rotator off the
// pass datapath: no cycles, no timing cost on the SRAM-to-DPU path.
//
//   even byte (x, column 2i): latched, with cos/sin for (pos, i) from the
//                             phase multiplier and a quarter-wave table
//   odd byte  (y, column 2i+1): x' = sat8((x c - y s + 128) >> 8),
//                               y' = sat8((x s + y c + 128) >> 8), written as one
//                               two-byte SRAM write in place of the host's byte
//
// cos/sin are Q1.8. The result is rounded and saturated to INT8, and the row
// keeps its exponent (a rotation preserves the pair's length, so saturation
// only clips a pair whose length is above 127).
// Phase is in turns, Q0.20: phase = pos * W_i mod 2^20, rounded to 10 bits.
// Only D = 16 (eight frequencies) is tabulated.
`default_nettype none
module raijin_rope (
    input  wire              clk,
    input  wire              we_even,   // host writes x of pair i
    input  wire [2:0]        pair,      // i
    input  wire [7:0]        pos,
    input  wire [7:0]        wdata,     // x (even) or y (odd)
    output wire [7:0]        xo,        // x'
    output wire [7:0]        yo         // y'
);
    function [17:0] wfreq;
        input [2:0] i;
        case (i)
            3'd0: wfreq = 18'd166886;
            3'd1: wfreq = 18'd52774;
            3'd2: wfreq = 18'd16689;
            3'd3: wfreq = 18'd5277;
            3'd4: wfreq = 18'd1669;
            3'd5: wfreq = 18'd528;
            3'd6: wfreq = 18'd167;
            3'd7: wfreq = 18'd53;
            default: wfreq = 18'd0;
        endcase
    endfunction

    // quarter-wave sine, Q1.8; entry 256 (= 256) is handled below
    function [8:0] qsin;
        input [7:0] k;
        case (k)
            8'd0: qsin = 9'd0;
            8'd1: qsin = 9'd2;
            8'd2: qsin = 9'd3;
            8'd3: qsin = 9'd5;
            8'd4: qsin = 9'd6;
            8'd5: qsin = 9'd8;
            8'd6: qsin = 9'd9;
            8'd7: qsin = 9'd11;
            8'd8: qsin = 9'd13;
            8'd9: qsin = 9'd14;
            8'd10: qsin = 9'd16;
            8'd11: qsin = 9'd17;
            8'd12: qsin = 9'd19;
            8'd13: qsin = 9'd20;
            8'd14: qsin = 9'd22;
            8'd15: qsin = 9'd24;
            8'd16: qsin = 9'd25;
            8'd17: qsin = 9'd27;
            8'd18: qsin = 9'd28;
            8'd19: qsin = 9'd30;
            8'd20: qsin = 9'd31;
            8'd21: qsin = 9'd33;
            8'd22: qsin = 9'd34;
            8'd23: qsin = 9'd36;
            8'd24: qsin = 9'd38;
            8'd25: qsin = 9'd39;
            8'd26: qsin = 9'd41;
            8'd27: qsin = 9'd42;
            8'd28: qsin = 9'd44;
            8'd29: qsin = 9'd45;
            8'd30: qsin = 9'd47;
            8'd31: qsin = 9'd48;
            8'd32: qsin = 9'd50;
            8'd33: qsin = 9'd51;
            8'd34: qsin = 9'd53;
            8'd35: qsin = 9'd55;
            8'd36: qsin = 9'd56;
            8'd37: qsin = 9'd58;
            8'd38: qsin = 9'd59;
            8'd39: qsin = 9'd61;
            8'd40: qsin = 9'd62;
            8'd41: qsin = 9'd64;
            8'd42: qsin = 9'd65;
            8'd43: qsin = 9'd67;
            8'd44: qsin = 9'd68;
            8'd45: qsin = 9'd70;
            8'd46: qsin = 9'd71;
            8'd47: qsin = 9'd73;
            8'd48: qsin = 9'd74;
            8'd49: qsin = 9'd76;
            8'd50: qsin = 9'd77;
            8'd51: qsin = 9'd79;
            8'd52: qsin = 9'd80;
            8'd53: qsin = 9'd82;
            8'd54: qsin = 9'd83;
            8'd55: qsin = 9'd85;
            8'd56: qsin = 9'd86;
            8'd57: qsin = 9'd88;
            8'd58: qsin = 9'd89;
            8'd59: qsin = 9'd91;
            8'd60: qsin = 9'd92;
            8'd61: qsin = 9'd94;
            8'd62: qsin = 9'd95;
            8'd63: qsin = 9'd97;
            8'd64: qsin = 9'd98;
            8'd65: qsin = 9'd99;
            8'd66: qsin = 9'd101;
            8'd67: qsin = 9'd102;
            8'd68: qsin = 9'd104;
            8'd69: qsin = 9'd105;
            8'd70: qsin = 9'd107;
            8'd71: qsin = 9'd108;
            8'd72: qsin = 9'd109;
            8'd73: qsin = 9'd111;
            8'd74: qsin = 9'd112;
            8'd75: qsin = 9'd114;
            8'd76: qsin = 9'd115;
            8'd77: qsin = 9'd117;
            8'd78: qsin = 9'd118;
            8'd79: qsin = 9'd119;
            8'd80: qsin = 9'd121;
            8'd81: qsin = 9'd122;
            8'd82: qsin = 9'd123;
            8'd83: qsin = 9'd125;
            8'd84: qsin = 9'd126;
            8'd85: qsin = 9'd128;
            8'd86: qsin = 9'd129;
            8'd87: qsin = 9'd130;
            8'd88: qsin = 9'd132;
            8'd89: qsin = 9'd133;
            8'd90: qsin = 9'd134;
            8'd91: qsin = 9'd136;
            8'd92: qsin = 9'd137;
            8'd93: qsin = 9'd138;
            8'd94: qsin = 9'd140;
            8'd95: qsin = 9'd141;
            8'd96: qsin = 9'd142;
            8'd97: qsin = 9'd144;
            8'd98: qsin = 9'd145;
            8'd99: qsin = 9'd146;
            8'd100: qsin = 9'd147;
            8'd101: qsin = 9'd149;
            8'd102: qsin = 9'd150;
            8'd103: qsin = 9'd151;
            8'd104: qsin = 9'd152;
            8'd105: qsin = 9'd154;
            8'd106: qsin = 9'd155;
            8'd107: qsin = 9'd156;
            8'd108: qsin = 9'd157;
            8'd109: qsin = 9'd159;
            8'd110: qsin = 9'd160;
            8'd111: qsin = 9'd161;
            8'd112: qsin = 9'd162;
            8'd113: qsin = 9'd164;
            8'd114: qsin = 9'd165;
            8'd115: qsin = 9'd166;
            8'd116: qsin = 9'd167;
            8'd117: qsin = 9'd168;
            8'd118: qsin = 9'd170;
            8'd119: qsin = 9'd171;
            8'd120: qsin = 9'd172;
            8'd121: qsin = 9'd173;
            8'd122: qsin = 9'd174;
            8'd123: qsin = 9'd175;
            8'd124: qsin = 9'd177;
            8'd125: qsin = 9'd178;
            8'd126: qsin = 9'd179;
            8'd127: qsin = 9'd180;
            8'd128: qsin = 9'd181;
            8'd129: qsin = 9'd182;
            8'd130: qsin = 9'd183;
            8'd131: qsin = 9'd184;
            8'd132: qsin = 9'd185;
            8'd133: qsin = 9'd186;
            8'd134: qsin = 9'd188;
            8'd135: qsin = 9'd189;
            8'd136: qsin = 9'd190;
            8'd137: qsin = 9'd191;
            8'd138: qsin = 9'd192;
            8'd139: qsin = 9'd193;
            8'd140: qsin = 9'd194;
            8'd141: qsin = 9'd195;
            8'd142: qsin = 9'd196;
            8'd143: qsin = 9'd197;
            8'd144: qsin = 9'd198;
            8'd145: qsin = 9'd199;
            8'd146: qsin = 9'd200;
            8'd147: qsin = 9'd201;
            8'd148: qsin = 9'd202;
            8'd149: qsin = 9'd203;
            8'd150: qsin = 9'd204;
            8'd151: qsin = 9'd205;
            8'd152: qsin = 9'd206;
            8'd153: qsin = 9'd207;
            8'd154: qsin = 9'd207;
            8'd155: qsin = 9'd208;
            8'd156: qsin = 9'd209;
            8'd157: qsin = 9'd210;
            8'd158: qsin = 9'd211;
            8'd159: qsin = 9'd212;
            8'd160: qsin = 9'd213;
            8'd161: qsin = 9'd214;
            8'd162: qsin = 9'd215;
            8'd163: qsin = 9'd215;
            8'd164: qsin = 9'd216;
            8'd165: qsin = 9'd217;
            8'd166: qsin = 9'd218;
            8'd167: qsin = 9'd219;
            8'd168: qsin = 9'd220;
            8'd169: qsin = 9'd220;
            8'd170: qsin = 9'd221;
            8'd171: qsin = 9'd222;
            8'd172: qsin = 9'd223;
            8'd173: qsin = 9'd224;
            8'd174: qsin = 9'd224;
            8'd175: qsin = 9'd225;
            8'd176: qsin = 9'd226;
            8'd177: qsin = 9'd227;
            8'd178: qsin = 9'd227;
            8'd179: qsin = 9'd228;
            8'd180: qsin = 9'd229;
            8'd181: qsin = 9'd229;
            8'd182: qsin = 9'd230;
            8'd183: qsin = 9'd231;
            8'd184: qsin = 9'd231;
            8'd185: qsin = 9'd232;
            8'd186: qsin = 9'd233;
            8'd187: qsin = 9'd233;
            8'd188: qsin = 9'd234;
            8'd189: qsin = 9'd235;
            8'd190: qsin = 9'd235;
            8'd191: qsin = 9'd236;
            8'd192: qsin = 9'd237;
            8'd193: qsin = 9'd237;
            8'd194: qsin = 9'd238;
            8'd195: qsin = 9'd238;
            8'd196: qsin = 9'd239;
            8'd197: qsin = 9'd239;
            8'd198: qsin = 9'd240;
            8'd199: qsin = 9'd241;
            8'd200: qsin = 9'd241;
            8'd201: qsin = 9'd242;
            8'd202: qsin = 9'd242;
            8'd203: qsin = 9'd243;
            8'd204: qsin = 9'd243;
            8'd205: qsin = 9'd244;
            8'd206: qsin = 9'd244;
            8'd207: qsin = 9'd245;
            8'd208: qsin = 9'd245;
            8'd209: qsin = 9'd245;
            8'd210: qsin = 9'd246;
            8'd211: qsin = 9'd246;
            8'd212: qsin = 9'd247;
            8'd213: qsin = 9'd247;
            8'd214: qsin = 9'd248;
            8'd215: qsin = 9'd248;
            8'd216: qsin = 9'd248;
            8'd217: qsin = 9'd249;
            8'd218: qsin = 9'd249;
            8'd219: qsin = 9'd249;
            8'd220: qsin = 9'd250;
            8'd221: qsin = 9'd250;
            8'd222: qsin = 9'd250;
            8'd223: qsin = 9'd251;
            8'd224: qsin = 9'd251;
            8'd225: qsin = 9'd251;
            8'd226: qsin = 9'd252;
            8'd227: qsin = 9'd252;
            8'd228: qsin = 9'd252;
            8'd229: qsin = 9'd252;
            8'd230: qsin = 9'd253;
            8'd231: qsin = 9'd253;
            8'd232: qsin = 9'd253;
            8'd233: qsin = 9'd253;
            8'd234: qsin = 9'd254;
            8'd235: qsin = 9'd254;
            8'd236: qsin = 9'd254;
            8'd237: qsin = 9'd254;
            8'd238: qsin = 9'd254;
            8'd239: qsin = 9'd255;
            8'd240: qsin = 9'd255;
            8'd241: qsin = 9'd255;
            8'd242: qsin = 9'd255;
            8'd243: qsin = 9'd255;
            8'd244: qsin = 9'd255;
            8'd245: qsin = 9'd255;
            8'd246: qsin = 9'd256;
            8'd247: qsin = 9'd256;
            8'd248: qsin = 9'd256;
            8'd249: qsin = 9'd256;
            8'd250: qsin = 9'd256;
            8'd251: qsin = 9'd256;
            8'd252: qsin = 9'd256;
            8'd253: qsin = 9'd256;
            8'd254: qsin = 9'd256;
            8'd255: qsin = 9'd256;
            default: qsin = 9'd0;
        endcase
    endfunction

    wire [25:0] ph_full = pos * wfreq(pair);
    wire [19:0] ph      = ph_full[19:0];
    wire [20:0] ph_r    = {1'b0, ph} + 21'd512;
    wire [9:0]  a       = ph_r[19:10];
    wire [1:0]  quad    = a[9:8];
    wire [7:0]  k       = a[7:0];
    wire [8:0]  s0      = qsin(k);
    wire [8:0]  c0      = (k == 8'd0) ? 9'd256 : qsin(8'd0 - k);   // sin(256 - k)
    reg  signed [9:0] c_n, s_n;
    always @(*) begin
        case (quad)
            2'd0: begin c_n =  $signed({1'b0, c0}); s_n =  $signed({1'b0, s0}); end
            2'd1: begin c_n = -$signed({1'b0, s0}); s_n =  $signed({1'b0, c0}); end
            2'd2: begin c_n = -$signed({1'b0, c0}); s_n = -$signed({1'b0, s0}); end
            default: begin c_n = $signed({1'b0, s0}); s_n = -$signed({1'b0, c0}); end
        endcase
    end

    reg signed [7:0] x_r;
    reg signed [9:0] c_r, s_r;
    always @(posedge clk)
        if (we_even) begin
            x_r <= wdata;
            c_r <= c_n;
            s_r <= s_n;
        end

    wire signed [7:0]  y  = wdata;
    wire signed [18:0] v1 = x_r * c_r - y * s_r + 19'sd128;
    wire signed [18:0] v2 = x_r * s_r + y * c_r + 19'sd128;
    wire signed [10:0] t1 = v1[18:8];
    wire signed [10:0] t2 = v2[18:8];
    assign xo = (t1 > 11'sd127) ? 8'h7F : (t1 < -11'sd128) ? 8'h80 : t1[7:0];
    assign yo = (t2 > 11'sd127) ? 8'h7F : (t2 < -11'sd128) ? 8'h80 : t2[7:0];
endmodule
`default_nettype wire
