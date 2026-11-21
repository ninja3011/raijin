// Raijin heavy-hitter tracker (feature 15): per-slot attention-weight
// accumulators for KV-cache eviction (H2O-style), with an on-chip argmin.
//
// During a single-row decode step the core passes every key's score s (Q.4
// fixed point, base-2 units) and its physical cache slot. When the step's keys
// have all streamed, the final running maximum m of that row is known and the
// tracker walks slots 0..NK-1, one per cycle, adding each captured key's
// max-normalised weight
//     w = 2^((s - 16 m) / 16)        in Q0.8, i.e. 256 for the top key,
// approximated as (frac >= 1/2 ? 181 : 256) >> floor((16 m - s) / 16)
// (181 = 256 / sqrt 2), zero once it falls below 2^-8. Accumulators saturate
// at 16 bits. The same walk finds the slot with the smallest accumulator among
// the slots that took part, skipping the first SKIP slots (attention sinks):
// that is the eviction candidate the host reads back.
//
// Bus (region 7, offsets):
//   0x000 + 2k, +1  ACC[k] low / high byte (read); any write clears ACC[k]
//   0x080  HHCTL   bit0 enable (kept), bit1 clear all, bit2 halve all (decay)
//   0x081  HHMIN   slot with the smallest ACC after the last step (read)
//   0x082  HHSKIP  slots 0..HHSKIP-1 never chosen by HHMIN (read/write)
//   0x083  HHRECENT keys among the newest HHRECENT positions of the step are
//                  never chosen either (H2O keeps a recent window) (read/write)
`default_nettype none
module raijin_hh #(
    parameter NK = 32,
    parameter LK = 5
) (
    input  wire              clk,
    input  wire              rst_n,
    // capture (core, during the decode stream)
    input  wire              cap_valid,
    input  wire [LK-1:0]     cap_slot,
    input  wire signed [11:0] cap_s,
    input  wire              cap_recent,     // key is in the step's recent window
    input  wire              cap_clear,      // start of a decode step
    // walk
    input  wire              go,             // all keys of the step have passed stage A
    input  wire signed [8:0] m_fin,
    output wire              busy,
    // bus
    input  wire              bus_we,
    input  wire [7:0]        bus_off,
    input  wire [7:0]        bus_wdata,
    output reg  [7:0]        bus_rdata,      // combinational from bus_off
    output wire              en,
    output wire [LK-1:0]     recent
);
    reg              en_r;
    reg  [LK-1:0]    skip;
    reg  [LK-1:0]    recent_r;
    reg  [NK-1:0]    rec;
    reg  [LK-1:0]    minslot;
    reg  [15:0]      acc [0:NK-1];
    reg signed [11:0] sc [0:NK-1];
    reg  [NK-1:0]    used;
    reg              walking;
    reg  [LK:0]      k;
    reg  [16:0]      best;
    reg  [LK-1:0]    best_k;
    reg signed [8:0] m_l;
    assign en   = en_r;
    assign recent = recent_r;
    assign busy = walking;

    wire [LK-1:0] kk     = k[LK-1:0];
    wire signed [13:0] d = ($signed({m_l[8], m_l}) <<< 4) - $signed({{2{sc[kk][11]}}, sc[kk]});
    wire [9:0]    di     = d[13] ? 10'd0 : d[13:4];
    wire [8:0]    w0     = (d[13] || !d[3]) ? 9'd256 : 9'd181;
    wire [8:0]    w      = (di > 10'd8) ? 9'd0 : (w0 >> di[3:0]);
    wire [16:0]   sum    = {1'b0, acc[kk]} + {8'd0, w};
    wire [15:0]   nacc   = sum[16] ? 16'hFFFF : sum[15:0];

    wire          ctl_w  = bus_we && bus_off == 8'h80;
    wire          acc_w  = bus_we && !bus_off[7];
    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            en_r    <= 1'b0;
            skip    <= {LK{1'b0}};
            recent_r <= {LK{1'b0}};
            minslot <= {LK{1'b0}};
            walking <= 1'b0;
            used    <= {NK{1'b0}};
            k       <= 0;
            for (i = 0; i < NK; i = i + 1) acc[i] <= 16'd0;
        end else begin
            if (ctl_w) begin
                en_r <= bus_wdata[0];
                for (i = 0; i < NK; i = i + 1) begin
                    if (bus_wdata[1])      acc[i] <= 16'd0;
                    else if (bus_wdata[2]) acc[i] <= acc[i] >> 1;
                end
            end
            if (bus_we && bus_off == 8'h82) skip <= bus_wdata[LK-1:0];
            if (bus_we && bus_off == 8'h83) recent_r <= bus_wdata[LK-1:0];
            if (acc_w) acc[bus_off[LK:1]] <= 16'd0;
            if (cap_clear) used <= {NK{1'b0}};
            if (en_r && cap_valid) begin
                used[cap_slot] <= 1'b1;
                rec[cap_slot]  <= cap_recent;
            end
            if (go && en_r && !walking) begin
                walking <= 1'b1;
                k       <= 0;
                m_l     <= m_fin;
                best    <= 17'h1FFFF;
                best_k  <= {LK{1'b0}};
            end else if (walking) begin
                if (used[kk]) begin
                    acc[kk] <= nacc;
                    if (kk >= skip && !rec[kk] && {1'b0, nacc} < best) begin
                        best   <= {1'b0, nacc};
                        best_k <= kk;
                    end
                end
                k <= k + 1'b1;
                if (k == NK - 1) begin
                    walking <= 1'b0;
                    minslot <= (kk >= skip && used[kk] && !rec[kk] && {1'b0, nacc} < best) ? kk : best_k;
                end
            end
        end
    end
    always @(posedge clk) begin
        if (en_r && cap_valid) sc[cap_slot] <= cap_s;
    end

    wire [15:0] acc_rd = acc[bus_off[LK:1]];
    always @(*) begin
        if (!bus_off[7])            bus_rdata = bus_off[0] ? acc_rd[15:8] : acc_rd[7:0];
        else if (bus_off == 8'h80)  bus_rdata = {7'd0, en_r};
        else if (bus_off == 8'h81)  bus_rdata = {{(8-LK){1'b0}}, minslot};
        else if (bus_off == 8'h82)  bus_rdata = {{(8-LK){1'b0}}, skip};
        else if (bus_off == 8'h83)  bus_rdata = {{(8-LK){1'b0}}, recent_r};
        else                        bus_rdata = 8'd0;
    end
endmodule
`default_nettype wire
