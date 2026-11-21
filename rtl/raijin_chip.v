// Raijin chip top (feature 11): SPI host port, idle clock gating, and the core.
//
// Pins: clk, rst_n, sck, cs_n, io[3:0] (bidirectional: io_in / io_out / io_oe
// to the pads; io[0] MOSI and io[1] MISO in single-lane SPI, all four in QSPI),
// done (IRQ), busy. The SPI target runs on the free-running clock; the core runs on a
// gated clock that is on only while a pass runs and for HOLD cycles after each
// bus access the SPI target makes, or after the pass ends. HOLD covers the
// core's two-cycle write and three-cycle read latency, and lets done settle.
// An SPI byte takes 64 clocks at SCK = f_clk / 8, so even during host traffic
// the core clock runs about 8 of every 64 cycles, and between tokens it stops.
// This removes the clock-tree and flop power of the whole datapath for that
// time. The gate is a sky130 integrated clock gate under
// RAIJIN_SKY130_CELLS, and a behavioural latch-and-AND otherwise.
// GATE_EN = 0 ties the gate on (for A/B comparisons).
`default_nettype none
module raijin_chip #(
    parameter N = 16,
    parameter D = 16,
    parameter P = 4,
    parameter GATE_EN = 1,
    parameter HOLD = 7
) (
    input  wire clk,
    input  wire rst_n,
    input  wire sck,
    input  wire cs_n,
    input  wire [3:0] io_in,        // io[0] MOSI, io[1] MISO, io[3:0] quad data
    output wire [3:0] io_out,
    output wire [3:0] io_oe,
    output wire done,
    output wire busy,
    output reg  [31:0] gated_cycles     // free-running count of cycles the core clock was off
);
    wire [15:0] addr;
    wire [7:0]  wdata, rdata;
    wire        we, re, active;

    // ---- reset synchroniser: rst_n asserts asynchronously and is released
    // two clk edges later, in step with clk, so every flop leaves reset in the
    // same cycle and the pin needs no timing (raijin_chip.sdc false-paths it) ----
    reg [1:0] rst_s;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) rst_s <= 2'b00;
        else        rst_s <= {rst_s[0], 1'b1};
    end
    wire srst_n = rst_s[1];

    raijin_spi u_spi (
        .clk(clk), .rst_n(srst_n), .sck(sck), .cs_n(cs_n),
        .io_in(io_in), .io_out(io_out), .io_oe(io_oe), .active(active),
        .addr(addr), .wdata(wdata), .we(we), .re(re), .rdata(rdata));

    // ---- clock gate enable ----
    reg [4:0] hold;
    always @(posedge clk or negedge srst_n) begin
        if (!srst_n)                hold <= HOLD;
        else if (we || re || busy)  hold <= HOLD;
        else if (hold != 5'd0)      hold <= hold - 5'd1;
    end
    wire cg_en = (GATE_EN == 0) || we || re || busy || (hold != 5'd0);

    always @(posedge clk or negedge srst_n) begin
        if (!srst_n)     gated_cycles <= 32'd0;
        else if (!cg_en) gated_cycles <= gated_cycles + 32'd1;
    end

    wire gclk;
`ifdef RAIJIN_SKY130_CELLS
    (* keep *) sky130_fd_sc_hd__dlclkp_1 u_icg (.CLK(clk), .GATE(cg_en), .GCLK(gclk));
`else
    reg cg_l;
    always @(clk or cg_en)
        if (!clk) cg_l = cg_en;
    assign gclk = clk & cg_l;
`endif

    raijin_core #(.N(N), .D(D), .P(P)) u_core (
        .clk(gclk), .rst_n(srst_n), .addr(addr), .wdata(wdata), .we(we), .re(re),
        .rdata(rdata), .busy(busy), .done(done));
endmodule
`default_nettype wire
