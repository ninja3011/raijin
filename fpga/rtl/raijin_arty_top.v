// Arty A7-100T top: 100 MHz board clock -> 50 MHz via MMCM, UART bridge, core.
//   led[0] busy, led[1] done, led[2] heartbeat, led[3] MMCM locked
//   ck_rst (red RESET button, active low) resets everything.
`default_nettype none
module raijin_arty_top #(
    parameter CLKS_PER_BIT = 434,   // 50 MHz / 115200 baud
    parameter N = 16,
    parameter D = 16
) (
    input  wire       CLK100MHZ,
    input  wire       ck_rst,
    input  wire       uart_txd_in,   // host -> FPGA
    output wire       uart_rxd_out,  // FPGA -> host
    output wire [3:0] led
);
    // ---------------- clocking ----------------
    wire clk_fb, clk50_unbuf, clk50, locked;
    MMCME2_BASE #(
        .CLKIN1_PERIOD(10.0),
        .CLKFBOUT_MULT_F(10.0),    // VCO 1000 MHz
        .CLKOUT0_DIVIDE_F(20.0),   // 50 MHz
        .DIVCLK_DIVIDE(1)
    ) u_mmcm (
        .CLKIN1(CLK100MHZ), .CLKFBIN(clk_fb), .CLKFBOUT(clk_fb),
        .CLKOUT0(clk50_unbuf), .LOCKED(locked),
        .RST(1'b0), .PWRDWN(1'b0),
        .CLKOUT0B(), .CLKOUT1(), .CLKOUT1B(), .CLKOUT2(), .CLKOUT2B(),
        .CLKOUT3(), .CLKOUT3B(), .CLKOUT4(), .CLKOUT5(), .CLKOUT6(), .CLKFBOUTB()
    );
    BUFG u_bufg (.I(clk50_unbuf), .O(clk50));

    // ---------------- reset synchroniser ----------------
    reg [3:0] rst_sh = 4'hF;
    always @(posedge clk50 or negedge locked) begin
        if (!locked) rst_sh <= 4'hF;
        else         rst_sh <= {rst_sh[2:0], ~ck_rst};
    end
    wire rst = rst_sh[3];

    // ---------------- UART + bridge + core ----------------
    wire       rx_valid;
    wire [7:0] rx_data;
    wire       tx_start, tx_busy;
    wire [7:0] tx_data;
    wire [15:0] addr;
    wire [7:0]  wdata, rdata;
    wire        we, re, busy, done;

    uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_rx (
        .clk(clk50), .rst(rst), .rx(uart_txd_in), .valid(rx_valid), .data(rx_data));
    uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_tx (
        .clk(clk50), .rst(rst), .start(tx_start), .data(tx_data), .tx(uart_rxd_out), .busy(tx_busy));
    raijin_bridge u_br (
        .clk(clk50), .rst(rst), .rx_valid(rx_valid), .rx_data(rx_data),
        .tx_start(tx_start), .tx_data(tx_data), .tx_busy(tx_busy),
        .addr(addr), .wdata(wdata), .we(we), .re(re), .rdata(rdata));
    raijin_core #(.N(N), .D(D)) u_core (
        .clk(clk50), .rst_n(~rst), .addr(addr), .wdata(wdata), .we(we), .re(re),
        .rdata(rdata), .busy(busy), .done(done));

    reg [24:0] hb;
    always @(posedge clk50) hb <= hb + 1'b1;
    assign led = {locked, hb[24], done, busy};
endmodule
`default_nettype wire
