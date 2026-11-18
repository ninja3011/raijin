// Simulation top for the FPGA datapath: UART rx/tx, bridge and core, without
// the Xilinx MMCM. Used by tb/test_uart.py with a short bit time.
`default_nettype none
module raijin_uart_tb_top #(
    parameter CLKS_PER_BIT = 8
) (
    input  wire clk,
    input  wire rst,
    input  wire rx,     // host -> FPGA
    output wire tx,     // FPGA -> host
    output wire busy,
    output wire done
);
    wire       rx_valid, tx_start, tx_busy, we, re;
    wire [7:0] rx_data, tx_data, wdata, rdata;
    wire [15:0] addr;
    uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_rx (
        .clk(clk), .rst(rst), .rx(rx), .valid(rx_valid), .data(rx_data));
    uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_tx (
        .clk(clk), .rst(rst), .start(tx_start), .data(tx_data), .tx(tx), .busy(tx_busy));
    raijin_bridge u_br (
        .clk(clk), .rst(rst), .rx_valid(rx_valid), .rx_data(rx_data),
        .tx_start(tx_start), .tx_data(tx_data), .tx_busy(tx_busy),
        .addr(addr), .wdata(wdata), .we(we), .re(re), .rdata(rdata));
    raijin_core u_core (
        .clk(clk), .rst_n(~rst), .addr(addr), .wdata(wdata), .we(we), .re(re),
        .rdata(rdata), .busy(busy), .done(done));
endmodule
`default_nettype wire
