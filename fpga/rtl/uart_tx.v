// 8N1 UART transmitter.
`default_nettype none
module uart_tx #(
    parameter CLKS_PER_BIT = 434
) (
    input  wire       clk,
    input  wire       rst,
    input  wire       start,
    input  wire [7:0] data,
    output reg        tx,
    output wire       busy
);
    localparam CW = $clog2(CLKS_PER_BIT + 1);
    reg [9:0]    sh;
    reg [3:0]    nbits;
    reg [CW-1:0] cnt;
    assign busy = (nbits != 0);

    always @(posedge clk) begin
        if (rst) begin
            tx    <= 1'b1;
            nbits <= 0;
            cnt   <= 0;
        end else if (nbits == 0) begin
            tx <= 1'b1;
            if (start) begin
                sh    <= {1'b1, data, 1'b0};
                nbits <= 4'd11;
                cnt   <= 0;
            end
        end else if (cnt == 0 && nbits == 4'd1) begin
            nbits <= 0;                      // stop bit has been held a full bit time
        end else if (cnt == 0) begin
            tx    <= sh[0];
            sh    <= {1'b1, sh[9:1]};
            nbits <= nbits - 1'b1;
            cnt   <= CLKS_PER_BIT - 1;
        end else begin
            cnt <= cnt - 1'b1;
        end
    end
endmodule
`default_nettype wire
