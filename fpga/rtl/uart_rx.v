// 8N1 UART receiver with a two-flop synchroniser and mid-bit sampling.
`default_nettype none
module uart_rx #(
    parameter CLKS_PER_BIT = 434      // 50 MHz / 115200
) (
    input  wire       clk,
    input  wire       rst,
    input  wire       rx,
    output reg        valid,
    output reg  [7:0] data
);
    localparam CW = $clog2(CLKS_PER_BIT + 1);
    reg rx_s1, rx_s2;
    always @(posedge clk) begin
        rx_s1 <= rx;
        rx_s2 <= rx_s1;
    end

    localparam IDLE = 2'd0, START = 2'd1, DATA = 2'd2, STOP = 2'd3;
    reg [1:0]    st;
    reg [CW-1:0] cnt;
    reg [2:0]    bitn;

    always @(posedge clk) begin
        if (rst) begin
            st    <= IDLE;
            valid <= 1'b0;
            cnt   <= 0;
            bitn  <= 0;
        end else begin
            valid <= 1'b0;
            case (st)
                IDLE: if (!rx_s2) begin
                    st  <= START;
                    cnt <= CLKS_PER_BIT / 2;
                end
                START: begin
                    if (cnt == 0) begin
                        if (!rx_s2) begin
                            st   <= DATA;
                            cnt  <= CLKS_PER_BIT - 1;
                            bitn <= 0;
                        end else st <= IDLE;
                    end else cnt <= cnt - 1'b1;
                end
                DATA: begin
                    if (cnt == 0) begin
                        data <= {rx_s2, data[7:1]};
                        cnt  <= CLKS_PER_BIT - 1;
                        if (bitn == 3'd7) st <= STOP;
                        bitn <= bitn + 1'b1;
                    end else cnt <= cnt - 1'b1;
                end
                STOP: begin
                    if (cnt == 0) begin
                        st    <= IDLE;
                        valid <= rx_s2;
                    end else cnt <= cnt - 1'b1;
                end
            endcase
        end
    end
endmodule
`default_nettype wire
