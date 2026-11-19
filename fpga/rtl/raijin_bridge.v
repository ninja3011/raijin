// UART command bridge: turns host bytes into raijin_core bus reads and writes.
//
// Commands (all addresses big endian):
//   'W' AH AL D            write one byte
//   'R' AH AL              read one byte          -> 1 byte reply
//   'B' AH AL LEN D0..Dn   burst write, LEN+1 bytes, address auto-increments
//   'X' AH AL LEN          burst read,  LEN+1 bytes reply
//   'P'                    ping                   -> 'K'
`default_nettype none
module raijin_bridge (
    input  wire        clk,
    input  wire        rst,
    // UART side
    input  wire        rx_valid,
    input  wire [7:0]  rx_data,
    output reg         tx_start,
    output reg  [7:0]  tx_data,
    input  wire        tx_busy,
    // core bus
    output reg  [15:0] addr,
    output reg  [7:0]  wdata,
    output reg         we,
    output reg         re,
    input  wire [7:0]  rdata
);
    localparam S_CMD = 4'd0, S_AH = 4'd1, S_AL = 4'd2, S_LEN = 4'd3,
               S_DATA = 4'd4, S_RD = 4'd5, S_RDWAIT = 4'd6, S_TX = 4'd7, S_RDCAP = 4'd8;
    reg [3:0] st;
    reg [7:0] cmd;
    reg [8:0] left;      // bytes remaining in a burst, minus one
    reg       tx_wait;   // one-cycle guard so busy is seen after start

    always @(posedge clk) begin
        if (rst) begin
            st       <= S_CMD;
            we       <= 1'b0;
            re       <= 1'b0;
            tx_start <= 1'b0;
            tx_wait  <= 1'b0;
            addr     <= 16'd0;
            left     <= 9'd0;
        end else begin
            we       <= 1'b0;
            re       <= 1'b0;
            tx_start <= 1'b0;
            tx_wait  <= 1'b0;
            case (st)
                S_CMD: if (rx_valid) begin
                    cmd <= rx_data;
                    if (rx_data == "P") begin
                        tx_data <= "K";
                        left    <= 9'd0;
                        st      <= S_TX;
                    end else if (rx_data == "W" || rx_data == "R" || rx_data == "B" || rx_data == "X")
                        st <= S_AH;
                end
                S_AH: if (rx_valid) begin
                    addr[15:8] <= rx_data;
                    st <= S_AL;
                end
                S_AL: if (rx_valid) begin
                    addr[7:0] <= rx_data;
                    left      <= 9'd0;
                    if (cmd == "W")      st <= S_DATA;
                    else if (cmd == "R") st <= S_RD;
                    else                 st <= S_LEN;
                end
                S_LEN: if (rx_valid) begin
                    left <= {1'b0, rx_data};
                    st   <= (cmd == "B") ? S_DATA : S_RD;
                end
                S_DATA: if (rx_valid) begin
                    wdata <= rx_data;
                    we    <= 1'b1;
                    if (left == 0) st <= S_CMD;
                    else           left <= left - 1'b1;
                end
                S_RD: begin
                    re <= 1'b1;
                    st <= S_RDWAIT;
                end
                S_RDWAIT: st <= S_RDCAP;       // core samples re this cycle
                S_RDCAP: begin
                    tx_data <= rdata;
                    st      <= S_TX;
                end
                S_TX: if (!tx_busy && !tx_wait && !tx_start) begin
                    tx_start <= 1'b1;
                    tx_wait  <= 1'b1;
                    if (cmd == "P" || left == 0) st <= S_CMD;
                    else begin
                        left <= left - 1'b1;
                        addr <= addr + 1'b1;
                        st   <= S_RD;
                    end
                end
            endcase
            // burst writes advance the address after each byte
            if (we && cmd == "B") addr <= addr + 1'b1;
        end
    end
endmodule
`default_nettype wire
