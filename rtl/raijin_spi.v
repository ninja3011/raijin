// Raijin SPI target: the chip's host port for tapeout (feature 11).
//
// SPI mode 0 (CPOL = 0, CPHA = 0), MSB first, byte commands in the style of
// SPI SRAMs, with an auto-incrementing 16-bit address:
//   WRITE  0x02, addr[15:8], addr[7:0], data, data, ...
//   READ   0x0B, addr[15:8], addr[7:0], dummy, data, data, ...  (MISO)
// Every data byte becomes one core bus write or read, so the whole core
// address map (Q/K/V, exponents, control registers, O) is reachable.
//
// SCK, CS_N and MOSI are sampled in the core clock domain through two-flop
// synchronizers, so SCK must be at most f_clk / 8 (6.25 MHz at 50 MHz). The
// read for each data byte is issued one byte ahead, so the core's 3-cycle
// read latency is hidden behind the dummy byte and then behind each byte.
`default_nettype none
module raijin_spi (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        sck,
    input  wire        cs_n,
    input  wire        mosi,
    output wire        miso,
    output wire        miso_oe,       // drive MISO only while selected
    output wire        active,        // a transaction is in progress
    // core bus
    output reg  [15:0] addr,
    output reg  [7:0]  wdata,
    output reg         we,
    output reg         re,
    input  wire [7:0]  rdata
);
    localparam [7:0] CMD_WRITE = 8'h02, CMD_READ = 8'h0B;
    localparam [1:0] PH_CMD = 2'd0, PH_AH = 2'd1, PH_AL = 2'd2, PH_DATA = 2'd3;

    reg [2:0] sck_s;
    reg [1:0] cs_s;
    reg [1:0] mosi_s;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sck_s  <= 3'b000;
            cs_s   <= 2'b11;
            mosi_s <= 2'b00;
        end else begin
            sck_s  <= {sck_s[1:0], sck};
            cs_s   <= {cs_s[0], cs_n};
            mosi_s <= {mosi_s[0], mosi};
        end
    end
    wire sel  = !cs_s[1];
    wire rise = sel && sck_s[1] && !sck_s[2];
    wire fall = sel && !sck_s[1] && sck_s[2];
    assign active = sel;

    reg  [1:0] phase;
    reg  [2:0] bitcnt;
    reg  [6:0] sh;
    reg  [7:0] cmd;
    reg        dummy;          // READ: the dummy byte is still to come
    reg        load;           // next falling edge starts a MISO byte
    reg  [7:0] tx, rbuf;
    reg  [2:0] rv;             // read in flight (core latency 3)
    wire [7:0] byte_in = {sh, mosi_s[1]};
    wire       rd_cmd  = (cmd == CMD_READ);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            phase  <= PH_CMD;
            bitcnt <= 3'd0;
            sh     <= 7'd0;
            cmd    <= 8'd0;
            dummy  <= 1'b0;
            load   <= 1'b0;
            tx     <= 8'd0;
            rbuf   <= 8'd0;
            rv     <= 3'd0;
            addr   <= 16'd0;
            wdata  <= 8'd0;
            we     <= 1'b0;
            re     <= 1'b0;
        end else begin
            we <= 1'b0;
            re <= 1'b0;
            rv <= {rv[1:0], re};
            if (rv[2]) rbuf <= rdata;
            if (!sel) begin
                phase  <= PH_CMD;
                bitcnt <= 3'd0;
                load   <= 1'b0;
                tx     <= 8'd0;
            end else begin
                if (rise) begin
                    sh     <= byte_in[6:0];
                    bitcnt <= bitcnt + 3'd1;
                    if (bitcnt == 3'd7) begin
                        case (phase)
                            PH_CMD: begin
                                cmd   <= byte_in;
                                phase <= PH_AH;
                            end
                            PH_AH: begin
                                addr[15:8] <= byte_in;
                                phase      <= PH_AL;
                            end
                            PH_AL: begin
                                addr[7:0] <= byte_in;
                                phase     <= PH_DATA;
                                dummy     <= rd_cmd;
                                if (rd_cmd) re <= 1'b1;      // prefetch the first byte
                            end
                            default: begin
                                if (rd_cmd) begin
                                    dummy <= 1'b0;
                                    load  <= 1'b1;
                                end else if (cmd == CMD_WRITE) begin
                                    wdata <= byte_in;
                                    we    <= 1'b1;
                                end
                            end
                        endcase
                    end
                end
                // a write byte advances the address once its write is issued
                if (we) addr <= addr + 16'd1;
                if (fall) begin
                    if (load) begin
                        tx   <= rbuf;
                        load <= 1'b0;
                        addr <= addr + 16'd1;
                        re   <= 1'b1;               // prefetch the byte after
                    end else begin
                        tx <= {tx[6:0], 1'b0};
                    end
                end
            end
        end
    end
    assign miso    = tx[7];
    assign miso_oe = sel;
endmodule
`default_nettype wire
