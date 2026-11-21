// Raijin SPI / QSPI target: the chip's host port for tapeout (features 11, 12).
//
// SPI mode 0 (CPOL = 0, CPHA = 0), MSB first, byte commands in the style of
// SPI SRAMs and flash, with an auto-incrementing 16-bit address. io[0] is
// MOSI and io[1] is MISO in single-lane mode; quad commands move data on
// io[3:0], high nibble first:
//   WRITE       0x02, AH, AL, data...             (all single lane)
//   READ        0x0B, AH, AL, dummy, data...      (data on io[1])
//   QUAD WRITE  0x32, AH, AL, data...             (data on io[3:0], 2 clocks per byte)
//   QUAD READ   0x6B, AH, AL, dummy, data...      (data driven on io[3:0])
// The command, address and dummy byte are always single lane, so a plain
// SPI host can use the chip and a QSPI host gets 4x the data bandwidth.
// Every data byte becomes one core bus write or read, so the whole core
// address map is reachable.
//
// SCK, CS_N and io are sampled in the core clock domain through two-flop
// synchronizers, so SCK must be at most f_clk / 8 (6.25 MHz at 50 MHz). The
// read for each data byte is issued one byte ahead, so the core's 3-cycle
// read latency is hidden.
`default_nettype none
module raijin_spi (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        sck,
    input  wire        cs_n,
    input  wire [3:0]  io_in,
    output wire [3:0]  io_out,
    output wire [3:0]  io_oe,
    output wire        active,        // a transaction is in progress
    // core bus
    output reg  [15:0] addr,
    output reg  [7:0]  wdata,
    output reg         we,
    output reg         re,
    input  wire [7:0]  rdata
);
    localparam [7:0] CMD_WRITE = 8'h02, CMD_READ = 8'h0B, CMD_QWRITE = 8'h32, CMD_QREAD = 8'h6B;
    localparam [1:0] PH_CMD = 2'd0, PH_AH = 2'd1, PH_AL = 2'd2, PH_DATA = 2'd3;

    reg [2:0] sck_s;
    reg [1:0] cs_s;
    reg [3:0] io_s0, io_s1;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sck_s <= 3'b000;
            cs_s  <= 2'b11;
            io_s0 <= 4'd0;
            io_s1 <= 4'd0;
        end else begin
            sck_s <= {sck_s[1:0], sck};
            cs_s  <= {cs_s[0], cs_n};
            io_s0 <= io_in;
            io_s1 <= io_s0;
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
    reg        dummy;          // read: the dummy byte is still to come
    reg        load;           // next falling edge starts a MISO byte
    reg  [7:0] tx, rbuf;
    reg  [2:0] rv;             // read in flight (core latency 3)
    wire       rd_cmd  = (cmd == CMD_READ) || (cmd == CMD_QREAD);
    wire       wr_cmd  = (cmd == CMD_WRITE) || (cmd == CMD_QWRITE);
    wire       quad    = (cmd == CMD_QREAD) || (cmd == CMD_QWRITE);
    // quad data phase (after the dummy byte for reads): 4 bits per clock
    wire       qdata   = quad && (phase == PH_DATA) && !dummy;
    wire [7:0] byte_in = {sh, io_s1[0]};
    wire [7:0] qbyte   = {sh[3:0], io_s1};
    wire       byte_end = qdata ? (bitcnt[0] == 1'b1) : (bitcnt == 3'd7);

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
                dummy  <= 1'b0;
                tx     <= 8'd0;
                cmd    <= 8'd0;
            end else begin
                if (rise) begin
                    sh     <= qdata ? {3'd0, io_s1} : byte_in[6:0];
                    bitcnt <= byte_end ? 3'd0 : bitcnt + 3'd1;
                    if (byte_end) begin
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
                                end else if (wr_cmd) begin
                                    wdata <= qdata ? qbyte : byte_in;
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
                        tx <= quad ? {tx[3:0], 4'd0} : {tx[6:0], 1'b0};
                    end
                end
            end
        end
    end
    // single lane: MISO on io[1]; quad read data phase: tx[7:4] on io[3:0]
    wire qout = (cmd == CMD_QREAD) && (phase == PH_DATA) && !dummy;
    assign io_out = qout ? tx[7:4] : {2'b00, tx[7], 1'b0};
    assign io_oe  = !sel ? 4'b0000 : (qout ? 4'b1111 : 4'b0010);
endmodule
`default_nettype wire
