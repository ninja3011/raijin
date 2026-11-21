// Cycle-accurate behavioural model of sky130_sram_1kbyte_1rw1r_32x256_8
// (OpenRAM, 256 words x 32 bits, byte write mask, one RW port and one R port).
//
// Port semantics match the macro at the clock-edge level: inputs are sampled
// on the rising edge; read data is valid before the next rising edge.
// The core synthesises with RAIJIN_SKY130_SRAM defined, which swaps this model
// for the real macro; the FPGA build keeps this model and infers block RAM.
// Never read and write the same address in the same cycle (the macro forbids it).
`default_nettype none
module raijin_mem (
    input  wire        clk0,
    input  wire        csb0,      // active-low select, port 0 (RW)
    input  wire        web0,      // active-low write enable
    input  wire [3:0]  wmask0,
    input  wire [7:0]  addr0,
    input  wire [31:0] din0,
    output reg  [31:0] dout0,
    input  wire        clk1,
    input  wire        csb1,      // active-low select, port 1 (R)
    input  wire [7:0]  addr1,
    output reg  [31:0] dout1
);
    reg [31:0] mem [0:255];
    integer bi;
    always @(posedge clk0) begin
        if (!csb0) begin
            if (!web0) begin
                for (bi = 0; bi < 4; bi = bi + 1)
                    if (wmask0[bi]) mem[addr0][bi*8 +: 8] <= din0[bi*8 +: 8];
            end else begin
                dout0 <= mem[addr0];
            end
        end
    end
    always @(posedge clk1) begin
        if (!csb1) dout1 <= mem[addr1];
    end
`ifndef SYNTHESIS
    always @(posedge clk0)
        if (!csb0 && !web0 && !csb1 && addr0 == addr1)
            $display("%t %m ERROR: write and read of address %0d in the same cycle", $time, addr0);
`endif
endmodule
`default_nettype wire
