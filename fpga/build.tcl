# Non-project batch build: synthesis, implementation, reports and bitstream.
# Run from the repo root:   vivado -mode batch -source fpga/build.tcl
# Outputs go to fpga/build/: raijin_arty.bit plus utilization and timing reports.
set root [file normalize [file join [file dirname [info script]] ..]]
set out  [file join $root fpga build]
file mkdir $out

set_part xc7a100tcsg324-1
read_verilog [glob [file join $root rtl *.v]]
read_verilog [glob [file join $root fpga rtl *.v]]
read_xdc [file join $root fpga constraints arty_a7_100t.xdc]

synth_design -top raijin_arty_top -part xc7a100tcsg324-1 -flatten_hierarchy rebuilt
report_utilization -file [file join $out post_synth_util.rpt]
opt_design
place_design
phys_opt_design
route_design

report_utilization -hierarchical -file [file join $out util_hier.rpt]
report_utilization                -file [file join $out util.rpt]
report_timing_summary -max_paths 20 -file [file join $out timing.rpt]
report_power                      -file [file join $out power.rpt]
write_bitstream -force [file join $out raijin_arty.bit]
puts "Bitstream: [file join $out raijin_arty.bit]"
