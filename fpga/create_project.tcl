# Create the Raijin Vivado project for the Arty A7-100T.
# Run from the repo root:   vivado -mode batch -source fpga/create_project.tcl
# Then open fpga/vivado/raijin_arty.xpr in the GUI, or run fpga/build.tcl.
set root [file normalize [file join [file dirname [info script]] ..]]
set proj_dir [file join $root fpga vivado]

create_project raijin_arty $proj_dir -part xc7a100tcsg324-1 -force
set_property target_language Verilog [current_project]

add_files -fileset sources_1 [glob [file join $root rtl *.v]]
add_files -fileset sources_1 [glob [file join $root fpga rtl *.v]]
add_files -fileset constrs_1 [file join $root fpga constraints arty_a7_100t.xdc]
set_property top raijin_arty_top [get_filesets sources_1]

# Simulation: the core-level cocotb tests live in tb/; this adds nothing extra.
update_compile_order -fileset sources_1

set_property strategy Flow_PerfOptimized_high [get_runs synth_1]
set_property strategy Performance_Explore      [get_runs impl_1]
puts "Created $proj_dir/raijin_arty.xpr"
