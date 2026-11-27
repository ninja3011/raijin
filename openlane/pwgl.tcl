set run $::env(RUN)
read_liberty /foss/pdks/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib
read_liberty /work/openlane/macros/sky130_sram_1kbyte_1rw1r_32x256_8_TT_1p8V_25C.lib
read_db /work/openlane/runs/$run/final/odb/raijin_core.odb
read_spef /work/openlane/runs/$run/final/spef/nom/raijin_core.nom.spef
read_sdc /work/openlane/runs/$run/final/sdc/raijin_core.sdc
set_propagated_clock [all_clocks]
read_vcd -scope raijin_core $::env(VCD)
puts "VCDSTART"
report_activity_annotation
report_power
