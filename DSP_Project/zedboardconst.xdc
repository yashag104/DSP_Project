## ZedBoard Pin Constraints for Conv1D Accelerator


## Clock Signal (100 MHz)
set_property PACKAGE_PIN Y9 [get_ports GCLK]
set_property IOSTANDARD LVCMOS33 [get_ports GCLK]
create_clock -period 10.000 -name sys_clk_pin -waveform {0.000 5.000} -add [get_ports GCLK]


## Push Buttons (Active High)

set_property PACKAGE_PIN P16 [get_ports {BTN[0]}];  # BTNC - Center (Start)
set_property PACKAGE_PIN R16 [get_ports {BTN[1]}];  # BTND - Down (Reset)
set_property PACKAGE_PIN N15 [get_ports {BTN[2]}];  # BTNL - Left
set_property PACKAGE_PIN R18 [get_ports {BTN[3]}];  # BTNR - Right
set_property PACKAGE_PIN T18 [get_ports {BTN[4]}];  # BTNU - Up

set_property IOSTANDARD LVCMOS33 [get_ports {BTN[*]}]
set_property PULLDOWN true [get_ports {BTN[*]}]


## Slide Switches

set_property PACKAGE_PIN F22 [get_ports {SW[0]}];   # SW0 - Auto-start enable
set_property PACKAGE_PIN G22 [get_ports {SW[1]}];   # SW1
set_property PACKAGE_PIN H22 [get_ports {SW[2]}];   # SW2
set_property PACKAGE_PIN F21 [get_ports {SW[3]}];   # SW3
set_property PACKAGE_PIN H19 [get_ports {SW[4]}];   # SW4
set_property PACKAGE_PIN H18 [get_ports {SW[5]}];   # SW5
set_property PACKAGE_PIN H17 [get_ports {SW[6]}];   # SW6
set_property PACKAGE_PIN M15 [get_ports {SW[7]}];   # SW7 - Continuous mode

set_property IOSTANDARD LVCMOS33 [get_ports {SW[*]}]

## ============================================================================
## LEDs
## ============================================================================
set_property PACKAGE_PIN T22 [get_ports {LED[0]}];  # LD0 - State bit 0
set_property PACKAGE_PIN T21 [get_ports {LED[1]}];  # LD1 - State bit 1
set_property PACKAGE_PIN U22 [get_ports {LED[2]}];  # LD2 - Valid signal
set_property PACKAGE_PIN U21 [get_ports {LED[3]}];  # LD3 - Reset status
set_property PACKAGE_PIN V22 [get_ports {LED[4]}];  # LD4 - Output bit 4
set_property PACKAGE_PIN W22 [get_ports {LED[5]}];  # LD5 - Output bit 5
set_property PACKAGE_PIN U19 [get_ports {LED[6]}];  # LD6 - Output bit 6
set_property PACKAGE_PIN U14 [get_ports {LED[7]}];  # LD7 - Output bit 7

set_property IOSTANDARD LVCMOS33 [get_ports {LED[*]}]
set_property DRIVE 12 [get_ports {LED[*]}]
set_property SLEW SLOW [get_ports {LED[*]}]

## ============================================================================
## Timing Constraints
## ============================================================================
set_property CLOCK_DEDICATED_ROUTE FALSE [get_nets GCLK_IBUF]

## Input Delay Constraints
set_input_delay -clock [get_clocks sys_clk_pin] -min 0.000 [get_ports {BTN[*]}]
set_input_delay -clock [get_clocks sys_clk_pin] -max 2.000 [get_ports {BTN[*]}]
set_input_delay -clock [get_clocks sys_clk_pin] -min 0.000 [get_ports {SW[*]}]
set_input_delay -clock [get_clocks sys_clk_pin] -max 2.000 [get_ports {SW[*]}]

## Output Delay Constraints
set_output_delay -clock [get_clocks sys_clk_pin] -min -1.000 [get_ports {LED[*]}]
set_output_delay -clock [get_clocks sys_clk_pin] -max 2.000 [get_ports {LED[*]}]

## ============================================================================
## Additional Timing
## ============================================================================
set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]
set_property BITSTREAM.CONFIG.CONFIGRATE 50 [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]
set_property CFGBVS VCCO [current_design]
