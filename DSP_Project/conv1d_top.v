// Top-Level Module for ZedBoard Implementation
// Maps conv1d_core to physical ZedBoard pins

module conv1d_top (
    // Clock and Reset
    input wire GCLK,              // 100MHz clock from ZedBoard
    input wire [4:0] BTN,         // Buttons: BTNC, BTND, BTNL, BTNR, BTNU
    input wire [7:0] SW,          // 8 Slide switches
    
    // Outputs
    output wire [7:0] LED         // 8 LEDs
);

    // Parameters
    parameter DATA_WIDTH = 16;
    parameter KERNEL_SIZE = 3;
    
    // Internal signals
    reg rst_n;
    reg start;
    reg signed [DATA_WIDTH-1:0] data_in;
    reg data_valid;
    reg signed [DATA_WIDTH-1:0] weight0, weight1, weight2, bias;
    
    wire signed [DATA_WIDTH-1:0] conv_out;
    wire out_valid;
    wire [1:0] state_out;
    
    // Button debouncing registers
    reg [19:0] btn_counter;
    reg btn_start_sync, btn_start_debounced;
    reg btn_rst_sync, btn_rst_debounced;
    
    // Clock divider for visible LED changes (100MHz -> ~1Hz)
    reg [26:0] clk_div;
    wire clk_slow;
    assign clk_slow = clk_div[22]; // Approx 23.8Hz for testing
    
    // Test data counter
    reg [3:0] test_counter;
    reg [3:0] input_step;
    
    
    // Clock Divider
    
    always @(posedge GCLK) begin
        clk_div <= clk_div + 1;
    end
    
   
    // Button Synchronization and Debouncing
    always @(posedge GCLK) begin
        // Synchronize buttons
        btn_start_sync <= BTN[0];  // BTNC - Start
        btn_rst_sync   <= BTN[1];  // BTND - Reset
        
        // Simple debouncing
        if (btn_counter == 0) begin
            btn_start_debounced <= btn_start_sync;
            btn_rst_debounced   <= btn_rst_sync;
            btn_counter <= 20'd999999; // ~10ms at 100MHz
        end else begin
            btn_counter <= btn_counter - 1;
        end
    end
    
    //reset logic
    always @(posedge GCLK) begin
        if (btn_rst_debounced)
            rst_n <= 0;
        else
            rst_n <= 1;
    end
    
    // Control Logic - Automatic Test Sequence
    always @(posedge clk_slow or negedge rst_n) begin
        if (!rst_n) begin
            start <= 0;
            data_valid <= 0;
            data_in <= 0;
            test_counter <= 0;
            input_step <= 0;
            
            // Initialize weights
            weight0 <= 16'h0100;  // 1.0 in Q8.8
            weight1 <= 16'h0080;  // 0.5
            weight2 <= 16'h0040;  // 0.25
            bias    <= 16'h0020;  // 0.125
        end else begin
            // Automatic test sequence
            case (input_step)
                4'd0: begin
                    // Prepare first input
                    if (btn_start_debounced || SW[0]) begin
                        start <= 1;
                        data_valid <= 1;
                        data_in <= 16'h0200;  // 2.0
                        input_step <= 4'd1;
                    end
                end
                
                4'd1: begin
                    data_in <= 16'h0180;  // 1.5
                    input_step <= 4'd2;
                end
                
                4'd2: begin
                    data_in <= 16'h0100;  // 1.0
                    input_step <= 4'd3;
                end
                
                4'd3: begin
                    data_valid <= 0;
                    start <= 0;
                    input_step <= 4'd4;
                end
                
                4'd4: begin
                    // Wait for computation
                    if (test_counter < 4'd10)
                        test_counter <= test_counter + 1;
                    else begin
                        test_counter <= 0;
                        input_step <= 4'd5;
                    end
                end
                
                4'd5: begin
                    // Second test sequence
                    start <= 1;
                    data_valid <= 1;
                    data_in <= 16'h0300;  // 3.0
                    input_step <= 4'd6;
                end
                
                4'd6: begin
                    data_in <= 16'h0280;  // 2.5
                    input_step <= 4'd7;
                end
                
                4'd7: begin
                    data_in <= 16'h0200;  // 2.0
                    input_step <= 4'd8;
                end
                
                4'd8: begin
                    data_valid <= 0;
                    start <= 0;
                    input_step <= 4'd9;
                end
                
                4'd9: begin
                    // Loop back or stop
                    if (SW[7])  // Continuous mode
                        input_step <= 4'd0;
                    // else stay in idle
                end
                
                default: input_step <= 4'd0;
            endcase
        end
    end

    // LED Output Mapping
    // Display conv_out on LEDs (lower 8 bits)
    // LED[7:6] - State indicator
    // LED[5:0] - Output value (lower 6 bits)
    assign LED[1:0] = state_out;           // State on LED1:0
    assign LED[2]   = out_valid;           // Valid indicator
    assign LED[3]   = rst_n;               // Reset status
    assign LED[7:4] = conv_out[7:4];       // Upper nibble of output
    
    // ========================================================================
    // Instantiate Conv1D Core
    // ========================================================================
    conv1d_core #(
        .DATA_WIDTH(DATA_WIDTH),
        .KERNEL_SIZE(KERNEL_SIZE),
        .INPUT_CHANNELS(3),
        .OUTPUT_FILTERS(2)
    ) conv1d_inst (
        .clk(GCLK),
        .rst_n(rst_n),
        .start(start),
        .data_in(data_in),
        .data_valid(data_valid),
        .weight0(weight0),
        .weight1(weight1),
        .weight2(weight2),
        .bias(bias),
        .conv_out(conv_out),
        .out_valid(out_valid),
        .state_out(state_out)
    );

endmodule
