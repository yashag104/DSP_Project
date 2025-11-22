// ============================================================================
// 1D Convolution Accelerator for Financial Time-Series CNN
// Based on Multi-Timeframe Forex Prediction Model
// ============================================================================

module conv1d_core #(
    parameter DATA_WIDTH = 16,
    parameter KERNEL_SIZE = 3,
    parameter INPUT_CHANNELS = 3,    // Simplified for testing
    parameter OUTPUT_FILTERS = 2     // Simplified for testing
)(
    input wire clk,
    input wire rst_n,
    input wire start,
    
    // Input data
    input wire signed [DATA_WIDTH-1:0] data_in,
    input wire data_valid,
    
    // Pre-loaded weights and bias (simplified)
    input wire signed [DATA_WIDTH-1:0] weight0,
    input wire signed [DATA_WIDTH-1:0] weight1,
    input wire signed [DATA_WIDTH-1:0] weight2,
    input wire signed [DATA_WIDTH-1:0] bias,
    
    // Output
    output reg signed [DATA_WIDTH-1:0] conv_out,
    output reg out_valid,
    output [1:0] state_out
);

    // Internal signals
    reg signed [DATA_WIDTH-1:0] buffer [0:KERNEL_SIZE-1];
    reg signed [31:0] accumulator;
    reg [1:0] state;
    reg [3:0] counter;
    integer i;
    
    // State machine
    localparam IDLE    = 2'b00;
    localparam COMPUTE = 2'b01;
    localparam OUTPUT  = 2'b10;
    
    assign state_out = state;
    
    // Input buffer (sliding window)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < KERNEL_SIZE; i = i + 1)
                buffer[i] <= 0;
        end else if (data_valid) begin
            buffer[0] <= data_in;
            buffer[1] <= buffer[0];
            buffer[2] <= buffer[1];
        end
    end
    
    // Main computation
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            accumulator <= 0;
            conv_out <= 0;
            out_valid <= 0;
            counter <= 0;
        end else begin
            case (state)
                IDLE: begin
                    out_valid <= 0;
                    if (start && data_valid) begin
                        accumulator <= bias << 8;  // Initialize with bias (Q8.8)
                        counter <= 0;
                        state <= COMPUTE;
                    end
                end
                
                COMPUTE: begin
                    // Multiply-Accumulate (buffered samples * weights)
                    accumulator <= accumulator + 
                                   (buffer[0] * weight0) +
                                   (buffer[1] * weight1) +
                                   (buffer[2] * weight2);
                    state <= OUTPUT;
                end
                
                OUTPUT: begin
                    // Apply ReLU and output
                    if (accumulator[31] == 1'b1)
                        conv_out <= 0;  // ReLU: negative -> 0
                    else
                        conv_out <= accumulator[DATA_WIDTH+7:8];  // Quantize (Q8.8 -> Q8.0)
                    
                    out_valid <= 1;
                    state <= IDLE;
                end
                
                default: state <= IDLE;
            endcase
        end
    end
    
endmodule

// ============================================================================
// Testbench with Proper Waveform Generation
// ============================================================================
module conv1d_tb;
    parameter DATA_WIDTH = 16;
    parameter KERNEL_SIZE = 3;
    
    // Testbench signals
    reg clk;
    reg rst_n;
    reg start;
    reg signed [DATA_WIDTH-1:0] data_in;
    reg data_valid;
    reg signed [DATA_WIDTH-1:0] weight0, weight1, weight2, bias;
    
    wire signed [DATA_WIDTH-1:0] conv_out;
    wire out_valid;
    wire [1:0] state_out;
    
    // Instantiate DUT
    conv1d_core #(
        .DATA_WIDTH(DATA_WIDTH),
        .KERNEL_SIZE(KERNEL_SIZE),
        .INPUT_CHANNELS(3),
        .OUTPUT_FILTERS(2)
    ) dut (
        .clk(clk),
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
    
    // Clock generation: 10ns period (100MHz)
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    // Simulation and waveform generation
    initial begin
        // Dump file + vars
        $dumpfile("dump.vcd");
        // Dump the testbench and the DUT hierarchy
        $dumpvars(0, conv1d_tb);
        $dumpvars(0, dut);
        
        // Header
        $display("\n============================================================");
        $display("1D CONVOLUTION ACCELERATOR TESTBENCH");
        $display("============================================================");
        $display("Simulating CNN Conv1D operation");
        $display("Kernel Size: %0d | Data Width: %0d bits", KERNEL_SIZE, DATA_WIDTH);
        $display("============================================================\n");
        
        // Initialize all signals
        rst_n = 0;
        start = 0;
        data_in = 0;
        data_valid = 0;
        weight0 = 0;
        weight1 = 0;
        weight2 = 0;
        bias = 0;
        
        // Apply reset
        $display("[%0t ns] Applying Reset...", $time);
        #20;
        rst_n = 1;
        $display("[%0t ns] Reset Released\n", $time);
        
        // Load weights (kernel)
        #10;
        $display("[%0t ns] Loading Convolution Kernel:", $time);
        weight0 = 16'h0100;  // 1.0 in Q8.8 fixed-point
        weight1 = 16'h0080;  // 0.5
        weight2 = 16'h0040;  // 0.25
        bias    = 16'h0020;  // 0.125
        $display("  Weight[0] = 0x%04h (1.00)", weight0);
        $display("  Weight[1] = 0x%04h (0.50)", weight1);
        $display("  Weight[2] = 0x%04h (0.25)", weight2);
        $display("  Bias      = 0x%04h (0.125)\n", bias);
        
        // Test Case 1: First convolution
        #20;
        $display("[%0t ns] TEST CASE 1: Processing Input Sequence", $time);
        $display("------------------------------------------------------------");
        
        start = 1;
        data_valid = 1;
        data_in = 16'h0200;  // 2.0
        $display("[%0t ns] Input[0] = 0x%04h (2.00)", $time, data_in);
        
        #10;
        data_in = 16'h0180;  // 1.5
        $display("[%0t ns] Input[1] = 0x%04h (1.50)", $time, data_in);
        
        #10;
        data_in = 16'h0100;  // 1.0
        $display("[%0t ns] Input[2] = 0x%04h (1.00)", $time, data_in);
        
        #10;
        data_valid = 0;
        $display("[%0t ns] Input sequence complete\n", $time);
        
        // Wait for computation
        #50;
        
        // Test Case 2: Second convolution with different data
        $display("\n[%0t ns] TEST CASE 2: New Input Sequence", $time);
        $display("------------------------------------------------------------");
        
        start = 1;
        data_valid = 1;
        data_in = 16'h0300;  // 3.0
        $display("[%0t ns] Input[0] = 0x%04h (3.00)", $time, data_in);
        
        #10;
        data_in = 16'h0280;  // 2.5
        $display("[%0t ns] Input[1] = 0x%04h (2.50)", $time, data_in);
        
        #10;
        data_in = 16'h0200;  // 2.0
        $display("[%0t ns] Input[2] = 0x%04h (2.00)", $time, data_in);
        
        #10;
        data_valid = 0;
        start = 0;
        
        // Wait for final computation
        #100;
        
        // Summary
        $display("\n============================================================");
        $display("SIMULATION COMPLETE");
        $display("============================================================");
        $display("VCD File Generated: dump.vcd");
        $display("View waveforms with: gtkwave dump.vcd");
        $display("============================================================\n");
        
        $finish;
    end
    
    // Monitor output
    always @(posedge clk) begin
        if (out_valid) begin
            $display("\n*** CONVOLUTION OUTPUT ***");
            $display("[%0t ns] Result = 0x%04h (Decimal: %0d)", 
                     $time, conv_out, conv_out);
            case (state_out)
                2'b00: $display("State: IDLE");
                2'b01: $display("State: COMPUTE");
                2'b10: $display("State: OUTPUT");
                default: $display("State: UNKNOWN");
            endcase
            $display("Valid: %b\n", out_valid);
        end
    end
    
    // State change monitor
    always @(state_out) begin
        $display("[%0t ns] State Change: %s", $time,
                 (state_out == 2'b00) ? "IDLE" :
                 (state_out == 2'b01) ? "COMPUTE" :
                 (state_out == 2'b10) ? "OUTPUT" : "UNKNOWN");
    end
    
    // Buffer monitor (for debugging)
    always @(posedge clk) begin
        if (data_valid) begin
            $display("[%0t ns] Buffer: [0]=%04h [1]=%04h [2]=%04h", 
                     $time, dut.buffer[0], dut.buffer[1], dut.buffer[2]);
        end
    end
    
endmodule