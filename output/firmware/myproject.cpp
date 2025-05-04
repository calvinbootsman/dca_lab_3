#include <iostream>

#include "myproject.h"
#include "parameters.h"


void myproject(
    hls::stream<input_t> &input_1,
    hls::stream<result_t> &layer13_out
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=input_1,layer13_out 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<_0_weight_t, 200704>(w10, "w10.txt");
        nnet::load_weights_from_txt<_0_bias_t, 256>(b10, "b10.txt");
        nnet::load_weights_from_txt<_3_weight_t, 32768>(w11, "w11.txt");
        nnet::load_weights_from_txt<_3_bias_t, 128>(b11, "b11.txt");
        nnet::load_weights_from_txt<_6_weight_t, 8192>(w12, "w12.txt");
        nnet::load_weights_from_txt<_6_bias_t, 64>(b12, "b12.txt");
        nnet::load_weights_from_txt<_9_weight_t, 640>(w13, "w13.txt");
        nnet::load_weights_from_txt<_9_bias_t, 10>(b13, "b13.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    hls::stream<layer9_t> layer9_out("layer9_out");
    #pragma HLS STREAM variable=layer9_out depth=784
    nnet::transpose<input_t, layer9_t, config9>(input_1, layer9_out); // transpose_input_for_input_1

    hls::stream<_0_result_t> layer10_out("layer10_out");
    #pragma HLS STREAM variable=layer10_out depth=256
    nnet::pointwise_conv_1d_cl<layer9_t, _0_result_t, config14>(layer9_out, layer10_out, w10, b10); // _0

    hls::stream<layer3_t> layer3_out("layer3_out");
    #pragma HLS STREAM variable=layer3_out depth=256
    nnet::relu<_0_result_t, layer3_t, relu_config3>(layer10_out, layer3_out); // _1

    hls::stream<_3_result_t> layer11_out("layer11_out");
    #pragma HLS STREAM variable=layer11_out depth=128
    nnet::pointwise_conv_1d_cl<layer3_t, _3_result_t, config15>(layer3_out, layer11_out, w11, b11); // _3

    hls::stream<layer5_t> layer5_out("layer5_out");
    #pragma HLS STREAM variable=layer5_out depth=128
    nnet::relu<_3_result_t, layer5_t, relu_config5>(layer11_out, layer5_out); // _4

    hls::stream<_6_result_t> layer12_out("layer12_out");
    #pragma HLS STREAM variable=layer12_out depth=64
    nnet::pointwise_conv_1d_cl<layer5_t, _6_result_t, config16>(layer5_out, layer12_out, w12, b12); // _6

    hls::stream<layer7_t> layer7_out("layer7_out");
    #pragma HLS STREAM variable=layer7_out depth=64
    nnet::relu<_6_result_t, layer7_t, relu_config7>(layer12_out, layer7_out); // _7

    nnet::pointwise_conv_1d_cl<layer7_t, result_t, config17>(layer7_out, layer13_out, w13, b13); // _9

}

