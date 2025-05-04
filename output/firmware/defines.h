#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 1
#define N_INPUT_2_1 784
#define OUT_HEIGHT_9 784
#define OUT_WIDTH_9 1
#define N_FILT_10 256
#define N_OUTPUTS_10 784
#define N_LAYER_2_2 256
#define N_LAYER_1_2 1
#define N_FILT_11 128
#define N_OUTPUTS_11 256
#define N_LAYER_2_4 128
#define N_LAYER_1_4 1
#define N_FILT_12 64
#define N_OUTPUTS_12 128
#define N_LAYER_2_6 64
#define N_LAYER_1_6 1
#define N_FILT_13 10
#define N_OUTPUTS_13 64


// hls-fpga-machine-learning insert layer-precision
typedef nnet::array<ap_fixed<16,6>, 784*1> input_t;
typedef nnet::array<ap_fixed<16,6>, 1*1> layer9_t;
typedef ap_fixed<33,13> _0_accum_t;
typedef nnet::array<ap_fixed<33,13>, 784*1> _0_result_t;
typedef ap_fixed<16,6> _0_weight_t;
typedef ap_fixed<16,6> _0_bias_t;
typedef nnet::array<ap_fixed<16,6>, 1*1> layer3_t;
typedef ap_fixed<18,8> _1_table_t;
typedef ap_fixed<33,13> _3_accum_t;
typedef nnet::array<ap_fixed<33,13>, 256*1> _3_result_t;
typedef ap_fixed<16,6> _3_weight_t;
typedef ap_fixed<16,6> _3_bias_t;
typedef nnet::array<ap_fixed<16,6>, 1*1> layer5_t;
typedef ap_fixed<18,8> _4_table_t;
typedef ap_fixed<33,13> _6_accum_t;
typedef nnet::array<ap_fixed<33,13>, 128*1> _6_result_t;
typedef ap_fixed<16,6> _6_weight_t;
typedef ap_fixed<16,6> _6_bias_t;
typedef nnet::array<ap_fixed<16,6>, 1*1> layer7_t;
typedef ap_fixed<18,8> _7_table_t;
typedef ap_fixed<33,13> _9_accum_t;
typedef nnet::array<ap_fixed<33,13>, 64*1> result_t;
typedef ap_fixed<16,6> _9_weight_t;
typedef ap_fixed<16,6> _9_bias_t;


#endif
