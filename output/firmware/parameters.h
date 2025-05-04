#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_fixed.h"
#include "ap_int.h"

#include "nnet_utils/nnet_code_gen.h"
#include "nnet_utils/nnet_helpers.h"
// hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_stream.h"
#include "nnet_utils/nnet_conv1d.h"
#include "nnet_utils/nnet_sepconv1d_stream.h"
#include "nnet_utils/nnet_transpose.h"
#include "nnet_utils/nnet_transpose_stream.h"

// hls-fpga-machine-learning insert weights
#include "weights/w10.h"
#include "weights/b10.h"
#include "weights/w11.h"
#include "weights/b11.h"
#include "weights/w12.h"
#include "weights/b12.h"
#include "weights/w13.h"
#include "weights/b13.h"


// hls-fpga-machine-learning insert layer-config
// transpose_input_for_input_1
struct config9 {
    static const unsigned dims = 2;
    static const unsigned N = 784;
    static const unsigned* const from_shape;
    static const unsigned* const to_shape;
    static const unsigned* const perm;
    static const unsigned* const perm_strides;
};

unsigned config9_from_shape[2] = {1, 784};
unsigned config9_to_shape[2] = {784, 1};
unsigned config9_perm[2] = {1, 0};
unsigned config9_perm_strides[2] = {1, 784};

const unsigned* const config9::from_shape = config9_from_shape;
const unsigned* const config9::to_shape = config9_to_shape;
const unsigned* const config9::perm = config9_perm;
const unsigned* const config9::perm_strides = config9_perm_strides;

// _0
struct config14_mult : nnet::dense_config {
    static const unsigned n_in = 1;
    static const unsigned n_out = 256;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef _0_accum_t accum_t;
    typedef _0_bias_t bias_t;
    typedef _0_weight_t weight_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config14 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 784;
    static const unsigned n_chan = 1;
    static const unsigned filt_width = 1;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 256;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = 784;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_width = 784;
    static const ap_uint<filt_width> pixels[min_width];
    static const unsigned n_partitions = 256;
    static const unsigned n_pixels = out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv1DBuffer<data_T, CONFIG_T>;
    typedef _0_accum_t accum_t;
    typedef _0_bias_t bias_t;
    typedef _0_weight_t weight_t;
    typedef config14_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_regular<K, S, W>;
    template<class data_T, class res_T, class CONFIG_T>
    using conv_kernel = nnet::Conv1DLatency<data_T, res_T, CONFIG_T>;
};
const ap_uint<config14::filt_width> config14::pixels[] = {0};

// _1
struct relu_config3 : nnet::activ_config {
    static const unsigned n_in = 256;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef _1_table_t table_t;
};

// _3
struct config15_mult : nnet::dense_config {
    static const unsigned n_in = 1;
    static const unsigned n_out = 128;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef _3_accum_t accum_t;
    typedef _3_bias_t bias_t;
    typedef _3_weight_t weight_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config15 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 256;
    static const unsigned n_chan = 1;
    static const unsigned filt_width = 1;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 128;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = 256;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_width = 256;
    static const ap_uint<filt_width> pixels[min_width];
    static const unsigned n_partitions = 128;
    static const unsigned n_pixels = out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv1DBuffer<data_T, CONFIG_T>;
    typedef _3_accum_t accum_t;
    typedef _3_bias_t bias_t;
    typedef _3_weight_t weight_t;
    typedef config15_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_regular<K, S, W>;
    template<class data_T, class res_T, class CONFIG_T>
    using conv_kernel = nnet::Conv1DLatency<data_T, res_T, CONFIG_T>;
};
const ap_uint<config15::filt_width> config15::pixels[] = {0};

// _4
struct relu_config5 : nnet::activ_config {
    static const unsigned n_in = 128;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef _4_table_t table_t;
};

// _6
struct config16_mult : nnet::dense_config {
    static const unsigned n_in = 1;
    static const unsigned n_out = 64;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef _6_accum_t accum_t;
    typedef _6_bias_t bias_t;
    typedef _6_weight_t weight_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config16 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 128;
    static const unsigned n_chan = 1;
    static const unsigned filt_width = 1;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 64;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = 128;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_width = 128;
    static const ap_uint<filt_width> pixels[min_width];
    static const unsigned n_partitions = 64;
    static const unsigned n_pixels = out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv1DBuffer<data_T, CONFIG_T>;
    typedef _6_accum_t accum_t;
    typedef _6_bias_t bias_t;
    typedef _6_weight_t weight_t;
    typedef config16_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_regular<K, S, W>;
    template<class data_T, class res_T, class CONFIG_T>
    using conv_kernel = nnet::Conv1DLatency<data_T, res_T, CONFIG_T>;
};
const ap_uint<config16::filt_width> config16::pixels[] = {0};

// _7
struct relu_config7 : nnet::activ_config {
    static const unsigned n_in = 64;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef _7_table_t table_t;
};

// _9
struct config17_mult : nnet::dense_config {
    static const unsigned n_in = 1;
    static const unsigned n_out = 10;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef _9_accum_t accum_t;
    typedef _9_bias_t bias_t;
    typedef _9_weight_t weight_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config17 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 64;
    static const unsigned n_chan = 1;
    static const unsigned filt_width = 1;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 10;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = 64;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_width = 64;
    static const ap_uint<filt_width> pixels[min_width];
    static const unsigned n_partitions = 10;
    static const unsigned n_pixels = out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::FillConv1DBuffer<data_T, CONFIG_T>;
    typedef _9_accum_t accum_t;
    typedef _9_bias_t bias_t;
    typedef _9_weight_t weight_t;
    typedef config17_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_regular<K, S, W>;
    template<class data_T, class res_T, class CONFIG_T>
    using conv_kernel = nnet::Conv1DLatency<data_T, res_T, CONFIG_T>;
};
const ap_uint<config17::filt_width> config17::pixels[] = {0};



#endif
