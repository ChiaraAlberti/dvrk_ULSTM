import tensorflow as tf
#import matplotlib.pyplot as plt

try:
    import tensorflow.python.keras as k
except AttributeError:
    import tensorflow.keras as k

from typing import List

from tensorflow.python.keras import regularizers

class DownBlock2D(k.Model):

    def __init__(self, conv_kernels: List[tuple], lstm_kernels: List[tuple], stride=2, data_format='NHWC', layer_ind = 0):
        super(DownBlock2D, self).__init__()
        data_format_keras = 'channels_first' if data_format[1] == 'C' else 'channels_last'
        channel_axis = 1 if data_format[1] == 'C' else -1
        self.ConvLSTM = []
        self.Conv = []
        self.BN = []
        self.LReLU = []
        self.total_stride = 1
        self.recurrent_dropout = 0.2
        self.bias = tf.keras.initializers.Constant(-0.198729)
        
        C = tf.keras.initializers.Constant
        weights_list = []
        for i in range(0, 7):
            weights_list.append(np.load("/home/stormlab/seg/layer_weights/block_%s_layer_%s_weights.npy" %(layer_ind, i)))

        for kxy_lstm, kout_lstm, dropout, reg, kernel_init in lstm_kernels:
            self.ConvLSTM.append(k.layers.ConvLSTM2D(filters=kout_lstm, kernel_size=kxy_lstm, strides=1,
                                                     padding='same', data_format=data_format_keras, kernel_initializer=C[weights_list[0]],
                                                     recurrent_initializer = C[weights_list[1]], bias_initializer = C[weights_list[2]],
                                                     return_sequences=True, stateful=True, recurrent_dropout=dropout, 
                                                     kernel_regularizer=regularizers.l1_l2(l1=reg[0], l2=reg[1])))

        for l_ind, (kxy, kout, dropout, reg, kernel_init) in enumerate(conv_kernels):
            _stride = stride if l_ind == 0 else 1
            self.total_stride *= _stride
            self.Conv.append(k.layers.Conv2D(filters=kout, kernel_size=kxy, strides=_stride, use_bias=True, kernel_initializer=C[weights_list[3 + l_ind*2]],
                                             bias_initializer = C[weights_list[4*l_ind*2]],
                                             data_format=data_format_keras, padding='same',
                                             kernel_regularizer=regularizers.l1_l2(l1=reg[0], l2=reg[1])))
            self.BN.append(k.layers.BatchNormalization(axis=channel_axis, beta_initializer = C[weights_list[7 + l_ind*4]],
                                                       gamma_initializer = C[weights_list[8 + l_ind*4]], moving_mean_initializer = C[weights_list[8 + l_ind*4]],
                                                       moving_variance_initializer = C[weights_list[8 + l_ind*4]]))
            self.LReLU.append(k.layers.LeakyReLU())

    def call(self, inputs, training=None, mask=None):
        convlstm = inputs
        for conv_lstm_layer in self.ConvLSTM:
            convlstm = conv_lstm_layer(convlstm)
        
        orig_shape = convlstm.shape
        conv_input = tf.reshape(convlstm, [orig_shape[0] * orig_shape[1], orig_shape[2], orig_shape[3], orig_shape[4]])
        activ = conv_input  # set input to for loop
        for conv_layer, bn_layer, lrelu_layer in zip(self.Conv, self.BN, self.LReLU):
            conv = conv_layer(activ)
            bn = bn_layer(conv, training)
            activ = lrelu_layer(bn)
        out_shape = activ.shape
        activ_down = tf.reshape(activ, [orig_shape[0], orig_shape[1], out_shape[1], out_shape[2], out_shape[3]])
        return activ_down, activ

    def reset_states_per_batch(self, is_last_batch):
        batch_size = is_last_batch.shape[0]
        is_last_batch = tf.reshape(is_last_batch, [batch_size, 1, 1, 1])
        for convlstm_layer in self.ConvLSTM:
            cur_state = convlstm_layer.states
            new_states = (cur_state[0] * is_last_batch, cur_state[1] * is_last_batch)
            convlstm_layer.states[0].assign(new_states[0])
            convlstm_layer.states[1].assign(new_states[1])

    def get_states(self):
        states = []
        for convlstm_layer in self.ConvLSTM:
            state = convlstm_layer.states
            states.append([s.numpy() if s is not None else s for s in state])

        return states

    def set_states(self, states):
        for convlstm_layer, state in zip(self.ConvLSTM, states):
            if None is state[0]:
                state = None
            convlstm_layer.reset_states(state)

class UpBlock2D(k.Model):

    def __init__(self, kernels: List[tuple], up_factor=2, data_format='NHWC', return_logits=False, layer_ind = 0):
        super(UpBlock2D, self).__init__()
        self.data_format_keras = 'channels_first' if data_format[1] == 'C' else 'channels_last'
        self.up_factor = up_factor
        self.channel_axis = 1 if data_format[1] == 'C' else -1
        self.Conv = []
        self.BN = []
        self.LReLU = []
        self.return_logits = return_logits
        self.bias = tf.keras.initializers.Constant(-0.198729)
        
        C = tf.keras.initializers.Constant
        weights_list = []
        for i in range(0, 7):
            weights_list.append(np.load("/home/stormlab/seg/layer_weights/block_%s_layer_%s_weights.npy" %(layer_ind + 4, i)))

        for kxy, kout, dropout, reg, kernel_init in kernels:
            self.Conv.append(k.layers.Conv2D(filters=kout, kernel_size=kxy, strides=1, use_bias=True, kernel_initializer=kernel_init,
                                             data_format=self.data_format_keras, padding='same', kernel_initializer=C[weights_list[0 + l_ind*2]],
                                             bias_initializer = C[weights_list[1*l_ind*2]],
                                             kernel_regularizer=regularizers.l1_l2(l1=reg[0], l2=reg[1])))
            self.BN.append(k.layers.BatchNormalization(axis=self.channel_axis, beta_initializer = C[weights_list[4 + l_ind*4]],
                                                       gamma_initializer = C[weights_list[5 + l_ind*4]], moving_mean_initializer = C[weights_list[6 + l_ind*4]],
                                                       moving_variance_initializer = C[weights_list[7 + l_ind*4]]))
            self.LReLU.append(k.layers.LeakyReLU())
            

    def call(self, inputs, training=None, mask=None):
        input_sequence, skip = inputs
        input_sequence = k.backend.resize_images(input_sequence, self.up_factor, self.up_factor, self.data_format_keras,
                                                 interpolation='bilinear')
        input_tensor = tf.concat([input_sequence, skip], axis=self.channel_axis)
        for conv_layer, bn_layer, lrelu_layer in zip(self.Conv, self.BN, self.LReLU):
            conv = conv_layer(input_tensor)
            if self.return_logits and conv_layer == self.Conv[-1]:
                return conv
            bn = bn_layer(conv, training)
            activ = lrelu_layer(bn)
            input_tensor = activ
        return input_tensor


class ULSTMnet2D(k.Model):
    def __init__(self, net_params=None, data_format='NHWC', pad_image=True, drop_input= False):
        super(ULSTMnet2D, self).__init__()
        self.data_format_keras = 'channels_first' if data_format[1] == 'C' else 'channels_last'
        self.channel_axis = 1 if data_format[1] == 'C' else -1
        self.DownLayers = []
        self.UpLayers = []
        self.total_stride = 1
        self.dropout_rate = 0.2
        self.drop_input = drop_input
        self.pad_image = pad_image

        if not len(net_params['down_conv_kernels']) == len(net_params['lstm_kernels']):
            raise ValueError('Number of layers in down path ({}) do not match number of LSTM layers ({})'.format(
                len(net_params['down_conv_kernels']), len(net_params['lstm_kernels'])))
        if not len(net_params['down_conv_kernels']) == len(net_params['up_conv_kernels']):
            raise ValueError('Number of layers in down path ({}) do not match number of layers in up path ({})'.format(
                len(net_params['down_conv_kernels']), len(net_params['up_conv_kernels'])))

        for layer_ind, (conv_filters, lstm_filters) in enumerate(zip(net_params['down_conv_kernels'],
                                                                     net_params['lstm_kernels'])):
            stride = 2 if layer_ind < len(net_params['down_conv_kernels']) - 1 else 1
            self.DownLayers.append(DownBlock2D(conv_filters, lstm_filters, stride, data_format, layer_ind))
            self.total_stride *= self.DownLayers[-1].total_stride
        
        for layer_ind, conv_filters in enumerate(net_params['up_conv_kernels']):
            up_factor = 2 if layer_ind > 0 else 1
            self.UpLayers.append(UpBlock2D(conv_filters, up_factor, data_format,
                                           return_logits=layer_ind + 1 == len(net_params['up_conv_kernels']), layer_ind))
            self.last_depth = conv_filters[-1][1]
            self.last_layer = conv_filters[-1]

    def call(self, inputs, training=None, mask=None):
        input_shape = inputs.shape
        if self.drop_input:
            inputs = k.layers.Dropout(self.dropout_rate)(inputs)
        min_pad_value = self.total_stride * int(self.pad_image) if self.pad_image else 0

        if self.channel_axis == 1:
            pad_y = [min_pad_value, min_pad_value + tf.math.mod(self.total_stride - tf.math.mod(input_shape[3],
                                                                                                self.total_stride),
                                                                self.total_stride)]
            pad_x = [min_pad_value, min_pad_value + tf.math.mod(self.total_stride - tf.math.mod(input_shape[4],
                                                                                                self.total_stride),
                                                                self.total_stride)]
            paddings = [[0, 0], [0, 0], [0, 0], pad_y, pad_x]
            crops = [[0, input_shape[0]], [0, input_shape[1]], [0, self.last_depth],
                     [pad_y[0], pad_y[0] + input_shape[3]], [pad_x[0], pad_x[0] + input_shape[4]]]
        else:
            pad_y = [min_pad_value, min_pad_value + tf.math.mod(self.total_stride - tf.math.mod(input_shape[2],
                                                                                                self.total_stride),
                                                                self.total_stride)]
            pad_x = [min_pad_value, min_pad_value + tf.math.mod(self.total_stride - tf.math.mod(input_shape[3],
                                                                                                self.total_stride),
                                                                self.total_stride)]
            paddings = [[0, 0], [0, 0], pad_y, pad_x, [0, 0]]
            crops = [[0, input_shape[0]], [0, input_shape[1]], [pad_y[0], input_shape[2] + pad_y[0]],
                     [pad_x[0], input_shape[3] + pad_x[0]], [0, self.last_depth]]
        inputs = tf.pad(inputs, paddings, "REFLECT")
        input_shape = inputs.shape
        skip_inputs = []
        out_down = inputs
        out_skip = tf.reshape(inputs, [input_shape[0] * input_shape[1], input_shape[2], input_shape[3], input_shape[4]])
        for down_layer in self.DownLayers:
            skip_inputs.append(out_skip)
            out_down, out_skip = down_layer(out_down, training=training, mask=mask)
#        out_skip = k.layers.SpatialDropout2D(self.dropout_rate, data_format=None)(out_skip)
        up_input = out_skip
        skip_inputs.reverse()
        assert len(skip_inputs) == len(self.UpLayers)
        for up_layer, skip_input in zip(self.UpLayers, skip_inputs):
#            if up_layer == self.last_layer:
#                up_layer = k.layers.SpatialDropout2D(self.dropout_rate, data_format=None)(up_layer)
            up_input = up_layer((up_input, skip_input), training=training, mask=mask)
        logits_output_shape = up_input.shape
        logits_output = tf.reshape(up_input, [input_shape[0], input_shape[1], logits_output_shape[1],
                                              logits_output_shape[2], logits_output_shape[3]])

        logits_output = logits_output[crops[0][0]:crops[0][1], crops[1][0]:crops[1][1], crops[2][0]:crops[2][1],
                        crops[3][0]:crops[3][1], crops[4][0]:crops[4][1]]
        output = k.activations.sigmoid(logits_output)
        return logits_output, output

#    def get_config(self):
#        config = super(ULSTMnet2D, self).get_config()
#        config.update({'': self.units})

    def reset_states_per_batch(self, is_last_batch):
        for down_block in self.DownLayers:
            down_block.reset_states_per_batch(is_last_batch)

    def get_states(self):
        states = []
        for down_block in self.DownLayers:
            states.append(down_block.get_states())
        return states

    def set_states(self, states):
        for down_block, state in zip(self.DownLayers, states):
            down_block.set_states(state)


if __name__ == "__main__":

    ULSTMnet2D.unit_test()
