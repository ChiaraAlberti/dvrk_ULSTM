#same content of "Networks.py" but the weights of each layer are initialized according to the pretrained model 
#available at this link: https://drive.google.com/file/d/1uQOdelJoXrffmW_1OCu417nHKtQcH3DJ/view?usp=sharing
import tensorflow as tf
import pickle
from typing import List
from tensorflow.python.keras import regularizers
import numpy as np

try:
    import tensorflow.python.keras as k
except AttributeError:
    import tensorflow.keras as k

class DownBlock2D(k.Model):

    def __init__(self, conv_kernels: List[tuple], lstm_kernels: List[tuple], stride=2, data_format='NHWC', weights_list = 0, pretraining = False):
        super(DownBlock2D, self).__init__()
        data_format_keras = 'channels_first' if data_format[1] == 'C' else 'channels_last'
        channel_axis = 1 if data_format[1] == 'C' else -1
        self.ConvLSTM = []
        self.Conv = []
        self.BN = []
        self.LReLU = []
        self.total_stride = 1
        self.recurrent_dropout = 0.2
        self.pretraining = pretraining
        
        first = True
        for kxy_lstm, kout_lstm, dropout, reg, kernel_init in lstm_kernels:
            if self.pretraining == 'cells':
                C = tf.keras.initializers.Constant
                if first:
                    self.ConvLSTM.append(k.layers.ConvLSTM2D(filters=kout_lstm, kernel_size=kxy_lstm, strides=1,
                                                             padding='same', data_format=data_format_keras, kernel_initializer=C(weights_list[0]),
                                                             recurrent_initializer = C(weights_list[1]), 
                                                             return_sequences=True, stateful=False, recurrent_dropout=dropout, 
                                                             kernel_regularizer=regularizers.l1_l2(l1=reg[0], l2=reg[1])))
                    first = False
                else:
                    self.ConvLSTM.append(k.layers.ConvLSTM2D(filters=kout_lstm, kernel_size=kxy_lstm, strides=1,
                                                             padding='same', data_format=data_format_keras, kernel_initializer=C(weights_list[0]),
                                                             recurrent_initializer = C(weights_list[1]), bias_initializer = C(weights_list[2]),
                                                             return_sequences=True, stateful=True, recurrent_dropout=dropout, 
                                                             kernel_regularizer=regularizers.l1_l2(l1=reg[0], l2=reg[1])))
            else:
                self.ConvLSTM.append(k.layers.ConvLSTM2D(filters=kout_lstm, kernel_size=kxy_lstm, strides=1,
                                     padding='same', data_format=data_format_keras, kernel_initializer=kernel_init,
                                     return_sequences=True, stateful=True, recurrent_dropout=dropout, 
                                     kernel_regularizer=regularizers.l1_l2(l1=reg[0], l2=reg[1])))

        for l_ind, (kxy, kout, dropout, reg, kernel_init) in enumerate(conv_kernels):
            _stride = stride if l_ind == 0 else 1
            self.total_stride *= _stride
            if self.pretraining== 'cells':
                C = tf.keras.initializers.Constant
                self.Conv.append(k.layers.Conv2D(filters=kout, kernel_size=kxy, strides=_stride, use_bias=True, kernel_initializer=C(weights_list[3 + l_ind*2]),
                                                 bias_initializer = C(weights_list[4 + l_ind*2]),
                                                 data_format=data_format_keras, padding='same',
                                                 kernel_regularizer=regularizers.l1_l2(l1=reg[0], l2=reg[1])))
                self.BN.append(k.layers.BatchNormalization(axis=channel_axis, beta_initializer = C(weights_list[7 + l_ind*4]),
                                                           gamma_initializer = C(weights_list[8 + l_ind*4]), moving_mean_initializer = C(weights_list[9 + l_ind*4]),
                                                           moving_variance_initializer = C(weights_list[10 + l_ind*4])))
            elif self.pretraining == 'image_net':
                C = tf.keras.initializers.Constant
                self.Conv.append(k.layers.Conv2D(filters=kout, kernel_size=kxy, strides=_stride, use_bias=True, kernel_initializer=C(weights_list[0 + l_ind*2]),
                                                 bias_initializer = C(weights_list[1 + l_ind*2]),
                                                 data_format=data_format_keras, padding='same',
                                                 kernel_regularizer=regularizers.l1_l2(l1=reg[0], l2=reg[1])))
                self.BN.append(k.layers.BatchNormalization(axis=channel_axis))
            
            else:
                self.Conv.append(k.layers.Conv2D(filters=kout, kernel_size=kxy, strides=_stride, use_bias=True, kernel_initializer=kernel_init,
                                 data_format=data_format_keras, padding='same',
                                 kernel_regularizer=regularizers.l1_l2(l1=reg[0], l2=reg[1])))
                self.BN.append(k.layers.BatchNormalization(axis=channel_axis))
            self.LReLU.append(k.layers.LeakyReLU())

    def call(self, inputs, training=None, mask=None):
        convlstm = inputs
        for conv_lstm_layer in self.ConvLSTM:
            convlstm = conv_lstm_layer(convlstm, training= training)
        
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

    def __init__(self, kernels: List[tuple], up_factor=2, data_format='NHWC',  layer_ind_up = 0, weights_list = 0, pretraining =False, pretraining_type = 'full', return_logits=False):
        super(UpBlock2D, self).__init__()
        self.data_format_keras = 'channels_first' if data_format[1] == 'C' else 'channels_last'
        self.up_factor = up_factor
        self.channel_axis = 1 if data_format[1] == 'C' else -1
        self.Conv = []
        self.BN = []
        self.LReLU = []
        self.return_logits = return_logits
        self.pretraining = pretraining
        
        for l_ind, (kxy, kout, dropout, reg, kernel_init) in enumerate(kernels):
            if self.pretraining == 'cells' and pretraining_type =='full':
                C = tf.keras.initializers.Constant
                if layer_ind_up ==3:
                    if l_ind != 2:
                        self.Conv.append(k.layers.Conv2D(filters=kout, kernel_size=kxy, strides=1, use_bias=True,
                                                         data_format=self.data_format_keras, padding='same', kernel_initializer=C(weights_list[0 + l_ind*2]),
                                                         bias_initializer = C(weights_list[1 + l_ind*2]),
                                                         kernel_regularizer=regularizers.l1_l2(l1=reg[0], l2=reg[1])))
                        self.BN.append(k.layers.BatchNormalization(axis=self.channel_axis, beta_initializer = C(weights_list[6 + l_ind*4]),
                                                                   gamma_initializer = C(weights_list[7 + l_ind*4]), moving_mean_initializer = C(weights_list[8 + l_ind*4]),
                                                                   moving_variance_initializer = C(weights_list[9 + l_ind*4])))
                    else:
                        self.Conv.append(k.layers.Conv2D(filters=kout, kernel_size=kxy, strides=1, use_bias=True,
                                                         data_format=self.data_format_keras, padding='same', kernel_initializer='he_normal',
                                                         kernel_regularizer=regularizers.l1_l2(l1=reg[0], l2=reg[1])))
                        self.BN.append(k.layers.BatchNormalization(axis=self.channel_axis))
                else:
                    self.Conv.append(k.layers.Conv2D(filters=kout, kernel_size=kxy, strides=1, use_bias=True,
                                                     data_format=self.data_format_keras, padding='same', kernel_initializer=C(weights_list[0 + l_ind*2]),
                                                     bias_initializer = C(weights_list[1 + l_ind*2]),
                                                     kernel_regularizer=regularizers.l1_l2(l1=reg[0], l2=reg[1])))
                    self.BN.append(k.layers.BatchNormalization(axis=self.channel_axis, beta_initializer = C(weights_list[4 + l_ind*4]),
                                                               gamma_initializer = C(weights_list[5 + l_ind*4]), moving_mean_initializer = C(weights_list[6 + l_ind*4]),
                                                               moving_variance_initializer = C(weights_list[7 + l_ind*4])))
#            elif self.pretraining == 'imagenet':
#                C = tf.keras.initializers.Constant
#                if l_ind != 2:
#                    self.Conv.append(k.layers.Conv2D(filters=kout, kernel_size=kxy, strides=1, use_bias=True,
#                                                     data_format=self.data_format_keras, padding='same', kernel_initializer=C(weights_list[0 + l_ind]),
#                                                     kernel_regularizer=regularizers.l1_l2(l1=reg[0], l2=reg[1])))
#                    self.BN.append(k.layers.BatchNormalization(axis=self.channel_axis, beta_initializer = C(weights_list[2 + l_ind*4]),
#                                                               gamma_initializer = C(weights_list[3 + l_ind*4]), moving_mean_initializer = C(weights_list[4 + l_ind*4]),
#                                                               moving_variance_initializer = C(weights_list[5 + l_ind*4])))
#                else:
#                    self.Conv.append(k.layers.Conv2D(filters=kout, kernel_size=kxy, strides=1, use_bias=True,
#                                                     data_format=self.data_format_keras, padding='same', kernel_initializer='he_normal',
#                                                     kernel_regularizer=regularizers.l1_l2(l1=reg[0], l2=reg[1])))
#                    self.BN.append(k.layers.BatchNormalization(axis=self.channel_axis))
            else:
                self.Conv.append(k.layers.Conv2D(filters=kout, kernel_size=kxy, strides=1, use_bias=True, kernel_initializer=kernel_init,
                                 data_format=self.data_format_keras, padding='same',
                                 kernel_regularizer=regularizers.l1_l2(l1=reg[0], l2=reg[1])))
                self.BN.append(k.layers.BatchNormalization(axis=self.channel_axis))      
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
    
class UnetGatingSignal(k.Model):
    def __init__(self, data_format='NHWC', num_layers = 256):
        super(UnetGatingSignal, self).__init__()
        self.data_format_keras = 'channels_first' if data_format[1] == 'C' else 'channels_last'
        self.channel_axis = 1 if data_format[1] == 'C' else -1
        self.Conv = None
        self.Batch = None
        self.ReLU = None        
              
        self.Conv = k.layers.Conv2D(num_layers, (1, 1), strides=(1, 1), padding='same',  kernel_initializer='he_normal')
        self.Batch = k.layers.BatchNormalization(axis = self.channel_axis)
        self.ReLU = k.layers.LeakyReLU()
        
    def call(self, x, is_batchnorm=True):
        x = self.Conv(x)
        x = self.Batch(x)
        x = self.ReLU(x)                
        return x

class AttnGatingBlock(k.Model):
    def __init__(self, data_format='NHWC', inter_shape= 256, num_filters= 128):
        super(AttnGatingBlock, self).__init__()
        self.data_format_keras = 'channels_first' if data_format[1] == 'C' else 'channels_last'
        self.channel_axis = 1 if data_format[1] == 'C' else -1
        self.inter_shape = inter_shape
        
        self.thetaConv =  k.layers.Conv2D(int(inter_shape), (2, 2), strides=(2, 2), padding='same')
        self.phiConv = k.layers.Conv2D(int(inter_shape), (1, 1), padding='same')
        self.psiConv = k.layers.Conv2D(1, (1, 1), padding='same')
        self.resultConv = k.layers.Conv2D(num_filters, (1, 1), padding='same')
        self.batch = k.layers.BatchNormalization(axis = self.channel_axis)

    def call(self, x, g):
        shape_x = x.shape # 32
#        shape_g = g.shape  # 16
#        print(shape_x)
#        print(shape_g)
        theta_x = self.thetaConv(x)  # 16
#        shape_theta_x = theta_x.shape
        phi_g = self.phiConv(g)
        #upsample_g = k.layers.Conv2DTranspose(int(self.inter_shape), (5, 5),strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),padding='same')(phi_g)  # 16
        concat_xg = k.layers.add([phi_g, theta_x])
        act_xg = k.layers.ReLU()(concat_xg)
        psi = self.psiConv(act_xg)
        sigmoid_xg = k.activations.sigmoid(psi)
        shape_sigmoid = sigmoid_xg.shape
        upsample_psi = k.layers.UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32
#        upsample_psi = k.layers.Lambda(lambda x, repnum: tf.keras.backend.repeat_elements(x, shape_x[3], axis=3), arguments={'repnum': shape_x[3]})(upsample_psi)
        result = k.layers.multiply([upsample_psi, x])
#        result = self.resultConv(y)
#        result_bn = self.batch(result)
#        print(result_bn.shape)
        return result


class ULSTMnet2D(k.Model):
    def __init__(self, net_params=None, data_format='NHWC', pad_image=True, drop_input= False, pretraining = False, pretraining_type = 'full', attention_gate = False):
        super(ULSTMnet2D, self).__init__()
        self.data_format_keras = 'channels_first' if data_format[1] == 'C' else 'channels_last'
        self.channel_axis = 1 if data_format[1] == 'C' else -1
        self.DownLayers = []
        self.UpLayers = []
        self.ConnectLayer = []
        self.AttentionBlock = []
        self.GateSignal = []
        self.total_stride = 1
        self.dropout_rate = 0.1
        self.drop_input = drop_input
        self.pad_image = pad_image
        self.pretraining = pretraining
        self.attention_gate = attention_gate

        if not len(net_params['down_conv_kernels']) == len(net_params['lstm_kernels']):
            raise ValueError('Number of layers in down path ({}) do not match number of LSTM layers ({})'.format(
                len(net_params['down_conv_kernels']), len(net_params['lstm_kernels'])))
        if not len(net_params['down_conv_kernels']) == len(net_params['up_conv_kernels']):
            raise ValueError('Number of layers in down path ({}) do not match number of layers in up path ({})'.format(
                len(net_params['down_conv_kernels']), len(net_params['up_conv_kernels'])))

        for layer_ind, (conv_filters, lstm_filters) in enumerate(zip(net_params['down_conv_kernels'],
                                                                     net_params['lstm_kernels'])):
            #loading the file with the weights list for each block
            weights_list = []
            if pretraining == 'cells':
                for i in range(0, 5):
                    with open("/home/stormlab/seg/layer_weights/%s/block_%s_layer_%s_weights.npy" %(pretraining, layer_ind, i), "rb") as fp:   # Unpickling
                        b = pickle.load(fp)
                    weights_list.extend(b)
            elif pretraining == 'imagenet':
                for i in range(0, 2):
                    with open("/home/stormlab/seg/layer_weights/%s/block_%s_layer_%s_weights.npy" %(pretraining, layer_ind, i), "rb") as fp:   # Unpickling
                        b = pickle.load(fp)
                    weights_list.extend(b)
            if self.attention_gate:
                stride =2
            else:
                stride = 2 if layer_ind < len(net_params['down_conv_kernels']) - 1 else 1
            self.DownLayers.append(DownBlock2D(conv_filters, lstm_filters, stride, data_format, weights_list, pretraining))
            self.total_stride *= self.DownLayers[-1].total_stride
        
        if pretraining == 'imagenet':
            for i in range(0, 3):
                self.ConnectLayer.append(k.layers.Conv2D(filters=512, kernel_size=3, strides=1, use_bias=True,
                                                         data_format=self.data_format_keras, padding='same'))
    
        if self.attention_gate:
            self.ConnectLayer = k.layers.Conv2D(filters=512, kernel_size=5, strides=2, use_bias=True,
                                                data_format=self.data_format_keras, padding='same')
            for i in [256, 256, 128, 128]:
                self.GateSignal.append(UnetGatingSignal(data_format, i))
            layers_num = 256
            for i in [128, 128, 64, 1]:
                self.AttentionBlock.append(AttnGatingBlock(data_format, layers_num, i))
                layers_num = layers_num/2

        for layer_ind, conv_filters in enumerate(net_params['up_conv_kernels']):
            if self.attention_gate:
                up_factor = 2
            else:
                up_factor = 2 if layer_ind > 0 else 1
            #loading the file with the weights list for each block
            weights_list = []
            if pretraining == 'cells':
                for i in range(0, 5):
                    with open("/home/stormlab/seg/layer_weights/%s/block_%s_layer_%s_weights.npy" %(pretraining, layer_ind +4, i), "rb") as fp:   # Unpickling
                        b = pickle.load(fp)
                    weights_list.extend(b)
            elif pretraining == 'imagenet':
                for i in range(0, 4):
                    with open("/home/stormlab/seg/layer_weights/%s/block_%s_layer_%s_weights.npy" %(pretraining, layer_ind +4, i), "rb") as fp:   # Unpickling
                        b = pickle.load(fp)
                    weights_list.extend(b)
            self.UpLayers.append(UpBlock2D(conv_filters, up_factor, data_format, layer_ind, weights_list, pretraining, pretraining_type,
                                           return_logits=layer_ind + 1 == len(net_params['up_conv_kernels'])))
            self.last_depth = conv_filters[-1][1]
            self.last_layer = conv_filters[-1]
            
#        self.Sigmoid = k.layers.Conv2D(1, 1, 1, use_bias=True, activation = 'sigmoid', data_format=self.data_format_keras, padding='same')


    def call(self, inputs, training=None, mask=None):
        input_shape = inputs.shape
#        if self.drop_input and training:
#            inputs = k.layers.Dropout(self.dropout_rate)(inputs)
        min_pad_value = self.total_stride * int(self.pad_image) if self.pad_image else 0

        if self.channel_axis == 1:
            pad_y = [min_pad_value, min_pad_value + tf.math.mod(self.total_stride - tf.math.mod(input_shape[3], self.total_stride), self.total_stride)]
            pad_x = [min_pad_value, min_pad_value + tf.math.mod(self.total_stride - tf.math.mod(input_shape[4], self.total_stride), self.total_stride)]
            paddings = [[0, 0], [0, 0], [0, 0], pad_y, pad_x]
            crops = [[0, input_shape[0]], [0, input_shape[1]], [0, self.last_depth],
                     [pad_y[0], pad_y[0] + input_shape[3]], [pad_x[0], pad_x[0] + input_shape[4]]]
        else:
            pad_y = [min_pad_value, min_pad_value + tf.math.mod(self.total_stride - tf.math.mod(input_shape[2], self.total_stride), self.total_stride)]
            pad_x = [min_pad_value, min_pad_value + tf.math.mod(self.total_stride - tf.math.mod(input_shape[3], self.total_stride), self.total_stride)]
            paddings = [[0, 0], [0, 0], pad_y, pad_x, [0, 0]]
            crops = [[0, input_shape[0]], [0, input_shape[1]], [pad_y[0], input_shape[2] + pad_y[0]],
                     [pad_x[0], input_shape[3] + pad_x[0]], [0, self.last_depth]]
        inputs = tf.pad(inputs, paddings, "REFLECT")
        input_shape = inputs.shape
        skip_inputs = []
        out_down = inputs
        out_skip = tf.reshape(inputs, [input_shape[0] * input_shape[1], input_shape[2], input_shape[3], input_shape[4]])
#        if self.attention_gate:
#            for down_layer in self.DownLayers:
#                out_down, out_skip = down_layer(out_down, training=training, mask=mask)
#                skip_inputs.append(out_skip)
#        else:
        for down_layer in self.DownLayers:
            skip_inputs.append(out_skip)
            out_down, out_skip = down_layer(out_down, training=training, mask=mask)
        if self.drop_input and training:
            out_skip = k.layers.SpatialDropout2D(self.dropout_rate, data_format=None)(out_skip)
        up_input = out_skip
#        if self.pretraining == 'imagenet':
#            for layer in self.ConnectLayer:
#                up_input = layer(up_input)
#                skip_inputs.append(up_input)
#                skip_inputs.pop(0)
#        if self.attention_gate:
#            up_input = self.ConnectLayer(up_input)
                
        skip_inputs.reverse()
        first = True
        assert len(skip_inputs) == len(self.UpLayers)
        if self.attention_gate:
            for up_layer, skip_input, signal_gate, attention_block in zip(self.UpLayers, skip_inputs, self.GateSignal, self.AttentionBlock):
                if up_layer == self.last_layer and training and self.drop_input:
                    skip_input = k.layers.SpatialDropout2D(self.dropout_rate, data_format=None)(skip_input) 
                up_input = signal_gate(up_input, is_batchnorm=True)
                attn = attention_block(skip_input, up_input)
                up_input = up_layer((up_input, attn), training=training, mask=mask)
        else:
            for up_layer, skip_input in zip(self.UpLayers, skip_inputs):
                up_input = up_layer((up_input, skip_input), training=training, mask=mask)
                if first and training and self.drop_input:
                    up_input = k.layers.SpatialDropout2D(self.dropout_rate, data_format=None)(skip_input) 
                    first = False

        logits_output_shape = up_input.shape
        logits_output = tf.reshape(up_input, [input_shape[0], input_shape[1], logits_output_shape[1],
                                              logits_output_shape[2], logits_output_shape[3]])

        logits_output = logits_output[crops[0][0]:crops[0][1], crops[1][0]:crops[1][1], crops[2][0]:crops[2][1],
                        crops[3][0]:crops[3][1], crops[4][0]:crops[4][1]]
        
        output = k.activations.sigmoid(logits_output)
#        output = self.Sigmoid(up_input)
#        output = tf.reshape(output, [input_shape[0], input_shape[1], logits_output_shape[1],
#                                     logits_output_shape[2], logits_output_shape[3]])
#        output = output[crops[0][0]:crops[0][1], crops[1][0]:crops[1][1], crops[2][0]:crops[2][1],
#                        crops[3][0]:crops[3][1], crops[4][0]:crops[4][1]]
        return logits_output, output


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

class DownConv(k.Model):
    def __init__(self, out_ch, data_format = 'NHWC'):
        super(DownConv, self).__init__()
        self.data_format_keras = 'channels_first' if data_format[1] == 'C' else 'channels_last'
        self.channel_axis = 1 if data_format[1] == 'C' else -1
        self.out_ch = out_ch

        self.Conv = k.layers.Conv2D(filters=self.out_ch, kernel_size=3)
        self.Batch = k.layers.BatchNormalization(axis = self.channel_axis)
        self.MaxPool = k.layers.MaxPool2D(pool_size=(2,2), strides=2)
        self.ReLU = k.layers.LeakyReLU()
        
    def __call__(self, x):
        x = self.Conv(x)
        x = self.Batch(x)
        x = self.MaxPool(x)
        x = self.ReLU(x)
        return x

class Discriminator(k.Model):
    def __init__(self, data_format = 'NHWC'):
        super(Discriminator, self).__init__()
        self.data_format_keras = 'channels_first' if data_format[1] == 'C' else 'channels_last'
        self.channel_axis = 1 if data_format[1] == 'C' else -1
        self.n_channels = [
            16,
            32,
            64,
            128,
        ]

        self.conv1 = DownConv(self.n_channels[0])
        self.conv2 = DownConv(self.n_channels[1])
        self.conv3 = DownConv(self.n_channels[2])
        self.conv4 = DownConv(self.n_channels[3])
        self.Dense = k.layers.Dense(1)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        dim = np.prod(x.shape[1:])
        x = tf.reshape(x, [-1, dim])
        x = self.Dense(x)
        x = k.activations.sigmoid(x)
        return x

if __name__ == "__main__":

    ULSTMnet2D.unit_test()
