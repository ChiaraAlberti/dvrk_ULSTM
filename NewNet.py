import tensorflow as tf
from typing import List
from tensorflow.python.keras import regularizers
import numpy as np

try:
    import tensorflow.python.keras as k
except AttributeError:
    import tensorflow.keras as k

#class that define downsampling blocks
class DownBlock2D(k.Model):

    def __init__(self, conv_kernels: List[tuple], lstm_kernels: List[tuple], data_format='NHWC', pretraining = False):
        super(DownBlock2D, self).__init__()
        data_format_keras = 'channels_first' if data_format[1] == 'C' else 'channels_last'
        channel_axis = 1 if data_format[1] == 'C' else -1
        self.ConvLSTM = []
        self.Conv = []
        self.BN = []
        self.LReLU = []
        self.total_stride = 1
        self.pretraining = pretraining
        
        #initialization of convLSTM2D layers
        for kxy_lstm, kout_lstm, dropout, reg, kernel_init in lstm_kernels:
            self.ConvLSTM.append(k.layers.ConvLSTM2D(filters=kout_lstm, kernel_size=kxy_lstm, strides=1,
                                 padding='same', data_format=data_format_keras, kernel_initializer=kernel_init,
                                 return_sequences=True, stateful=False, recurrent_dropout=dropout, 
                                 kernel_regularizer=regularizers.l1_l2(l1=reg[0], l2=reg[1])))
        #initialization of Conv + Batch + ReLU
        for l_ind, (kxy, kout, dropout, reg, kernel_init) in enumerate(conv_kernels):
            self.Conv.append(k.layers.Conv2D(filters=kout, kernel_size=kxy, strides=1, use_bias=True, kernel_initializer=kernel_init,
                             data_format=data_format_keras, padding='same',
                             kernel_regularizer=regularizers.l1_l2(l1=reg[0], l2=reg[1])))
            self.BN.append(k.layers.BatchNormalization(axis=channel_axis))
            self.LReLU.append(k.layers.LeakyReLU())
        
        #initialization of maxpooling layer
        self.MaxPool =  k.layers.MaxPool2D(pool_size=(2, 2))

    def call(self, inputs, training=None, mask=None):
        
        convlstm = inputs
        for conv_lstm_layer in self.ConvLSTM:
            convlstm = conv_lstm_layer(convlstm, training= training)
        #reshape in order to feed in the convolutional layers
        orig_shape = convlstm.shape
        conv_input = tf.reshape(convlstm, [orig_shape[0] * orig_shape[1], orig_shape[2], orig_shape[3], orig_shape[4]])
        activ = conv_input  # set input to for loop
        for conv_layer, bn_layer, lrelu_layer in zip(self.Conv, self.BN, self.LReLU):
            conv = conv_layer(activ)
            bn = bn_layer(conv, training)
            activ = lrelu_layer(bn)
        activ_down = self.MaxPool(activ)
        out_shape = activ_down.shape
        activ_down = tf.reshape(activ_down, [orig_shape[0], orig_shape[1], out_shape[1], out_shape[2], out_shape[3]])
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


#class that define upsampling blocks
class UpBlock2D(k.Model):

    def __init__(self, kernels: List[tuple], lstm_kernels: List[tuple], up_factor=2, data_format='NHWC',  layer_ind_up = 0, pretraining =False, return_logits=False):
        super(UpBlock2D, self).__init__()
        self.data_format_keras = 'channels_first' if data_format[1] == 'C' else 'channels_last'
        self.up_factor = up_factor
        self.channel_axis = 1 if data_format[1] == 'C' else -1
        self.ConvLSTM = []
        self.Conv = []
        self.BN = []
        self.LReLU = []
        self.return_logits = return_logits
        self.pretraining = pretraining
        
        if lstm_kernels is not None:
            for kxy_lstm, kout_lstm, dropout, reg, kernel_init in lstm_kernels:
                self.ConvLSTM.append(k.layers.ConvLSTM2D(filters=kout_lstm, kernel_size=kxy_lstm, strides=1,
                                     padding='same', data_format=self.data_format_keras, kernel_initializer=kernel_init,
                                     return_sequences=True, stateful=False, recurrent_dropout=dropout, 
                                     kernel_regularizer=regularizers.l1_l2(l1=reg[0], l2=reg[1])))
        
        #initialization of Conv + Batch + ReLU
        for l_ind, (kxy, kout, dropout, reg, kernel_init) in enumerate(kernels):
            self.Conv.append(k.layers.Conv2D(filters=kout, kernel_size=kxy, strides=1, use_bias=True, kernel_initializer=kernel_init,
                             data_format=self.data_format_keras, padding='same',
                             kernel_regularizer=regularizers.l1_l2(l1=reg[0], l2=reg[1])))
            self.BN.append(k.layers.BatchNormalization(axis=self.channel_axis))      
            self.LReLU.append(k.layers.LeakyReLU())
            

    def call(self, inputs, training=None, mask=None, shape = None):
        input_sequence, skip = inputs
        #bilinear interpolation
        input_sequence = k.backend.resize_images(input_sequence, self.up_factor, self.up_factor, self.data_format_keras,
                                                 interpolation='bilinear')
        #concatenation
        input_tensor = tf.concat([input_sequence, skip], axis=self.channel_axis)
        
        if len(self.ConvLSTM) != 0:            
            lstm_input = tf.reshape(input_tensor, [shape[0], shape[1], input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]])
            for conv_lstm_layer in self.ConvLSTM:
                lstm_tensor = conv_lstm_layer(lstm_input, training= training)           
            input_tensor = tf.reshape(lstm_tensor, [input_tensor.shape[0], lstm_tensor.shape[2], lstm_tensor.shape[3], lstm_tensor.shape[4]])
            
        for conv_layer, bn_layer, lrelu_layer in zip(self.Conv, self.BN, self.LReLU):
            conv = conv_layer(input_tensor)
            if self.return_logits and conv_layer == self.Conv[-1]:
                return conv
            bn = bn_layer(conv, training)
            activ = lrelu_layer(bn)
            input_tensor = activ
        return input_tensor
    
#class that define the gating signal, input for the attention gate
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

#class that define the attention gate block
class AttnGatingBlock(k.Model):
    def __init__(self, data_format='NHWC', inter_shape= 256, num_filters= 128):
        super(AttnGatingBlock, self).__init__()
        self.data_format_keras = 'channels_first' if data_format[1] == 'C' else 'channels_last'
        self.channel_axis = 1 if data_format[1] == 'C' else -1
        self.inter_shape = inter_shape
        
        self.thetaConv =  k.layers.Conv2D(int(inter_shape), (2, 2), strides=(2, 2), padding='same')
        self.phiConv = k.layers.Conv2D(int(inter_shape), (1, 1), padding='same')
        self.psiConv = k.layers.Conv2D(1, (1, 1), padding='same')
        self.resultConv = k.layers.Conv2D(inter_shape, (1, 1), padding='same')
        self.batch = k.layers.BatchNormalization(axis = self.channel_axis)

    def call(self, x, g):
        shape_x = x.shape
        #downsampling of the x signal
        theta_x = self.thetaConv(x)
        phi_g = self.phiConv(g)
        #concatenation
        concat_xg = k.layers.add([phi_g, theta_x])
        #ReLU activation
        act_xg = k.layers.ReLU()(concat_xg)
        psi = self.psiConv(act_xg)
        #sigmoid activation
        sigmoid_xg = k.activations.sigmoid(psi)
        shape_sigmoid = sigmoid_xg.shape
        #upsampling
        upsample_psi = k.layers.UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32
        #multiplication between input and results of attention gate
        result = k.layers.multiply([upsample_psi, x])
        result = self.resultConv(result)
        result = self.batch(result)
        return result

#class that define the network architecture
class ULSTMnet2D(k.Model):
    def __init__(self, net_params=None, data_format='NHWC', pad_image=True, dropout= 0, pretraining = False, lstm = 'full', attention_gate = False):
        super(ULSTMnet2D, self).__init__()
        self.data_format_keras = 'channels_first' if data_format[1] == 'C' else 'channels_last'
        self.channel_axis = 1 if data_format[1] == 'C' else -1
        self.DownLayers = []
        self.UpLayers = []
        self.ConnectLayer = []
        self.AttentionBlock = []
        self.GateSignal = []
        self.total_stride = 1
        self.dropout_rate = dropout
        self.pad_image = pad_image
        self.pretraining = pretraining
        self.attention_gate = attention_gate
        self.total_stride = 2^(len(net_params['down_conv_kernels']))

        if not len(net_params['down_conv_kernels']) == len(net_params['lstm_kernels']):
            raise ValueError('Number of layers in down path ({}) do not match number of LSTM layers ({})'.format(
                len(net_params['down_conv_kernels']), len(net_params['lstm_kernels'])))
        if not len(net_params['down_conv_kernels']) == len(net_params['up_conv_kernels']):
            raise ValueError('Number of layers in down path ({}) do not match number of layers in up path ({})'.format(
                len(net_params['down_conv_kernels']), len(net_params['up_conv_kernels'])))

        #create a list of downsampling blocks
        for layer_ind, (conv_filters, lstm_filters) in enumerate(zip(net_params['down_conv_kernels'],
                                                                     net_params['lstm_kernels'])):
            self.DownLayers.append(DownBlock2D(conv_filters, lstm_filters, data_format, pretraining))
        
        #bottleneck layer connecting the two branches
        self.ConnectLayer = k.layers.Conv2D(filters=512, kernel_size=3, strides=1, use_bias=True,
                                            data_format=self.data_format_keras, padding='same')
        
        self.Dropout = k.layers.Dropout(self.dropout_rate)  
        
        #attention gate filter values
        if self.attention_gate:
            for i in [512, 256, 128, 128]:
                self.GateSignal.append(UnetGatingSignal(data_format, i))
            for i in [256, 128, 128, 64]:
                self.AttentionBlock.append(AttnGatingBlock(data_format, i))
        
        #create a list of downsampling blocks
        if lstm == 'full':
            for layer_ind, (conv_filters, lstm_filters) in enumerate(zip(net_params['up_conv_kernels'],
                                                                         net_params['lstm_kernels'][::-1])):
                up_factor = 2
                self.UpLayers.append(UpBlock2D(conv_filters, lstm_filters, up_factor, data_format, layer_ind, pretraining,
                                               return_logits=layer_ind + 1 == len(net_params['up_conv_kernels'])))
                self.last_depth = conv_filters[-1][1]
                self.last_layer = conv_filters[-1]
        else:
            for layer_ind, conv_filters in enumerate(net_params['up_conv_kernels']):
                up_factor = 2
                self.UpLayers.append(UpBlock2D(conv_filters, None, up_factor, data_format, layer_ind, pretraining,
                                               return_logits=layer_ind + 1 == len(net_params['up_conv_kernels'])))
                self.last_depth = conv_filters[-1][1]
                self.last_layer = conv_filters[-1]    

    def call(self, inputs, training=None, mask=None):
        input_shape = inputs.shape
#        min_pad_value = self.total_stride * int(self.pad_image) if self.pad_image else 0
#
#        if self.channel_axis == 1:
#            pad_y = [min_pad_value, min_pad_value + tf.math.mod(self.total_stride - tf.math.mod(input_shape[3], self.total_stride), self.total_stride)]
#            pad_x = [min_pad_value, min_pad_value + tf.math.mod(self.total_stride - tf.math.mod(input_shape[4], self.total_stride), self.total_stride)]
#            paddings = [[0, 0], [0, 0], [0, 0], pad_y, pad_x]
#            crops = [[0, input_shape[0]], [0, input_shape[1]], [0, self.last_depth],
#                     [pad_y[0], pad_y[0] + input_shape[3]], [pad_x[0], pad_x[0] + input_shape[4]]]
#        else:
#            pad_y = [min_pad_value, min_pad_value + tf.math.mod(self.total_stride - tf.math.mod(input_shape[2], self.total_stride), self.total_stride)]
#            pad_x = [min_pad_value, min_pad_value + tf.math.mod(self.total_stride - tf.math.mod(input_shape[3], self.total_stride), self.total_stride)]
#            paddings = [[0, 0], [0, 0], pad_y, pad_x, [0, 0]]
#            crops = [[0, input_shape[0]], [0, input_shape[1]], [pad_y[0], input_shape[2] + pad_y[0]],
#                     [pad_x[0], input_shape[3] + pad_x[0]], [0, self.last_depth]]
#        inputs = tf.pad(inputs, paddings, "REFLECT")
        input_shape = inputs.shape
        skip_inputs = []
        out_down = inputs
        #downsampling branch
        for down_layer in self.DownLayers:
            out_down, out_skip = down_layer(out_down, training=training, mask=mask)
            skip_inputs.append(out_skip)
        
        #connecting layer with dropout
        up_input = tf.reshape(out_down, [input_shape[0]* input_shape[1], out_down.shape[2], out_down.shape[3], out_down.shape[4]])
        up_input = self.ConnectLayer(up_input)
        up_input = self.Dropout(up_input, training)               
        skip_inputs.reverse()
        #upsampling layer with and without attention gate
        assert len(skip_inputs) == len(self.UpLayers)
        if self.attention_gate:
            for up_layer, skip_input, signal_gate, attention_block in zip(self.UpLayers, skip_inputs, self.GateSignal, self.AttentionBlock):
                up_input = signal_gate(up_input, is_batchnorm=True)
                attn = attention_block(skip_input, up_input)
                up_input = up_layer((up_input, attn), training=training, mask=mask, shape = input_shape)
        else:
            for up_layer, skip_input in zip(self.UpLayers, skip_inputs):
                up_input = up_layer((up_input, skip_input), training=training, mask=mask, shape = input_shape)

        logits_output_shape = up_input.shape
        logits_output = tf.reshape(up_input, [input_shape[0], input_shape[1], logits_output_shape[1],
                                              logits_output_shape[2], logits_output_shape[3]])

#        logits_output = logits_output[crops[0][0]:crops[0][1], crops[1][0]:crops[1][1], crops[2][0]:crops[2][1],
#                        crops[3][0]:crops[3][1], crops[4][0]:crops[4][1]]
        
        output = k.activations.sigmoid(logits_output)
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
