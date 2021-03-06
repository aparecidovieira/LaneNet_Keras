####
import keras
import keras.backend as K
from keras.layers import *
from keras.activations import *
from keras.models import *
from pywt import wavedec2
import pywt
import numpy as np

def wavelet(inputs, level=4):
    coeffs = wavedec2(inputs, 'db1', level=level)
    return coeffs

def multiply(tensor_a,tensor_b):
    return Multiply()([tensor_a,tensor_b])

def lanenet_wavelet(inputs):
    coeffs = wavelet(inputs)
    cA, C4, C3, C2, C1 = coeffs

    w1 = np.stack((C1), axis=-1)
    w2 = np.stack((C2), axis=-1)
    w3 = np.stack((C3), axis=-1)
    w4 = np.stack((C4), axis=-1)
    return w1, w2, w3, w4

def convBlock_v2(inputs, n_filters, kernel=3, strides=1):
    net = Conv2D(n_filters, kernel_size=[kernel, kernel], activation = 'relu', strides=1, padding ='same')(inputs)
    return net

def upBlock(inputs, n_filters, blocks=1):
    net = Upsampling_v1(inputs)
    for _ in range(blocks):
        net = convBlock(net, n_filters, kernel=2)
    return net

def convBlock(inputs, n_filters, kernel=3, strides=1):
    net = Conv2D(n_filters, kernel_size=[kernel, kernel], activation = None, strides=1, padding ='same')(inputs)
    net = BatchNormalization()(net)
    net = Activation("relu")(net)
    return net

def conv_block(inputs, n_filters, blocks=3):
    for _ in range(blocks):
        inputs = convBlock(inputs, n_filters)
    return inputs


def resnetBlock_v1(inputs, n_filters, blocks=1, kernel=3, rate=1, stride=1, first=False, n_conv=3):
    skip = inputs
    if first:
        skip = convBlock_v1(skip, n_filters * 4, stride=stride, kernel=1)
    for i in range(n_conv):
        inputs = convBlock_v1(inputs, n_filters * (1 if i < 2 else 4),\
         kernel = 1 if i % 2  == 0 else 3, \
         stride=1 if i != 0 else stride, rate=1 if i < n_conv-1 else rate)
    inputs = add(inputs, skip)
    return inputs

def conv_block_v2(inputs, n_filters, blocks=3):
    for _ in range(blocks):
        inputs = convBlock_v2(inputs, n_filters)
    return inputs

def convBlock_v1(inputs, n_filters, kernel=3,rate=1, stride=1):
    net = Conv2D(n_filters, kernel_size=[kernel, kernel], dilation_rate=rate, activation = None, strides=stride, padding ='same')(inputs)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    return net

def conv1x1(inputs, n_filters, stride=1, act=None):
    net = Conv2D(n_filters, kernel_size=1, strides=stride, activation = act,padding ='same')(inputs)
    return net

def shortcut(net, res, n_filters, not_equal=False):
    if not_equal:
        res = conv1x1(res, n_filters)
    net = Add()([net, res])
    return net

def encoder_block(inputs, n_filters, blocks=3):
    for _ in range(blocks):
        inputs = convBlock(inputs, n_filters)
    return inputs

def resBlock(inputs, n_filters, n_conv=3):
    skip = inputs
    skip = convBlock(inputs, n_filters)
    for _ in range(n_conv):
        inputs = convBlock(inputs, n_filters)
    skip = add(inputs, skip)
    return convBlock(skip, n_filters)

def encoder_block_v2(inputs, n_filters, blocks=3):
    for _ in range(blocks):
        inputs = convBlock_v2(inputs, n_filters)
    return inputs


def transBlock(inputs, n_filters):
    net = Conv2DTranspose(n_filters, kernel_size=[2, 2], strides=1, activation='relu')(inputs)
    return net

def sepConvBlock(inputs, n_filters):
    net = SeparableConv2D(n_filters, kernel=(3, 3), activation='relu')(inputs)
    net = conv1x1(net, n_filters, act='relu')
    net = SeparableConv2D(n_filters, kernel=(3, 3), dilation=2, activation='relu')(net)
    net = conv1x1(net, n_filters,act='relu')
    net = SeparableConv2D(n_filters, kernel=(3, 3), dilation=4, activation='relu')(net)
    net = conv1x1(net, n_filters, act='relu')
    return net

def pixelShuffle(inputs):
    inputs = x
    return inputs

def add(tensor_a, tensor_b):
    return Add()([tensor_a, tensor_b])

def add_v1(tensorList):
    result = tensorList[0]
    for tensor in tensorList[1:]:
        result = add(tensor, result)
    return result

def pool(inputs, p=[2, 2], stride=[2, 2], pooling_type='MAX', padding='same'):
    pll = MaxPooling2D(pool_size=p, strides=stride, padding=padding)(inputs)
    return pll

def xffBlock(tensor_g, tensor_x, tensor_add):
    add_ = Lambda(lambda x : x+1.0)(tensor_g)
    sub_ = Lambda(lambda x : 1.0 - x)(tensor_g)
    return add(multiply(add_,tensor_x), multiply(sub_,tensor_add))

def sub(tensor_a, tensor_b):
    return Subtract()([tensor_a, tensor_b])

def concat(input_A, input_B, axis=-1):
    net = Concatenate(axis)([input_A, input_B])
    return net

def concat_v2(inputs_list, axis=-1):
    for i,inputs in enumerate(inputs_list):
        if i ==0 :
            inputs_res = inputs
        else:
            inputs_res = concat(inputs_res, inputs)
    return inputs_res

def interBlock(inputs, level, feature_map, pooling_type):
    kernel_size = [int(np.round(float(feature_map[0])/float(level))), int(np.round(float(feature_map[1])/float(level)))]
    net = pool(inputs, kernel_size, stride=kernel_size)
    net = convBlock_v1(net, 512, kernel=1)
    net = Upsampling_v1(net, kernel_size)
    return net
    # net = BatchNormalization()(net)


def PyramidPoolingModule(inputs, feature_map, pooling_type):
    interBlock1 = interBlock(inputs, 1, feature_map, pooling_type)
    interBlock2 = interBlock(inputs, 2, feature_map, pooling_type)
    interBlock3 = interBlock(inputs, 4, feature_map, pooling_type)
    interBlock4 = interBlock(inputs, 6, feature_map, pooling_type)
    net = concat_v2([inputs, interBlock1, interBlock2, interBlock3, interBlock4], axis=-1)

    return net

def FuseGFFConvBlock(inputs, n_filters, kernel=3, stride=1, blocks=2):
    for _ in range(blocks):
        inputs = convBlock_v1(inputs, n_filters)
    return inputs


def desconv_v3(inputs, n_filters, rate=2):

    net = Conv2DTranspose(n_filters, kernel_size=rate, strides=rate)(inputs)
    return net

def desconv_v2(inputs, n_filters, rate=2):
    #def upBlock(inputs, n_filters, rate=2):
    up = UpSampling2D(rate)(inputs)
    net =  convBlock_v2(up, n_filters, kernel=2)
    return net


def desconv(inputs, n_filters, rate=2):
    #def upBlock(inputs, n_filters, rate=2):
    up = UpSampling2D(rate)(inputs)
    net =  convBlock(up, n_filters, kernel=2)
    return net

def Upsampling_v1(inputs, rate=2):
    return UpSampling2D(rate)(inputs)

def Upsampling_v2(inputs, n_filters, rate=2, layers=3):
    for i in range(layers):
        inputs = convUpBlock(inputs, n_filters * (4//(i+1)), rate)
        inputs = convBlock(inputs, n_filters* ((4//(i+1))))

    return inputs


def convUpBlock(inputs, n_filters, rate=2):
    net = BatchNormalization()(inputs)
    net = Activation('relu')(net)
    # net = transBlock(net, n_filters)
    # net = Upsampling_v1(net)
    # print(H, W, 'XTTTTXX')
    return desconv_v3(net, n_filters, rate)
    # return net
