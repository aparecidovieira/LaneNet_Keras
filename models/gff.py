###
import keras
from keras.layers import *
import keras.backend as K
import common as C
from keras.models import *

n_filters = 64

filters = [n_filters, 2 * n_filters, 4 * n_filters, 8 * n_filters, 16 * n_filters, 32 * n_filters]


def build_gff(input_shape, one_hot_label=False, pooling_type='MAX'):
    print(input_shape, 'Input size shape ~')
    input_layer = Input((input_shape))
    # n_filters = 64

    inputs_ = C.convBlock_v1(input_layer, filters[0])
    inputs_ = C.convBlock_v1(inputs_, filters[0], stride=2)##128, 128

    # inputs_ = C.pool(inputs_)##64, 64

    enc1 = C.resnetBlock_v1(inputs_, filters[0], first=True)
    enc1 = C.resnetBlock_v1(enc1, filters[0])
    x1 = C.resnetBlock_v1(enc1, filters[0]) ## (64, 64)
    x1n = C.conv1x1(x1, filters[2], stride=2) ##  (32, 32)
    g1 = C.conv1x1(x1n, filters[2])
    g1 = Activation('sigmoid')(g1) ##  (32, 32)

    # skip1 = C.pool(enc1)
    # _skip1 = C.concat(skip1, inputs_waves1)

    enc2 = C.resnetBlock_v1(x1, filters[1], first=True, stride=2) #(32, 32)
    enc2 = C.resnetBlock_v1(enc2, filters[1])
    enc2 = C.resnetBlock_v1(enc2, filters[1])
    x2 = C.resnetBlock_v1(enc2, filters[1]) #(32, 32)
    x2n = C.conv1x1(x2, filters[2])
    g2 = C.conv1x1(x2n, filters[2])
    g2 = Activation('sigmoid')(g2)## (32, 32)
    # _skip2 = C.concat(skip2, inputs_waves2)

    enc3 = C.resnetBlock_v1(x2, filters[2], first=True, rate=2)## (32, 32)
    enc3 = C.resnetBlock_v1(enc3, filters[2])
    enc3 = C.resnetBlock_v1(enc3, filters[2])
    enc3 = C.resnetBlock_v1(enc3, filters[2])
    enc3 = C.resnetBlock_v1(enc3, filters[2])
    x3 = C.resnetBlock_v1(enc3, filters[2]) #(32, 32)
    x3n = C.conv1x1(x3, filters[2])
    g3 = C.conv1x1(x3n, filters[2])
    g3 = Activation('sigmoid')(g3) ## (32,32)
    # skip3 = C.pool(enc3)
    # _skip3 = C.concat(skip3, inputs_waves3)

    enc4 = C.resnetBlock_v1(x3, filters[3], first=True, rate=4)
    enc4 = C.resnetBlock_v1(enc4, filters[3])
    x4 = C.resnetBlock_v1(enc4, filters[3]) #(32, 32)
    x4n = C.conv1x1(x4, filters[2])
    g4 = C.conv1x1(x4n, filters[2])
    g4 = Activation('sigmoid')(g4)


    m1 = C.multiply(g1, x1n)
    m2 = C.multiply(g2 , x2n)
    m3  = C.multiply(g3, x3n)
    m4 = C.multiply(g4, x4n)

    add1 = C.add_v1([m2, m3, m4])
    add2 = C.add_v1([m1, m3, m4])
    add3 = C.add_v1([m2, m1 ,m4])
    add4 = C.add_v1([m2, m3 , m1])


    x1gff = C.xffBlock(g1, x1n, add1)#(C.add(1, g1),x1n) + (1-g1)*(add1)
    x2gff = C.xffBlock(g2, x2n, add2)#(1+g2)*x2n + (1-g2)*(add2)
    x3gff = C.xffBlock(g3, x3n, add3)#(1+g3)*x3n + (1-g3)*(add3)
    x4gff = C.xffBlock(g4, x4n, add4)#(1+g4)*x4n + (1-g4)*(add4)
    # skip4 = C.pool(enc4)
    # _skip4 = C.concat(skip4, inputs_waves4)

    # enc5 = C.resBlock_v1(_skip4, 8 * n_filters)
    # skip5 = C.pool(enc5)


    full_block = C.concat_v2([x1gff, x2gff, x3gff, x4gff], axis=-1)


    x1gff = C.FuseGFFConvBlock(x1gff, 256)
    x2gff = C.FuseGFFConvBlock(x2gff, 256)
    x3gff = C.FuseGFFConvBlock(x3gff, 256)
    x4gff = C.FuseGFFConvBlock(x4gff, 256)


    label = [n_filters, n_filters]
    feature_map = [x/8 for x in label]

    psp = C.PyramidPoolingModule(x4, feature_map, pooling_type)

    d5 = C.concat_v2([psp, x1gff, x2gff, x3gff, x4gff], axis=-1)
    d4 = C.concat_v2([x1gff, x2gff, x3gff, x4gff], axis=-1)
    d3 = C.concat_v2([x1gff, x2gff, x3gff], axis=-1)
    d2 = C.concat_v2([x1gff, x2gff], axis=-1)
    d1 = x1gff

    full_block = C.concat_v2([d1, d2, d3, d4, d5], axis=-1)
    net = C.convBlock_v1(full_block, filters[3])
    net = C.Upsampling_v2(net, filters[0], layers=2)
    print(net.shape.as_list(), 'Shape')

    if one_hot_label:
        net = Conv2D(2, 1, 1, activation='relu', border_mode='same')(net)
        net = Reshape((2,input_shape[0]*input_shape[1]))(net)
        net = Permute((2,1))(net)
        net = Activation('softmax')(net)
    else:
        net = Conv2D(1, 1, 1, activation='sigmoid')(net)

    model = Model(inputs=input_layer, outputs=net)

    return model
