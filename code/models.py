import lasagne
import lasagne.layers as lyr
from lasagne import nonlinearities


def cnn_autoencoder(input_var=None):
    """
    Build the network using Lasagne library
    """

    ##################
    # Network config #
    ##################

    input_channels = 3
    weight_init = lasagne.init.Normal()

    # encoder
    conv1_nb_filt = 32
    conv1_sz_filt = (9, 9)
    conv1_sz_padd = 2
    # conv1 output size = (60, 60)

    pool1_sz = (2, 2)
    # pool1 output size = (30, 30)

    conv2_nb_filt = 64
    conv2_sz_filt = (7, 7)
    conv2_sz_padd = 0
    # conv2 output size = (24, 24)

    pool2_sz = (4, 4)
    # pool2 size = (6, 6)

    conv3_nb_filt = 128
    conv3_sz_filt = (5, 5)
    conv3_sz_padd = 0
    # conv3 output size = (2, 2)

    pool3_sz = (2, 2)
    # pool3 output size = (32, 1, 1)

    dens1_nb_unit = 256
    # dense1 output (vector 256)

    dens2_nb_unit = 256
    # dense2 output (vector 256)

    rshp_sz = 1
    # reshape output (256, 1, 1)

    # decoder
    tconv1_nb_filt = 64
    tconv1_sz_filt = (5, 5)
    tconv1_sz_strd = (1, 1)
    # conv1 output size = (5, 5)

    upsamp1_sz = (2, 2)
    # upsamp1 output size = (10, 10)

    tconv2_nb_filt = 32
    tconv2_sz_filt = (4, 4)
    tconv2_sz_strd = (1, 1)
    # tconv2 output size = (13, 13)

    upsamp2_sz = (2, 2)
    # upsamp2 output size = (26, 26)

    tconv3_nb_filt = 32
    tconv3_sz_filt = (5, 5)
    tconv3_sz_strd = (1, 1)
    # tconv3 output size = (30, 30)

    tconv4_nb_filt = 3
    tconv4_sz_filt = (3, 3)
    tconv4_sz_strd = (1, 1)
    # tconv4 output size = (32, 32)

    # final output = (3 channels, 32 x 32)

    #####################
    # Build the network #
    #####################

    # Add input layer
    network = lyr.InputLayer(
        shape=(None, input_channels, 64, 64), input_var=input_var)

    # Add convolution layer
    network = lyr.Conv2DLayer(incoming=network, num_filters=conv1_nb_filt,
                              filter_size=conv1_sz_filt, pad=conv1_sz_padd,
                              W=weight_init)
    # Add pooling layer
    network = lyr.MaxPool2DLayer(incoming=network, pool_size=pool1_sz)

    # Add convolution layer
    network = lyr.Conv2DLayer(incoming=network, num_filters=conv2_nb_filt,
                              filter_size=conv2_sz_filt, pad=conv2_sz_padd,
                              W=weight_init)
    # Add pooling layer
    network = lyr.MaxPool2DLayer(incoming=network, pool_size=pool2_sz)

    # Add convolution layer
    network = lyr.Conv2DLayer(incoming=network, num_filters=conv3_nb_filt,
                              filter_size=conv3_sz_filt, pad=conv3_sz_padd,
                              W=weight_init)
    # Add pooling layer
    network = lyr.MaxPool2DLayer(incoming=network, pool_size=pool3_sz)

    network = lyr.FlattenLayer(network)

    # Add dense layer
    network = lyr.DenseLayer(network, dens1_nb_unit, W=weight_init)
    network = lyr.DenseLayer(network, dens2_nb_unit, W=weight_init)

    network = lyr.ReshapeLayer(
        network, (input_var.shape[0], dens2_nb_unit / (rshp_sz ** 2), rshp_sz, rshp_sz))

    # Add transposed convolution layer
    network = lyr.TransposedConv2DLayer(incoming=network,
                                        num_filters=tconv1_nb_filt,
                                        filter_size=tconv1_sz_filt,
                                        stride=tconv1_sz_strd,
                                        W=weight_init)
    # Add upsampling layer
    network = lyr.Upscale2DLayer(incoming=network, scale_factor=upsamp1_sz)

    # Add transposed convolution layer
    network = lyr.TransposedConv2DLayer(incoming=network,
                                        num_filters=tconv2_nb_filt,
                                        filter_size=tconv2_sz_filt,
                                        stride=tconv2_sz_strd,
                                        W=weight_init)
    # Add upsampling layer
    network = lyr.Upscale2DLayer(incoming=network, scale_factor=upsamp2_sz)

    # Add transposed convolution layer
    network = lyr.TransposedConv2DLayer(incoming=network,
                                        num_filters=tconv3_nb_filt,
                                        filter_size=tconv3_sz_filt,
                                        stride=tconv3_sz_strd,
                                        W=weight_init)

    # Add transposed convolution layer
    network = lyr.TransposedConv2DLayer(incoming=network,
                                        num_filters=tconv4_nb_filt,
                                        filter_size=tconv4_sz_filt,
                                        stride=tconv4_sz_strd,
                                        W=weight_init,
                                        nonlinearity=lasagne.nonlinearities.sigmoid)

    return network


def small_cnn_autoencoder(input_var=None):
    """
    Build the network using Lasagne library
    """

    ##################
    # Network config #
    ##################

    input_channels = 3
    weight_init = lasagne.init.Normal()

    # encoder
    conv1_nb_filt = 32
    conv1_sz_filt = (9, 9)
    conv1_sz_padd = 2
    # conv1 output size = (32, 60, 60)

    pool1_sz = (2, 2)
    # pool1 output size = (32, 30, 30)

    conv2_nb_filt = 32
    conv2_sz_filt = (7, 7)
    conv2_sz_padd = 0
    # conv2 output size = (32, 24, 24)

    pool2_sz = (4, 4)
    # pool2 size = (32, 6, 6)

    conv3_nb_filt = 32
    conv3_sz_filt = (5, 5)
    conv3_sz_padd = 0
    # conv3 output size = (32, 2, 2)

    pool3_sz = (2, 2)
    # pool3 output size = (32, 1, 1)

    dens1_nb_unit = 256
    # dense1 output (vector 256)

    dens2_nb_unit = 256
    # dense2 output (vector 256)

    rshp_sz = 4
    # reshape output (64, 4, 4)

    # decoder
    tconv1_nb_filt = 32
    tconv1_sz_filt = (4, 4)
    tconv1_sz_strd = (1, 1)
    # conv1 output size = (32, 7, 7)

    upsamp1_sz = (2, 2)
    # upsamp1 output size = (32, 14, 14)

    tconv2_nb_filt = 3
    tconv2_sz_filt = (3, 3)
    tconv2_sz_strd = (1, 1)
    # tconv2 output size = (3, 16, 16)

    upsamp2_sz = (2, 2)
    # upsamp2 output size = (3, 32, 32)

    # final output = (3 channels, 32 x 32)

    #####################
    # Build the network #
    #####################

    # Add input layer
    network = lyr.InputLayer(
        shape=(None, input_channels, 64, 64), input_var=input_var)

    # Add convolution layer
    network = lyr.Conv2DLayer(incoming=network, num_filters=conv1_nb_filt,
                              filter_size=conv1_sz_filt, pad=conv1_sz_padd,
                              W=weight_init)
    # Add pooling layer
    network = lyr.MaxPool2DLayer(incoming=network, pool_size=pool1_sz)

    # Add convolution layer
    network = lyr.Conv2DLayer(incoming=network, num_filters=conv2_nb_filt,
                              filter_size=conv2_sz_filt, pad=conv2_sz_padd,
                              W=weight_init)
    # Add pooling layer
    network = lyr.MaxPool2DLayer(incoming=network, pool_size=pool2_sz)

    # Add convolution layer
    network = lyr.Conv2DLayer(incoming=network, num_filters=conv3_nb_filt,
                              filter_size=conv3_sz_filt, pad=conv3_sz_padd,
                              W=weight_init)
    # Add pooling layer
    network = lyr.MaxPool2DLayer(incoming=network, pool_size=pool3_sz)

    network = lyr.FlattenLayer(network)

    # Add dense layer
    network = lyr.DenseLayer(network, dens1_nb_unit, W=weight_init)
    network = lyr.DenseLayer(network, dens2_nb_unit, W=weight_init)

    network = lyr.ReshapeLayer(
        network, (input_var.shape[0], dens2_nb_unit / (rshp_sz ** 2), rshp_sz, rshp_sz))

    # Add transposed convolution layer
    network = lyr.TransposedConv2DLayer(incoming=network,
                                        num_filters=tconv1_nb_filt,
                                        filter_size=tconv1_sz_filt,
                                        stride=tconv1_sz_strd,
                                        W=weight_init)
    # Add upsampling layer
    network = lyr.Upscale2DLayer(incoming=network, scale_factor=upsamp1_sz)

    # Add transposed convolution layer
    network = lyr.TransposedConv2DLayer(incoming=network,
                                        num_filters=tconv2_nb_filt,
                                        filter_size=tconv2_sz_filt,
                                        stride=tconv2_sz_strd,
                                        W=weight_init,
                                        nonlinearity=lasagne.nonlinearities.sigmoid)
    # Add upsampling layer
    network = lyr.Upscale2DLayer(incoming=network, scale_factor=upsamp2_sz)

    return network


class DCGAN:
    """
    DCGAN implementation based on Yeh et al. 2016
    http://arxiv.org/pdf/1607.07539v2
    - with batch norm
    - with leaky relus in discriminator only (0.2 slope)
    - with regular relus in generator
    """

    def __init__(self):
        pass

    def init_discriminator(self, input_var=None):
        """
        Initialize the DCGAN discriminator network using lasagne
        Returns the network
        """

        lrelu = nonlinearities.LeakyRectify(0.2)

        network = lyr.InputLayer((None, 3, 64, 64), input_var)
        print 'discr input layer shape:\t', network.output_shape

        network = lyr.Conv2DLayer(
            incoming=network, num_filters=64, filter_size=5, stride=2, pad=2,
            nonlinearity=lrelu
        )
        print 'discr layer output shape:\t', network.output_shape

        network = lyr.batch_norm(lyr.Conv2DLayer(
            incoming=network, num_filters=128, filter_size=5, stride=2, pad=2,
            nonlinearity=lrelu
        ))
        print 'discr layer output shape:\t', network.output_shape

        network = lyr.batch_norm(lyr.Conv2DLayer(
            incoming=network, num_filters=256, filter_size=5, stride=2, pad=2,
            nonlinearity=lrelu
        ))
        print 'discr layer output shape:\t', network.output_shape

        network = lyr.batch_norm(lyr.Conv2DLayer(
            incoming=network, num_filters=512, filter_size=5, stride=2, pad=2,
            nonlinearity=lrelu
        ))
        print 'discr layer output shape:\t', network.output_shape

        network = lyr.batch_norm(lyr.DenseLayer(
            incoming=network, num_units=1,
            nonlinearity=nonlinearities.sigmoid
        ))
        print 'discr layer output shape:\t', network.output_shape

        return network

    def init_generator(self, input_var=None):
        """
        Initialize the DCGAN generator network using lasagne
        Returns the network
        """

        network = lyr.InputLayer((None, 100), input_var)

        network = lyr.batch_norm(lyr.DenseLayer(
            incoming=network, num_units=4*4*1024, nonlinearity=nonlinearities.rectify
        ))

        network = lyr.ReshapeLayer(
            incoming=network, shape=(input_var.shape[0], 1024, 4, 4)
        )

        network = lyr.batch_norm(lyr.TransposedConv2DLayer(
            incoming=network, num_filters=512, filter_size=5, stride=2, crop=2,
            nonlinearity=nonlinearities.rectify
        ))

        network = lyr.batch_norm(lyr.TransposedConv2DLayer(
            incoming=network, num_filters=256, filter_size=5, stride=2, crop=2,
            nonlinearity=nonlinearities.rectify
        ))

        network = lyr.batch_norm(lyr.TransposedConv2DLayer(
            incoming=network, num_filters=128, filter_size=5, stride=2, crop=2,
            nonlinearity=nonlinearities.rectify
        ))

        network = lyr.batch_norm(lyr.TransposedConv2DLayer(
            incoming=network, num_filters=3, filter_size=5, stride=2, crop=2,
            nonlinearity=nonlinearities.tanh
        ))
        print 'generator output shape:', network.output_shape
        return network
