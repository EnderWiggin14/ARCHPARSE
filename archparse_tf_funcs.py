# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 16:30:28 2020

@author: Michael
"""
import tensorflow.keras as krs
import tensorflow.keras.activations as acts
import archparse_custom_tf as custom

regChoices = ['L1','L2','L1L2','l1','l2','l1l2']
regLayerChoices = ['CONV','CONVT','CONV2','CONV2T','FC']
typeChoices = ['FC','CONV']
actChoices = ['relu','sig','elu','exp','selu','spls','ssgn','None',
              'lin','smax','swsh','hsig','tanh','slisin','slicos',
              'repolu','relpolu','nlrelu','lognrelu']
initChoices = ['xavu','xavn','glu','gln','heu','hen','orth',
               'rndn','vsc','zeros']

actMap = {
            'elu':  'elu',
            'exp':  'exponential',
            'hsig': 'hard_sigmoid',
            'lin':  'linear',
            'relu': 'relu',
            'selu': 'selu',
            'sig':  'sigmoid',
            'smax': 'softmax',
            'spls': 'softplus',
            'ssgn': 'softsign',
            'swsh': 'swish',
            'tanh': 'tanh',
            'slisin': custom.Activation_Generator.slisin(),
            'slicos': custom.Activation_Generator.slicos(),
            'nlrelu': custom.Activation_Generator.nlrelu(),
            'repolu': custom.Activation_Generator.repolu(),
            'relpolu': custom.Activation_Generator.relpolu(),
            'lognrelu': custom.Activation_Generator.lognrelu(),
            'None': None
            }
initMap = {
            'xavu': 'GlorotUniform',
            'xavn': 'GlorotNormal',
            'glu':  'GlorotUniform',
            'gln':  'GlorotNormal',
            'orth': 'Orthogonal',
            'heu':  'he_uniform',
            'hen':  'he_normal',
            'rndn': 'RandomNormal',
            'rndu': 'RandomUniform',
            'vsc':  'VarianceScaling',
            'zeros': 'zeros'
            }
padMap = {
            'v': 'valid',
            's': 'same',
            'c': 'causal'
            }
regMap = {
            'L1': krs.regularizers.L1,
            'L2': krs.regularizers.L2,
            'L1L2': krs.regularizers.L1L2
            }

def convert_args(args):
    """
    Convert arguments to tensorflow layers.

    This function uses a dictionary to map the layer types in args['type'] to
    the appropriate function which generates a corresponding tensorflow layer.

    Parameters
    ----------
    args : dict
        dictionary of arguments necessary to construct the layer.

    Returns
    -------
    tensorflow.keras.layers.Layer
        The constructed layer.

    """
    switcher = {
                'AVP1': get_AvgPool1D,
                'CONV': get_Conv1D,
                'CONVT': get_Conv1DTranspose,
                'CONV2': get_Conv2D,
                'CONV2T': get_Conv2DTranspose,
                'FC':   get_Dense,
                'FLAT': get_Flatten,
                'RSHP': get_Reshape,
                'MXP1': get_MaxPool1D,
                'UPS1': get_UpSample1D,
                'POLREG': get_PolynomialRegression
                }
    return switcher[args['type']](args)


def assign_Regularizers(args):
    """
    Convert regularizer arguments into regularizers to be used in layers.

    Parameters
    ----------
    args : dict
        dictionary of arguments necessary to construct the layer.

    Returns
    -------
    None.

    """
    regList = ['kernelReg','biasReg','actReg']
    for i in regList:
        if args[i] is None:
            pass
        elif args[i][0] == 'L1L2':
            args[i] = regMap[args[i][0]](l1=args[i][1],l2=args[i][2])
        else:
            args[i] = regMap[args[i][0]](args[i][1])

def get_Conv1D(args):
    """
    Create tensorflow Conv1D layer.

    Parameters
    ----------
    args : dict
        dictionary of arguments necessary to construct the layer.

    Returns
    -------
    layer : tensorflow.keras.layers.Conv1D
        The constructed Conv1D layer.

    """
    assign_Regularizers(args)
    layer = krs.layers.Conv1D(filters = args['filters'],
                              kernel_size = args['size'],
                              strides = args['stride'],
                              padding = padMap[args['pad']],
                              data_format = 'channels_last',
                              dilation_rate = 1,
                              activation = actMap[(args['act'])],
                              use_bias = args['use_bias'],
                              kernel_initializer = initMap[args['init']],
                              bias_initializer = initMap[args['bias_init']],
                              kernel_regularizer = args['kernelReg'],
                              bias_regularizer = args['biasReg'],
                              activity_regularizer = args['actReg'],
                              kernel_constraint = None,
                              bias_constraint = None)
    return layer

def get_Conv1DTranspose(args):
    """
    Create tensorflow Conv1DTranspose layer.

    Parameters
    ----------
    args : dict
        dictionary of arguments necessary to construct the layer.

    Returns
    -------
    layer : tensorflow.keras.layers.Conv1DTranspose
        The constructed Conv1D layer.

    """
    assign_Regularizers(args)
    layer = krs.layers.Conv1DTranspose(filters = args['filters'],
                                       kernel_size = args['size'],
                                       strides = args['stride'],
                                       padding= padMap[args['pad']],
                                       output_padding = args['oPad'],
                                       data_format = 'channels_last',
                                       dilation_rate = 1,
                                       activation = actMap[args['act']],
                                       use_bias = args['use_bias'],
                                       kernel_initializer = initMap[args['init']],
                                       bias_initializer = initMap[args['bias_init']],
                                       kernel_regularizer = args['kernelReg'],
                                       bias_regularizer = args['biasReg'],
                                       activity_regularizer = args['actReg'],
                                       kernel_constraint = None,
                                       bias_constraint = None)
    return layer

def get_Conv2D(args):
    """
    Create tensorflow Conv2D layer.

    Parameters
    ----------
    args : dict
        dictionary of arguments necessary to construct the layer.

    Returns
    -------
    layer : tensorflow.keras.layers.Conv2D
        The constructed Conv2D layer.

    """
    assign_Regularizers(args)
    layer = krs.layers.Conv2D(filters = args['filters'],
                              kernel_size = tuple(args['size']),
                              strides = args['stride'],
                              padding = padMap[args['pad']],
                              data_format = 'channels_last',
                              dilation_rate = 1,
                              activation = actMap[(args['act'])],
                              use_bias = args['use_bias'],
                              kernel_initializer = initMap[args['init']],
                              bias_initializer = initMap[args['bias_init']],
                              kernel_regularizer = args['kernelReg'],
                              bias_regularizer = args['biasReg'],
                              activity_regularizer = args['actReg'],
                              kernel_constraint = None,
                              bias_constraint = None)
    return layer

def get_Conv2DTranspose(args):
    """
    Create tensorflow Conv2DTranspose layer.

    Parameters
    ----------
    args : dict
        dictionary of arguments necessary to construct the layer.

    Returns
    -------
    layer : tensorflow.keras.layers.Conv2DTranspose
        The constructed Conv2DTranpose layer.

    """
    assign_Regularizers(args)
    layer = krs.layers.Conv2DTranspose(filters = args['filters'],
                                       kernel_size = tuple(args['size']),
                                       strides = args['stride'],
                                       padding= padMap[args['pad']],
                                       output_padding = args['oPad'],
                                       data_format = 'channels_last',
                                       dilation_rate = 1,
                                       activation = actMap[args['act']],
                                       use_bias = args['use_bias'],
                                       kernel_initializer = initMap[args['init']],
                                       bias_initializer = initMap[args['bias_init']],
                                       kernel_regularizer = args['kernelReg'],
                                       bias_regularizer = args['biasReg'],
                                       activity_regularizer = args['actReg'],
                                       kernel_constraint = None,
                                       bias_constraint = None)
    return layer

def get_Dense(args):
    """
    Create tensorflow Dense layer.

    Parameters
    ----------
    args : dict
        dictionary of arguments necessary to construct the layer.

    Returns
    -------
    layer : tensorflow.keras.layers.Dense
        The constructed Dense layer.

    """
    assign_Regularizers(args)
    layer = krs.layers.Dense(units = args['size'],
                             activation = actMap[args['act']],
                             use_bias = args['use_bias'],
                             kernel_initializer = initMap[args['init']],
                             bias_initializer = initMap[args['bias_init']],
                             kernel_regularizer = args['kernelReg'],
                             bias_regularizer = args['biasReg'],
                             activity_regularizer = args['actReg'],
                             kernel_constraint = None,
                             bias_constraint = None)
    return layer

def get_Flatten(args):
    """
    Create tensorflow Flatten layer.

    Parameters
    ----------
    args : dict
        dictionary of arguments necessary to construct the layer.

    Returns
    -------
    layer : tensorflow.keras.layers.Flatten
        The constructed Flatten layer.

    """
    layer = krs.layers.Flatten()
    return layer

def get_Reshape(args):
    """
    Create tensorflow Reshape layer.

    Parameters
    ----------
    args : dict
        dictionary of arguments necessary to construct the layer.

    Returns
    -------
    layer : tensorflow.keras.layers.Reshape
        The constructed Reshape layer.

    """
    layer = krs.layers.Reshape(target_shape = args['shape'])
    return layer

def get_MaxPool1D(args):
    """
    Create tensorflow MaxPool1D layer.

    Parameters
    ----------
    args : dict
        dictionary of arguments necessary to construct the layer.

    Returns
    -------
    layer : tensorflow.keras.layers.MaxPool1D
        The constructed MaxPool1D layer.

    """
    layer = krs.layers.MaxPool1D(pool_size=args['size'],
                                 strides = args['stride'],
                                 padding = padMap[args['pad']],
                                 data_format = 'channels_last')
    return layer

def get_AvgPool1D(args):
    """
    Create tensorflow AveragePool1D layer.

    Parameters
    ----------
    args : dict
        dictionary of arguments necessary to construct the layer.

    Returns
    -------
    layer : tensorflow.keras.layers.AveragePool1D
        The constructed AveragePool1D layer.

    """
    layer = krs.layers.AveragePooling1D(pool_size=args['size'],
                                 strides = args['stride'],
                                 padding = padMap[args['pad']],
                                 data_format = 'channels_last')
    return layer

def get_UpSample1D(args):
    """
    Create tensorflow UpSample1D layer.

    Parameters
    ----------
    args : dict
        dictionary of arguments necessary to construct the layer.

    Returns
    -------
    layer : tensorflow.keras.layers.UpSample1D
        The constructed UpSample1D layer.

    """
    layer = krs.layers.UpSampling1D(size = args['size'])
    return layer

def get_PolynomialRegression(args):
    """
    Create PolynomialRegression (custom) layer.

    Parameters
    ----------
    args : dict
        dictionary of arguments necessary to construct the layer.

    Returns
    -------
    layer : archparse_custom_tf.PolynomialRegression
        The constructed PolynomialRegression layer.

    """
    assign_Regularizers(args)
    layer = custom.PolynomialRegression(n_poly = args['n_poly'],
                                        order = args['order'],
                                        padding = args['poly_pad'],
                                        use_bias = args['use_bias'],
                                        kernel_initializer = initMap[args['init']],
                                        bias_initializer = initMap[args['bias_init']],
                                        kernel_regularizer = args['kernelReg'],
                                        bias_regularizer = args['biasReg'],
                                        activity_regularizer = args['actReg'],
                                        kernel_constraint = None,
                                        bias_constraint = None)
    return layer