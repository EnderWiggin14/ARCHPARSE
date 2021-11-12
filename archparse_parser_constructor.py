# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 16:21:53 2020

@author: Michael
"""

import argparse
typeChoices = ['FC','CONV']
actChoices = ['relu','sig','elu','exp','selu','spls','ssgn','None',
              'lin','smax','swsh','hsig','tanh','slisin','slicos',
              'repolu','relpolu','nlrelu','lognrelu']
initChoices = ['xavu','xavn','glu','gln','heu','hen','orth',
               'rndn','vsc','zeros']

def construct_parsers():
    """
    Construvct parser for parsing layer arguments.

    Parameters
    ----------
    parser : argparse.parser
        parser to which arguments will be added

    Returns
    -------
    None.

    """
    parser = argparse.ArgumentParser()
    parser2 = argparse.ArgumentParser()
    group = parser2.add_mutually_exclusive_group()
    group.add_argument("-i",metavar = 'initializer', dest = 'init',
                        default = 'xavn',choices = initChoices,
                        help="Kernel initializer function.")
    group.add_argument('-ki', metavar = 'kernel_init', dest = 'init', help = "Kernel initializer function.")

    parser2.add_argument('-bi', metavar = 'bias_init', dest = 'bias_init', default = 'zeros', help = 'The bias initializer. Default is \'zeros\'')

    parser2.add_argument('-kr',metavar='kernelReg',dest='kernelReg',nargs='+',default=None,
                          help='Option to add a kernel regularizer. Choices are L1, L2, or L1L2. L1 and L2 each need one additional float parameter. L1L2 needs two additional float parameters.')
    parser2.add_argument('-br',metavar='biasReg',dest='biasReg',nargs='+',default=None,
                          help='Option to add a bias regularizer. Choices are L1, L2, or L1L2. L1 and L2 each need one additional float parameter. L1L2 needs two additional float parameters.')
    parser2.add_argument('-ar',metavar='actReg',dest='actReg',nargs='+',default=None,
                          help='Option to add a activity regularizer. Choices are L1, L2, or L1L2. L1 and L2 each need one additional float parameter. L1L2 needs two additional float parameters.')
    parser2.add_argument('-b', metavar = 'use_bias', dest = 'use_bias', default = True, type = bool, help = "Sets whether a bias is used. Default is True")

    subparsers = parser.add_subparsers(title='type',dest='type',help="")

    ## The Fully Connected Layer
    parse_fc = subparsers.add_parser('FC',parents=[parser2],add_help=False,help="Option to make a Fully Connected (Dense) layer.")
    parse_fc.add_argument('size',type=int,
                          help="The size of the layer. It must be an integer greater than zero")
    parse_fc.add_argument('-a',metavar = 'activation', dest = 'act', default = ['lin'], nargs='+',
                            help="Activation function choice. The options are ["+ ','.join(actChoices)+']')

    ## The Conv1D Layer
    parse_conv = subparsers.add_parser('CONV',parents=[parser2],add_help=False,help="Option to make a 1D-convolutional layer.")
    conv_parser(parse_conv)

    ## The Conv1DTranspose Layer
    parse_convT = subparsers.add_parser('CONVT',parents=[parser2],add_help=False,help="Option to make a 1D deconvolutional layer.")
    convt_parser(parse_convT)


    ## The Conv2D Layer
    parse_conv2 = subparsers.add_parser('CONV2',parents=[parser2],add_help=False,help="Option to make a 2D-convolutional layer")
    conv2d_parser(parse_conv2)

    ## The Conv2DTranspose Layer
    parse_conv2t = subparsers.add_parser('CONV2T',parents=[parser2],add_help=False,help="Option to make a 2D-convolutional transpose layer")
    conv2dt_parser(parse_conv2t)

    ##  The Flatten Layer
    parse_flat = subparsers.add_parser("FLAT", help="Add a flatten layer")

    ## The Reshape Layer
    parse_rshp = subparsers.add_parser("RSHP", help="Add a reshape layer")
    parse_rshp.add_argument('shape',metavar='dimensions', nargs='+',type=int,help="A list of the dimension sizes")

    ## The MaxPool1D Layer
    parse_mxp1 = subparsers.add_parser("MXP1",help="Add a MaxPool1D layer")
    maxPool1d_parser(parse_mxp1)

    ## The AveragePool1D Layer
    parse_avp1 = subparsers.add_parser("AVP1",help="Add an AveragePool1D layer")
    avgPool1d_parser(parse_avp1)

    ## The UpSampling1D Layer
    parse_ups1 = subparsers.add_parser("UPS1",help="Add an UpSampling1D layer")
    upSample1D_parser(parse_ups1)

    #The PolynomialRegression Layer
    parse_polreg = subparsers.add_parser("POLREG", parents=[parser2], add_help = False, help = "Add a PolynomialRegression layer")
    polynomialRegression_parser(parse_polreg)

    return parser

def conv_parser(parser):
    """
    Create 1D convolution parser.

    Parameters
    ----------
    parser : argparse.parser
        parser to which arguments will be added

    Returns
    -------
    None.

    """
    parser.add_argument('filters',type=int,help="The number of filters to be used.")
    parser.add_argument('size',type=int,help="The size of the convolutional kernel." )

    parser.add_argument('-a',metavar = 'activation', dest = 'act', default = ['lin'], nargs='+',
                            help="Activation function choice. The options are ["+ ','.join(actChoices)+']')
    parser.add_argument('-pad',metavar='padType',dest='pad',default='v',
                            choices = ['v','s','c'],
                            help="These options correspond to the allowable 'padding' options for tf.keras.layers.Conv1D()")
    parser.add_argument('-s',metavar="stride",dest='stride',default=1,type=int,
                            help="The length of the stride between filter applications.")
    parser.add_argument('-d',metavar='dilation',dest='dilation',default=1,type=int,
                            help="The rate of dilation.")

def convt_parser(parser):
    """
    Create 1D deconvolution (convolution transpose) parser.

    Parameters
    ----------
    parser : argparse.parser
        parser to which arguments will be added

    Returns
    -------
    None.

    """
    parser.add_argument('filters',type=int,help="The number of filters to be used.")
    parser.add_argument('size',type=int,help="The size of the convolutional kernel." )

    parser.add_argument('-a',metavar = 'activation', dest = 'act', default = ['lin'], nargs='+',
                            help="Activation function choice. The options are ["+ ','.join(actChoices)+']')
    parser.add_argument('-pad',metavar='padType',dest='pad',default='v',
                            choices = ['v','s'],
                            help="These options correspond to the allowable 'padding' options for tf.keras.layers.Conv1D()")
    parser.add_argument('-s',metavar="stride",dest='stride',default=1,type=int,
                            help="The length of the stride between filter applications.")
    parser.add_argument('-o',metavar="outputPadding",dest = 'oPad',default='None',help='The amount of padding along the non-channel dimension. The value must be less that the size of stride.' )
    parser.add_argument('-d',metavar='dilation',dest='dilation',default=1,type=int,
                            help="The rate of dilation.")

def conv2d_parser(parser):
    """
    Create 2D convolution parser.

    Parameters
    ----------
    parser : argparse.parser
        parser to which arguments will be added

    Returns
    -------
    None.

    """
    ## The Conv2D Layer

    parser.add_argument('filters',type=int,help="The number of filters to be used.")
    parser.add_argument('size',nargs=2,type=int,help="The size of the convolutional kernel." )

    parser.add_argument('-a',metavar = 'activation', dest = 'act', default = ['lin'], nargs='+',
                            help="Activation function choice. The options are ["+ ','.join(actChoices)+']')
    parser.add_argument('-pad',metavar='padType',dest='pad',default='v',
                            choices = ['v','s','c'],
                            help="These options correspond to the allowable 'padding' options for tf.keras.layers.Conv2DTranspose()")
    parser.add_argument('-s',metavar="stride",dest='stride',default=1,nargs='+',type=int,
                            help="The length of the stride between filter applications.")
    parser.add_argument('-d',metavar='dilation',dest='dilation',default=1,type=int,nargs='+',
                            help="The rate of dilation.")

def conv2dt_parser(parser):
    """
    Create 2D deconvolution (convolution transpose) parser.

    Parameters
    ----------
    parser : argparse.parser
        parser to which arguments will be added

    Returns
    -------
    None.

    """
    parser.add_argument('filters',type=int,help="The number of filters to be used.")
    parser.add_argument('size',nargs=2,type=int,help="The size of the convolutional kernel." )

    parser.add_argument('-a',metavar = 'activation', dest = 'act', default = ['lin'], nargs='+',
                            help="Activation function choice. The options are ["+ ','.join(actChoices)+']')
    parser.add_argument('-pad',metavar='padType',dest='pad',default='v',
                            choices = ['v','s','c'],
                            help="These options correspond to the allowable 'padding' options for tf.keras.layers.Conv2DTranspose()")
    parser.add_argument('-s',metavar="stride",dest='stride',default=1,nargs='+',type=int,
                            help="The length of the stride between filter applications.")
    parser.add_argument('-d',metavar='dilation',dest='dilation',default=1,type=int,nargs='+',
                            help="The rate of dilation.")
    parser.add_argument('-o',metavar="outputPadding",dest = 'oPad',default='None',
                        nargs='*',help='The amount of padding along the non-channel dimension. The value must be less that the size of stride.' )

def maxPool1d_parser(parser):
    """
    Create 1D MaxPooling parser.

    Parameters
    ----------
    parser : argparse.parser
        parser to which arguments will be added

    Returns
    -------
    None.

    """
    parser.add_argument('size',type=int,default=2,help="The size of the pool. Default size is 2.")
    parser.add_argument('-pad',metavar='padType',dest='pad',default='v',
                            choices = ['v','s','c'],
                            help="These options correspond to the allowable 'padding' options for tf.keras.layers.MaxPool1D()")
    parser.add_argument('-s',metavar="stride",dest='stride',default=1,type=int,
                            help="The length of the stride between filter applications.")

# def maxPool2d_parser(parser):
#     """
#     Create 2D MaxPooling parser.

#     Parameters
#     ----------
#     parser : argparse.parser
#         parser to which arguments will be added

#     Returns
#     -------
#     None.

#     """
#     parser.add_argument('size',type=int,nargs='+',default=2,help="The size of the pool. Default size is 2.")
#     parser.add_argument('-pad',metavar='padType',dest='pad',default='v',
#                             choices = ['v','s','c'],
#                             help="These options correspond to the allowable 'padding' options for tf.keras.layers.MaxPool2D()")
#     parser.add_argument('-s',metavar="stride",dest='stride',nargs='+',default=1,type=int,
#                             help="The length of the stride between filter applications.")

def avgPool1d_parser(parser):
    """
    Create 1D AveragePooling parser.

    Parameters
    ----------
    parser : argparse.parser
        parser to which arguments will be added

    Returns
    -------
    None.

    """
    parser.add_argument('size',type=int,default=2,help="The size of the pool. Default size is 2.")
    parser.add_argument('-pad',metavar='padType',dest='pad',default='v',
                            choices = ['v','s','c'],
                            help="These options correspond to the allowable 'padding' options for tf.keras.layers.AveragePooling1D()")
    parser.add_argument('-s',metavar="stride",dest='stride',default=1,type=int,
                            help="The length of the stride between filter applications.")

def upSample1D_parser(parser):
    """
    Create 1D Up-Sampling parser.

    Parameters
    ----------
    parser : argparse.parser
        parser to which arguments will be added

    Returns
    -------
    None.

    """
    parser.add_argument('size',type=int,default=2,help="The upsampling width. A size of 3 means that each value will have 2 extra copies. Default size is 2.")

def upSample2D_parser(parser):
    """
    Create 2D Up-Sampling parser.

    Parameters
    ----------
    parser : argparse.parser
        parser to which arguments will be added

    Returns
    -------
    None.

    """
    parser.add_argument('size',type=int,nargs='+',default=2,help="The upsampling of rows and columns. A size of 3 means that each value will have 2 extra copies. Default size is 2.")

def polynomialRegression_parser(parser):
    """
    Create Polynomial Regression parser.

    Parameters
    ----------
    parser : argparse.parser
        parser to which arguments will be added

    Returns
    -------
    None.

    """
    parser.add_argument( '-n', metavar = "n_poly", dest = 'n_poly', type = int, default = 1, help= "The number of polynomial repetitions.")
    parser.add_argument( '-o', metavar = "order", dest = 'order', type = int, default = 2, help = "The order of the polynomial activation functions.")
    parser.add_argument( '-pad', metavar = "node_padding", dest = "poly_pad", type = int, default = 0,
                        help = "The number of nodes to append to layer to allow any number of inputs or outputs when used as an output layer (i.e. for autoencoders). The activation function of these nodes is relu")
    parser.add_argument( '-a', metavar = "activation", dest = "poly_act", type = str, default = 'relpolu', choices = ['repolu,relpolu'], help = "The choice of activation class.")

