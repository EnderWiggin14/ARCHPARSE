# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 09:19:35 2020

@author: vanderwal1

summary

description

:REQUIRES:

:TODO: Add support for non-sequential models and multiple inputs and outputs.
   1. Add support for more model types. (VAE,DJINN)
   2. Give model types their own construction logic.
   3. Add support for non-SGD-based optimizers if possible (X-BFGS)
   4. Make functions that reduce the number of commands needed by user to build
        a model.

:AUTHOR: Michael Vander Wal ( vanderwal1 )
:ORGANIZATION: Lawrence-Livermore National Laboratory (LLNL)
:CONTACT:
:SINCE: Tue Jun  9 09:19:35 2020
:VERSION: 0.1
"""
#===============================================================================
# PROGRAM METADATA
#===============================================================================
__author__ = 'Michael Vander Wal ( vanderwal1 )'
__contact__ = ''
__copyright__ = ''
__license__ = ''
__date__ = 'Tue Jun  9 09:19:35 2020'
__version__ = '0.2.0'

import tensorflow as tf
import tensorflow.keras as krs
# import numpy as np
# import os
from sys import exit
import argparse

try:
    from archparse_parameter_check import parameter_check
    from archparse_parser_constructor import construct_parsers
    import archparse_parameter_check as arparc
    from archparse_tf_funcs import convert_args
    from archparse_custom_tf import MeanInfNormMSE, custom_objects
except:
    raise

TYPE_LINE = 0
NAME_LINE = TYPE_LINE + 1
INPUT_DIM_LINE = NAME_LINE + 1
LATENT_DIM_LINE = INPUT_DIM_LINE + 1
END_OF_HEADER = LATENT_DIM_LINE + 3

supportedLayers = ['FC','CONV','CONVT','CONV2','CONV2T','FLAT','RSHP','MXP1'
                   'AVP1','UPS1','POLREG']
regChoices = ['L1','L2','L1L2','l1','l2','l1l2']
regLayerChoices = ['CONV','CONVT','CONV2','CONV2T','FC']
typeChoices = ['FC','CONV']
actChoices = ['relu','sig','elu','exp','selu','spls','ssgn','None',
              'lin','smax','swsh','hsig','tanh','slisin','slicos',
              'repolu','relpolu','nlrelu','lognrelu']
initChoices = ['xavu','xavn','glu','gln','orth']

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
            'vsc':  'VarianceScaling'
            }
padMap = {
            'v': 'valid',
            's': 'same',
            'c': 'causal'
            }
lossMap = {
            'BINCR': krs.losses.BinaryCrossentropy,
            'CATCR': krs.losses.CategoricalCrossentropy,
            'CATH':  krs.losses.CategoricalHinge,
            'COSIM': krs.losses.CosineSimilarity,
            'HINGE': krs.losses.Hinge,
            'HUBER': krs.losses.Huber,
            'KLD':   krs.losses.KLDivergence,
            'LOGC':  krs.losses.LogCosh,
            'MAE':   krs.losses.MeanAbsoluteError,
            'MAPE':  krs.losses.MeanAbsolutePercentageError,
            'MSE':   krs.losses.MeanSquaredError,
            'MSLE':  krs.losses.MeanSquaredLogarithmicError,
            'POIS':  krs.losses.Poisson,
            'SCATC': krs.losses.SparseCategoricalCrossentropy,
            'SQHIN': krs.losses.SquaredHinge,
            'MIMSE' : MeanInfNormMSE
            }
regMap = {
            'L1': krs.regularizers.L1,
            'L2': krs.regularizers.L2,
            'L1L2': krs.regularizers.L1L2
            }

class InputSyntaxError(Exception):
    pass

def read_lines(filePath,fileName):
    """
    Read lines of input file.

    Parameters
    ----------
    filePath : str
        Path to directory of input file.
    fileName : str
        Name of input file.

    Returns
    -------
    lines : list
        List of lines from input file.

    """
    file = open("%s%s"%(filePath,fileName),"r")
    lines = file.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].partition('#')[0].strip()
    return lines

def parse_loss(line):
    """
    Parse input arguments of a line and return a loss function.

    Builds parser for loss function inputs and then parses the line containing
    the loss funciton.

    Parameters
    ----------
    line : str
        The line from the input file that specifies the loss functio to be used.

    Returns
    -------
    loss :
        The loss function to be used.

    TODO
    ----
    1. Instead of using large if-else if tree, utilize a dict to functions that
        use **kwargs

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-reduct',metavar='reduction',dest='reduct',type=str,choices = ['none','sum','sum_over_batch','sum_over_batch_size','auto'],default='auto', help = "Reduction method, Default is AUTO")
    subparser = parser.add_subparsers(title = "LOSS FUNCS",dest = 'loss')

    binCross = subparser.add_parser('BINCR',add_help=True, help = "Binary Cross-Entropy")
    binCross.add_argument('-logits',action='store_true', help="Boolean - whether to interpret as logits")
    binCross.add_argument('-smooth',metavar='smooth',dest='smooth',type=float,default=0.0, help="label smoothing - in range [0,1]. 0 is not smoothing, 1 is heaviest smothing.")

    catCross = subparser.add_parser('CATCR',add_help=True, help = "Categorical Cross-Entropy")
    catCross.add_argument('-logits',action='store_true', help="Boolean - whether to interpret as logits")
    catCross.add_argument('-smooth',metavar='smooth',dest='smooth',type=float,default=0.0, help="label smoothing - in range [0,1]. 0 is not smoothing, 1 is heaviest smothing.")

    catHinge = subparser.add_parser('CATH',add_help=True, help = "Categorical Hinge")

    cosSim = subparser.add_parser('COSIM',add_help=True, help = "Cosine Similarity")
    cosSim.add_argument('-axis',metavar='axis',dest='axis',type=int,default=-1, help = "Axis along which cosine similarity is computed")

    hinge = subparser.add_parser('HINGE',add_help=True, help = "Hinge")

    huber = subparser.add_parser('HUBER',add_help=True, help = "Huber")
    huber.add_argument('-delta',metavar='delta',dest='delta',type=float,default=1.0, help = "a float that is the point where the Huber loss function changes from quadratic to linear.")

    kld = subparser.add_parser('KLD',add_help=True, help="Kulback-Liebler Divergence")
    logc = subparser.add_parser('LOGC',add_help=True, help = "LogCosh")
    mae = subparser.add_parser('MAE',add_help=True, help = "Mean Absolute Error")
    mape = subparser.add_parser('MAPE',add_help=True, help = "Mean Absolute Percent Error")
    mse = subparser.add_parser('MSE',add_help=True, help = "Mean Square Error")
    msle = subparser.add_parser('MSLE',add_help=True, help = "Mean Square Logarithmic Error")
    poisson = subparser.add_parser('POIS',add_help=True, help = "Poisson")

    mimse = subparser.add_parser('MIMSE',add_help=True, help = "Mean-Infinity Norm + Mean Square Error")
    mimse.add_argument('-mse','--mse_weight',metavar="mse_weight",dest="mse_weight",type=float,default=.5, help = "MSE weight")
    mimse.add_argument('-inf','--inf_weight',metavar="inf_weight",dest="inf_weight",type=float,default=.5, help = "Mean-Inf Norm weight")

    sparseCatCross = subparser.add_parser('SCATC',add_help=True, help = "Sparse Categorical Cross-Entropy")
    sparseCatCross.add_argument('-logits',action='store_true', help="Boolean - whether to interpret as logits")

    sqHinge = subparser.add_parser('SQHIN',add_help=True, help = "Square Hinge")

    args = vars(parser.parse_args(line))

    if args['reduct'] == 'auto':
        reduct = krs.losses.Reduction.AUTO
    elif args['reduct'] == 'none':
        reduct = krs.losses.Reduction.NONE
    elif args['reduct'] == 'sum':
        reduct = krs.losses.Reduction.SUM
    else:
        reduct = krs.losses.Reduction.SUM_OVER_BATCH_SIZE

    lossFunc = args['loss']
    if lossFunc == 'BINCR':
        loss = lossMap[lossFunc](from_logits=args['logits'],label_smoothing=args['smooth'],reduction=reduct)
    elif lossFunc == 'CATCR':
        loss = lossMap[lossFunc](from_logits=args['logits'],label_smoothing=args['smooth'],reduction=reduct)
    elif lossFunc == 'COSIM':
        loss = lossMap[lossFunc](axis=args['axis'],reduction=reduct)
    elif lossFunc == 'HUBER':
        loss = lossMap[lossFunc](delta=args['delta'],reduction=reduct)
    elif lossFunc == 'SCATC':
        loss = lossMap[lossFunc](from_logits=args['logits'],reduction=reduct)
    elif lossFunc == 'MIMSE':
        loss = lossMap[lossFunc](mse_weight=args['mse_weight'],infNorm_weight=args['inf_weight'])
    else:
        loss = lossMap[lossFunc](reduction=reduct)

    return loss

def parse_body(lines):
    """
    Extract the various sections of the input file.

    This extracts the header information and some of the training information
    as well as the architecture information in blocks/sections. These blocks
    undergo further processing either in this function or another function.

    Parameters
    ----------
    lines : list
        The lines from the input file.

    Raises
    ------
    InputSyntaxError
        If there is a line with unknown structure/syntax or there is one or
        more blocks missing or in a an unexpected order, an error is raised.
    ValueError
        If the first lines is not recognized abbreviation for a supported
        network, an error is raised.

    Returns
    -------
    param_dict : dict
        A dictionary of the network parameters/blocks.

    TODO
    ----
    Break the model structure parsing into different functions for different
    network types.

    Use a dict to addressed by the abbreviation in the first line of input
    which declares the network type to access those functions.

    """
    if 'AE' in lines and 'NN' in lines:
        raise InputSyntaxError("AE and NN cannot both be present in the same file.")

    networkType = lines[TYPE_LINE]
    name = lines[NAME_LINE]

    inputDim = lines[INPUT_DIM_LINE].split()[1:]
    for i in range(len(inputDim)):
        inputDim[i] = int(inputDim[i])

    latentDim = lines[LATENT_DIM_LINE].split()[1:]
    for i in range(len(latentDim)):
        latentDim[i] = int(latentDim[i])

    if 'LOSS' in lines[LATENT_DIM_LINE+1].upper():
        global END_OF_HEADER
        END_OF_HEADER = LATENT_DIM_LINE+4
        # print("END_OF_HEADER",END_OF_HEADER)
        sep_loss = lines[LATENT_DIM_LINE+1].split()
        lossLine = [sep_loss[1].upper()]
        if len(sep_loss) > 2:
            lossLine += sep_loss[2:]

        # print("-----------LossLine----------",lossLine)
        loss = parse_loss(lossLine)
    else:
        lossLine = ['MSE']
        loss = parse_loss(lossLine)



    inputDim = tuple(inputDim)
    latentDim = tuple(latentDim)
    networkSets = []

    if networkType == 'AE':
        if not "ENCOD" in lines or not "DECOD" in lines:
            raise InputSyntaxError("ENCOD and DECOD are both expected in the input file when AE input provided. One or both were not provided")
        encoderStart=lines.index('ENCOD')
        decoderStart=lines.index('DECOD')
        encoderLines = []
        decoderLines = []

        for i in range(encoderStart+1,decoderStart-1):
            encoderLines.append(lines[i])
        for i in range(decoderStart+1,len(lines)):
            decoderLines.append(lines[i])
        networkSets = (encoderLines,decoderLines)

    elif networkType == 'NN':
        if "ENCOD" in lines or "DECOD" in lines:
            raise InputSyntaxError("ENCOD and DECOD cannot be present when NN is provided.")
        if not "NTWRK" in lines:
            raise InputSyntaxError("NTWRK block must be present when NN is provided")
        networkStart = lines.index('NTWRK')
        networkLines = []
        for i in range(networkStart+1,len(lines)):
            networkLines.append(lines[i])
        networkSets = (networkLines,)

    else:
        raise ValueError("The first line of input file must be either NN or AE, but %s was given"%(networkType))

    custom_model_components = {}
    if isinstance(loss,custom_objects):
        for i in range(len(custom_objects)):
            if isinstance(loss,custom_objects[i]):
                key = str(type(loss))
                key = key.split('\'')[1]
                key = key.rpartition('.')[2]
                custom_model_components[key] = custom_objects[i]
                # custom_model_components[key] = loss
        # print(custom_model_components)

    # USE FOR OPTIMIZER LATER
    # if isinstance(opt,custom_objects):
    #     key = str(loss)
    #     key = key.split('\'')[1]
    #     key = key.rpartition('.')[2]
    #     custom_model_components[key] = opt

    param_dict = {    'type': networkType,
                      'name': name,
                      'inputDim': inputDim,
                      'latentDim': latentDim,
                      'unparsed_model': networkSets,
                      'loss': loss,
                      'optimizer': None,
                      'callbacks': None,
                      'algorithms': None,
                      'nInputs': 1,
                      'learningRate': .001,
                      'custom': custom_model_components}

    return param_dict

def parse_line(line,lineNumber=-1):
    """
    Parse input file line corresponding to a network layer type.

    Parameters
    ----------
    line : list
        List of strings that were the substrings in the input file separated
        by whitespace.
    lineNumber : int
        The line number of the file from which the command came from.
        Default is -1.

    Raises
    ------
    ValueError
        If the input for an activation function is not recognized,
        an error is raised.

    Returns
    -------
    args : dict
        A dictionary with arguments as keys which point to their
        associated values.

    TODO
    ----
    Pull contruct_parsers() out of this function and either pass the parser as
    a function argument or as a global. Possibly make it an overridable global.

    """
    if '-a' in line and not line[line.index('-a')+1] in actChoices:
        raise ValueError('The choice given for an activation function on line %i is not supported.'%(lineNumber))
    parser = construct_parsers()

    args = parser.parse_args(line)
    # print(line)
    # print(args)
    args = vars(args)

    parameter_check(args,lineNumber)

    # print(args['act'],args['init'])
    # print(args['actOptions'])
    # parser.print_help()
    # exit()
    # for item in args.items():
    #     print(item)
    return args

def link_layers(argList,input_shape):
    """
    Link the layers of the model/sub-model together.


    Parameters
    ----------
    argList : list
        List of argument dictionaries for layers.
    input_shape : tuple
        Shape of input for the model.

    Returns
    -------
    tensorflow.keras.Model
        A linked model.

    Notes
    -----
    This is currently only capable of supporting single inputs and
    sequential models.

    TODO
    ----
    Add multiple input support

    """
    layers = []
    inputs = krs.Input(shape=input_shape)
    x = inputs
    for args in argList:
        layers.append(convert_args(args))

    for layer in layers:
        x = layer(x)

    return krs.Model(inputs,x)

# def process_arch_file(filePath,fileName,debug=False):
#     """
#     Process the architecture input file.

#     This function handles the passing of input to the various stages of
#     parsing and translation.

#     The model isn't directly built with this function. Instead a list of the
#     translated inputs is returned to be done with as the user wishes.

#     Parameters
#     ----------
#     filePath : str
#         The path to the directory of the input file.
#     fileName : str
#         The name of the input file.
#     debug : bool, optional
#         Used to set whether debugging messages will be printed.
#         The default is False.

#     Raises
#     ------
#     Exception
#         If the debug=True and an error occurs during parsing due to an
#         invalid option.

#     Returns
#     -------
#     layers : list of layers
#         DESCRIPTION.

#     """
#     lines = read_lines(filePath,fileName)
#     lineNumber = 1
#     layers = []
#     if not debug:
#         try:
#             for line in lines:
#                 print(line.partition('\n')[0])
#                 args = parse_line(line.split(),lineNumber)
#                 layers.append(convert_args(args))
#                 print(type(layers[-1]))
#                 lineNumber += 1
#             return layers
#         except ValueError:
#             raise
#         except:
#             raise Exception("Invalid option(s) on line %i of %s%s."%(lineNumber,filePath,fileName))
#     else:
#         for line in lines:
#             print(line.partition('\n')[0])
#             args = parse_line(line.split(),lineNumber)
#             layers.append(convert_args(args))
#             print(type(layers[-1]))
#             lineNumber += 1
#         return layers

def retrieve_args_parsedBody(parsedBody):
    """
    Retrieve args from parsedBody.

    This function extracts the arguments by calling `parse_line` on the lines
    stored in parsedBody dict. The arguments are grouped by the sub-model to
    which they belong.

    Parameters
    ----------
    parsedBody : dict
        Dictionary containing the model information..

    Returns
    -------
    argSet : tuple
        The tuple contains the arguments for all of the sub-models of the model.

    """
    netType = parsedBody['type']
    networkSets = parsedBody['unparsed_model']
    if netType == 'AE':
        encArgs = []
        decArgs = []
        lineNumber = END_OF_HEADER+1
        print("line number",lineNumber, "end of header",END_OF_HEADER+1)
        for line in networkSets[0]:
            print(line.partition('\n')[0])
            encArgs.append(parse_line(line.split(),lineNumber))
            lineNumber += 1
        lineNumber += 3
        for line in networkSets[1]:
            print(line.partition('\n')[0])
            decArgs.append(parse_line(line.split(),lineNumber))
            lineNumber += 1
        argSet = (encArgs,decArgs)

    elif netType == 'NN':
        args = []
        lineNumber = END_OF_HEADER
        for line in networkSets[0]:
            print(line.partition('\n')[0])
            args.append(parse_line(line.split(),lineNumber))
            lineNumber += 1
        argSet = (args,)
    return argSet

def retrieve_args(parsedBody = None, lines = None, filePath=None, fileName=None, debug=False):
    """
    Extract layer construction arguments from parsedBody.

    Retrieves the layer construction arguments from one of many different input
    methods. Only of of parsedBody, lines, or filePath & fileName should be
    passed. If more than one is is passed priority is as follows

    parsedBody > lines > filePath & fileName

    parsedBody should be used as the other input options currently do nothing.
    The other options may be removed in the future.

    Parameters
    ----------
    parsedBody : dict, optional
        Dictionary containing the model information. The default is None.
    lines : list, optional
        List of strings that are the lines of input file. The default is None.
    filePath : str, optional
        Path to the directory of the input file. The default is None.
    fileName : str, optional
        Name of the input file. The default is None.
    debug : bool, optional
        Used to set whether debug statements are printed. The default is False.

    Raises
    ------
    Exception
        If an invalid option is encountered during parsing, and error is raised.

    Returns
    -------
    tuple
        The tuple contains the arguments for all of the sub-models of the model.

    TODO
    ----
    Either remove lines, filePath, and fileName, or properly re-implement them.

    """
    # Priority is parsedBody > lines > filePath & fileName
    if not parsedBody is None:
        if not debug:
            try:
                return retrieve_args_parsedBody(parsedBody)
            except:
                raise
        else:
            return retrieve_args_parsedBody(parsedBody)


    # else:
    #     if lines is None:
    #         lines = read_lines(filePath,fileName)

    #     lineNumber = 1
    #     args = []
    #     if not debug:
    #         try:
    #             for line in lines:
    #                 print(line.partition('\n')[0])
    #                 args.append(parse_line(line.split(),lineNumber))
    #                 lineNumber += 1
    #             return args
    #         except ValueError:
    #             raise
    #         except:
    #             raise Exception("Invalid option(s) on line %i of %s%s."%(lineNumber,filePath,fileName))
    #     else:
    #         for line in lines:
    #             print(line.partition('\n')[0])
    #             args.append(parse_line(line.split(),lineNumber))
    #             lineNumber += 1
    #         return args

# def generate_layers(args,debug=False):
#     lineNumber = 1
#     layers = []
#     if not debug:
#         try:
#             for line in args:
#                 layers.append(convert_args(line))
#                 print(type(layers[-1]))
#                 lineNumber += 1
#             return layers
#         except:
#             print("\033[0;31;48m Attention: \033[0;37;48m Invalid option(s) on line %i of input"%(lineNumber))
#             raise
#     else:
#         for line in args:
#             layers.append(convert_args(line))
#             print(type(layers[-1]))
#             lineNumber += 1
#         return layers

def generate_model( parsedBody, argSet):
    """
    Translate arguments to layers and link layers.

    The arguments in argSet are converted into tensorflow.keras layers.
    The layers are then linked together to make the full model. The models and
    its sub-models, if there are sub-models, are returned in a tuple.

    Parameters
    ----------
    parsedBody : dict
        Dictionary containing the model information.
    argSet : tuple
        The arguments of the sub-models.

    Returns
    -------
    modelSet : tuple
        The full model and its sub-models. The sub-models are in order of
        appearance in the full model. The last item in the tuple is the
        full model.

    TODO
    ----
    Possibly add non-sequential model support.

    """
    # modelSet = (0,)
    netType = parsedBody['type']
    name = parsedBody['name']
    inDim = parsedBody['inputDim']
    latDim = parsedBody['latentDim']
    loss = parsedBody['loss']
    # networkSets = parsedBody[NETWORK_SET_INDEX]

    print(f"inDim :  {inDim}   latDim : {latDim}")
    if netType == 'AE':
        encoder = link_layers(argSet[0],inDim)
        encoder._name = "encoder"
        # encoder.summary()

        decoder = link_layers(argSet[1],latDim)
        decoder._name = "decoder"
        # decoder.summary()

        inputs = krs.Input(shape=inDim)
        autoencoder = krs.Model(inputs,decoder(encoder(inputs)),name="%s_%s"%(name,'AE'))
        autoencoder.run_eagerly=True
        extract_custom_layers(parsedBody['custom'],autoencoder)

        modelSet = (encoder,decoder,autoencoder)
    elif netType == 'NN':
        neuralNet = link_layers(argSet[0],inDim)
        neuralNet._name = "%s_%s"%(name,'NN')
        modelSet = (neuralNet,)

    return modelSet

def extract_custom_layers(custom_model_components,layer):
    """
    Extract custom layers types from the model.

    This function searches for any custom layers in the model so that they can
    be passed as the custom_objects argument in `tf.keras.models.load_model`.

    Parameters
    ----------
    custom_model_components : dict
        This is dict where the handles to custom_objects will be stored.
    layer : tensorflow.keras.layers.Layer
        The current layers being checked to see if it is a custom object.

    Returns
    -------
    None.

    """
    try:
        for layer in layer.layers:
            if isinstance(layer,custom_objects):
                key = str(type(layer))
                key = key.split('\'')[1]
                key = key.rpartition('.')[2]
                custom_model_components[key] = layer
            extract_custom_layers(custom_model_components, layer)
    except:
        return

# def extract_shape(args):
#     shape = []
#     for line in args:
#         if line['type'] == 'FC':
#             shape.append(line['size'])
#         elif line['type'] == 'CONV':
#             shape.append(-1)
#     return shape

# def extract_activations(args):
#     acts = []
#     for line in args:
#         acts.append(line['act'])
#     return acts

def main():
    """
    Use for testing.

    Returns
    -------
    None.

    """
    krs.backend.clear_session()
    # filePath = "./"
    # fileName = "test.arch"
    # lines = read_lines(filePath,fileName)
    # parsedBody = parse_body(lines)

    # combArgs = retrieve_args(parsedBody = parsedBody, debug = True)
    # modelSet = generate_model(parsedBody, combArgs)


if __name__ == "__main__":
    main()