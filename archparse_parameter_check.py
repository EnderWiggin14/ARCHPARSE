# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 13:59:21 2020

@author: Michael
"""
actChoices = ['relu','sig','elu','exp','selu','spls','ssgn','None',
              'lin','smax','swsh','hsig','tanh','slisin','slicos',
              'repolu','relpolu','nlrelu','lognrelu']

def parameter_check(args,lineNumber):
    """
    Check input parameters for archparse.

    Uses a dict mapped to the function handles of each layer's verification
    function. The keys for mapping are in args['type'].

    Parameters
    ----------
    args : dict
        Dictionary of arguments to be verified.
    lineNumber : int
        Line number in .arch file.

    Raises
    ------
    ValueError
        If an invalid activation function option is present.

    Returns
    -------
    None.

    """
    switcher = {
                'AVP1': check_AvgPool1D,
                'CONV': check_Conv1D,
                'CONVT': check_Conv1DTranspose,
                'CONV2': check_Conv2D,
                'CONV2T': check_Conv2DTranspose,
                'FC':   check_Dense,
                'FLAT': check_Flatten,
                'RSHP': check_Reshape,
                'MXP1': check_MaxPool1D,
                'UPS1': check_UpSample1D,
                'POLREG': check_PolynomialRegression
                }
    switcher[args['type']](args,lineNumber)

    args['actOptions'] = []
    if 'act' in args.keys():
        if not args['act'][0] in actChoices:
            raise ValueError("Invalid activation option provided on line %i"%(lineNumber))
        if len(args['act']) == 1:
            args['act'] = args['act'][0]
        elif len(args['act']) > 1:
            args['actOptions'] = args['act'][1:]
            args['act'] = args['act'][0]

def check_AvgPool1D(args,lineNumber):
    """
    Check 1D Average Pooling inputs.

    Parameters
    ----------
    args : dict
        Dictionary of arguments to be verified.
    lineNumber : int
        Line number in .arch file.

    Raises
    ------
    ValueError
        If pool size option is less two.
        If the stride length is less than one.

    Returns
    -------
    None.

    """
    if args['size'] < 2:
        raise ValueError("The pool size on line %i must greater than or equal to two."%(lineNumber))
    if args['stride'] < 1:
        raise ValueError("The stride length on line %i must be an integer greater than or equal to one."%(lineNumber))
    check_regularizers(args,lineNumber)

def check_Conv1D(args,lineNumber):
    """
    Check 1D Convolution inputs.

    Parameters
    ----------
    args : dict
        Dictionary of arguments to be verified.
    lineNumber : int
        Line number in .arch file.

    Raises
    ------
    ValueError
        If the stride length is less than one.
        If the dilation is less than one.

    Returns
    -------
    None.

    """
    if args['stride'] < 1:
        raise ValueError("The value of the stride on line %i must be greater than zero."%(lineNumber))
    if args['dilation'] < 1:
        raise ValueError("The value of the dilation on line %i must be greater than zero."%(lineNumber))
    check_regularizers(args,lineNumber)

def check_Conv1DTranspose(args,lineNumber):
    """
    Check 1D Deconvolution inputs.

    Parameters
    ----------
    args : dict
        Dictionary of arguments to be verified.
    lineNumber : int
        Line number in .arch file.

    Raises
    ------
    ValueError
        If the stride length is less than one.
        If the dilation is less than one.
        If output padding type is neither None or int.
        If output padding larger than the stride length.
        If output padding is less than zero.

    Returns
    -------
    None.

    """
    if args['stride'] < 1:
        raise ValueError("The value of the stride on line %i must be greater than zero."%(lineNumber))
    if args['dilation'] < 1:
        raise ValueError("The value of the dilation on line %i must be greater than zero."%(lineNumber))

    if (args['oPad'] == 'None'):
        args['oPad'] = None
    else:
        try:
            args['oPad'] = int(args['oPad'])
        except:
            raise TypeError("The dataType for -o ( i.e. outputPadding) on line %i must be either None or int."%(lineNumber))
        if isinstance(args['oPad'],int):
            if args['oPad'] >= args['stride']:
                raise ValueError('The amount of output padding (option -o) on line %i must less than the size of stride'%(lineNumber))
            if args['oPad'] < 0:
                raise ValueError('The amount of output padding (option -o) on line %i must greater than or equal to zero.'%(lineNumber))
    check_regularizers(args,lineNumber)

def check_Conv2D(args,lineNumber):
    """
    Check 2D Convolution Inputs.

    Parameters
    ----------
    args : dict
        Dictionary of arguments to be verified.
    lineNumber : int
        Line number in .arch file.

    Raises
    ------
    ValueError
        If the stride length is less than one.
        If the dilation is less than one.
        If the stride dimensions are neither 1 or 2.
        If the dilation dimensions are niether 1 or 2.

    Returns
    -------
    None.

    """
    if len(args['stride']) == 1:
        if args['stride'][0] < 1:
            raise ValueError("The value of the stride on line %i must be greater than zero."%(lineNumber))
        args['stride'] = [args['stride'][0],args['stride'][0]]
    elif len(args['stride']) > 2:
        raise ValueError('The number of dimensions provided for the stride of a Conv2D layer on line %i must be 2 or 1.'%(lineNumber))
    for i in args['stride']:
        if i < 1:
            raise ValueError("The value of the stride on line %i must be greater than zero."%(lineNumber))
    if len(args['dilation']) == 1:
        if args['dilation'][0] < 1:
            raise ValueError("The value of the dilation on line %i must be greater than zero."%(lineNumber))
        args['dilation'] = [args['dilation'][0],args['dilation'][0]]
    elif len(args['dilation']) > 2:
        raise ValueError('The number of dimensions provided for the dilation of a Conv2D layer on line %i must be 2 or 1.'%(lineNumber))
    for i in args['dilation']:
        if i < 1:
            raise ValueError("The value of the dilation on line %i must be greater than zero."%(lineNumber))
    check_regularizers(args,lineNumber)

def check_Conv2DTranspose(args,lineNumber):
    """
    Check 2D Deconvolution inputs.

    Parameters
    ----------
    args : dict
        Dictionary of arguments to be verified.
    lineNumber : int
        Line number in .arch file.

    Raises
    ------
    ValueError
        If the stride length is less than one.
        If the dilation is less than one.
        If output padding is less than zero.
        If the stride dimensions are neither 1 or 2.
        If the dilation dimensions are niether 1 or 2.
        If the output dimensions are niether 1 or 2.
        If output padding type is neither None or int.

    Returns
    -------
    None.

    """
    if len(args['stride']) == 1:
        if args['stride'] < 1:
            raise ValueError("The value of the stride on line %i must be greater than zero."%(lineNumber))
        args['stride'] = [args['stride'],args['stride']]
    elif len(args['stride']) > 2:
        raise ValueError('The number of dimensions provided for the stride of a Conv2DTranspose layer on line %i must be 2 or 1.'%(lineNumber))
    for i in args['stride']:
        if i < 1:
            raise ValueError("The value of the stride on line %i must be greater than zero."%(lineNumber))
    if len(args['dilation']) == 1:
        if args['dilation'][0] < 1:
            raise ValueError("The value of the dilation on line %i must be greater than zero."%(lineNumber))
        args['dilation'] = [args['dilation'][0],args['dilation'][0]]
    elif len(args['dilation']) > 2:
        raise ValueError('The number of dimensions provided for the dilation of a Conv2DTranspose layer on line %i must be 2 or 1.'%(lineNumber))
    for i in args['dilation']:
        if i < 1:
            raise ValueError("The value of the dilation on line %i must be greater than zero."%(lineNumber))
    print("length of oPad is ",len(args['oPad']))
    if args['oPad'] == 'None':
        args['oPad'] = None
    else:
        if len(args['oPad']) == 1:
            print("length of oPad is 1")
            args['oPad'] += args['oPad'][0]
            print("length of oPad is %i"%(len(args['oPad']))," and is ",args['oPad'])
        elif len(args['oPad']) > 2:
            raise ValueError('The number of dimensions provided for the output padding of a Conv2DTranspose layer on line %i must be 2 or 1.'%(lineNumber))

        try:
            for i in range(len(args['oPad'])):
                args['oPad'][i] = int(args['oPad'][i])
                if args['oPad'][i] < 0:
                    raise ValueError("The value of the output padding on line %i must be greater than or equal to zero."%(lineNumber))
                elif args['oPad'][i] >= args['stride'][i]:
                    raise ValueError('The amount output padding (option -o) on lin %i must less than the size of stride'%(lineNumber))
        except ValueError:
            raise
        except:
            raise TypeError("The dataType for -o ( i.e. outputPadding) on line %i must be either None or int."%(lineNumber))
    check_regularizers(args,lineNumber)

def check_Dense(args,lineNumber):
    """
    Check Dense inputs.

    Parameters
    ----------
    args : dict
        Dictionary of arguments to be verified.
    lineNumber : int
        Line number in .arch file.

    Raises
    ------
    ValueError
        If layer size is less than one.

    Returns
    -------
    None.

    """
    if args['size'] < 1:
        raise ValueError("The layer size on line %i must greater than or equal to 1."%(lineNumber))
    check_regularizers(args,lineNumber)

def check_Flatten(args,lineNumber):
    """
    Check Flatten.

    Does nothing. Here for sake of completeness and ensure functioning code.

    Parameters
    ----------
    args : dict
        Dictionary of arguments to be verified.
    lineNumber : int
        Line number in .arch file.

    Returns
    -------
    None.

    """
    pass

def check_Reshape(args,lineNumber):
    """
    Checks Reshape inputs.

    Converts args['shape'] to a tuple.

    Parameters
    ----------
    args : dict
        Dictionary of arguments to be verified.
    lineNumber : int
        Line number in .arch file.

    Returns
    -------
    None.

    """
    args['shape'] = tuple(args['shape'])

def check_MaxPool1D(args,lineNumber):
    """
    Check 1D Max Pooling inputs.

    Parameters
    ----------
    args : dict
        Dictionary of arguments to be verified.
    lineNumber : int
        Line number in .arch file.

    Raises
    ------
    ValueError
        If size is less than two.
        If stride length is less than one.

    Returns
    -------
    None.

    """
    if args['size'] < 2:
        raise ValueError("The pool size on line %i must be an integer greater than or equal to two."%(lineNumber))
    if args['stride'] < 1:
        raise ValueError("The stride length on line %i must be an integer greater than or equal to one."%(lineNumber))

def check_UpSample1D(args,lineNumber):
    """
    Check 1D Up-Sampling inputs.

    Parameters
    ----------
    args : dict
        Dictionary of arguments to be verified.
    lineNumber : int
        Line number in .arch file.

    Raises
    ------
    ValueError
        If size is less than two.

    Returns
    -------
    None.

    """
    if args['size'] < 2:
        raise ValueError("The pool size on line %i must be an integer greater than or equal to two."%(lineNumber))

def check_regularizers(args,lineNumber):
    """
    Check regularizer inputs.

    Parameters
    ----------
    args : dict
        Dictionary of arguments to be verified.
    lineNumber : int
        Line number in .arch file.

    Raises
    ------
    ValueError
        If neither zero or two values are provided along with L1L2 option.
        If niether zero or one value are provided along with L1 or L2 options.

    Returns
    -------
    None.

    """
    regList = ['kernelReg','biasReg','actReg']
    for i in regList:
        if not args[i] is None:
            args[i][0] = args[i][0].upper()

            if args[i][0] == 'L1L2':
                if len(args[i]) != 3 and len(args[i]) != 1:
                    raise ValueError('Either no values or two values must be given for L1L2 regularization on line %i. No values given will result in use of TF default values.'%(lineNumber))
                elif len(args[i]) == 3:
                    args[i][1] = float(args[i][1])
                    args[i][2] = float(args[i][2])
                elif len(args[i]) == 1:
                    args[i].append( 0.0 )
                    args[i].append( 0.0 )

            if args[i][0] == 'L1' or args[i][0] == 'L2' :
                print(args[i])
                if len(args[i]) != 2 and len(args[i]) != 1:
                    raise ValueError('Either no values or one value must be given for L1 or L2 regularization on line %i. No values given will result in use of TF default values.'%(lineNumber))
                elif len(args[i]) == 2:
                    args[i][1] = float(args[i][1])
                elif len(args[i]) == 1:
                    args[i].append( 0.0 )

def check_PolynomialRegression( args, lineNumber ):
    """
    Check Polynomial Regression inputs.

    Parameters
    ----------
    args : dict
        Dictionary of arguments to be verified.
    lineNumber : int
        Line number in .arch file.

    Raises
    ------
    ValueError
        If the 'n_poly' option is less than 1.
        If the order option is less than 1.

    Returns
    -------
    None.

    """
    if args['n_poly'] < 1:
        raise ValueError('-n (n_poly) argument cannot be less than 1.')
    if args['order'] < 1:
        raise ValueError('-o (order) argument cannot be less than 1.')
