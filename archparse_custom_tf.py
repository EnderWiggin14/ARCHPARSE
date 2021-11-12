# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 09:41:47 2020

@author: vanderwal1

summary

description

:REQUIRES: tensorflow, numpy

:TODO:
    1. Rewrite repolu and relpolu to move the logic outside of the activation
        funtion if possible

:AUTHOR: Michael Vander Wal
:ORGANIZATION:
:CONTACT:
:SINCE: Tue Dec 15 09:41:47 2020
:VERSION: 0.1
"""

import tensorflow as tf
import tensorflow.keras as krs
from numpy import inf
# import numpy as np
# from tensorflow.keras import backend as K

class Activations:
    """
    Custom activation functions.

    Methods
    -------
    nlrelu
    lognrelu
    repolu
    relpolu
    slisin
    slicos

    """
    @staticmethod
    def nlrelu(x,beta = 1.0):
        """
        Evaluate nlrelu activation for input.

        Natural Log Relu [nlrelu]

        Parameters
        ----------
        x : numpy.ndarray,float,int
            Function input.
        beta : float, optional
            Scaling factor. The default is 1.0.

        Returns
        -------
        numpy.ndarray, float, int
            nlrelu activation output.

        Notes
        -----
        .. [nlrelu] insert citation

        """
        return tf.math.log1p( tf.math.multiply( beta, tf.math.maximum(x,0) ) )

    @staticmethod
    def lognrelu( x, beta = 1.0, base = 10.0 ):
        """
        Evaluate lognrelu activation for input.

        log_n Relu. See nlrelu for more information as this is a variant
        of nlrelu.

        Parameters
        ----------
        x : numpy.ndarray,float,int
            Function input.
        beta : float, optional
            Scaling factor. The default is 1.0.
        base : float, optional
            Base of logarithm used. The default is 10.0.

        Returns
        -------
        numpy.ndarray, float, int
            lognrelu activation output.

        """
        inv_base = 1/tf.math.log(base)
        return tf.multiply(tf.math.log1p( tf.math.multiply(beta,tf.math.maximum(x,0.) ) ), tf.math.log(inv_base) )

    @staticmethod
    def repolu(x, order=2, alpha = 0., max_value = None, threshold = 0. ):
        """
        Evaluate repolu activation for input.

        Rectified Polynomial Unit

        Piecewise function
        if x <= threshold   , alpha*x
        else                , (1/order) * x^order

        Parameters
        ----------
        x : numpy.ndarray,float,int
            Function input.
        order : int, optional
            Order of polynomial used. The default is 2.
        alpha : float, optional
            Slope of function for values less than the threshold.
            The default is 0.
        max_value : float, None, optional
            Sets the maximum value the function can return. The default is None.
        threshold : float, optional
            The value that must be exceeded in order for the ouput values to not
            be set to 0 or otherwise damped. The default is 0.

        Returns
        -------
        numpy.ndarray, float, int
            repolu activation output.

        """
        if order == 0:
            return tf.keras.activations.linear( x )
        elif order == 1:
            return tf.keras.activations.relu( x, alpha = 0., max_value = None, threshold = 0. )
        else:
            coef = 1/order
            if max_value is None and threshold == 0. and alpha == 0.:
                return tf.math.maximum( 0. , x )
            elif max_value is None:
                if alpha == 0.:
                    if threshold == 0. :
                        x = tf.math.maximum( 0. , x )
                        return tf.multiply(tf.math.pow( x, order ),coef)
                    else:
                        index = tf.where( tf.less( x, threshold ))
                        x[index] = 0.
                        x = tf.math.pow( x, order )
                        return tf.multiply( x, order )
                else:
                    index  = tf.where( tf.less(x, threshold ), x)
                    x_temp = x
                    x_temp[index] = tf.multiply( x_temp, alpha )
                    x[index] = 0.
                    x = tf.math.pow( x, order )
                    x[index] = x_temp[index]
                    return tf.multiply( x, order )
            else:
                if alpha == 0.:
                    if threshold == 0. :
                        x = tf.math.maximum( 0. , x )
                        return tf.multiply( tf.math.pow( x, order ), coef )
                    else:
                        index = tf.where( tf.less( x, threshold ))
                        x[index] = 0.
                        x = tf.math.maximum( x, max_value )
                        x = tf.math.pow( x, order )
                        return tf.multiply( x, order )
                else:
                    index  = tf.where( tf.less(x, threshold ), x)
                    x_temp = x
                    x_temp[index] = tf.multiply( x_temp, alpha )
                    x[index] = 0.
                    x = tf.math.maximum( x, max_value )
                    x = tf.math.pow( x, order )
                    x[index] = x_temp[index]
                    return tf.multiply( x, order )

    @staticmethod
    def relpolu(x, order=2, alpha = 0., max_value = None, threshold = 0. ):
        """
        Evaluate relpolu activation for input.

        Rectified Linear-Polynomial Unit

        Piecewise function
        if x <= threshold   , alpha*x
        else                , x + (1/order) * x^order

        The exception to this piecewise function is for order=1 which just
        produces the relu activation function.

        Parameters
        ----------
        x : numpy.ndarray,float,int
            Function input.
        order : int, optional
            Order of polynomial used. The default is 2.
        alpha : float, optional
            Slope of function for values less than the threshold.
            The default is 0.
        max_value : float, None, optional
            Sets the maximum value the function can return. The default is None.
        threshold : float, optional
            The value that must be exceeded in order for the ouput values to not
            be set to 0 or otherwise damped. The default is 0.

        Returns
        -------
        numpy.ndarray, float, int
            relpolu activation output.

        """
        if order == 0:
            return tf.keras.activations.linear( x )
        elif order == 1:
            return tf.keras.activations.relu( x, alpha = 0., max_value = None, threshold = 0. )
        else:
            # print("order: ",order)
            coef = 1/order
            if max_value is None and threshold == 0. and alpha == 0.:
                return tf.math.maximum( 0. , x )
            elif max_value is None:
                if alpha == 0.:
                    if threshold == 0. :
                        x = tf.math.maximum( 0. , x )
                        return tf.add(tf.multiply(tf.math.pow( x, order ),coef), x )
                    else:
                        index = tf.where( tf.less( x, threshold ))
                        x[index] = 0.
                        x = tf.add( tf.multiply( tf.math.pow( x, order ), coef ), x )
                        return x
                else:
                    index  = tf.where( tf.less( x, threshold ), x )
                    x_temp = x
                    x_temp[index] = tf.multiply( x_temp, alpha )
                    x[index] = 0.
                    x = tf.add( tf.multiply( tf.math.pow( x, order ), coef ), x )
                    x[index] = x_temp[index]
                    return x
            else:
                if alpha == 0.:
                    if threshold == 0. :
                        x = tf.math.maximum( 0. , x )
                        return tf.add( tf.multiply( tf.math.pow( x, order ), coef ), x)
                    else:
                        index = tf.where( tf.less( x, threshold ))
                        x[index] = 0.
                        x = tf.math.maximum( x, max_value )
                        x = tf.add( tf.multiply( tf.math.pow( x, order ), coef ), x )
                        return x
                else:
                    index  = tf.where( tf.less(x, threshold ), x)
                    x_temp = x
                    x_temp[index] = tf.multiply( x_temp, alpha )
                    x[index] = 0.
                    x = tf.math.maximum( x, max_value )
                    x = tf.add( tf.multiply( tf.math.pow( x, order ), order ), x )
                    x[index] = x_temp[index]
                    return x

    @staticmethod
    def slisin( x, alpha = 1. ):
        """
        Evaluate slisin activation function for input.

        f(x) = alpha * (beta * x + sin( beta * x ) )
        beta = 1 / alpha

        Parameters
        ----------
        x : numpy.ndarray,float,int
            Function input.
        alpha : TYPE, optional
            Scaling coefficient. The default is 1..

        Returns
        -------
        numpy.ndarray, float, int
            slisin activation output.

        """
        beta = 1/alpha
        return tf.multiply( alpha, tf.add( tf.multiply( beta, x ), tf.math.sin( tf.multiply( beta, x ) ) ) )

    @staticmethod
    def slicos( x, alpha = 1. ):
        """
        Evaluate slicos activation function for input.

        f(x) = alpha * (beta * x + cos( beta * x ) + 1 )
        beta = 1 / alpha

        Parameters
        ----------
        x : numpy.ndarray,float,int
            Function input.
        alpha : TYPE, optional
            Scaling coefficient. The default is 1..

        Returns
        -------
        numpy.ndarray, float, int
            slicos activation output.

        """
        beta = 1/alpha
        return tf.multiply( alpha, tf.add( tf.subtract( tf.multiply( beta, x ), tf.math.cos( tf.multiply( beta, x ) ) ), 1. ) )


class Activation_Generator:
    """
    Generates activation functions

    Methods
    -------
    nlrelu
    lognrelu
    repolu
    relpolu
    slisin
    slicos
    """
    @staticmethod
    def nlrelu( beta = 1.0):
        """
        Return handle to nlrelu activation function

        Parameters
        ----------
        beta : float, optional
            Scaling factor. The default is 1.0.

        Returns
        -------
        function
            nlrelu activation function with parameters set.

        """
        def activation_fetch(x):
            return Activations.nlrelu( x, beta = beta)
        return activation_fetch

    @staticmethod
    def lognrelu( beta = 1.0, base = 10.0 ):
        """
        Return handle to lognrelu activation function



        Parameters
        ----------
        beta : float, optional
            Scaling factor. The default is 1.0.
        base : float, optional
            Logarithmic base used. If equal to zero, natural log is used.
            The default is 10.0.

        Raises
        ------
        ValueError
            If base is less than 0.

        Returns
        -------
        function
            lognrelu activation function with parameters set.

        """
        if base < 0:
            raise ValueError("base must be >= 0")
        if base == 0:
            return Activations.nrelu( beta=beta )
        else:
            def activation_fetch(x):
                return Activations.lognrelu( x, beta = beta, base = base )
        return activation_fetch

    @staticmethod
    def repolu( order = 2, alpha = 0., max_value = None, threshold = 0.  ):
        """
        Return a handle to the repolu activation function.

        Parameters
        ----------
        order : int, optional
            Order of polynomial used. The default is 2.
        alpha : float, optional
            Slope of function for values less than the threshold.
            The default is 0..
        max_value : float, None, optional
            Sets the maximum value the function can return. The default is None.
        threshold : float, optional
            The value that must be exceeded in order for the ouput values to not
            be set to 0 or otherwise damped. The default is 0..

        Returns
        -------
        function
            repolu activation function with parameters set.
        """
        def activation_fetch(x):
            return Activations.repolu( x, order, alpha, max_value, threshold )
        return activation_fetch

    @staticmethod
    def relpolu( order = 2 , alpha = 0., max_value = None, threshold = 0. ):
        """
        Return a handle to the relpolu activation function.

        Parameters
        ----------
        order : int, optional
            Order of polynomial used. The default is 2.
        alpha : float, optional
            Slope of function for values less than the threshold.
            The default is 0..
        max_value : float, None, optional
            Sets the maximum value the function can return. The default is None.
        threshold : float, optional
            The value that must be exceeded in order for the ouput values to not
            be set to 0 or otherwise damped. The default is 0..

        Returns
        -------
        function
            relpolu activation function with parameters set.

        """
        # print('order top of relpolu: ', order)
        def activation_fetch(x):
            # print('order activation fetch: ',order)
            return Activations.relpolu( x, order, alpha, max_value, threshold)
        return activation_fetch

    @staticmethod
    def slisin( alpha = 1. ):
        """
        Return a handle to the slisin activation function.

        Parameters
        ----------
        alpha : float, optional
            Scaling coefficient. The default is 1..

        Returns
        -------
        function
            slisin activation function with parameters set.

        """
        def activation_fetch(x):
            return Activations.slisin( x, alpha = alpha )
        return activation_fetch

    @staticmethod
    def slicos( alpha = 1. ):
        """
        Return a handle to the slicos activation function.

        Parameters
        ----------
        alpha : float, optional
            Scaling coefficient. The default is 1..

        Returns
        -------
        function
            slicos activation function with parameters set.

        """
        def activation_fetch(x):
            return Activations.slicos( x, alpha = alpha )
        return activation_fetch

class PolynomialRegression( krs.layers.Layer ):
    """
    Create a layer utilizing heterogeneous polynomial activation functions.

    Attributes
    ----------
    poly_list : list of dense layers
        List of dense layers that utilize heterogenous polynomial activation
        functions
    concat : tensorflow.keras.layers.Layer
        A concatenation layer that concatenates along axis=1

    """
    def __init__(self, n_poly = 1, order = 2, padding = 0,
                 activation = 'relpolu', use_bias=True,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None, kernel_constraint=None,
                 bias_constraint=None, **kwargs ):
        """


        Parameters
        ----------
        n_poly : int, optional
            Number of polynomial sets to use. The default is 1.
        order : int, optional
            The order of each polynomial set. The default is 2.
        padding : int, optional
            The size of the dense layer utilizing a relu activation function
            that is used to pad the output. This is used to reach a desired
            size of layer. The default is 0.
        activation : {'relpolu','repolu'}, optional
            Specifies which of the two polynomial activation functions will be
            used. The default is 'relpolu'.
        use_bias : bool, optional
            Sets whether to use biases. The default is True.
        kernel_initializer : tensorflow.keras.initializers.Initializer, str, optional
            Specifies the kernel initializer. The default is 'glorot_uniform'.
        bias_initializer : tensorflow.keras.initializers.Initializer, str, optional
            Specifies the bias initializer. The default is 'zeros'.
        kernel_regularizer : tensorflow.keras.regularizers.Regularizer, str, optional
            Specifies the kernel regularizer. The default is None.
        bias_regularizer : tensorflow.keras.regularizers.Regularizer, str, optional
            Specifies the bias regularizer. The default is None.
        activity_regularizer : tensorflow.keras.regularizers.Regularizer, str, optional
            Specifies the activity regularizer. The default is None.
        kernel_constraint : tensorflow.keras.constraints.Constraint, str, optional
            Specifies the kernel constraint. The default is None.
        bias_constraint : tensorflow.keras.constraints.Constraint, str, optional
            Specifies the bias constraint. The default is None.
        **kwargs
            keyword arguments.

        Returns
        -------
        None.

        """
        assert isinstance( n_poly, int )
        assert isinstance( order, int )
        super( PolynomialRegression, self).__init__()
        self.output_size = order
        self.n_poly = n_poly
        self.order = order
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.kwargs = kwargs
        self.poly_list = []
        self.activation_map = { 'repolu': Activation_Generator.repolu,
                           'relpolu': Activation_Generator.relpolu
                         }
        self.padding = padding
        for i in range( 0, int(self.order)+1 ):
            self.poly_list.append( krs.layers.Dense( self.n_poly,
                            activation = self.activation_map[activation]( i ),
                            use_bias = self.use_bias,
                            kernel_initializer = self.kernel_initializer,
                            bias_initializer = self.bias_initializer,
                            kernel_regularizer = self.kernel_regularizer,
                            bias_regularizer = self.bias_regularizer,
                            activity_regularizer = self.activity_regularizer,
                            kernel_constraint = self.kernel_constraint,
                            bias_constraint = self.kernel_constraint,
                            **self.kwargs ) )
        if self.padding:
            self.poly_list.append( krs.layers.Dense( self.padding,
                            activation = self.activation_map[activation]( 1 ),
                            use_bias = self.use_bias,
                            kernel_initializer = self.kernel_initializer,
                            bias_initializer = self.bias_initializer,
                            kernel_regularizer = self.kernel_regularizer,
                            bias_regularizer = self.bias_regularizer,
                            activity_regularizer = self.activity_regularizer,
                            kernel_constraint = self.kernel_constraint,
                            bias_constraint = self.kernel_constraint,
                            **self.kwargs ) )
        self.concat = krs.layers.Concatenate(axis = 1)
        # self.outputs = []
        # for i in range(order):
        #     self.outputs.append( tf.Variable(validate_shape=False) )

    def call(self, inputs ):
        """
        Evaluate the layer

        Parameters
        ----------
        inputs : tensorflow.Tensor, numpy.ndarray
             inputs to be evaluated.

        Returns
        -------
        output : tensorflow.Tensor, numpy.ndarray
            layer outputs.

        """
        # for i in range(len(self.poly_list)):
        #     self.outputs[i] = self.poly_list[i](inputs)
        # output = self.concat( self.outputs )
        output = self.concat( [ i(inputs) for i in self.poly_list ] )
        # output = tf.concat( [ i(inputs) for i in self.poly_list ],axis=1)
        return output

    def get_config(self):
        """
        Get object configuration.

        Returns
        -------
        data : TYPE
            DESCRIPTION.

        """
        data = { "output_size":           self.output_size,
                 "n_poly":                self.n_poly,
                 "order":                 self.order,
                 "use_bias":              self.use_bias,
                 "kernel_initializer":    self.kernel_initializer,
                 "bias_initializer":      self.bias_initializer,
                 "kernel_regularizer":    self.kernel_regularizer,
                 "bias_regularizer":      self.bias_regularizer,
                 "acitivity_initializer": self.activity_regularizer,
                 "kernel_constraint":     self.kernel_constraint,
                 "kwargs":                self.kwargs,
                 "poly_list":             self.poly_list,
                 "activation_map":        self.activation_map,
                 "padding":               self.padding}

        return data

class MeanInfNormMSE( krs.losses.Loss ):
    """
    Mean Infinity Norm + Mean Square Error.

    Methods
    -------

    """
    def __init__(self, mse_weight = 1., infNorm_weight = 1., reduction = krs.losses.Reduction.AUTO, name = None ):
        """


        Parameters
        ----------
        mse_weight : float, optional
            The value with which to weight the MSE. The default is 1..
        infNorm_weight : float, optional
            The valkue with which to weight the Inf Norm. The default is 1..
        reduction : tensorflow.keras.losses.Reduction, optional
            Type of reduction to use. The default is krs.losses.Reduction.AUTO.
        name : str, optional
            Name of the loss object. The default is None.

        Returns
        -------
        None.

        """
        super( MeanInfNormMSE, self).__init__(reduction = reduction, name = name)
        self.mse_weight= mse_weight
        self.infNorm_weight = infNorm_weight

    def call(self, y_true, y_pred):
        """


        Parameters
        ----------
        y_true : tensorflow.Tensor
            Expected values.
        y_pred : tensorflow.Tensor
            Predicted value.

        Returns
        -------
        result : tensorflow.Tensor
            Loss result.

        """
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        mse = tf.multiply(self.mse_weight,tf.reduce_mean(tf.math.squared_difference(y_pred, y_true), axis=-1))
        reduced_sum = tf.math.reduce_sum(y_true,axis=-1)
        reduced_mean = tf.math.reduce_mean(y_true, axis=-1)
        multed = tf.multiply(reduced_sum,reduced_mean)
        reduced_inf_norm = tf.multiply(multed,tf.norm((y_pred - y_true),ord=inf,axis=-1))
        result = tf.add(mse,tf.multiply(self.infNorm_weight,reduced_inf_norm))
        return result

    def get_config(self):
        """
        Retreive the configuration of the class

        Returns
        -------
        data : dict
            Parameters of the class that are not part of the parent class.

        """
        data = { "mse_weight": self.mse_weight,
                  "infNorm_weight": self.infNorm_weight
                }
        return data


# custom_objects = (Activation_Generator,Activations,MeanInfNormMSE,PolynomialRegression)
custom_objects = (Activations,MeanInfNormMSE)
custom_obj_map = {}
for i in custom_objects:
    key = str(i)
    key = key.split('\'')[1]
    key = key.rpartition('.')[2]
    custom_obj_map[key] = i
# for key in custom_obj_map.keys():
#     print(key,custom_obj_map[key])