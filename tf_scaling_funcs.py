# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 14:38:22 2021

@author: Michael
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras as krs
# from sklearn.preprocessing import MinMaxScaler


class FeatureMinMaxScaler( krs.layers.Layer ):
    """
    Featurewise Scaling Minmax Scaling - Transform.

    Attributes
    ----------
    _max_data : numpy.ndarray
        The maximum values used to calculate minmax scaling
    _min_data : numpy.ndarrayThe minimum values used to calculate minmax scaling
    _limits : numpy.ndarray
        A tuple, list, or numpy.ndarray containing to values
        that set the lower and upper limits of the range of
        values that data will be rescaled to.
    num_ouputs : int
        Number features being scaled and thus being output.

    Methods
    -------
    __init__
    build
    call
    get_config

    """
    def __init__(self, max_data, min_data, limits = (0.,1.),
                 name = 'FeatureMnMxScale', **kwargs ):
        """
        Initial FeatureMinMaxScaler

        Parameters
        ----------
        max_data : numpy.ndarray
            The maximum values used to calculate minmax scaling
        min_data : numpy.ndarray
            The minimum values used to calculate minmax scaling
        limits : tuple, list, numpy.ndarray, optional
            A tuple, list, or numpy.ndarray containing to values
            that set the lower and upper limits of the range of
            values that data will be rescaled to.
            The default is (0.,1.).
        name : str, optional
            name to give the layer. The default is "FeatureMnMxScale".
        kwargs : dict
            keyword arguments.

        Returns
        -------
        None.

        """
        super(FeatureMinMaxScaler, self).__init__(trainable=False,
                                                  name = name, **kwargs)
        self._max_data = max_data
        self._min_data = min_data
        self._limits = np.array(limits)
        self.num_outputs = max_data.size


    def build(self, input_shape):
        """
        Build the FeatureMinMaxScaler object.

        Values and arrays used to speed up computation are pre-calculated here.

        Parameters
        ----------
        input_shape : tensorflow.TensorShape
            The shape of the expected incoming tensors.

        Returns
        -------
        None.

        """
        self.max_data = tf.convert_to_tensor(self._max_data,dtype='float64')
        self.min_data = tf.convert_to_tensor(self._min_data,dtype='float64')
        self.limits = tf.convert_to_tensor(np.array(self._limits),dtype='float64')
        self.range = tf.constant(self.limits[1]-self.limits[0],dtype='float64')

        self.inv_denominator = tf.divide(tf.constant(1.,dtype='float64'), tf.subtract(self.max_data,self.min_data) )
        self.inv_denominator = tf.where(tf.math.is_inf(self.inv_denominator),tf.constant(0.,dtype='float64'),self.inv_denominator)

        self.scaling_multiplier = tf.multiply(self.inv_denominator,self.range)
        self.range_min = tf.constant(self.limits[0],dtype='float64')

    def call(self, input):
        """
        Compute the Featurewise Minmax transform.

        Parameters
        ----------
        input : tensorflow.Tensor
            Input data.

        Returns
        -------
        tensorflow.Tensor
            Rescaled data.

        """
        return tf.cast(tf.add( tf.multiply( tf.subtract(tf.cast(input,dtype='float64'), self.min_data), self.scaling_multiplier ), self.range_min ),dtype='float32')

    def get_config(self):
        """
        Create config dict.

        Returns
        -------
        data : dict
            Dictionary of custom class attributes.

        """
        data = {    '_max_data': self._max_data,
                    '_min_data': self._min_data,
                    '_limits': self._limits,
                    'num_outputs': self.num_outputs}
        return data

class FeatureMinMaxDescaler( krs.layers.Layer ):
    """
    Featurewise Scaling Minmax Scaling - Inverse Transform.

    Attributes
    ----------
    _max_data : numpy.ndarray
        The maximum values used to calculate minmax scaling
    _min_data : numpy.ndarrayThe minimum values used to calculate minmax scaling
    _limits : numpy.ndarray
        A tuple, list, or numpy.ndarray containing to values
        that set the lower and upper limits of the range of
        values that data will be rescaled to.
    num_ouputs : int
        Number features being scaled and thus being output.

    Methods
    -------
    __init__
    build
    call
    get_config

    """
    def __init__(self, max_data, min_data, limits = (0.,1.),
                 name = 'FeatureMnMxDescale', **kwargs ):
        """


        Parameters
        ----------
        max_data : numpy.ndarray
            The maximum values used to calculate minmax scaling
        min_data : numpy.ndarray
            The minimum values used to calculate minmax scaling
        limits : tuple, list, numpy.ndarray, optional
            A tuple, list, or numpy.ndarray containing to values
            that set the lower and upper limits of the range of
            values that data will be rescaled to.
            The default is (0.,1.).
        name : str, optional
            name to give the layer. The default is "FeatureMnMxDescale".
        kwargs : dict
            keyword arguments.

        Returns
        -------
        None.

        """
        super(FeatureMinMaxDescaler, self).__init__(trainable=False,name = name, **kwargs)
        self._max_data = max_data
        self._min_data = min_data
        self._limits = np.array(limits)
        self.num_outputs = max_data.size


    def build(self, input_shape):
        """
        Build the FeatureMinMaxDescaler object.

        Values and arrays used to speed up computation are pre-calculated here.

        Parameters
        ----------
        input_shape : tensorflow.TensorShape
            The shape of the expected incoming tensors.

        Returns
        -------
        None.

        """
        self.max_data = tf.convert_to_tensor(self._max_data,dtype='float64')
        self.min_data = tf.convert_to_tensor(self._min_data,dtype='float64')
        self.limits = tf.convert_to_tensor(np.array(self._limits),dtype='float64')
        self.range = tf.constant(self.limits[1]-self.limits[0],dtype='float64')

        self.inv_denominator = tf.divide(tf.constant(1.,dtype='float64'), self.range)

        self.scaling_multiplier = tf.multiply( self.inv_denominator,
                                    tf.subtract(self.max_data,self.min_data))
        self.range_min = tf.constant(self.limits[0],dtype='float64')
        self.num_outputs = self.max_data.shape[0]

    def call(self, input):
        """
        Compute the inverse transform of Featurewise Minmax Scaling.

        Parameters
        ----------
        input : tensorflow.Tensor
            Input data.

        Returns
        -------
        tensorflow.Tensor
            Rescaled data.

        """
        return tf.cast(tf.add( tf.multiply( tf.subtract(tf.cast(input,dtype='float64'), self.range_min), self.scaling_multiplier ), self.min_data ),dtype='float32')

    def get_config(self):
        """
        Create config dict.

        Returns
        -------
        data : dict
            Dictionary of custom class attributes.

        """
        data = {    '_max_data': self._max_data,
                    '_min_data': self._min_data,
                    '_limits': self._limits,
                    'num_outputs': self.num_outputs}
        return data

class SampleMinMaxScaler(tf.keras.layers.Layer):
    """
    Samplewise Minmax Scaler

    Works but currently cannot save models that include this layer.

    Attributes
    ----------
    limits : numpyp.ndarray
        A tuple, list, or numpy.ndarray containing to values
        that set the lower and upper limits of the range of
        values that data will be rescaled to.

    Methods
    -------
    __init__
    build
    call
    scale
    descale
    get_config


    """
    def __init__(self, limits=(0.0, 1.0), name="SampleMnMxScale", **kwargs):
        """
        Initialize Samplewise Minmax Scaler

        Parameters
        ----------
        limits : tuple, list, or numpy.ndarray, optional
            A tuple, list, or numpy.ndarray containing to values
            that set the lower and upper limits of the range of
            values that data will be rescaled to.
            The default is (0.,1.).
        name : str, optional
            name to give the layer. The default is "SampleMnMxScale".
        kwargs : dict
            keyword arguments.

        Returns
        -------
        None.

        """
        super(SampleMinMaxScaler, self).__init__(trainable=False, name = name, **kwargs)
        self.limits = tf.convert_to_tensor(np.array(limits), dtype="float64")
        self.range = tf.constant(limits[1] - limits[0], dtype="float64")
        self.range_min = tf.constant(limits[0], dtype="float64")
        self.max_data = tf.zeros((5, 10), dtype="float64")
        self.min_data = tf.zeros((5, 10), dtype="float64")

    def scale(self, input):
        """
        Perform samplewise minmax scaling transform.

        Parameters
        ----------
        input : tensorflow.Tensor
            Input data.

        Returns
        -------
        tensorflow.Tensor
            Rescaled data.

        """
        input_data = tf.cast(input, dtype="float64")
        self.max_data = tf.math.reduce_max(input_data, axis=1)
        self.min_data = tf.math.reduce_min(input_data, axis=1)

        inv_denominator = tf.divide(
            tf.constant(1.0, dtype="float64"), tf.subtract(self.max_data, self.min_data)
        )
        inv_denominator = tf.where(
            tf.math.is_inf(inv_denominator),
            tf.constant(0.0, dtype="float64"),
            inv_denominator,
        )

        scaling_multiplier = tf.multiply(inv_denominator, self.range)

        return tf.transpose(
            tf.cast( tf.add( tf.multiply( tf.subtract( tf.transpose(tf.cast(input, dtype="float64")), self.min_data), scaling_multiplier),self.range_min),dtype="float32"))

    def descale(self, input):
        """
        Perform samplewise minmax scaling inverse transform.


        Parameters
        ----------
        input : tensorflow.Tensor
            Input data.

        Returns
        -------
        tensorflow.Tensor
            Rescaled data.

        """
        inv_denominator = tf.divide(tf.constant(1.0, dtype="float64"), self.range)

        scaling_multiplier = tf.multiply(
            inv_denominator, tf.subtract(self.max_data, self.min_data)
        )

        return tf.transpose(
            tf.cast(tf.add(tf.multiply(tf.subtract(tf.transpose(tf.cast(input, dtype="float64")),self.range_min),scaling_multiplier),self.min_data),dtype="float32"))

    def call(self, input, descale=False):
        """
        Compute the Samplewise Minmax transform or inverse transform.

        The parameter `descale` determines if the transform or inverse
        transform will be used.

        descale=False -> transform
        descale=True -> inverse transform

        Parameters
        ----------
        input : tensorflow.Tensor
            Input data.
        descale : bool, optional
            If true, perform inverse transform on data. The default is False.

        Returns
        -------
        tensorflow.Tensor
            Rescaled data.
        """
        if descale:
            return self.descale(input)
        return self.scale(input)

    def get_config(self):
        """
        Create config dict.

        Returns
        -------
        data : dict
            Dictionary of custom class attributes.

        """
        data = {    'limits': self.limits,
                    'range': self.range,
                    'range_min': self.range_min}
        return data



class Log10Scaler( krs.layers.Layer ):
    """
    Log10( x + 1) transform.

    Attributes
    ----------
    log10 : float
        1 / log(10)

    Methods
    -------
    __init__
    build
    call
    get_config

    """
    def __init__( self, name = 'Log10Scale', **kwargs ):
        """
        Initialize Log10Scaler object.

        Parameters
        ----------
            name : str, optional
                name to give the layer. The default is "Log10Scale".
            kwargs : dict
                keyword arguments

        Returns
        -------
        None.

        """
        super(Log10Scaler, self).__init__(trainable=False, name = name, **kwargs)


    def build(self, input_shape):
        """
        Build the Log10Scaler object.

        1/log(10) is precalculated here for faster computation..

        Parameters
        ----------
        input_shape : tensorflow.TensorShape
            The shape of the expected incoming tensors.

        Returns
        -------
        None.

        """
        self.log10 = np.divide(1.,np.log(10.,dtype='float64'))

    def call(self, input):
        """
        Compute the log_10(x+1) transform of the input data.

        Parameters
        ----------
        input : tensorflow.Tensor
            Input data.

        Returns
        -------
        tensorflow.Tensor
            Rescaled data.

        """
        return tf.cast(tf.multiply( tf.math.log1p( tf.cast(input,dtype='float64') ), self.log10 ),dtype='float32')

    def get_config(self):
        """
        Create config dict.

        Returns
        -------
        data : dict
            Dictionary of custom class attributes.

        """
        data = {'log10': self.log10}
        return data

class Log10Descaler( krs.layers.Layer ):
    """
    Log10( x + 1 ) inverse transfrom.

    Methods
    -------
    __init__
    call

    """
    def __init__( self, name = 'Log10Descale', **kwargs ):
        """
        Initialize Log10Descaler object.

        Parameters
        ----------
            name : str, optional
                name to give the layer. The default is "Log10Descale".
            kwargs : dict
                keyword arguments

        Returns
        -------
        None.

        """
        super(Log10Descaler, self).__init__(trainable=False, name = name, **kwargs)

    def call(self, input):
        """
        Compute the inverse log_10(x+1) transform of the input data.

        Parameters
        ----------
        input : tensorflow.Tensor
            Input data.

        Returns
        -------
        tensorflow.Tensor
            Rescaled data.

        """
        return tf.cast(tf.subtract( tf.math.pow( tf.constant(10.,dtype='float64'), tf.cast(input,dtype='float64') ), tf.constant( 1.,dtype='float64' ) ),dtype='float32')

class CubeRootScaler( krs.layers.Layer ):
    """
    Cube root transform.

    Methods
    -------
    __init__
    call

    """
    def __init__( self, name = 'CubeRootScale', **kwargs ):
        """
        Initialize CubeRootScaler object.

        Parameters
        ----------
            name : str, optional
                name to give the layer. The default is "CubeRootScale".
            kwargs : dict
                keyword arguments

        Returns
        -------
        None.

        """
        super(CubeRootScaler, self).__init__(trainable=False, name = name, **kwargs)


    def call(self, input):
        """
        Compute the cube root of the input data

        Parameters
        ----------
        input : tensorflow.Tensor
            Input data.

        Returns
        -------
        tensorflow.Tensor
            Rescaled data.

        """
        return tf.cast(tf.math.pow( tf.cast(input,dtype='float64'), tf.constant( 1./3.,dtype='float64' ) ),dtype='float32')

class CubeRootDescaler( krs.layers.Layer ):
    """
    Cube root inverse transform.

    Methods
    -------
    __init__
    call

    """
    def __init__( self, name = 'CubeRootDescale', **kwargs ):
        """
        Initialize CubeRootDescaler object.

        Parameters
        ----------
            name : str, optional
                name to give the layer. The default is "CubeRootDescale".
            kwargs : dict
                keyword arguments

        Returns
        -------
        None.

        """
        super(CubeRootDescaler, self).__init__(trainable=False, name = name, **kwargs)

    def call(self, input):
        """
        Compute the inverse cube root of the input data.

        Parameters
        ----------
        input : tensorflow.Tensor
            Input data.

        Returns
        -------
        tensorflow.Tensor
            Rescaled data.

        """
        return tf.cast(tf.math.pow( tf.cast(input,dtype='float64'), tf.constant(3.,dtype='float64') ),dtype='float32')


