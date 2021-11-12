# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 08:46:43 2021

@author: Michael
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler


class Scalers():
    """
    Create scalers and scale with scalers.

    """
    @staticmethod
    def log_scale( X, scaler_data = None ):
        """
        Perform log(x+1) scaling.

        Parameters
        ----------
        X : numpy.ndarray
            Data to be scaled/transformed.
        scaler_data : None, optional
            Unused. Objects or values necessary for scaling operations if needed.
            The default is None.

        Returns
        -------
        X : numpy.ndarray
            Scaled data.

        """
        X = np.log(np.add(X,1.))
        return X

    @staticmethod
    def log_descale( X, scaler_data = None ):
        """
        Peform inverse log(x+1) scaling.

        Parameters
        ----------
        X : numpy.ndarray
            Data to be scaled/transformed.
        scaler_data : None, optional
            Unused. Objects or values necessary for scaling operations if needed.
            The default is None.

        Returns
        -------
        X : numpy.ndarray
            Inverse scaled data.

        """
        X = np.add(np.exp(X),-1.)
        return X

    @staticmethod
    def create_log_scaler( scaler_list, scaler_data, scaler_map, num = 0,
                          io = 'in', X = None, unused = None ):
        """
        Create log(x+1) scaler.

        Parameters
        ----------
        scaler_list : list
            List of scalers to which this scaler's label will be added.
        scaler_data : dict
            Dict of objects or values necessary for scaling operations to which
            this scaler's data will be added.
        scaler_map : dict
            Dict that maps scaler labels to the appropriate function handles to
            which this the handle to this scaler's functions will be added.
        num : int, optional
            The number that will be appended to the scaler's label.
            The default is 0.
        io : str, optional
            The string representing whether the scaler is intended for the
            input or output data. The default is 'in'.
        X : numpy.ndarray, optional
            unused. The data that will eventually be scaled/transformed if it
            is needed to create the scaling process. The default is None.
        unused : None, optional
            unused. The default is None.

        Returns
        -------
        None.

        """
        scaler_list.append( '%s_log_%i'%(io,num) )
        scaler_data['%s_log_%i'%(io,num)] = None
        scaler_map['%s_log_%i'%(io,num)] = 'log'

    @staticmethod
    def log10_scale( X , scaler_data = None ):
        """
        Perform log_10(x+1) scaling.

        Parameters
        ----------
        X : numpy.ndarray
            Data to be scaled/transformed.
        scaler_data : None, optional
            unused. Objects or values necessary for scaling operations if needed.
            The default is None.

        Returns
        -------
        X : numpy.ndarray
            scaled data.

        """
        X = np.log10(np.add(X,1.))
        return X

    @staticmethod
    def log10_descale( X, scaler_data = None ):
        """
        Perform inverse log_10(x+1) scaling.

        Parameters
        ----------
        X : numpy.ndarray
            Data to be scaled/transformed.
        scaler_data : None, optional
            unused. Objects or values necessary for scaling operations if needed.
            The default is None.

        Returns
        -------
        X : numpy.ndarray
            Inverse scaled data.

        """
        # X = np.add(np.exp(np.multiply(X,np.log(10.))),-1.)
        X = np.add(np.power(10., X),-1.)
        return X

    @staticmethod
    def create_log10_scaler( scaler_list, scaler_data, scaler_map, num = 0,
                            io = 'in', X = None, unused = None ):
        """
        Create log_10(x+1) scaler.

        Parameters
        ----------
        scaler_list : list
            List of scalers to which this scaler's label will be added.
        scaler_data : dict
            Dict of objects or values necessary for scaling operations to which
            this scaler's data will be added.
        scaler_map : dict
            Dict that maps scaler labels to the appropriate function handles to
            which this the handle to this scaler's functions will be added.
        num : int, optional
            The number that will be appended to the scaler's label.
            The default is 0.
        io : str, optional
            The string representing whether the scaler is intended for the
            input or output data. The default is 'in'.
        X : numpy.ndarray, optional
            The data that will eventually be scaled/transformed if it is needed
            to create the scaling process. The default is None.
        unused : None, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        scaler_list.append( '%s_log10_%i'%(io,num) )
        scaler_data['%s_log10_%i'%(io,num)] = None
        scaler_map['%s_log10_%i'%(io,num)] = 'log10'

    @staticmethod
    def logn_scale( X, base = None ):
        """
        Perform log_n(x+1) scaling.

        Parameters
        ----------
        X : numpy.ndarray
            Data to be scaled/transformed.
        base : float, optional
            The logarthmic base that is used. The default is None.

        Returns
        -------
        X : numpy.ndarray
            scaled data.

        """
        if base != 0.:
            X = np.divide(np.log(np.add(X,1.)),np.log(base))
        else:
            X = np.log(np.add(X,1.))
        return X

    @staticmethod
    def logn_descale( X, base = None ):
        """
        Peform inverse log_n(x+1) scaling.

        Parameters
        ----------
        X : numpy.ndarray
            Data to be scaled/transformed.
        base : float, optional
            The logarthmic base that is used. The default is None.

        Returns
        -------
        X : numpy.ndarray
            Inverse scaled data.

        """
        if base != 0.:
            X = np.add(np.power(base,X),-1.)
        else:
            X = np.add(np.exp(X),-1.)
        return X

    @staticmethod
    def create_logn_scaler( scaler_list, scaler_data, scaler_map, num = 0, io = 'in', X = None, base = 10.0 ):
        """
        Create log_n(x+1) scaler.

        Parameters
        ----------
        scaler_list : list
            List of scalers to which this scaler's label will be added.
        scaler_data : dict
            Dict of objects or values necessary for scaling operations to which
            this scaler's data will be added.
        scaler_map : dict
            Dict that maps scaler labels to the appropriate function handles to
            which this the handle to this scaler's functions will be added.
        num : int, optional
            The number that will be appended to the scaler's label.
            The default is 0.
        io : str, optional
            The string representing whether the scaler is intended for the
            input or output data. The default is 'in'.
        X : numpy.ndarray, optional
            The data that will eventually be scaled/transformed if it is needed
            to create the scaling process. The default is None.
        base : float, optional
            The logarithmic base that will be used. If base=0 then the
            natural logarithm will be performed. The default is 10.0.

        Raises
        ------
        ValueError
            If base is less than 0. Result is undefined.

        Returns
        -------
        None.

        """
        if base < 0:
            raise ValueError("Argument 'base' cannot be less than 0.")
        scaler_list.append( '%s_logn_%i'%(io,num) )
        scaler_data['%s_logn_%i'%(io,num)] = base
        scaler_map['%s_logn_%i'%(io,num)] = 'logn'

    @staticmethod
    def signed_logn_scale( X , scaler_data = None ):
        """
        Perform signed log_n(x+1) scaling.

        Extracts the sign of the values in X, performs log_n( \|x\| + 1), then
        applies the sign to the result.

        i.e. -12 for base 10 -> -log_10(12+1) -> -2.07918

        Parameters
        ----------
        X : numpy.ndarray
            Data to be scaled/transformed.
        scaler_data : list, optional
            The list that contains the sign of the data bein scaled as well as
            the logarithmic base used. The sign of the data is stored in
            scaler_data[0], and the base used is stored in scaler_data[1].
            The default is None.

        Returns
        -------
        X : numpy.ndarray
            Scaled data.

        """
        base = scaler_data[0]
        scaler_data[1] = np.sign(X)
        X = np.multiply(X,scaler_data[1])
        if base != 0.:
            X = np.multiply(scaler_data[1],np.divide(np.log(np.add(X,1.)),np.log(base)))
        else:
            X = np.multiply(scaler_data[1],np.log(np.add(X,1.)))
        return X

    @staticmethod
    def signed_logn_descale( X, scaler_data = None ):
        """
        Perform inverse signed log_n(x+1) scaling.

        See docstring for signed_logn_scale for an explanation of signed log_n
        scaling.

        Parameters
        ----------
        X : numpy.ndarray
            Data to be scaled/transformed.
        scaler_data : TYPE, optional
            Objects or values necessary for scaling operations if needed.
            The default is None.

        Returns
        -------
        X : numpy.ndarray
            Inverse scaled data.

        """
        base = scaler_data[0]
        sign = scaler_data[1]
        X = np.multiply(X,sign)
        if base != 0.:
            X = np.add(np.exp(np.multiply(X,np.log(base))),-1.)
        else:
            X = np.add(np.exp(X),-1.)
        X = np.multiply(X,sign)
        return X

    @staticmethod
    def create_signed_logn_scaler( scaler_list, scaler_data, scaler_map, num = 0, io = 'in', X = None, base = 10.0 ):
        """
        Create signed log_n(x+1) scaler.

        Parameters
        ----------
        scaler_list : list
            List of scalers to which this scaler's label will be added.
        scaler_data : dict
            Dict of objects or values necessary for scaling operations to which
            this scaler's data will be added.
        scaler_map : dict
            Dict that maps scaler labels to the appropriate function handles to
            which this the handle to this scaler's functions will be added.
        num : int, optional
            The number that will be appended to the scaler's label.
            The default is 0.
        io : str, optional
            The string representing whether the scaler is intended for the
            input or output data. The default is 'in'.
        X : numpy.ndarray, optional
            The data that will eventually be scaled/transformed if it is needed
            to create the scaling process. The default is None.
        base : TYPE, optional
            The logarithmic base that will be used. If base=0 then the
            natural logarithm will be performed. The default is 10.0.

        Returns
        -------
        None.

        """
        scaler_list.append( '%s_signed_logn_%i'%(io,num) )
        scaler_data['%s_signed_logn_%i'%(io,num)] = [base, np.array([])]
        scaler_map['%s_signed_logn_%i'%(io,num)] = 'signed_logn'

    @staticmethod
    def sample_minmax_scale( X, scaler_data = None ):
        """
        Perform samplewise minmax scaling.

        Parameters
        ----------
        X : numpy.ndarray
            Data to be scaled/transformed.
        scaler_data : MinMaxScaler, optional
            The MinMaxScaler that is used to scale data.
            The default is None.

        Raises
        ------
        TypeError
            If scaler_data is not a sklearn MinMaxScaler, an error is raised.

        Returns
        -------
        X : numpy.ndarray
            Scaled data.

        """
        if not isinstance(scaler_data,MinMaxScaler):
            raise TypeError('scaler_data was %s and MinMaxScaler was expected'%(type(scaler_data)))
        scaler_data.fit(X.T)
        return scaler_data.transform(X.T).T

    @staticmethod
    def sample_minmax_descale( X, scaler_data = None ):
        """
        Perform inverse samplewise minmax scaling.

        Parameters
        ----------
        X : numpy.ndarray
            Data to be scaled/transformed.
        scaler_data : MinMaxScaler, optional
            The MinMaxScaler that is used to scale data.
            The default is None.

        Returns
        -------
        X : numpy.ndarray
            Inverse scaled data.

        """
        return scaler_data.inverse_transform(X.T).T

    @staticmethod
    def create_sample_minmax_scaler( scaler_list, scaler_data, scaler_map, num = 0, io = 'in', X = None, unused = None ):
        """
        Create samplewise minmax scaler.

        In a 2D array of data, a the samplewise minmax scaler scales each row
        in the array based on the minimum and maximum found in that given row.

        Parameters
        ----------
        scaler_list : list
            List of scalers to which this scaler's label will be added.
        scaler_data : dict
            Dict of objects or values necessary for scaling operations to which
            this scaler's data will be added.
        scaler_map : dict
            Dict that maps scaler labels to the appropriate function handles to
            which this the handle to this scaler's functions will be added.
        num : int, optional
            The number that will be appended to the scaler's label.
            The default is 0.
        io : str, optional
            unused. The string representing whether the scaler is intended for the
            input or output data. The default is 'in'.
        X : numpy.ndarray, optional
            The data that will eventually be scaled/transformed if it is needed
            to create the scaling process. The default is None.
        unused : None, optional
            unused. The default is None.

        Notes
        -----
        This scaler does not prepend the string stored in `io` as the
        samplewise minmax scaler cannot[*]_ be used as a scaler for a network
        output during inference. This is because, in most use cases, there will
        be no knowledge of what the minimum and maximum are supposed to be. An
        exception to this is for an autoencoder that is used to compress data
        for storage where the minimum and maximum values can be stored for each
        sample.

        ..[*] usually cannot be used

        Special care should be taken when using this for scaler as an input for
        any neural network unless absolute magnitude of is of no consequence.
        A samplewise minmax scaler retains shape/profile information and
        the relative scaling between input dimensions of the same sample, but
        all relative scaling between samples is effectively lost.

        Returns
        -------
        None.

        """
        if not 'samp_minmax_%i'%(num) in scaler_data.keys():
            scaler = MinMaxScaler()
            scaler_data['samp_minmax_%i'%(num)] = scaler
            scaler_map[ 'samp_minmax_%i'%(num) ] = 'samp_minmax'
        scaler_list.append('samp_minmax_%i'%(num))

    @staticmethod
    def feature_minmax_scale( X, scaler_data = None ):
        """
        Perform featurewise scaling.

        Parameters
        ----------
        X : numpy.ndarray
            Data to be scaled/transformed.
        scaler_data : MinMaxScaler, optional
            MinMaxScaler object fitted on the original input data.
            The default is None.

        Raises
        ------
        TypeError
            If scaler_data is not of type MinMaxScaler, an error is raised.

        Returns
        -------
        X : numpy.ndarray
            Scaled data.

        """
        if not isinstance(scaler_data,MinMaxScaler) :
            raise TypeError("feature_minmax_scale can only take MinMaxScaler objects from scikit-learn in the scaler_data argument.")
        return scaler_data.transform(X)

    @staticmethod
    def feature_minmax_descale( X, scaler_data ):
        """
        Perform inverse featurewise scaling.

        Parameters
        ----------
        X : numpy.ndarray
            Data to be scaled/transformed.
        scaler_data : MinMaxScaler, optional
            MinMaxScaler object fitted on the original input data.
            The default is None.

        Returns
        -------
        X : numpy.ndarray
            Inverse scaled data.

        """
        if not isinstance(scaler_data,type(MinMaxScaler())) :
            raise TypeError("feature_minmax_descale can only take MinMaxScaler objects from scikit-learn in the scaler_data argument.")
        return scaler_data.inverse_transform(X)

    @staticmethod
    def create_feature_minmax_scaler( scaler_list, scaler_data, scaler_map, num = 0, io = 'in', X = None, unused = None ):
        """
        Create featurewise minmax scaler.

        In a 2D array of data, a the featurewise minmax scaler scales each
        column in the array based on the minimum and maximum found in that
        given column.

        Parameters
        ----------
        scaler_list : list
            List of scalers to which this scaler's label will be added.
        scaler_data : dict
            Dict of objects or values necessary for scaling operations to which
            this scaler's data will be added.
        scaler_map : dict
            Dict that maps scaler labels to the appropriate function handles to
            which this the handle to this scaler's functions will be added.
        num : int, optional
            The number that will be appended to the scaler's label.
            The default is 0.
        io : str, optional
            The string representing whether the scaler is intended for the
            input or output data. The default is 'in'.
        X : numpy.ndarray, optional
            The data that will eventually be scaled/transformed if it is needed
            to create the scaling process. The default is None.
        unused : None, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        scaler = MinMaxScaler()
        scaler.fit(X)
        scaler_list.append('%s_feat_minmax_%i'%(io,num))
        scaler_data['%s_feat_minmax_%i'%(io,num)] = scaler
        scaler_map['%s_feat_minmax_%i'%(io,num)] = 'feat_minmax'

    @staticmethod
    def unit_norm_scale( X, scaler_data = None ):
        """
        Perform unit norm (L2) scaling.


        Parameters
        ----------
        X : numpy.ndarray
            Data to be scaled/transformed.
        scaler_data : numpy.ndarray, optional
            Normalizing constants
            The default is None.

        Raises
        ------
            If scaler_data is not a numpy.ndarray, an error is raised.

        Returns
        -------
        X : numpy.ndarray
            Scaled data.

        """
        if not isinstance(scaler_data, np.ndarray):
            raise TypeError('scaler_data received %s when it expected a numpy.ndarray.'%(str(type(scaler_data))))
        X = np.multiply(X,scaler_data[0])
        return X

    @staticmethod
    def unit_norm_descale( X, scaler_data = None ):
        """
        Inverse unit norm (L2) scaling.

        Parameters
        ----------
        X : numpy.ndarray
            Data to be scaled/transformed.
        scaler_data : numpy.ndarray, optional
            Normalizing constants.
            The default is None.

        Raises
        ------
            If scaler_data is not a numpy.ndarray, an error is raised.

        Returns
        -------
        X : numpy.ndarray
            Inverse scaled data.

        """
        if not isinstance(scaler_data, np.ndarray):
            raise TypeError('scaler_data received %s when it expected a numpy.ndarray.'%(str(type(scaler_data))))
        X = np.divide(X,scaler_data[0])
        return X

    @staticmethod
    def create_unit_norm_scaler( scaler_list, scaler_data, scaler_map, num = 0, io = 'in', X = None, unused = None ):
        """
        Create unit norm (L2) scaler.

        Divides each sample by its respective L2-Norm.

        Parameters
        ----------
        scaler_list : list
            List of scalers to which this scaler's label will be added.
        scaler_data : dict
            Dict of objects or values necessary for scaling operations to which
            this scaler's data will be added.
        scaler_map : dict
            Dict that maps scaler labels to the appropriate function handles to
            which this the handle to this scaler's functions will be added.
        num : int, optional
            The number that will be appended to the scaler's label.
            The default is 0.
        io : str, optional
            The string representing whether the scaler is intended for the
            input or output data. The default is 'in'.
        X : numpy.ndarray, optional
            The data that will eventually be scaled/transformed if it is needed
            to create the scaling process. The default is None.
        unused : None, optional
            DESCRIPTION. The default is None.

        Raises
        ------
        Exception
            If an individual sample has more than one dimension an error
            is raised.

        Notes
        -----
        This should not be used for an output scaler because there will,
        in most cases, be no normalization values produced by the
        neural network with which to descale with during inference.

        Returns
        -------
        None.

        """
        if len(X.shape) > 2:
            raise Exception("The unit norm scaler does not currently support more than one dimension of data.")
        temp = np.square(X)
        unitNormScalars = np.zeros((X.shape[0],1))
        for i in range(X.shape[0]):
            unitNormScalars[i,0] = temp[i,:].sum()
        unitNormScalars = np.divide(1,np.sqrt(unitNormScalars))
        scaler_list.append( '%s_unit_norm_%i'%(io,num) )
        scaler_data['%s_unit_norm_%i'%(io,num)] = [unitNormScalars,1]
        scaler_map['%s_unit_norm_%i'%(io,num)] = 'unit_norm'

    @staticmethod
    def constant_multiplier_scale( X, scaler_data = None ):
        """
        Apply a constant multiplier.

        Parameters
        ----------
        X : numpy.ndarray
            Data to be scaled/transformed.
        scaler_data : float,int, optional
            Value to multiply X by.
            The default is None.

        Raises
        ------
        TypeError
            If scaler data is neither an int or float, an error is raised.

        Returns
        -------
        X : numpy.ndarray
            Scaled data.

        """
        if not isinstance(scaler_data,(int,float)):
            raise TypeError('scaler_data is %s when it expected an int or float.'%(str(type(scaler_data))))
        X = np.multiply(X,scaler_data)
        return X

    @staticmethod
    def constant_multiplier_descale( X, scaler_data = None ):
        """
        Apply inverse constant multiplier.


        Parameters
        ----------
        X : numpy.ndarray
            Data to be scaled/transformed.
        scaler_data : float,int, optional
            Value to multiply X by.
            The default is None.

        Raises
        ------
        TypeError
            If scaler data is neither an int or float, an error is raised.

        Returns
        -------
        X : numpy.ndarray
            Inverse scaled data.

        """
        if not isinstance(scaler_data,(int,float)):
            raise TypeError('scaler_data is %s when it expected an int or float.'%(str(type(scaler_data))))
        X = np.divide(X,scaler_data)
        return X

    @staticmethod
    def create_constant_multiplier_scaler( scaler_list, scaler_data, scaler_map, num = 0, io = 'in', X = None, scale_coef = 1.0 ):
        """


        Parameters
        ----------
        scaler_list : list
            List of scalers to which this scaler's label will be added.
        scaler_data : dict
            Dict of objects or values necessary for scaling operations to which
            this scaler's data will be added.
        scaler_map : dict
            Dict that maps scaler labels to the appropriate function handles to
            which this the handle to this scaler's functions will be added.
        num : int, optional
            The number that will be appended to the scaler's label.
            The default is 0.
        io : str, optional
            The string representing whether the scaler is intended for the
            input or output data. The default is 'in'.
        X : numpy.ndarray, optional
            The data that will eventually be scaled/transformed if it is needed
            to create the scaling process. The default is None.
        scale_coef : float,int, optional
            Value to multiply X by.
            The default is 1.0.

        Raises
        ------
        TypeError
            If scale_coef is neither an int or float, an error is raised.

        Returns
        -------
        None.

        """
        if not isinstance(scale_coef,(int,float)):
            raise TypeError("scale_coef expected an int or float but recieved %s"%(type(scale_coef)))
        scaler_list.append('%s_const_mult_%i'%(io,num))
        scaler_data['%s_const_mult_%i'%(io,num)] = scale_coef
        scaler_map['%s_const_mult_%i'%(io,num)] = 'const_mult'

    @staticmethod
    def z_number_scale( X, scaler_data = None ):
        """
        Perform z-number dependent scaling.

        Parameters
        ----------
        X : numpy.ndarray
            Data to be scaled/transformed.
        scaler_data : numpy.ndarray, optional
            z-numbers of samples
            The default is None.

        NOTES
        -----
        This has never been tested.

        Returns
        -------
        X : numpy.ndarray
            Scaled data.

        """
        vals = np.divide(np.add(4000.,np.power(scaler_data,3)),np.add(4.,np.power(scaler_data,3)))
        X = np.multiply(X,vals)
        return X

    @staticmethod
    def z_number_descale( X, scaler_data = None ):
        """
        Perform inverse z-number dependent scaling.

        Parameters
        ----------
        X : numpy.ndarray
            Data to be scaled/transformed.
        scaler_data : numpy.ndarray, optional
            z-numbers samples
            The default is None.

        Returns
        -------
        X : numpy.ndarray
            Inverse scaled data.

        """
        vals = np.divide(np.add(4.,np.power(scaler_data,3)),np.add(4000.,np.power(scaler_data,3)))
        X = np.multiply(X,vals)
        return X

    @staticmethod
    def create_z_number_scaler( scaler_list, scaler_data, scaler_map, num = 0, io = 'in', X = None, zvals = None ):
        """
        Create z-number dependent scaler.

        Parameters
        ----------
        scaler_list : list
            List of scalers to which this scaler's label will be added.
        scaler_data : dict
            Dict of objects or values necessary for scaling operations to which
            this scaler's data will be added.
        scaler_map : dict
            Dict that maps scaler labels to the appropriate function handles to
            which this the handle to this scaler's functions will be added.
        num : int, optional
            The number that will be appended to the scaler's label.
            The default is 0.
        io : str, optional
            The string representing whether the scaler is intended for the
            input or output data. The default is 'in'.
        X : numpy.ndarray, optional
            The data that will eventually be scaled/transformed if it is needed
            to create the scaling process. The default is None.
        zvals : numpy.ndarray,list, optional
            z-numbers of samples.
            The default is None.

        Raises
        ------
        TypeError
            If zvals is neither a numpy.ndarray or list, an error is raised.

        Returns
        -------
        None.

        """
        if not isinstance(zvals,[list,np.ndarray]):
            raise TypeError("Received type %s for zvals but expected either a list of numbers or a numpy.ndarray"%(type(zvals)))
        if isinstance( zvals,list):
            zvals = np.array(zvals)
        scaler_list.append('%s_z_num_%i'%(io,num))
        scaler_data['%s_z_num_%i'%(io,num)] = zvals
        scaler_map['%s_z_num_%i'%(io,num)] = 'z_num'

    @staticmethod
    def cube_root_scale( X, scaler_data = None ):
        """
        Perform cube root scaling.

        Parameters
        ----------
        X : numpy.ndarray
            Data to be scaled/transformed.
        scaler_data : None, optional
            unused. Objects or values necessary for scaling operations if needed.
            The default is None.

        Returns
        -------
        X : numpy.ndarray
            Scaled data.

        """
        X = np.cbrt(X)
        return X

    @staticmethod
    def cube_root_descale( X, scaler_data = None ):
        """
        Perform inverse cube root scaling.

        Parameters
        ----------
        X : numpy.ndarray
            Data to be scaled/transformed.
        scaler_data :  None, optional
            unused. Objects or values necessary for scaling operations if needed.
            The default is None.

        Returns
        -------
        X : numpy.ndarray
            Inverse scaled data.

        """
        X = np.power(X,3)
        return X

    @staticmethod
    def create_cube_root_scaler( scaler_list, scaler_data, scaler_map, num = 0, io = 'in', X = None, unused = None ):
        """
        Create cube root scaler.

        Parameters
        ----------
        scaler_list : list
            List of scalers to which this scaler's label will be added.
        scaler_data : dict
            Dict of objects or values necessary for scaling operations to which
            this scaler's data will be added.
        scaler_map : dict
            Dict that maps scaler labels to the appropriate function handles to
            which this the handle to this scaler's functions will be added.
        num : int, optional
            The number that will be appended to the scaler's label.
            The default is 0.
        io : str, optional
            The string representing whether the scaler is intended for the
            input or output data. The default is 'in'.
        X : numpy.ndarray, optional
            The data that will eventually be scaled/transformed if it is needed
            to create the scaling process. The default is None.
        unused : None, optional
            Unused. The default is None.

        Returns
        -------
        None.

        """
        scaler_list.append('%s_cbrt_%i'%(io,num))
        scaler_data['%s_cbrt_%i'%(io,num)] = None
        scaler_map['%s_cbrt_%i'%(io,num)] = 'cbrt'

    @staticmethod
    def square_root_scale( X, scaler_data = None ):
        """
        Perform square root scaling.

        Parameters
        ----------
        X : numpy.ndarray
            Data to be scaled/transformed.
        scaler_data : None, optional
            Unused. The default is None.

        Returns
        -------
        X : numpy.ndarray
            Scaled data.

        """
        X = np.sqrt(X)
        return X

    @staticmethod
    def square_root_descale( X, scaler_data = None ):
        """
        Perform inverse square root scaling.

        Parameters
        ----------
        X : numpy.ndarray
            Data to be scaled/transformed.
        scaler_data : None, optional
            Unused. The default is None.

        Returns
        -------
        X : numpy.ndarray
            Inverse scaled data.

        """
        X = np.square(X)
        return X

    @staticmethod
    def create_square_root_scaler( scaler_list, scaler_data, scaler_map, num = 0, io = 'in', X = None, unused = None ):
        """
        Create square root scaler.

        Parameters
        ----------
        scaler_list : list
            List of scalers to which this scaler's label will be added.
        scaler_data : dict
            Dict of objects or values necessary for scaling operations to which
            this scaler's data will be added.
        scaler_map : dict
            Dict that maps scaler labels to the appropriate function handles to
            which this the handle to this scaler's functions will be added.
        num : int, optional
            The number that will be appended to the scaler's label.
            The default is 0.
        io : str, optional
            The string representing whether the scaler is intended for the
            input or output data. The default is 'in'.
        X : numpy.ndarray, optional
            The data that will eventually be scaled/transformed if it is needed
            to create the scaling process. The default is None.
        unused : None, optional
            unused. The default is None.

        Returns
        -------
        None.

        """
        scaler_list.append('%s_sqrt_%i'%(io,num))
        scaler_data['%s_sqrt_%i'%(io,num)] = None
        scaler_map['%s_sqrt_%i'%(io,num)] = 'sqrt'

    @staticmethod
    def nth_root_scale( X, scaler_data = 2. ):
        """
        Perform nth-root scaling.

        Parameters
        ----------
        X : numpy.ndarray
            Data to be scaled/transformed.
        scaler_data : float,int, optional
            The value for which n is set in an nth-root operation.
            The default is None.

        Returns
        -------
        X : numpy.ndarray
            Scaled data.

        """
        X = np.power( X, 1./scaler_data )
        return X

    @staticmethod
    def nth_root_descale( X, scaler_data = 2. ):
        """
        Perform inverse nth-root scaling.

        Parameters
        ----------
        X : numpy.ndarray
            Data to be scaled/transformed.
        scaler_data : float,int, optional
            The value for which n is set in an nth-root operation.
            The default is None.

        Returns
        -------
        X : numpy.ndarray
            Inverse scaled data.

        """
        X = np.power( X, scaler_data )
        return X

    @staticmethod
    def create_nth_root_scaler( scaler_list, scaler_data, scaler_map, num = 0, io = 'in', X = None, root = 2. ):
        """
        Create nth-root scaler.

        Parameters
        ----------
        scaler_list : list
            List of scalers to which this scaler's label will be added.
        scaler_data : dict
            Dict of objects or values necessary for scaling operations to which
            this scaler's data will be added.
        scaler_map : dict
            Dict that maps scaler labels to the appropriate function handles to
            which this the handle to this scaler's functions will be added.
        num : int, optional
            The number that will be appended to the scaler's label.
            The default is 0.
        io : str, optional
            The string representing whether the scaler is intended for the
            input or output data. The default is 'in'.
        X : numpy.ndarray, optional
            The data that will eventually be scaled/transformed if it is needed
            to create the scaling process. The default is None.
        root : float,int, optional
            The root value for which n is set in an nth-root operation.
            The default is 2.

        Returns
        -------
        None.

        """
        scaler_list.append('%s_nroot_%i'%(io,num))
        scaler_data['%s_nroot_%i'%(io,num)] = root
        scaler_map['%s_nroot_%i'%(io,num)] = 'nroot'

    @staticmethod
    def mult_vector_scale( X, scaler_data = None ):
        """
        Perform multiplicative scaling by a constant vector.

        Parameters
        ----------
        X : numpy.ndarray
            Data to be scaled/transformed.
        scaler_data : numpy.ndarray, optional
            The vector by which X will multiplied across axis=1.
            `X.shape[1] == scaler_data.size` must be true.
            The default is None.

        Returns
        -------
        X : numpy.ndarray
            Scaled data.

        """
        X = np.multiply(X,scaler_data)
        return X

    @staticmethod
    def mult_vector_descale( X, scaler_data = None ):
        """
        Perform inverse multiplicative scaling by a constant vector.

        Parameters
        ----------
        X : numpy.ndarray
            Data to be scaled/transformed.
        scaler_data : numpy.ndarray, optional
            The vector by which X will multiplied across axis=1.
            `X.shape[1] == scaler_data.size` must be true.
            The default is None.

        Returns
        -------
        X : numpy.ndarray
            Inverse scaled data.

        """
        X = np.divide(X,scaler_data)
        return X

    @staticmethod
    def create_mult_vector_scaler( scaler_list, scaler_data, scaler_map, num = 0, io = 'in', X = None, vector = None ):
        """
        Create multiplicative vector scaler.

        Parameters
        ----------
        scaler_list : list
            List of scalers to which this scaler's label will be added.
        scaler_data : dict
            Dict of objects or values necessary for scaling operations to which
            this scaler's data will be added.
        scaler_map : dict
            Dict that maps scaler labels to the appropriate function handles to
            which this the handle to this scaler's functions will be added.
        num : int, optional
            The number that will be appended to the scaler's label.
            The default is 0.
        io : str, optional
            The string representing whether the scaler is intended for the
            input or output data. The default is 'in'.
        X : numpy.ndarray, optional
            The data that will eventually be scaled/transformed if it is needed
            to create the scaling process. The default is None.
        vector : numpy.ndarray,list, optional
            The vector by which X will multiplied across axis=1.
            `X.shape[1] == scaler_data.size` must be true.
            The default is None.

        Raises
        ------
        TypeError
            If vector is neither a numpy.ndarray or a list,
            an error will be raised.

        Returns
        -------
        None.

        """
        if not isinstance(vector,(np.ndarray,list)):
            raise TypeError( "create_mult_vector_scaler expects a numpy.ndarray or list not %s"%(type(scaler_data)))
        elif isinstance(vector,list):
            vector = np.array(vector)
        scaler_list.append('%s_mult_vec_%i'%(io,num))
        scaler_data['%s_mult_vec_%i'%(io,num)] = vector
        scaler_map['%s_mult_vec_%i'%(io,num)] = 'mult_vec'

    @staticmethod
    def div_vector_scale( X, scaler_data = None ):
        """
        Perform divisionary constant vector scaling.

        Parameters
        ----------
        X : numpy.ndarray
            Data to be scaled/transformed.
        scaler_data : TYPE, optional
            The vector by which X will divided across axis=1.
            `X.shape[1] == scaler_data.size` must be true.
            The default is None.

        Returns
        -------
        X : numpy.ndarray
            Scaled data.

        """
        X = np.divide(X,scaler_data)
        return X

    @staticmethod
    def div_vector_descale( X, scaler_data = None ):
        """
        Perform inverse divisionary constant vector scaling.

        Parameters
        ----------
        X : numpy.ndarray
            Data to be scaled/transformed.
        scaler_data : TYPE, optional
            The vector by which X will divided across axis=1.
            `X.shape[1] == scaler_data.size` must be true.
            The default is None.

        Returns
        -------
        X : numpy.ndarray
            Inverse scaled data.

        """
        X = np.multiply(X,scaler_data)
        return X

    @staticmethod
    def create_div_vector_scaler( scaler_list, scaler_data, scaler_map, num = 0, io = 'in', X = None, vector = None ):
        """
        Create divisionary vector scaler.

        Parameters
        ----------
        scaler_list : list
            List of scalers to which this scaler's label will be added.
        scaler_data : dict
            Dict of objects or values necessary for scaling operations to which
            this scaler's data will be added.
        scaler_map : dict
            Dict that maps scaler labels to the appropriate function handles to
            which this the handle to this scaler's functions will be added.
        num : int, optional
            The number that will be appended to the scaler's label.
            The default is 0.
        io : str, optional
            The string representing whether the scaler is intended for the
            input or output data. The default is 'in'.
        X : numpy.ndarray, optional
            The data that will eventually be scaled/transformed if it is needed
            to create the scaling process. The default is None.
        vector : numpy.ndarray, optional
            The vector by which X will divided across axis=1.
            `X.shape[1] == scaler_data.size` must be true.
            The default is None.

        Raises
        ------
        TypeError
            If vector is not a numpy.ndarray or list.

        Returns
        -------
        None.

        """
        if not isinstance(vector,(np.ndarray,list)):
            raise TypeError( "create_mult_vector_scaler expects a numpy.ndarray or list not %s"%(type(vector)))
        if isinstance(vector,list):
            vector = np.array(vector)
        scaler_list.append('%s_div_vec_%i'%(io,num))
        scaler_data['%s_div_vec_%i'%(io,num)] = vector
        scaler_map['%s_div_vec_%i'%(io,num)] = 'div_vec'

scaler_functions = {'log_scale': Scalers.log_scale,
                    'log_descale': Scalers.log_descale,
                    'log10_scale': Scalers.log10_scale,
                    'log10_descale': Scalers.log10_descale,
                    'logn_scale': Scalers.logn_scale,
                    'logn_descale': Scalers.logn_descale,
                    'signed_logn_scale': Scalers.logn_scale,
                    'signed_logn_descale': Scalers.logn_descale,
                    'feat_minmax_scale': Scalers.feature_minmax_scale,
                    'feat_minmax_descale': Scalers.feature_minmax_descale,
                    'samp_minmax_scale': Scalers.sample_minmax_scale,
                    'samp_minmax_descale': Scalers.sample_minmax_descale,
                    'unit_norm_scale': Scalers.unit_norm_scale,
                    'unit_norm_descale': Scalers.unit_norm_descale,
                    'const_mult_scale': Scalers.constant_multiplier_scale,
                    'const_mult_descale': Scalers.constant_multiplier_descale,
                    'z_num_scale': Scalers.z_number_scale,
                    'z_num_descale': Scalers.z_number_descale,
                    'cbrt_scale': Scalers.cube_root_scale,
                    'cbrt_descale': Scalers.cube_root_descale,
                    'nroot_scale': Scalers.nth_root_scale,
                    'nroot_descale': Scalers.nth_root_descale,
                    'mult_vec_scale': Scalers.mult_vector_scale,
                    'mult_vec_descale': Scalers.mult_vector_descale,
                    'div_vec_scale': Scalers.div_vector_scale,
                    'div_vec_descale': Scalers.div_vector_descale,
                    None: None}

scaler_creators = {'log': Scalers.create_log_scaler,
                   'log10': Scalers.create_log10_scaler,
                   'logn': Scalers.create_logn_scaler,
                   'signed-logn': Scalers.create_signed_logn_scaler,
                   'feat-minmax': Scalers.create_feature_minmax_scaler,
                   'samp-minmax': Scalers.create_sample_minmax_scaler,
                   'unit-norm': Scalers.create_unit_norm_scaler,
                   'const-mult': Scalers.create_constant_multiplier_scaler,
                   'z-num': Scalers.create_z_number_scaler,
                   'cbrt': Scalers.create_cube_root_scaler,
                   'nroot': Scalers.create_nth_root_scaler,
                   'mult-vec': Scalers.create_mult_vector_scaler,
                   'div-vec': Scalers.create_div_vector_scaler,
                   None: None}