# -*- coding: utf-8 -*-
"""
Created on Wed May 27 15:07:49 2020

@author: mvander
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import tensorflow.keras as krs
import numpy as np
from archparse_np_scalers import scaler_functions, scaler_creators
from archparse_custom_tf import *
import _pickle as cPickle
from sys import exit
# import shutil
import archparse
# import time
# import sys

__version = '0.2.0'

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

def get_version():
    """
    Returns the version number of the package

    Returns
    -------
    __version : str
        DESCRIPTION.

    """
    global __version
    return __version

# def __convert_version( state ):
#     return state

def load(modelName, modelPath="./", prefix = 'ae'):
    """
    Load the pickled Neural Network object or derived class object.

    Parameters
    ----------
    modelName : str
        The base name of the model that will loaded.
    modelPath : str, optional
        The path the folder that the pickled object and Tensorflow saved_model
        that will be loaded. The default is "./".
    prefix : str, optional
        The prefix of the model type to prepended to the model name
        ( i.e. 'ae' denotes an Autoencoder model). The default is 'ae'.

    Returns
    -------
    None.

    """
    try:
        print("=======From autoencoder --- modelPath,modelName : ",modelPath,modelName)
        modelName2 = "%s_"%(prefix) + modelName
        with open("%s%s.pkl"%(modelPath,modelName2), "rb") as f:
            model=cPickle.load(f)
        model.load_model(modelPath,modelName)
        return(model)
    except:
        print("Error loading model ", modelPath + modelName2 )
        # raise
        return()

def train( x, y, model, learningRate, batchSize, nEpochs, modelName,
          modelPath='./', loss = krs.losses.MSE, save_model = True,
          validStep = 1, validation_split = 0.0, forceGPU = False,
          continue_training = False, gpuDevice=0, prefix = 'ae'):
    """
    Trains the passed model. Currently only handles single dataset inputs.

    Parameters
    ----------
    x : numpy.ndarray
        input data to be trained on
    y : numpy.ndarray
        output data to be trained on
    model : tensorflow.keras.Model
        model that will be trained
    learningRate : float
        the learning rate to be used during optimization
    batchSize : int
        the batch size to be used during training
    nEpochs : int
        number of epochs to train for
    modelName : str
        base name of model being trained
    modelPath : str, optional
        path to folder where the Tensorflow model will be saved.
        The default is './'.
    loss : tensorflow.keras.losses.Loss, functional optional
        The loss function to be ussed during training.
        A user defined function can also be used with the signature
        ``def custom_loss(y_true,y_pred)``

        It should use Tensor/EagerTensor compatible operations.

        The default is krs.losses.MSE.
    save_model : bool, optional
        The flag,  that if true, causes the model to be saved.
        The default is True.
    validStep : int, optional
        How often, in number of epochs, should validation be performed.
        The default is 1.
    validation_split : float, optional
        The fraction of the data passed to fit that will be set aside for
        validation. The default is 0.0
    forceGPU : bool, optional
        Setting to 'True' causes training to fail if a GPU is unavailable.
        The default is False.
    continue_training : bool, optional
        The flag, that if true, declares that the existing model will be
        continued to be trained rather than training a brandnew model.
        The default is False.
    gpuDevice : int, optional
        Sets which gpu device to use in Tensorflow.
        The default is 0.
    prefix : str, optional
        The prefix that will be prepended to the model name when saving.
        The prefix corresponds to the type of model being saved.
        i.e. 'ae' <-> Autoencoder
             'nn' <-> Neural Network

        The default is 'ae'.

    Raises
    ------
    Exception
        Exception is raised if the forceGPU==True and a GPU is unavailable.

    Returns
    -------
    avgCost : numpy.ndarray
        Unused, returns a numpy array of zeros
    costTrain : numpy.ndarray
        returns the total training cost at each epoch
    costValid : numpy.ndarray
        returns the total validation cost at each epoch
    validEpochs : numpy.ndarray
        returns an array holding the epoch numbers at which the costs were
        collected

    TODO
    ----
    [1] Fix force_gpu and search for gpu.

    """
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        if forceGPU:
            raise Exception("Unable to use GPU.")

    modelName="%s_"%(prefix) + modelName
    avgCost=np.zeros(nEpochs)
    validEpochs = np.zeros(nEpochs)
    costTrain=np.zeros(nEpochs)
    costValid=np.zeros(nEpochs)

    try:
        with tf.device("/device:GPU:%i"%gpuDevice):
            history = model.fit( x, y, batch_size = batchSize, epochs = nEpochs,
                            validation_split = validation_split, verbose = 2,
                            validation_freq = validStep )
    except:
        print("Failed to assign a specific GPU")
        history = model.fit( x, y, batch_size = batchSize, epochs = nEpochs,
                            validation_split = validation_split, verbose = 2,
                            validation_freq = validStep )

    costTrain = np.array(history.history['loss'])
    if 'val_loss' in history.history.keys():
        costValid = np.array(history.history['val_loss'])
    else:
        costValid = np.zeros(costTrain.shape)
    validEpochs = np.linspace(1,nEpochs,nEpochs,dtype=int)

    if save_model == True:
        attempts = 1
        success = False
        while attempts <= 5 and not success:
            krs.models.save_model(model,"%s%s.mdl"%(modelPath,modelName))
            try:
                krs.models.load_model("%s%s.mdl"%(modelPath,modelName))
                success = True
                print("%s%s.mdl was successfully saved."%(modelPath,modelName))
            except:
                if attempts == 5:
                    print("%s%s failed save on 5 separate occasions. Program will now proceed to finish")
                attempts += 1
    return avgCost,costTrain,costValid,validEpochs

def freeze_layers_except(layers,unfrozen_layers=None):
    """
    Freeze all layers except for the specified layers.

    Freezes all layers passed in the argument 'layers' except for the layers
    with the indices specified in 'unfrozen_layers'.

    Parameters
    ----------
    layers : list of tensorflow.keras.layers.Layer objects
        The tensorflow.keras.layers.Layer objects that will have their
        'trainable' parameter set to False except for those specified by
        'unfrozen_layers'.
    unfrozen_layers : list of integers, int, None, optional
        The list of integers corresponding to the indices of the layers in
        'layers' that will be kept trainable while all other layers are made
        untrainable. If a single 'int', only the layer corresponding to that
        index will be left trainable. If 'None', all layers will be frozen.
        The default is None.

    Raises
    ------
    TypeError
        If unfrozen_layers is not a list of ints, an int, or None, an exception
        is raised.

    Returns
    -------
    None.

    """
    for i in layers:
        i.trainable = False
    if isinstance(unfrozen_layers,int):
        layers[unfrozen_layers].trainable = True
    elif isinstance(unfrozen_layers,list):
        if any([ not isinstance(i,int) for i in unfrozen_layers ]):
            raise TypeError("Invalid type for object in unfrozen_layers. unfrozen_layers must be None, %s, or a list of %s"%(str(int),str(int)))
        for i in unfrozen_layers:
            layers[i].trainable = True
    elif unfrozen_layers is None:
        pass
    else:
        raise TypeError("Invalid type for unfrozen_layers.")

def accumulate_layers(model):
    """
    Traverse layers and collect layers of model into a single list.

    Traverses the sub-models and layers of a model depth-first and returns a
    flattened list of the layers in model except for the layers of type
    tensorflow.keras.InputLayer.

    Parameters
    ----------
    model : tensorflow.keras.Model
        the model to be traversed and layers extracted

    Returns
    -------
    all_layers : list of tensorflow.keras.layers.Layer objects
        The list of layers extracted from 'model'

    Notes
    -----
    Behavior on models consisting of parallel sub-models is untested.

    """
    all_layers = []
    for i in model.layers:
        try:
            if isinstance(i,krs.layers.Layer) and not isinstance(i,krs.layers.InputLayer):
                all_layers += accumulate_layers(i)
        except:
            all_layers.append(i)

    return all_layers


#############################################################################

def set_scalers( scaling_list, scaler_order, scaler_data, scaler_map, data, params, io='in' ):
    """
    Create scaling/transformation operations.

    This function takes a list of scaling/transformation labels stored in
    scaling_list and uses the labels to access scaler creation function handles
    stored in the dict scaler_creators.

    if `scaling_list = ['cbrt','feat-minmax']` and `io=in` then
    ```
    scaler_order = ['in_cbrt_0','in_feat_minmax_0']
    scaler_map = { 'in_cbrt_0': 'cbrt', 'in_feat_minmax_0': 'feat_minmax'}
    scaler_data = { 'in_cbrt_0': None, 'in_feat_minmax_0': MinMaxScaler }
    ```

    Parameters
    ----------
    scaling_list : list
        List of strings corresponding to scaling/transformation operations.
    scaler_order : list
        List where the order of scaling operations will be stored. The
        difference between the entries in scaling_list and scaler_order is that
        the entries in `scaler_order` have a prefix prepended and suffix appended
        to the labels that exist in `scaling_list`.
    scaler_data : dict
        Dictionary in which data/object associated with the
        scaling/transformation operations will be stored.
    scaler_map : dict
        Dictionary in which associations between strings stored in scaling_list
        and the appropriate key for a given scaling function is stored. The
        key-strings stored in this dict must have '_scale' or '_descale' appended to
        the key-string before it can be used to access the function handles
        stored in scaler_functions.
    data : numpy.ndarray
        The data that will be scaled. The creation of some scalers
        (i.e. a featurewise minmax scaler) depend on the data on which the
        model will initially trained.
    params : list
        The parameters associated with the scaling operations
        (i.e. base=5 for a logn operation meaning a base 5 log transformation).
    io : {'in','out'}, optional
        A string that specifies whether the scaling functions currently being
        processed are for input or output data. The default is 'in'.

        io='in' -> input data scaler
        io='out' -> output data scaler

    Returns
    -------
    None.

    """
    scaler_counter = {}
    for i in scaling_list:
        scaler_counter[i] = 0

    for index,i in enumerate(scaling_list):
        if not i is None:
            scaler_creators[i]( scaler_order, scaler_data, scaler_map, scaler_counter[i], io, data, params[index] )
            scaler_counter[i] += 1

            j = scaler_order[index]
            data = scaler_functions[scaler_map[j]+'_scale'](data,scaler_data[j])


def check_input_scaler_list(scaling_list):
    """
    Check that the list of scalers passed to `NeuralNetwork.train()` is valid.

    If `scaling_list` arrives as a list of lists, the list is verified to
    contain only one or two sub-lists. If there is on sub-list, it is
    duplicated and inserted as the second sub-list.

    If `scaling_list` arrives as a list of scaling operations, it is
    duplicated, and both are placed in a list.

    Parameters
    ----------
    scaling_list : list
        List that either contains labels corresponding scaling methods, or it
        contains lists that contain labels corresponding scaling methods.

    Raises
    ------
    Exception
        If there are more than two sub-lists in the `scaling_list`.

        If there is some unknown structure not immediately known to be
        invalid but also isn't known to be valid.

    Returns
    -------
    scaling_list : list of lists
        The validated and possibly modified list of scaling operation labels.

    """
    if not scaling_list is None and isinstance(scaling_list,type(list())):
        if isinstance(scaling_list[0],list):
            if len(scaling_list) > 2:
                raise Exception("Input for scaling_list should be of format [['input scalings'],['output scalings']]")
            if len(scaling_list) == 1:
                scaling_list.append(scaling_list[0])
        else:
            scaling_list = [scaling_list,scaling_list]
    elif scaling_list is None:
        scaling_list = [[None],[None]]
    else:
        raise Exception("Unknown state stemming from scaling_list input.")

    return scaling_list


def check_input_scaler_params( scaling_params, scaling_list ):
    """
    Verify that input arguments have valid structures.

    Verify that input scaling_parameters and scaling_list have the same
    general structure.

    scaling_params and scaling_list must both have the same number of elements
    and/or the same number of sub-elements. i.e. [ [1,2,3], [1,2,3] ] or
    [1,2,3].

    Parameters
    ----------
    scaling_params : list
        List of parameters, or list of lists of parameters than have 1:1
        relation to the entries in `scaling_list`.
    scaling_list : list
        List that either contains labels corresponding scaling methods, or it
        contains lists that contain labels corresponding scaling methods.

    Raises
    ------
    Exception
        If the first entry is a list, and that sub-list does not have the same
        length as the corresponding sub-list in `scaling_list`.

        If the number of sub-lists exceeds 2.

        If there is some unknown structure not immediately known to be
        invalid but also isn't known to be valid.

    Returns
    -------
    scaling_params : lis
        List of lists containing the parameters necessary to perform the
        scaling operations specified in `scaling_list`.

    """
    if not scaling_params is None and isinstance(scaling_params,type(list())):
        if isinstance(scaling_params[0], type(list())):
            if len(scaling_params[0]) > len(scaling_list[0]):
                raise Exception("The length of scaling_params[0] must be less than or equal to the length of scaling_list[0] ")
            if len(scaling_params) > 2:
                raise Exception("Input for scaling_params should be of format [['input scaling params'],['output scaling params']]")
            if len(scaling_params) == 1:
                scaling_params.append(scaling_params[0])
        else:
            scaling_params = [scaling_params,scaling_params]
    elif scaling_params is None and not scaling_list is None:
        n = len(scaling_list[0])
        scaling_params = [None]*n
        scaling_params = [scaling_params,scaling_params]
    else:
        raise Exception("Unknown state stemming from scaling_params and scaling_list inputs.")

    return scaling_params

def apply_scalers( data, scaler_order, scaler_data, scaler_map ):
    """
    Perform scaling/transformation operations on data.

    Parameters
    ----------
    data : numpy.ndarray
        data to be scaled or transformed.
    scaler_order : list
        ordered list containing the scaling/transformation operations that will
        be performed on data.
    scaler_data : dict
        dictionary containing any pertinent scaling parameters or objects,
        i.e. a sklearn MinMaxScaler would have its scaling object stored there
        a logn transformation would have the base of the log_n transformation
        stored in scaler_data.
    scaler_map : dict
        dictionary that maps the labels in `scaler_order` to the function
        handles of the appropriate scaling functions.

    Returns
    -------
    data : numpy.ndarray
        The scaled data.

    Notes
    -----
    This function does not offer the capacity to descale data. This function
    is used prior to training. All scaling and descaling during inference
    occurs in the functions performing inference.

    """
    for index,i in enumerate(scaler_order):
        if not i is None:
            data = scaler_functions[scaler_map[i]+'_scale'](data,scaler_data[i])

    return data

class NeuralNetwork():
    """
    Multi-layer Autoencoder implemented using TensorFlow.

    Attributes
    ----------
    modelName : str,
        name of the model.
    modelPath : str
        path to the directory of the model
    model : tensorflow.keras.Model
        the model.
    args : dict
        dictionary that hold archparse construction info
    custom_objects : dict
        custom objects used in the network construction
    learningRate : float
        learning rate
    train_cost : numpy.ndarray
        array of training costs
    valid_cost : numpy.ndarray
        array of validation costs
    avg_cost : numpy.ndarray
        array of average costs (currently is just zeros)
    epochs_evaluation : numpy.ndarray
        epochs at which costs are calculated
    precision : string
        string that denotes the primitive data type to do calculations with
    scaler_data : dict
        dictionary hold the data/objects needed for scaling operations
    scaler_order : dict
        dictionary with two lists as entries
        keys=['input_scalers','output_scalers']
    scaler_map : dict
        dictionary with labels that map to scaling function handles
    prefix : 'nn'
        prefix that is preprended to the file name of the model
    _prebuilt : bool
        flag stating whether the model was initialize with a passed model.
        currently not fully implemented.

    Methods
    -------
    version
    construct
    train
    compile
    freeze_layers_except
    save_model
    load_model
    close_model
    count_parameters
    summary
    plot_model
    predict
    __getstate__
    __setstate__


    """
    def __init__(self,model=None):
        """


        Parameters
        ----------
        model : TYPE, optional
            DESCRIPTION. The default is None.

        Raises
        ------
        TypeError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if not isinstance( model, type(None) ):
            raise TypeError("model must be of type %s"%(str(krs.Model)))
        self.__version = get_version()
        self.modelName = None
        self.modelPath = None
        self.model = model
        self.args = None
        self.custom_objects = {}
        self.learningRate = None
        self.train_cost = None
        self.valid_cost = None
        self.avgCost = None
        self.epochs_evaluation = None
        self.precision = 'float32'
        input_scalers = []
        output_scalers = []
        self.scaler_order={'input_scalers':input_scalers,
                           'output_scalers':output_scalers}
        input_scalers = []
        output_scalers = []
        self.scaler_data = {}
        self.scaler_map = {}
        self.prefix = "nn"
        self._prebuilt = False
        if not self.model is None:
            self.prebuilt = True


    def __getstate__(self):
        """
        Get object state.

        This must be implemented separately for each sub-classed model. This
        necessary to make sure the object is able to be pickled.

        At a minimum, `self.model` and all of the sub-models must be deleted
        prior to pickling.

        Returns
        -------
        state : dict
            a dict with unserializable data objects removed

        """
        state = self.__dict__.copy()
        del state['model']
        return state

    def __setstate__(self, state):
        """
        Set object state.

        Parameters
        ----------
        state : dict
            Incoming state.

        Returns
        -------
        None.

        """
        self.__dict__.update(state)
        # try:
        #     if self.__version != get_version():
        #         print("Updating Version Info")
        #         self.__version = get_version()
        # except:
        #     __convert_version(state)
        #     self.__dict__.update(state)
        #     self.__version == get_version()
        self.model = None

    def version(self):
        """
        Get version number of this class object.

        Returns
        -------
        str
            version number/string

        """
        return self.__version

    def construct(self):
        """
        Construct model from archparse arguments.

        Calls self.compile after archparse construction.

        This must be implemented separately for all sub-classed models to
        ensure all sub-models are readily accessible via the API. If not
        implemented, only the full model will be immediately accessible via
        the API.

        Returns
        -------
        None.

        """
        argSets = archparse.retrieve_args(parsedBody = self.args)
        self.model = archparse.generate_model(self.args,argSets)[-1]
        self.custom_objects = self.args['custom']
        self.compile()

    def compile(self):
        """
        Set optimizer, loss, and then compile model.

        Returns
        -------
        None.

        """
        opt = krs.optimizers.Adam(learning_rate=self.learningRate)
        loss = self.args['loss']
        self.model.compile(opt,loss)

    def freeze_layers_except(self,unfrozen_layers=None):
        """
        Freeze all layers except specified layers.

        Indices passed via unfrozen layers should NOT account for the
        InputLayers. InputLayers are not included in the list of layers that
        can be  frozen/unfrozen.

        This must be implemented separately for sub-classed models if
        additional string shortcuts are desired to be used.

        Parameters
        ----------
        unfrozen_layers : str,int,list of str/int, optional
            Layers that are to be kept trainable. A single int or str will
            keep only that single corresponding layer trainable.
            Acceptable string values are ['all','first','last'].
            If `unfrozen_layers` is None, all layers will be made untrainable.
            The default is None.

        Raises
        ------
        TypeError
            If unfrozen_layers is not an int,s tr, list of int and/or str.
        ValueError
            If unfrozen_layers is or contains a str other than 'first',
            'last', or 'all'.

        Returns
        -------
        None.

        TODO
        ----
        [1] Enable usage to specify named sub-models to only affect those
            models. This should enable the ability to freeze layers in
            arbitrary models. Defualt would still be to flatten the whole model.
        [2] Use class specific predefined words, so they may be passed to an
            external function and mapped to a specifc layer. May be difficult
            for multiple sub-models. (i.e. latent space layer of AE) This would
            allow for the removal of this function from sub-classed models.
        [3] Possibly allow the use of a prefix (i.e. 'name:') to specify a
            layer's/model's name as opposed to an index.
        [4] Add an original state object/function to the class to return the
            to its original trainability state.

        """
        layers = accumulate_layers(self.model)
        if not isinstance(unfrozen_layers,(int,list,str,type(None))):
            raise TypeError("frozen_layers was expecting type int, list, or str but received type %s"%(type(unfrozen_layers)))
        elif isinstance(unfrozen_layers,(str,list)):
            if isinstance(unfrozen_layers,str):
                if not unfrozen_layers in ['first','last','all']:
                    raise ValueError("Invalid string value.")
            else:
                for i in unfrozen_layers:
                    if not isinstance(i,(int,str)):
                        raise TypeError("An argument of type 'list' must contain only integers exclusive or strings.")

        if not isinstance(unfrozen_layers,list):
            unfrozen_layers = [unfrozen_layers]

        for i in range(len(unfrozen_layers)):
            if isinstance(unfrozen_layers[i],int):
                pass
            elif isinstance(unfrozen_layers[i],str):
                if unfrozen_layers[i] == 'first':
                    unfrozen_layers[i] = 0
                elif unfrozen_layers[i] == 'last':
                    unfrozen_layers[i] = -1
                elif unfrozen_layers[i] == 'all':
                    unfrozen_layers = [i for i in range(len(layers))]
                else:
                    raise ValueError("Invalid str value unfrozen_layers.")
            else:
                raise TypeError("unfrozen_layers must be str,int, or list of str and int.")
        freeze_layers_except(layers,unfrozen_layers)

    def train( self, xtrain, ytrain, args, learningRate, batchSize,
              training_epochs, modelName, modelPath = './', save_model = True,
              valid_step=1, validation_split = 0.0, continue_training = False, forceGPU = False,
              gpuDevice = 0, scaling_list = None, scaling_params = None,
              newPath = './', newName = None, transfer_learn = False ):
        """
        Train model and save (if save_model=True) in ae_modelName.pkl.

        Parameters
        ----------
        xtrain : numpy.ndarry
            Input training data.
        ytrain : numpy.ndarray
            Output training data.
        args : dict
            Arguments received from `archparse.parse_body`. Will become
            optional in the future.
        learningRate : float
            learning rate.
        batchSize : int
            batch size.
        training_epochs : int
            number of epochs to train for.
        modelName : str
            name of the model.
        modelPath : str, optional
            path to the diretory where the model will be stored.
            The default is './'.
        save_model : bool, optional
            Sets whether the model will be saved. The default is True.
        valid_step : int, optional
            How often, in number of epochs, should validation be performed.
            The default is 1.
        validation_split : float, optional
            A float value in range [0,1) specifying what fraction of input
            data will be used for validation. The default is 0.0.
        continue_training : bool, optional
            Sets whether the session will continue training from previous state
            or restart training from scratch. The default is False.
        forceGPU : bool, optional
            Setting to 'True' causes training to fail if a GPU is unavailable.
            The default is False.
        gpuDevice : int, optional
            Sets which gpu device to use in Tensorflow.
            The default is 0.
        scaling_list : list
            List that either contains labels corresponding scaling methods, or it
            contains lists that contain labels corresponding scaling methods.

            Valid Formats: [ <labels> ] or
                        [ [ <input labels> ], [ <output labels> ] ]

            The default is None.
        scaling_params : list, optional
            List of parameters, or list of lists of parameters than have 1:1
            relation to the entries in `scaling_list`. The default is None.
        newPath : str, optional
            Path to the new directory where the model will be saved. The default is './'.
        newName : str, optional
            New name of the model. The default is None.
        transfer_learn : bool, optional
            States whether current session is for transfer learning.
            The default is False.

        Returns
        -------
        None.

        TODO
        ----
        [1] Improve the logic regarding whether to compile or not. Possibly
        remove `transfer_learn` paramter.

        """
        if tf.test.gpu_device_name():
            print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
        elif forceGPU and not tf.test.gpu_device_name():
            print("Please install GPU version of TF")
        else:
            print("Please install GPU version of TF")
        krs.backend.clear_session()

        try:
            phys_dev = tf.config.list_physical_devices('GPU')
            for i in phys_dev:
                tf.config.experimental.set_memory_growth(i,True)
        except:
            pass

        print("continue training: ",continue_training)
        if not continue_training:
            self.args = args
            self.modelName = modelName
            self.modelPath = modelPath

            scaling_list = check_input_scaler_list(scaling_list)
            scaling_params = check_input_scaler_params(scaling_params,
                                                       scaling_list)

            set_scalers( scaling_list[0],
                         self.scaler_order['input_scalers'],
                         self.scaler_data, self.scaler_map,
                         xtrain, scaling_params[0], io = 'in' )

            set_scalers( scaling_list[1],
                         self.scaler_order['output_scalers'],
                         self.scaler_data, self.scaler_map,
                         ytrain,
                         scaling_params[1], io = 'out' )

        xtrain = apply_scalers( xtrain, self.scaler_order['input_scalers'],
                               self.scaler_data, self.scaler_map )
        ytrain = apply_scalers( ytrain, self.scaler_order['output_scalers'],
                               self.scaler_data, self.scaler_map )


        if not continue_training:
            self.learningRate = learningRate
            self.construct()
        else:
            if not newName is None:
                self.modelName = newName
                self.modelPath = newPath
            if self.learningRate != learningRate:
                self.learningRate = learningRate
                self.compile()
        if transfer_learn:
            self.compile()

        print('AE from SAVE_PY_4HYDRA')
        # Train
        avgCost,costTrain,costValid,epochsEval = train(xtrain, ytrain,
                            self.model, learningRate, batchSize,
                            training_epochs, self.modelName, self.modelPath,
                            save_model = save_model, validStep = valid_step,
                            validation_split = validation_split, forceGPU = forceGPU,
                            continue_training = continue_training,
                            gpuDevice=gpuDevice, prefix = self.prefix )

        if not continue_training:
            self.train_cost = costTrain
            self.valid_cost = costValid
            self.avgCost = avgCost
            self.epochs_evaluation = epochsEval
        else:
            self.train_cost = np.concatenate((self.train_cost,costTrain))
            self.valid_cost = np.concatenate((self.valid_cost,costValid))
            self.avgCost = np.concatenate((self.avgCost,avgCost))
            epochsEval += self.epochs_evaluation[-1]
            self.epochs_evaluation = np.concatenate((self.epochs_evaluation,epochsEval))


        if not continue_training and save_model:
            modelName = '%s_'%(self.prefix) + modelName
            with open('%s%s.pkl'%(modelPath, modelName), 'wb') as f:
                cPickle.dump(self,f)
        elif save_model:
            if newName is None:
                modelName = '%s_'%(self.prefix) + modelName
                with open('%s%s.pkl'%(modelPath, modelName), 'wb') as f:
                    cPickle.dump(self,f)
            else:
                name = '%s_'%(self.prefix) + self.modelName
                with open('%s%s.pkl'%(self.modelPath, name), 'wb') as f:
                    cPickle.dump(self,f)

    def save_model(self, newModelPath, newModelName ):
        """
        Save the model.

        The actual file/folder names will be format
        <prefix>_<newModelName>.pkl and
        <prefix>_<newModelName>.mdl

        Parameters
        ----------
        newModelPath : str
            Path to the in which to save the model.
        newModelName : str
            Name with which to name the model.

        Returns
        -------
        None.

        """
        self.modelName = newModelName
        self.modelPath = newModelPath
        name = '%s_'%(self.prefix) + self.modelName
        krs.models.save_model(self.model,"%s/%s.mdl"%(self.modelPath, name))
        with open('%s%s.pkl'%(self.modelPath, name), 'wb') as f:
            cPickle.dump(self,f)


    def load_model(self, modelPath=None, modelName=None):
        """
        Load model <prefix>_<modelName>.mdl.

        This must be implemented separately for all sub-classed models to
        ensure all sub-models are readily accessible via the API.

        Parameters
        ----------
        modelPath : str, optional
            Path to the directory of the model to be loaded.
            If modelPath is None the modelPath will be set to the modelPath
            stored in the class object.
            The default is None.
        modelName : str, optional
            The name of the model to be loaded (Not the file name).
            If modelName is None the modelPath will be set to the modelName
            stored in the class object.
            The default is None.

        Returns
        -------
        None.

        """
        krs.backend.clear_session()
        if modelPath == None: modelPath = self.modelPath
        if modelName == None: modelName = self.modelName
        modelName= self.prefix + "_" +modelName

        self.model = krs.models.load_model("%s%s.mdl"%(modelPath,modelName), custom_objects = self.custom_objects )

        try:
            phys_dev = tf.config.list_physical_devices('GPU')
            for i in phys_dev:
                tf.config.experimental.set_memory_growth(i,True)
        except:
            pass
        print("Model %s%s.mdl successfully restored."%(modelPath,modelName))

    def close_model(self):
        """
        Close the model. Clear keras session.

        Calls tensorflow.keras.backend.clear_session()

        Returns
        -------
        None.

        """
        tf.keras.backend.clear_session()
        print('Model closed')

    def count_parameters(self):
        """
        Count the number of parameters.

        Returns
        -------
        total : int
            Number of parameters in the model.

        """
        total = 0
        for i in self.model.get_weights():
            total += i.size
        print("Total Parameters: ", total)

        return total

    def summary(self):
        """
        Prints the model summary.

        Calls the `summary` method of a tensorflow.keras.Model object.

        Must be implmented by sub-classed models if full summaries of each
        sub-model is desired.

        Returns
        -------
        None.

        """
        self.model.summary()

    def plot_model(self,filePath='./', fileName='model.png'):
        """
        Plot and save an image of the model.

        Calls tensorflow.keras.utils.plot_model.

        Parameters
        ----------
        filePath : str, optional
            The path to the directory where the image will be saved.
            The default is './'.
        fileName : str, optional
            The name that will be given to the image file.
            The default is 'model.png'.

        Returns
        -------
        None.

        """
        krs.utils.plot_model(self.model,to_file='%s%s'%(filePath,fileName),show_shapes=True,show_layer_names=True,expand_nested = True)

    def get_weights(self):
        """
        Retrieve the model weights.

        This needs to be implemented separately for all sub-classed models
        if the weights from just a specific sub-model are wanted.

        Raises
        ------
        Exception
            If self.model is not a tensorflow.keras.Model.

        Returns
        -------
        List of numpy.ndarray
            Model weights.

        """
        if not isinstance(self.model,krs.Model):
            raise Exception('The model has not been defined.')

        return self.model.get_weights()

    def predict(self,X,**kwargs):
        """
        Perform inference.

        Scaling of data automatically occurs prior to and after inference.

        Parameters
        ----------
        X : numpy.ndarray
            The input data.
        **kwargs :
            kwargs which currently have no use.

        Returns
        -------
        Z : numpy.ndarray
            Model output.

        """
        for index,i in enumerate(self.scaler_order['input_scalers']):
            if not i is None:
                X = scaler_functions[self.scaler_map[i]+'_scale'](X,self.scaler_data[i])

        Z = self.model.predict(X)

        for index,i in enumerate(self.scaler_order['output_scalers'][::-1]): # Loops in reverse
            if not i is None:
                Z = scaler_functions[self.scaler_map[i] + '_descale' ](Z,self.scaler_data[i])

        return Z


class Autoencoder(NeuralNetwork):
    """
    Autoencoder implemented using TensorFlow.

    For a full list of attributes and methods, see the
    documentation of the NeuralNetwork class.

    Attributes
    ----------
    encoder : tensorflow.keras.Model
        The encoder.
    decoder : tensorflow.keras.Model
        The decoder.

    Methods
    -------
    encode
    transform
    decode
    reconstruct

    """
    def __init__(self, model = (None,None,None)):
        """
        Initialize Autoencoder class inheriting from NeuralNetwork class.

        Parameters
        ----------
        model : tuple of tensorflow.keras.Models, optional
            model(s) to initialize the Autoencoder object with.
            The required structure is ( <encoder>, <decoder>, <full model>)
            The default is (None,None,None).

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if len(model) != 3:
            raise Exception("model must be a tuple in format (autoencoder,encoder,decoder)")
        if all([ not i is None for i in model]) and all([ not isinstance(i,krs.Model) for i in model]):
            raise Exception("model must be a tuple in format (autoencoder,encoder,decoder) and all objects must be of type %s"%(str(krs.Model)))
        super(Autoencoder,self).__init__(model = model[2])
        self.prefix = "ae"
        self.encoder = model[0]
        self.decoder = model[1]



    def __getstate__(self):
        """
        Get object state.

        This must be implemented separately for each sub-classed model. This
        necessary to make sure the object is able to be pickled.

        At a minimum, `self.model` and all of the sub-models must be deleted
        prior to pickling.

        Returns
        -------
        state : dict
            a dict with unserializable data objects removed

        """
        state = self.__dict__.copy()
        del state['decoder']
        del state['encoder']
        del state['model']
        return state

    def __setstate__(self, state):
        """
        Set object state.

        Parameters
        ----------
        state : dict
            Incoming state.

        Returns
        -------
        None.

        """
        self.__dict__.update(state)
        self.decoder = None
        self.encoder = None
        self.model = None


    def construct(self):
        """
        Construct model from archparse arguments.

        Calls self.compile after archparse construction.

        This must be implemented separately for all sub-classed models to
        ensure all sub-models are readily accessible via the API. If not
        implemented, only the full model will be immediately accessible via
        the API.

        Returns
        -------
        None.

        """
        argSets = archparse.retrieve_args(parsedBody = self.args)
        self.encoder,self.decoder,self.model = archparse.generate_model(self.args,argSets)
        self.custom_objects = self.args['custom']
        self.compile()

    def freeze_layers_except(self,unfrozen_layers=None):
        """
        Freeze all layers except specified layers.

        Indices passed via unfrozen layers should NOT account for the
        InputLayers. InputLayers are not included in the list of layers that
        can be  frozen/unfrozen.

        This must be implemented separately for sub-classed models if
        additional string shortcuts are desired to be used.

        Parameters
        ----------
        unfrozen_layers : str,int,list of str/int, optional
            Layers that are to be kept trainable. A single int or str will
            keep only that single corresponding layer trainable.
            Acceptable string values are ['all','first','last', 'latent'].
            If `unfrozen_layers` is None, all layers will be made untrainable.
            The default is None.

        Raises
        ------
        TypeError
            If unfrozen_layers is not an int,s tr, list of int and/or str.
        ValueError
            If unfrozen_layers is or contains a str other than 'first',
            'last', 'latent', or 'all'.

        Returns
        -------
        None.

        """
        layers = accumulate_layers(self.model)
        if not isinstance(unfrozen_layers,(int,list,str,type(None))):
            raise TypeError("frozen_layers was expecting type int, list, or str but received type %s"%(type(unfrozen_layers)))
        elif isinstance(unfrozen_layers,(str,list)):
            if isinstance(unfrozen_layers,str):
                if not unfrozen_layers in ['first','last','latent','all']:
                    raise ValueError("Invalid string value.")
            else:
                for i in unfrozen_layers:
                    if not isinstance(i,(int,str)):
                        raise TypeError("An argument of type 'list' must contain only integers exclusive or strings.")

        if not isinstance(unfrozen_layers,list):
            unfrozen_layers = [unfrozen_layers]

        for i in range(len(unfrozen_layers)):
            if isinstance(unfrozen_layers[i],int):
                pass
            elif isinstance(unfrozen_layers[i],str):
                if unfrozen_layers[i] == 'first':
                    unfrozen_layers[i] = 0
                elif unfrozen_layers[i] == 'last':
                    unfrozen_layers[i] = -1
                elif unfrozen_layers[i] == 'latent':
                    unfrozen_layers[i] = len(self.encoder.layers)-2
                elif unfrozen_layers[i] == 'all':
                    unfrozen_layers = [i for i in range(len(layers))]
                else:
                    raise ValueError("Invalid str value unfrozen_layers.")
            else:
                raise TypeError("unfrozen_layers must be str,int, or list of str and int.")

        freeze_layers_except(layers,unfrozen_layers)

    def load_model(self, modelPath=None, modelName=None):
        """
        Load model <prefix>_<modelName>.mdl.

        This must be implemented separately for all sub-classed models to
        ensure all sub-models are readily accessible via the API.

        Parameters
        ----------
        modelPath : str, optional
            Path to the directory of the model to be loaded.
            If modelPath is None the modelPath will be set to the modelPath
            stored in the class object.
            The default is None.
        modelName : str, optional
            The name of the model to be loaded (Not the file name).
            If modelName is None the modelPath will be set to the modelName
            stored in the class object.
            The default is None.

        Returns
        -------
        None.

        """
        krs.backend.clear_session()
        if modelPath == None: modelPath = self.modelPath
        if modelName == None: modelName = self.modelName
        modelName="ae_"+modelName

        self.model = krs.models.load_model("%s%s.mdl"%(modelPath,modelName),
                                           custom_objects = self.custom_objects )

        self.encoder = self.model.get_layer('encoder')
        self.decoder = self.model.get_layer('decoder')
        try:
            phys_dev = tf.config.list_physical_devices('GPU')
            for i in phys_dev:
                tf.config.experimental.set_memory_growth(i,True)
        except:
            pass
        print("Model %s%s.mdl successfully restored."%(modelPath,modelName))

    # def count_parameters(self):
    #     total = 0
    #     for i in self.model.get_weights():
    #         total += i.size
    #     print("Total Parameters: ", total)

    def summary(self):
        """
        Prints the model summary.

        Calls the `summary` method of a tensorflow.keras.Model object.

        This is done for the encoder, decoder, and finally the full model.

        Must be implmented by sub-classed models if full summaries of each
        sub-model is desired.

        Returns
        -------
        None.

        """
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def decode(self,Z):
        """
        Decode the latent space of the autoencoder.

        Calls `self.reconstruct`.

        Parameters
        ----------
        Z : numpy.ndarray
            Input data from a latent space.

        Returns
        -------
        numpy.ndarray
            Decoder output.

        """
        return self.reconstruct(Z)

    def encode(self,X):
        """
        Encode the input of the autoencoder.

        Parameters
        ----------
        X : numpy.ndarray
            Input data to be encoded.

        Returns
        -------
        numpy.ndarray
            Encoder output ( the latent space ).

        """
        return self.transform(X)

    def transform(self, X, **kwargs):
        """
        Returns the transformation of data X (latent space of the autoencoder).

        Parameters
        ----------
        X : numpy.ndarray
            Input data to be encoded.
        **kwargs :
            kwargs which currently have no use.

        Returns
        -------
        numpy.ndarray
            Encoder output ( the latent space ).

        """
        for index,i in enumerate(self.scaler_order['input_scalers']):
            if not i is None:
                X = scaler_functions[self.scaler_map[i]+'_scale'](X,self.scaler_data[i])

        X = self.encoder.predict(X)
        return X

    def get_weights(self):
        """
        Returns weights and biases of the encoder and decoder.

        Usage: encoderWeights, decoderWeights = self.get_weights()

        This needs to be implemented separately for all sub-classed models
        if the weights from just a specific sub-model are wanted.

        Raises
        ------
        Exception
            An exception is raised if either self.decoder or self.encoder are
            not a tensorflow.keras.Model object.

        Returns
        -------
        encoderWeights : list of numpy.ndarray
            The list of weights and biases of the encoder
        decoderWeights : list of numpy.ndarray
            The list of weights and biases of the decoder

        """
        if not isinstance(self.encoder,krs.Model) or not isinstance(self.decoder,krs.Model):
            raise Exception('The decoder and/or encoder has not been defined.')
        encoderWeights = self.encoder.get_weights()
        decoderWeights = self.decoder.get_weights()

        return encoderWeights, decoderWeights

    def reconstruct(self, Z, **kwargs):
        """
        Returns the decoding of Z (reconstructs from the model latent space).

        Automatically performs output scaling.

        Parameters
        ----------
        Z : numpy.ndarray
            Input data from a latent space.
        **kwargs :
            kwargs which currently have no use.

        Returns
        -------
        numpy.ndarray
            Decoder output.

        """
        Z = self.decoder.predict(Z)
        # n = len(self.scaler_data['output_scalers'])-1
        for index,i in enumerate(self.scaler_order['output_scalers'][::-1]): # Loops in reverse
            if not i is None:
                Z = scaler_functions[self.scaler_map[i] + '_descale' ](Z,self.scaler_data[i])

        return Z