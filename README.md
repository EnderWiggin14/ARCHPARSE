==========================================================================
||				ARCHPARSE				||
==========================================================================

--------------------------------------------------------------------------
			Python Packages Needed				
--------------------------------------------------------------------------
numpy >= 1.19.0
tensorflow >= 2.4.0
tensorflow-gpu >= 2.4.0 (optional)
scikit-learn >= 0.24.2

--------------------------------------------------------------------------
			Example Instructions
--------------------------------------------------------------------------

1. Install all necessary Python Packages.

2. Place the Archparse directory wherever you please.

3. Run example.py.

--------------------------------------------------------------------------
			Instructions For Use				
--------------------------------------------------------------------------

1. Create an architecture ".arch" file. Further instrctions on this will 
	come later. example.py utilizes the example.arch file. Use this 
	as a reference point.

2. There are two primary commands that will be needed from the 
	archparse.py file. They are "archparse.read_lines" and 
	archparse.parse_body. ( Yes the author knows they should be 
	be combined into one function as the intention for having it this
	way would still be accomplished.)

	first pass the architecture file name and the path to 
	`read_lines` and pass the returned object to `parse_body` as 
	such.

	```
	lines = archparse.read_lines(archPath,archName)
	parsedBody = archparse.parse_body(lines)
	```

3. As of right now archparse.py is fully intended to be used with 
	nn4n_autoencoder.py. This will eventually be renamed to be reflect
	that there is more than just an autoencoder in the file. Also 
	nn4n_autoencoder.py is fully intended to be used with archparse 
	for the time being. This will hopefully be changed in the future 
	to improve the usablility / increase number of use cases.

	Moving on...

	Create an Autoencoder object/instance,
	
	```
	model = nn4n_autoencoder.Autoencoder()
	```

4. Set the scaling methods desired and the parameters necessary for 
	the scaling methods to be performed. Further information 
	will come later.
	
	Note: NeuralNetwork and the classes that inherit from it
		are designed to perform scaling automatically.

	Set the scaling methods and scaling parameters like below. 
	here there are two scaling methods used. First, a constant 
	multiplier is used. Second, the data is featurewise minmax 
	scale. ( This is an example set up, not realistic set up. )

	The constant multiplier needs a mulitplier value, so 2 is 
	provided in the in the same index location of scaling_parms.

	If a scaling method needs no addtional input like featurewise
	minmax scaling, <None> must be passed.

	```
	scalings = ['const-mult','feat-minmax']
	scaling_parms = [2, None]
	```

5. Train the model with the data and hyperparameters of your choice.
	
	```
	model.train( x, x, parsedBody, learningRate, batchSize = batchSize,
            training_epochs=epochs, modelName = modelName, save_model = True,
            validation_split = 0.0, forceGPU=False, scaling_list = scalings,
            scaling_params = scaling_parms)
	```

6. For an Autoencoder, there are two ( technically 5 ) different evaluation 
	functions.

	Autoencoder
	-----------

	Autoencoder.transform(x)	: evaluates just the encoder
	Autoencoder.reconstruct(x)	: evaluates just the decoder
	Autoencoder.encode(x)		: calls Autoencoder.transform(x)
	Autoencoder.decode(x)		: calls Autoencoder.reconstruct(x)
	Autoencoder.predict(x)		: evaluates the whole model, this is
					inherited from NeuralNetwork. The
					latent space is not obtained with 
					this function.

	NeuralNetwork
	-------------

	NeuralNetwork.predict(x)	: evaluates the neural network.

7. Load a saved model with `nn4n_autoencoder.load( <model name>, <file path>, prefix = <prefix> )`.
	Extensions should not be included in the <model name> string.
	<prefix> currently can only take the values 'ae' and 'nn'.

8. If it is desired to freeze layers (make them untrainable) call 
	`Autoencoder.freeze_layers_except( <layers> )`

	where <layers> is a list of the indices or strings that 
	represent the layers in the model that are NOT to be frozen 
	(i.e. still be trainable). This design choice is because the 
	author is of the understanding that during transfer learning 
	it is far more common to freeze almost all layers except for 
	a select few.

	The indices start at 0 and are supposed to represent the layers 
	of the network as the network were drawn on a paper 
	(i.e. the input layer is usually not drawn ). This means that 
	accessing the layers of the model via the Tensorflow API will 
	not result in the same layers being frozen. During the evaluation 
	of `freeze_layers_except()` the layers of the model are 'flattened' 
	into a single list with all `InputLayer` objects removed. These 
	layers will always be trainable after using this function. Users 
	should not worry about this because `InputLayer` objects do not 
	have any parameters to train, so there will be no effect on 
	training by leaving them trainable.

	The recognized strings are:

	For Autoencoder and Neural Network
	----------------------------------
	'first'	 ->	first layer of full model
	'last' 	 ->	last layer of full model
	'all'    ->	all layers of the full model

	Just Autoencoder
	----------------
	'latent' ->	latent space layer (last layer of encoder)

9. To change the save location of the model Autoencoder.save_model() 
	can be used. Do not add extensions to the model name.
	
	NOTE: The behavior of `save_model()` is that it will also 
		change the model's class attributes `self.modelName` 
		`self.modelPath` to correspond to the new name and 
		save location.

--------------------------------------------------------------------------
			Architecture File Writing
--------------------------------------------------------------------------
	-Commenting can be done with by preceding comment with '#'.
	-Formatting is relatively easy but is strict.

	Header ( this is not to be used in actual '.arch' files )
	------
line 1.	model type - either 'AE' or 'NN'
line 2. name - letters and numbers, no spaces
line 3.	INDIM - input dimensions, space separated, 1x13 would just be "13"
		4x4 would be "4 4"

		Example:	INDIM 13
				INDIM 4 4
line 4. LTDIM - latent dimensions, space separated, format same as INDIM
line 5. LOSS - loss functions (more info later)
		
		Options:
		BINCR	: binary cross-entropy
		CATCR	: categorical cross-entropy
		CATH	: categorial hinge
		COSIM	: cosine similarity
		HINGE	: hinge
		HUBER	: Huber
		KLD	: Kullback-Liebler divergence
		LOGC	: log cosh
		MAE	: mean absolute error
		MAPE	: mean absolute percent error
		MSE	: mean square error
		MSLE	: mean squared log error
		POIS	: Poisson
		SCATC	: sparse categorical cross-entropy
		SQHIN	: squared hinge
		MIMSE	: Mean-Inf Norm + MSE (custom)

line 6. Empty or Commented Line (only one, must exist)
line 7. If NeuralNetwork	:"NTWRK"
	If Autoencoder		:"ENCOD"

lines 8 to line x.
	Network layers and options, Encoder layers and options

If NeuralNetwork done.
If Autoencoder continue,

line x+1. Empty or Commented Line (only one, must exist)
line x+2 = y. "DECOD"
lines y to end.
	Decoder layers and options.

--------------------------------------------------------------------------
			Layers and Options
--------------------------------------------------------------------------

	Input Command		Options
	-------------		-------
	FC - fully connected
				positional arguments:
  				size           	The size of the layer. It must be an integer greater
                        			than zero

				optional arguments:
				  -h, --help            show this help message and exit
				  -i initializer        Kernel initializer function.
  				  -ki kernel_init       Kernel initializer function.
  				  -bi bias_init         The bias initializer. Default is 'zeros'
  				  -kr kernelReg [kernelReg ...]
                        			  	Option to add a kernel regularizer. Choices are L1,
                        			  	L2, or L1L2. L1 and L2 each need one additional float
                        			  	parameter. L1L2 needs two additional float parameters.
  				  -br biasReg [biasReg ...]
                        			  	Option to add a bias regularizer. Choices are L1, L2,
                        			  	or L1L2. L1 and L2 each need one additional float
                        			  	parameter. L1L2 needs two additional float parameters.
  				  -ar actReg [actReg ...]
                        			  	Option to add a activity regularizer. Choices are L1,
                       				  	L2, or L1L2. L1 and L2 each need one additional float
                        			  	parameter. L1L2 needs two additional float parameters.
  				  -b use_bias     	Sets whether a bias is used. Default is True
  				  -a activation [activation ...]
                        				Activation function choice. The options are [relu,sig,
                        				elu,exp,selu,spls,ssgn,None,lin,smax,swsh,hsig,tanh,sl
                        				isin,slicos,repolu,relpolu,nlrelu,lognrelu]

	CONV - 1d convolution
				positional arguments:
				  filters               The number of filters to be used.
				  size                  The size of the convolutional kernel.

				optional arguments:
				  -h, --help            show this help message and exit
				  -i initializer        Kernel initializer function.
				  -ki kernel_init       Kernel initializer function.
				  -bi bias_init         The bias initializer. Default is 'zeros'
				  -kr kernelReg [kernelReg ...]
				                        Option to add a kernel regularizer. Choices are L1,
				                        L2, or L1L2. L1 and L2 each need one additional float
				                        parameter. L1L2 needs two additional float parameters.
				  -br biasReg [biasReg ...]
				                        Option to add a bias regularizer. Choices are L1, L2,
				                        or L1L2. L1 and L2 each need one additional float
				                        parameter. L1L2 needs two additional float parameters.
				  -ar actReg [actReg ...]
				                        Option to add a activity regularizer. Choices are L1,
				                        L2, or L1L2. L1 and L2 each need one additional float
				                        parameter. L1L2 needs two additional float parameters.
				  -b use_bias           Sets whether a bias is used. Default is True
				  -a activation [activation ...]
				                        Activation function choice. The options are [relu,sig,
				                        elu,exp,selu,spls,ssgn,None,lin,smax,swsh,hsig,tanh,sl
				                        isin,slicos,repolu,relpolu,nlrelu,lognrelu]
				  -pad padType          These options correspond to the allowable 'padding'
				                        options for tf.keras.layers.Conv1D()
							{ s = 'same', v = 'valid }
				  -s stride             The length of the stride between filter applications.
				  -d dilation           The rate of dilation.

	CONVT - 1d deconvolution
				positional arguments:
				  filters               The number of filters to be used.
				  size                  The size of the convolutional kernel.

				optional arguments:
				  -h, --help            show this help message and exit
				  -i initializer        Kernel initializer function.
				  -ki kernel_init       Kernel initializer function.
				  -bi bias_init         The bias initializer. Default is 'zeros'
				  -kr kernelReg [kernelReg ...]
				                        Option to add a kernel regularizer. Choices are L1,
				                        L2, or L1L2. L1 and L2 each need one additional float
				                        parameter. L1L2 needs two additional float parameters.
				  -br biasReg [biasReg ...]
				                        Option to add a bias regularizer. Choices are L1, L2,
				                        or L1L2. L1 and L2 each need one additional float
				                        parameter. L1L2 needs two additional float parameters.
				  -ar actReg [actReg ...]
				                        Option to add a activity regularizer. Choices are L1,
				                        L2, or L1L2. L1 and L2 each need one additional float
				                        parameter. L1L2 needs two additional float parameters.
				  -b use_bias           Sets whether a bias is used. Default is True
				  -a activation [activation ...]
				                        Activation function choice. The options are [relu,sig,
				                        elu,exp,selu,spls,ssgn,None,lin,smax,swsh,hsig,tanh,sl
				                        isin,slicos,repolu,relpolu,nlrelu,lognrelu]
				  -pad padType          These options correspond to the allowable 'padding'
				                        options for tf.keras.layers.Conv1D()
				  -s stride             The length of the stride between filter applications.
				  -o outputPadding      The amount of padding along the non-channel dimension.
				                        The value must be less that the size of stride.
				  -d dilation           The rate of dilation.

	CONV2 - 2d convolution
				positional arguments:
				  filters               The number of filters to be used.
				  size                  The size of the convolutional kernel.

				optional arguments:
				  -h, --help            show this help message and exit
				  -i initializer        Kernel initializer function.
				  -ki kernel_init       Kernel initializer function.
				  -bi bias_init         The bias initializer. Default is 'zeros'
				  -kr kernelReg [kernelReg ...]
				                        Option to add a kernel regularizer. Choices are L1,
				                        L2, or L1L2. L1 and L2 each need one additional float
				                        parameter. L1L2 needs two additional float parameters.
				  -br biasReg [biasReg ...]
				                        Option to add a bias regularizer. Choices are L1, L2,
				                        or L1L2. L1 and L2 each need one additional float
				                        parameter. L1L2 needs two additional float parameters.
				  -ar actReg [actReg ...]
				                        Option to add a activity regularizer. Choices are L1,
				                        L2, or L1L2. L1 and L2 each need one additional float
				                        parameter. L1L2 needs two additional float parameters.
				  -b use_bias           Sets whether a bias is used. Default is True
				  -a activation [activation ...]
				                        Activation function choice. The options are [relu,sig,
				                        elu,exp,selu,spls,ssgn,None,lin,smax,swsh,hsig,tanh,sl
				                        isin,slicos,repolu,relpolu,nlrelu,lognrelu]
				  -pad padType          These options correspond to the allowable 'padding'
				                        options for tf.keras.layers.Conv2DTranspose()
				  -s stride [stride ...]
				                        The length of the stride between filter applications.
				  -d dilation [dilation ...]
				                        The rate of dilation.

	CONV2T - 2d deconvolution
				positional arguments:
				  filters               The number of filters to be used.
				  size                  The size of the convolutional kernel.

				optional arguments:
				  -h, --help            show this help message and exit
				  -i initializer        Kernel initializer function.
				  -ki kernel_init       Kernel initializer function.
				  -bi bias_init         The bias initializer. Default is 'zeros'
				  -kr kernelReg [kernelReg ...]
				                        Option to add a kernel regularizer. Choices are L1,
				                        L2, or L1L2. L1 and L2 each need one additional float
				                        parameter. L1L2 needs two additional float parameters.
				  -br biasReg [biasReg ...]
				                        Option to add a bias regularizer. Choices are L1, L2,
				                        or L1L2. L1 and L2 each need one additional float
				                        parameter. L1L2 needs two additional float parameters.
				  -ar actReg [actReg ...]
				                        Option to add a activity regularizer. Choices are L1,
				                        L2, or L1L2. L1 and L2 each need one additional float
				                        parameter. L1L2 needs two additional float parameters.
				  -b use_bias           Sets whether a bias is used. Default is True
				  -a activation [activation ...]
				                        Activation function choice. The options are [relu,sig,
				                        elu,exp,selu,spls,ssgn,None,lin,smax,swsh,hsig,tanh,sl
				                        isin,slicos,repolu,relpolu,nlrelu,lognrelu]
				  -pad padType          These options correspond to the allowable 'padding'
				                        options for tf.keras.layers.Conv2DTranspose()
				  -s stride [stride ...]
				                        The length of the stride between filter applications.
				  -d dilation [dilation ...]
				                        The rate of dilation.
				  -o [outputPadding [outputPadding ...]]
				                        The amount of padding along the non-channel dimension.
				                        The value must be less that the size of stride.

	FLAT - flatten
				no options

	RSHP - reshape
				positional arguments:
				  dimensions  		A list of the dimension sizes

				optional arguments:
				  -h, --help  		show this help message and exit

	MXP1 - 1d max pooling
				positional arguments:
				  size          	The size of the pool. Default size is 2.

				optional arguments:
				  -h, --help    	show this help message and exit
				  -pad padType  	These options correspond to the allowable 'padding' options
                					for tf.keras.layers.MaxPool1D()
				  -s stride     	The length of the stride between filter applications.

	AVP1 - 1d average pooling
				positional arguments:
				  size          	The size of the pool. Default size is 2.

				optional arguments:
				  -h, --help    	show this help message and exit
				  -pad padType  	These options correspond to the allowable 'padding' options
				                	for tf.keras.layers.AveragePooling1D()
				  -s stride     	The length of the stride between filter applications.

	UPS1 - 1d up-sampling
				positional arguments:
				  size			The upsampling width. A size of 3 means that each value will
					              	have 2 extra copies. Default size is 2.

				optional arguments:
				  -h, --help  		show this help message and exit

	POLREG - polynomial regression
				optional arguments:
				  -h, --help            show this help message and exit
				  -i initializer        Kernel initializer function.
				  -ki kernel_init       Kernel initializer function.
				  -bi bias_init         The bias initializer. Default is 'zeros'
				  -kr kernelReg [kernelReg ...]
				                        Option to add a kernel regularizer. Choices are L1,
				                        L2, or L1L2. L1 and L2 each need one additional float
				                        parameter. L1L2 needs two additional float parameters.
				  -br biasReg [biasReg ...]
				                        Option to add a bias regularizer. Choices are L1, L2,
				                        or L1L2. L1 and L2 each need one additional float
				                        parameter. L1L2 needs two additional float parameters.
				  -ar actReg [actReg ...]
				                        Option to add a activity regularizer. Choices are L1,
				                        L2, or L1L2. L1 and L2 each need one additional float
				                        parameter. L1L2 needs two additional float parameters.
				  -b use_bias           Sets whether a bias is used. Default is True
				  -n n_poly             The number of polynomial repetitions.
				  -o order              The order of the polynomial activation functions.
				  -pad node_padding     The number of nodes to append to layer to allow any
				                        number of inputs or outputs when used as an output
				                        layer (i.e. for autoencoders). The activation function
				                        of these nodes is relu
				  -a activation         The choice of activation class. Non-standard allowances.
							Only two options. 'repolu' or 'relpolu'

--------------------------------------------------------------------------
			Activation Functions
--------------------------------------------------------------------------
The capacity to specify additional arguments for activation functions 
exists, but it isn't currently implemented.

	relu		: rectified linear unit
	sig		: sigmoid
	elu		: exponential linear unit
	selu		: scaled exponential linear unit
	spls		: softplus
	ssgn		: soft sign
	swsh		: swish
	tanh		: tanh
	slisin		: scaled linear-sine (custom)
	slicos		: scaled linear-cosine (custom)
	nlrelu		: ( natural ) logarithmic rectified linear unit (custom,from a paper)
	repolu		: rectified polynomial unit (custom)
	relpolu		: recitfied linear-polynomial unit (custom)
	lognrelu	: log_n rectified linear unit

--------------------------------------------------------------------------
			Initializer Options
--------------------------------------------------------------------------
	
	xavu		: Xavier uniform
	xavn		: Xavier normal
	glu		: Glorot uniform
	gln		: Glorot normal
	heu		: He Uniform
	hen		: He Normal
	orth		: orthogonal
	rndn		: random normal
	rndu		: random uniform
	vsc		: variance scaling
	zeros		: zeros

--------------------------------------------------------------------------
			Padding Options
--------------------------------------------------------------------------
	
	v		: 'valid'
	s		: 'same'
	c		: 'causal' (this one is particularly well supported in tensorflow)

--------------------------------------------------------------------------
			Regularizer Options
--------------------------------------------------------------------------
	
	L1 [weight]	: L1 regularizer
	L2 [weight]	: L2 regularizer
	L1L2 [w1] [w2]	: L1L2 regularizer

--------------------------------------------------------------------------
			Loss Options
--------------------------------------------------------------------------
usage: [-h] [-reduct reduction]
                  {BINCR,CATCR,CATH,COSIM,HINGE,HUBER,KLD,LOGC,MAE,MAPE,MSE,MSLE,POIS,MIMSE,SCATC,SQHIN}

				optional arguments:
				  -h, --help            show this help message and exit
				  -reduct reduction     Reduction method, Default is AUTO

				LOSS FUNCS:
				  {BINCR,CATCR,CATH,COSIM,HINGE,HUBER,KLD,LOGC,MAE,MAPE,MSE,MSLE,POIS,MIMSE,SCATC,SQHIN}
				    BINCR               Binary Cross-Entropy
				    CATCR               Categorical Cross-Entropy
				    CATH                Categorical Hinge
				    COSIM               Cosine Similarity
				    HINGE               Hinge
				    HUBER               Huber
				    KLD                 Kulback-Liebler Divergence
				    LOGC                LogCosh
				    MAE                 Mean Absolute Error
				    MAPE                Mean Absolute Percent Error
				    MSE                 Mean Square Error
				    MSLE                Mean Square Logarithmic Error
				    POIS                Poisson
				    MIMSE               Mean-Infinity Norm + Mean Square Error
				    SCATC               Sparse Categorical Cross-Entropy
				    SQHIN               Square Hinge

	Input Command		Options
	-------------		-------

	BINCR - binary cross-entropy
				optional arguments:
				  -h, --help      	show this help message and exit
				  -logits         	Boolean - whether to interpret as logits
				  -smooth smooth  	label smoothing - in range [0,1]. 0 is not smoothing, 1 is
					                heaviest smothing.

	CATCR - categorical cross-entropy
				optional arguments:
				  -h, --help      	show this help message and exit
				  -logits         	Boolean - whether to interpret as logits
				  -smooth smooth  	label smoothing - in range [0,1]. 0 is not smoothing, 1 is
					        	heaviest smothing.

	CATH - categorial hinge
				no options

	COSIM - cosine similarity
				optional arguments:
				  -h, --help  		show this help message and exit
				  -axis axis  		Axis along which cosine similarity is computed

	HINGE - hinge
				no options

	HUBER - Huber
					optional arguments:
					  -h, --help    show this help message and exit
					  -delta delta  a float that is the point where the Huber loss function
					                changes from quadratic to linear.

	KLD - Kullback-Liebler divergence
				no options

	LOGC - log cosh
				no options

	MAE - mean absolute error
				no options

	MAPE - mean absolute percent error
				no options

	MSE - mean square error
				no options

	MSLE - mean squared log error
				no options

	POIS - Poisson
				no options

	MIMSE - Mean-Inf Norm + MSE (custom)
				optional arguments:
				  -h, --help            show this help message and exit
				  -mse mse_weight, --mse_weight mse_weight
                        				MSE weight
				  -inf inf_weight, --inf_weight inf_weight
				                        Mean-Inf Norm weight

	SCATC - sparse categorical cross-entropy
				optional arguments:
				  -h, --help  		show this help message and exit
				  -logits     		Boolean - whether to interpret as logits

	SQHIN - squared hinge
				no options

--------------------------------------------------------------------------
			Scaling Options
--------------------------------------------------------------------------
NOTE: The following are all numpy scaling options. In the package, there 
	currently exists a file,  "tf_scaling_funcs.py," that has 
	some scaling methods implemented as Tensorflow layers. They will 
	eventually be added to the archparser, but how they will interact
	with the numpy scaling options is unknown as of right now.

	Scaling Method	String Label	Params
	--------------  ------------	------
	natural log	log		None
	log10		log10		None
	log_n		logn		<base n>
	signed log_n	signed_logn	<base n>
	feature minmax	feat-minmax	None
	sample minmax	samp-minmax	None
	unit (l2) norm	unit-norm	None
	consant mult	const-mult	<multiplier>
	z number based	z-num		<z number vector>
	cube root	cbrt		None
	n-th root	nroot		<n value>
	vector mult	mult-vec	<vector of multipliers (featurwise)>
	vector div	div-vec		<vector of divisors (featurwise)>

--------------------------------------------------------------------------
			Release
--------------------------------------------------------------------------
LLNL-CODE- 827607

Title: Archparse, Version: 0.2.0

Author(s) Michael D. Vander Wal

CP02489