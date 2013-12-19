'''
Created on Jul 4, 2013

@author: Gregory Kramida
'''
import theano
import theano.tensor as T
import numpy as np
from data_prep import prep_data
from theano.ifelse import ifelse

THEANO_FLOAT_X = np.dtype(np.double).name
if(hasattr(theano.config,'floatX')):
    THEANO_FLOAT_X = getattr(theano.config,'floatX')

class Layer(object):
    """
    Layer of a neural network: units are fully-connected and use the specified
    link function. Weight matrix W is of shape (num_input,num_output)
    and the bias vector b is of shape (num_output).

    NOTE : The nonlinearity used here is tanh
    Hidden unit activation is given by: tanh(dot(input,W) + b),

    @type weights: int, theano tensor, numpy array, or None
    @param weights: initial weights for the hidden layer.

    @type random_gen: numpy.random.RandomState
    @param random_gen: a random state used by the random generator
        to initialize weights

    @type data_in: theano.tensor.dmatrix
    @param data_in: a symbolic tensor of shape (batch_size, num_input)

    @type num_input: int
    @param num_input: dimensionality of input

    @type num_output: int
    @param num_output: number of hidden units

    @type link_function: theano.Op or function
    @defualt: T.tanh
    @param link_function: used to make the hidden layer's
        output a continous function
    """
    def __init__(self, data_in, num_input, num_output,
                 weights=None, bias=None, random_gen = None,
                 link_function = T.tanh):

        #initialize weights if initial weights aren't given
        if weights is None:
            if random_gen is not None:
                np.random.set_state(random_gen)
            bound = np.sqrt(0.6 /  num_input + num_output)
            self.weights = theano.shared(np.asarray(np.random.uniform(
                                        low=-bound,high=bound,
                                        size= (num_input,num_output)),
                                        dtype=THEANO_FLOAT_X), name="layer_weights")
        elif weights is 0:
            self.weights = theano.shared(np.zeros((num_input, num_output),dtype=THEANO_FLOAT_X),
                name="layer_weights")
        elif type(weights) is np.ndarray:
            self.weights = theano.shared(weights, name="layer_weights")
        else:
            #assume theano variable
            self.weights = weights

        #initialize bias vector if it's not given
        if bias is None:
            self.bias = theano.shared(np.zeros((num_output,), dtype=THEANO_FLOAT_X),name="layer_bias")
        elif type(bias) is np.ndarray:
            self.bias = theano.shared(bias, name="layer_bias")
        else:
            #assume theano variable
            self.bias = bias

        #function to compute linear output (activation)
        #produces output of size (batch_size,num_output),
        linear_output = T.dot(data_in,self.weights) + self.bias

        #differentiable output
        if link_function is None:
            self.output = linear_output
        else:
            self.output = link_function(linear_output)
        self.params = [self.weights,self.bias]



class SCGLayer(Layer):
    def __init__(self,data_in, num_input, num_output,
                 weights=None, bias=None, random_gen = None, link_function = T.tanh):
        super(SCGLayer,self).__init__(data_in,num_input,num_output,weights,bias,random_gen,link_function)

        self.w_search_dir = theano.shared(np.zeros((num_input, num_output),dtype=THEANO_FLOAT_X),name="weight_search_direction")
        self.b_search_dir = theano.shared(np.zeros((num_output),dtype=THEANO_FLOAT_X),name="bias_search_direction")
        self.w_reciprocal = theano.shared(np.zeros((num_input, num_output),dtype=THEANO_FLOAT_X),name="weight_reciprocal")
        self.b_reciprocal = theano.shared(np.zeros((num_output),dtype=THEANO_FLOAT_X),name="bias_reciprocal")
        self.search = [self.w_search_dir,self.b_search_dir]
        self.reciprocals = [self.w_reciprocal,self.b_reciprocal]


class NeuralNet(object):
    '''
    A feed-forward neural network with a single hidden layer.
    Consists of an arbitrary number layers: input, any
    number of hidden, and output.

    Input layer is simply the data being passed in.

    Expected data is multiple samples with multiple features,
    i.e. an n-by-m matrix, where n is the number of samples, and
    m is the number of features.

    When a single sample of data is fed into the first hidden layer,
    eash feature is multiplied by a weight that corresponds to the transition
    from a single input node to a single hidden layer node. Let h represent
    the number of nodes in the hidden layer.

    Then, every node of the first hidden layer would have m weights
    associated with it, and the hidden weights may be represented by
    a m-by-h matrix W.

    The hidden layer would produce Wx + b, where
    vector b of size 1xh is the bias. Each entry of the output
    would then be fed into the link_function to make it continuously
    differentiable and then be passed on as input to the next hidden layer.

    @type link_function_hidden theano function
    @defualt link_function_hidden: T.tanh
    @param link_function_hidden: function to make the outputs
        of the hidden nodes continuously differentiable

    @type link_function_output theano funciton
    @defualt link_function_output: T.tanh
    @param link_function_output: function to make the
        network outputs continuously differentiable
    '''
    def __init__(self, hidden_dimensions,
        link_function_hidden = T.tanh,
        link_function_output = None):
        if type(hidden_dimensions) is int:
            self.hidden_dimensions = [hidden_dimensions]
        else:
            self.hidden_dimensions = hidden_dimensions
        self.link_function_hidden = link_function_hidden
        self.link_function_output = link_function_output


    def __init_data(self,data,split_sizes,randomize):
        return prep_data(data,split_sizes,randomize)


    def __create_node_set(self, LayerType, num_features, num_output, data_in,
            weightsFunc = None):
        prev_out = data_in
        prev_dim = num_features
        layers = []
        num_weights = 0
        for i_h_layer in range(0,len(self.hidden_dimensions)):
            n_hidden_nodes = self.hidden_dimensions[i_h_layer]
            weights = None
            bias = None
            if weightsFunc is not None:
                weights,bias = weightsFunc(i_h_layer)
            #acutal hidden layer
            hidden_layer = LayerType(data_in=prev_out,
                                    num_input=prev_dim,
                                    num_output=n_hidden_nodes,
                                    link_function=self.link_function_hidden,
                                    weights=weights,
                                    bias=bias)

            layers.append(hidden_layer)
            num_weights += (prev_dim+1)*n_hidden_nodes
            prev_out = hidden_layer.output
            prev_dim = n_hidden_nodes


        weights = 0
        bias = None
        if weightsFunc is not None:
            weights,bias = weightsFunc(len(self.hidden_dimensions))
        output_layer = LayerType(
            data_in=prev_out,
            num_input=prev_dim,
            num_output=num_output,
            link_function=self.link_function_output,
            weights=weights,
            bias=bias)
        num_weights += (prev_dim+1)*num_output
        layers.append(output_layer)
        return layers,num_weights

    def __build_architecture(self, num_features, num_output, data_in, data_out, train_in, train_out, batch_size):
        #create layers
        #this set is only used for actual prediction (it has the true weights)
        self.layers,num_weights = \
            self.__create_node_set(SCGLayer,num_features,num_output,data_in)
        ol = self.layers[len(self.layers)-1]

        success = theano.shared(np.int8(1),name="Reduction Success")
        lambda_ = theano.shared(self.params["lambda"],name="Lambda")
        #scale factor normalizer
        lambda_bar = theano.shared(0.0,name="Lambda Bar")
        batch_index = T.lscalar()
        epoch = T.lscalar()

        
        #retained error from previous iteration
        error = theano.shared(0.0,name="Mean Squared Error")
        
        self.train_error = error

        #aggregate all layer weights, reciprocals, and search directions
        weights = self.layers[0].params
        reciprocals = self.layers[0].reciprocals
        search_dirs = self.layers[0].search
        for layer in self.layers[1:]:
            weights += layer.params
            reciprocals += layer.reciprocals
            search_dirs += layer.search
        self.weights = weights
        self.search_dirs = search_dirs
        self.reciprocals = reciprocals


        #============(Step 2)=====================#
        #2nd-order information 
        sigma = self.params['sigma']

        pL2norm = T.sum(search_dirs[0]**2)
        for search_dir in search_dirs[1:]:
            pL2norm = pL2norm + T.sum(T.sqr(search_dir))
        pnorm = T.sqrt(pL2norm)
        calc_sigma = sigma / pnorm

        def hes_weight_func(i):
            layer = self.layers[i]
            weights=layer.weights + calc_sigma * layer.w_search_dir
            weights.name = "Hessian Approx. Weights"
            bias =layer.bias + calc_sigma * layer.b_search_dir
            bias.name = "Hessian Approx. Bias"
            return weights, bias
        
        #create mirror node set for Hessian approximation
        self.hes_approx_layers = \
            self.__create_node_set(Layer,num_features,num_output,data_in,hes_weight_func)[0]
        hess_ol = self.hes_approx_layers[len(self.hes_approx_layers) - 1]
        hess_error = T.mean((data_out - hess_ol.output)**2)

        hess_error.name = "Hess. Approx. MSE"

        #aggregate hessian approximation weights
        hess_weights = self.hes_approx_layers[0].params
        for layer in self.hes_approx_layers[1:]:
            hess_weights += layer.params

        #compute gradients
        hess_grads = T.grad(hess_error,hess_weights)
        
        #for storing / re-using old delta in case a reduction wasn't possible
        delta_cache = theano.shared(0.0,name="Old Delta")

        #estimate of products of hessian components with the search direction
        #s ~~ Hp
        #this delta indicator is approx. (p^T)Hp, wich has to be positive for
        #positive-definite H
        calc_delta = 0

        #note that the reciprocals are negative gradients and are computed in the previous iteration
        for reciprocal, hess_grad, search_dir in zip(reciprocals,hess_grads,search_dirs):
            #the numerator is subtracting the gradients from the modified gradients
            #hessian_comp is a part of the "s" vector
            #takes advantage that reciprocal is actually -E'(w_k)
            hessian_comp = (hess_grad + reciprocal) / calc_sigma
            #hessian_prod.append(hessian_comp)
            #equivalent to dot product of p with s
            calc_delta += T.sum(hessian_comp*search_dir)
        calc_delta.name = "New Delta"

        #scaling to get effects of a positive-definite H
        #calculate new delta if last reduction was successfull, use old one otherwise
        #============(Step 3)=====================#
        delta = ifelse(success, calc_delta, delta_cache) + (lambda_ - lambda_bar)*pL2norm
        delta.name = "Scaled Delta"
        
        #============(Step 4)=====================#
        #if delta < 0, we need to make the Hessian positive definite 
        #recompute the scaling factor normalizer
        posdef_scale_factor = 2*(lambda_- (delta / pL2norm))
        posdef_scale_factor.name = "Lambda After pos-def Updates"
        posdef_updates = ifelse(T.gt(delta,0),
            [delta,lambda_],
            [-delta + lambda_*pL2norm, posdef_scale_factor])
        #note - the delta is still calculated with the old lambda
        delta = posdef_updates[0]
        delta.name = "Pos-def Delta"
        #lambda and lambda bar either stay the same or both change to the same value
        calc_lambda = posdef_updates[1]
        calc_lambda.name = "Pos-def Lambda"

        #============(Step 5)=====================#
        #build step-size function 
        mu =  T.sum(search_dirs[0]*reciprocals[0])
        mu.name = "Mu"
        #dot product of search direction
        for search_dir,reciprocal in zip(search_dirs[1:],reciprocals[1:]):
            mu += T.sum(search_dir * reciprocal)

        step_size = mu / delta
        step_size.name = "Alpha (Step Size)"

        def try_weight_func(i):
            layer=self.layers[i]
            weights=layer.weights + step_size * layer.w_search_dir
            weights.name = "Tryout Weights"
            bias =layer.bias + step_size * layer.b_search_dir
            bias.name = "Tryout Bias"
            return weights, bias
        
        #create mirror node set for trying a new step
        self.try_layers = \
            self.__create_node_set(Layer,num_features,num_output,data_in,try_weight_func)[0]
        try_ol = self.try_layers[len(self.try_layers)-1]
        #new error
        try_error = T.mean((data_out - try_ol.output)**2)
        #aggregate weights
        try_weights = self.try_layers[0].params
        for layer in self.try_layers[1:]:
            try_weights += layer.params
        #compute gradeints
        try_grads = T.grad(try_error,try_weights)

        # don't recompute error - instead used the cached version
        com_param = 2*delta*(error - try_error)/(mu**2)
        #comparison parameters
        com_param.name = "Comparison Parameter"

        #====================Step 7 & 8===================#
        #successful updates
        #updates to the weights
        reduction_updates = try_weights
        #these will become new reciprocals
        reduction_updates += [-try_grad for try_grad in try_grads]

        #search direction updates
        new_reciprocal_norm = 0
        for try_grad in try_grads:
            new_reciprocal_norm += T.sum(T.sqr(try_grad))

        direction_update_ops = []
        beta = 0
        
        #beta is the magnitude we need to multiply the previous search_direction by to cancel it out of the new verctor
        for try_grad, reciprocal in zip(try_grads, reciprocals):
            beta += T.sum(try_grad * reciprocal)

        beta = (new_reciprocal_norm + beta) / mu
        beta.name = "Beta"
        
        for try_grad, search_dir in zip(try_grads, search_dirs):
            direction_update_ops.append(-try_grad + beta*search_dir)

        #if the epoch reaches # of weights, restart the algorithm (Step 7a)
        direction_update_ops = ifelse(T.eq(epoch % num_weights,0),[-try_grad for try_grad in try_grads],direction_update_ops)

        reduction_updates += direction_update_ops

        rluo_partial = ifelse(T.lt(com_param,0.25),4*calc_lambda,calc_lambda)
        reduction_lambda_update_op = ifelse(
            T.lt(com_param,0.75),
            rluo_partial,
            calc_lambda/2)

        #updates to weights, reciprocals, search_dirs, lambda, lambda_bar, success, delta_cache, error
        learning_update_ops = ifelse(T.lt(com_param,0.0),
            weights + reciprocals + search_dirs + [4*calc_lambda, calc_lambda, 0, delta, error],
            reduction_updates +  [reduction_lambda_update_op, T.zeros_like(calc_lambda), 1, delta, try_error])

        vals_to_update = weights + reciprocals + search_dirs + [lambda_, lambda_bar, success, delta_cache, error]
        self.success = success

        learning_updates = []
        for val, update in zip (vals_to_update, learning_update_ops):
            learning_updates.append((val,update))

        trainSCG = theano.function(
            inputs=[batch_index,epoch],
            outputs=try_error,
            updates=learning_updates,
            givens={
                    data_in: train_in[batch_index * batch_size:(batch_index + 1) * batch_size],
                    data_out: train_out[batch_index * batch_size:(batch_index + 1) * batch_size]
                    },
            name='trainSCG'
        )

        trainSCG_Debug = theano.function(
            inputs=[batch_index,epoch],
            outputs=try_error,
            updates=learning_updates,
            givens={
                    data_in: train_in[batch_index * batch_size:(batch_index + 1) * batch_size],
                    data_out: train_out[batch_index * batch_size:(batch_index + 1) * batch_size]
                    },
            name='trainSCG'
        )
        
        self.train_func = trainSCG_Debug

    default_params = {
                'max_epochs': 100,
                'disp_interval':25,
                'print_interval': True,
                'show_gui':False,
                'goal': 0.0,
                'max_seconds': float("inf"),
                'min_grad': 1e-6,
                'max_fail': 5,
                'sigma': 5.0e-5,
                'lambda':5.0e-7
            }


    '''
    Trains the current network on the given input.

    @type data: tuple of numpy arrays
     @param data: input data in one of the following forms:
    (train_input,train_output,validation_input,validation_output, test_input, test_output)
    (train_input,train_output, test_input, test_output) - no validation will be conducted
    (train_input,train_output) - if split_sizes are not passed
    (entire_data_input,entire_data_output) - if split_sizes are passed,
    the dataset will be split up accordingly

    @type randomize: Boolean
    @default randomize: True
    @param randomize: if set to true when train_input/train_output are predefined,
    randomizes just those. If set to true when entire data is given and split_sizes are specified,
    randomizes the whole data first, and then proceeds with the split.

    @type split_sizes: tuple of floats
    @param split_sizes: ratios for training, validation, and testing set. Ignored
    unless data has length 2. The values in this tuple should add up to 1.
    If one value is omitted or the second value low enough to yield zero validation samples,
    the function won't perform validation. If the last value yields 0 when
    multiplied by total data size and truncated to the nearest whole, the
    function won't perform testing.
    '''
    def train(self,x,y,params=None,randomize=True,
         split_sizes=None, batch_size=None):
        #TODO: add support for different error functions
        if params is None:
            self.params = NeuralNet.default_params
        else:
            for param in NeuralNet.default_params:
                if(param not in params):
                    params[param] = NeuralNet.default_params[param]
            self.params = params

        #prepare the data
        (train_in,train_out,
         val_in,val_out,
         test_in,test_out,
         do_validation,do_test) =\
        self.__init_data(x,y, split_sizes, randomize)

        #input size is number
        num_features = train_in.get_value().shape[1]
        num_output = train_out.get_value().shape[1]

        data_in = T.matrix('data_in')
        data_out = T.matrix('data_out')

        #default batchsize is the whole train dataset
        n_samples = len(train_in.get_value())
        if(batch_size is None):
            batch_size = n_samples

        self.__build_architecture(num_features,
            num_output,data_in, data_out,
            train_in, train_out, batch_size)
        
        

        #initialize first iteration     
        ol = self.layers[len(self.layers)-1]
        #actual first error
        calc_error = T.mean((data_out - ol.output)**2)
        calc_error.name = "Current MSE"
        self.calc_error = calc_error
        #compute gradients - first iteration only
        grads = T.grad(calc_error,self.weights)
        self.grads = grads
        first_dir_updates = []

        for search_dir,reciprocal, grad in zip(self.search_dirs,self.reciprocals,grads):
            first_dir_updates.append((search_dir,-grad))
            first_dir_updates.append((reciprocal,-grad))
        first_dir_updates.append((self.train_error,calc_error))

        initial_run = theano.function(
            inputs=[],
            outputs=[],
            updates=first_dir_updates,
            givens={
                data_in: train_in,
                data_out: train_out
            },
        )
        #set initial conditions
        initial_run()

        # set up validation if necessary
        if(val_in is not None):
            validation_func = theano.function(
                inputs=[],
                outputs=calc_error,
                givens={
                    data_in: val_in,
                    data_out: val_out
                }
            )

        train_error = theano.function(
            inputs=[],
            outputs=calc_error,
            givens={
                data_in: train_in,
                data_out: train_out
            }
        )

        trainSCG = self.train_func
        

        performance_goal = self.params["goal"]
        max_fail = self.params["max_fail"]
        max_epochs = self.params["max_epochs"]
        verbose = self.params["print_interval"]
        report_interval = self.params["disp_interval"]
        num_batches = n_samples / batch_size
        epoch = 1
        done = False
        success = self.success
        
        if(val_in is None):
            while epoch <= max_epochs and not done:
                for iteration in xrange(0, num_batches):
                    #err = trainSCG(iteration,epoch)
                    err = trainSCG(iteration,epoch)
                if(success.get_value() == True):
                    if(verbose and epoch % report_interval):
                        print "epoch %d, train error %f" % (epoch,err)
                    done = err <= performance_goal
                    epoch+=1
        else:
            cur_fails = 0
            last_val_err = float("infinity")
            while epoch <= max_epochs and not done:
                for iteration in xrange(0, num_batches):
                    err = trainSCG(iteration,epoch)
                if(success.get_value() == True):
                    val_err = validation_func()
                    done = err <= performance_goal
                    if(val_err > last_val_err):
                        cur_fails +=1
                        if cur_fails == max_fail:
                            done = True
                    last_val_err = val_err
                    if(verbose and epoch % report_interval):
                        print "epoch %d, train error %f, validation error %f" % (epoch,err,val_err)
                    epoch+=1

        self.predict = theano.function(
            inputs = [data_in],
            outputs = ol.output
        )

        self.prediction_error = theano.function(
            inputs = [data_in,data_out],
            outputs = calc_error
        )
        
        
        if(verbose):
            print "Done training at epoch %d" % epoch
            print "Training MSE: %f" % train_error()
            if (val_in is not None):
                print "Validation MSE: %f" % validation_func()
            if(test_in is not None):
                test_error = theano.function(
                    inputs=[],
                    outputs=calc_error,
                    givens={
                        data_in: test_in,
                        data_out: test_out
                    }
                )
                print "Testing MSE: %f" % test_error()