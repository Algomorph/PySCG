'''
Created on Jul 4, 2013

@author: Gregory Kramida
'''
import theano
import theano.tensor as T
import numpy as np
import data_prep as dp
import time
from trainscg_params import TrainSCGParams
from theano.ifelse import ifelse

THEANO_FLOAT_X = np.dtype(np.double).name
if(hasattr(theano.config,'floatX')):
    THEANO_FLOAT_X = getattr(theano.config,'floatX')

def theano_f(x):
    if THEANO_FLOAT_X == 'float64':
        return np.float64(x)
    else:
        return np.float32(x) 

class Layer(object):
    """
    Layer of a neural network: units are fully-connected and use the specified
    link function. Weight matrix W is of shape (n_input,n_output)
    and the bias vector b is of shape (n_output).

    NOTE : The nonlinearity used here is tanh
    Hidden unit activation is given by: tanh(dot(input,W) + b),

    @type weights: int, theano tensor, numpy array, or None
    @param weights: initial weights for the hidden layer.

    @type random_gen: numpy.random.RandomState
    @param random_gen: a random state used by the random generator
        to initialize weights

    @type data_in: theano.tensor.matrix
    @param data_in: a symbolic tensor of shape (batch_size, n_input)

    @type n_input: int
    @param n_input: dimensionality of input

    @type n_output: int
    @param n_output: number of hidden units

    @type link_function: theano.Op or function
    @defualt: T.tanh
    @param link_function: used to make the hidden layer's
        output a continous function
    """
    def __init__(self, data_in, n_input, n_output,
                 weights=None, bias=None, random_gen = None,
                 link_function = T.tanh, name = None):
        self.n_input = n_input
        self.n_output = n_output
        self.name = name
        if name is None:
            name = ""
            self.name = ""
        else:
            name += " "
        
        #initialize weights if initial weights aren't given
        #use Nguyen-Widrow initialization
        if weights is None:
            if random_gen is not None:
                np.random.set_state(random_gen)
            self.reset_weights_nw()
            bias = self.bias#avoid reinitialization
        elif weights is 0:
            self.weights = theano.shared(np.zeros((n_input, n_output),dtype=THEANO_FLOAT_X),
                name="layer_weights")
        elif type(weights) is np.ndarray:
            self.weights = theano.shared(weights.astype(THEANO_FLOAT_X), name=name + "Layer Weights",borrow=True)
        else:
            #assume theano tensor
            self.weights = weights
            self.weights.name = name + "Layer Weights"

        #initialize bias vector if it's not given
        if bias is None or bias is 0:
            self.bias = theano.shared(np.zeros((n_output,), dtype=THEANO_FLOAT_X),name=name + "Layer Bias")
        elif type(bias) is np.ndarray:
            self.bias = theano.shared(bias.astype(THEANO_FLOAT_X), name=name + "Layer Bias",borrow=True)
        else:
            #assume theano tensor
            self.bias = bias
            self.bias.name = name + "Layer Bias"

        #function to compute linear output (activation)
        #produces output of size (batch_size,n_output),
        linear_output = T.dot(data_in,self.weights) + self.bias

        #differentiable output
        if link_function is None:
            self.output = linear_output
        else:
            self.output = link_function(linear_output)

    def reset_weights_nw(self):
        n_input = self.n_input
        n_output = self.n_output
        name = self.name
        bound = 0.5
        w_arr = np.random.uniform(low=-bound,high=bound,size= (n_input,n_output))
        b = np.random.uniform(low=-bound,high=bound, size = (n_output,))
        beta = 0.7 * float(n_output)**(1.0/n_input)
        norms = np.sqrt((w_arr * w_arr).sum(axis = 0) + b*b)
        w_arr = (beta * w_arr) / norms
        b = (beta * b) / norms
        self.weights = theano.shared(w_arr.astype(dtype=THEANO_FLOAT_X), name=name + " Layer Weights",borrow=True)
        self.bias = theano.shared(b.astype(dtype=THEANO_FLOAT_X), name=name + " Layer Bias",borrow=True)


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
    def __init__(self, hidden_dimensions, x, y,
                 link_function_hidden = T.tanh, link_function_output = None, 
                 train_params = None, use_matlab_lambda_updates = False):
        if type(hidden_dimensions) is int:
            #integer provided, assume only one hiddel layer
            self.hidden_dimensions = [hidden_dimensions]
        else:
            self.hidden_dimensions = hidden_dimensions
        self.link_function_hidden = link_function_hidden
        self.link_function_output = link_function_output
        if type(x) is int:
            #integer provided, assume it's feature count
            n_features = x
        elif type(x) is np.ndarray:
            #numpy array provided, assume it's actual training input
            n_features = x.shape[1]
        else:
            raise ValueError("Expected %s or %s for argument x, got %s" % (str(int),str(np.ndarray),str(type(x))))
        if type(y) is int:
            #integer provided, assume it's output node count
            n_output = y
        elif type(y) is np.ndarray:
            #numpy array provided, assume it's actual training output
            n_output = y.shape[1]
        else:
            raise ValueError("Expected %s or %s for argument y, got %s" % (str(int),str(np.ndarray),str(type(y))))
        self.n_features = n_features
        self.n_output = n_output

        #discard given x & y for now, create symbolic vars instead
        self.__x = x = T.matrix('x')
        self.__y = y = T.matrix('y')

        if train_params is None:
            #create default parameter set
            train_params = TrainSCGParams();
        self.train_params = train_params

        self.__build_architecture(n_features,n_output,x,y,use_matlab_lambda_updates)

    def __init_data(self,x, y,split_sizes,randomize):
        return dp.prep_data(x, y,split_sizes,randomize)


    def __create_node_set(self, n_features, n_output, data_in, note_set_name, weightsFunc = None,):
        prev_out = data_in
        prev_dim = n_features
        layers = []
        n_weights = 0
        weights_list = []
        state = None
        for i_h_layer in range(0,len(self.hidden_dimensions)):
            n_hidden_nodes = self.hidden_dimensions[i_h_layer]
            weights = None
            #weights = np.ones((prev_dim,n_hidden_nodes)) - 0.5
            bias = None
            if weightsFunc is not None:
                weights,bias,state = weightsFunc(i_h_layer,state)
            #acutal hidden layer
            hidden_layer = Layer(data_in=prev_out,
                                    n_input=prev_dim,
                                    n_output=n_hidden_nodes,
                                    link_function=self.link_function_hidden,
                                    weights=weights,
                                    bias=bias,
                                    name=note_set_name + " Hidden Layer")
            weights_list.append(hidden_layer.weights)
            weights_list.append(hidden_layer.bias)
            layers.append(hidden_layer)
            n_weights += (prev_dim+1)*n_hidden_nodes
            prev_out = hidden_layer.output
            prev_dim = n_hidden_nodes

        weights = None
        #weights = np.ones((prev_dim,n_output)) - 0.5
        bias = None
        if weightsFunc is not None:
            weights,bias,state = weightsFunc(len(self.hidden_dimensions),state)
        output_layer = Layer(
            data_in=prev_out,
            n_input=prev_dim,
            n_output=n_output,
            link_function=self.link_function_output,
            weights=weights,
            bias=bias,
            name=note_set_name + " Output Layer")
        weights_list.append(output_layer.weights)
        weights_list.append(output_layer.bias)
        layers.append(output_layer)
        n_weights += (prev_dim+1)*n_output

        #concatenate weights into one huge vector
        flat_weights = T.concatenate([T.flatten(item) for item in weights_list])
        flat_weights.name = "Network " + note_set_name + " Weights"
        #compute MSE
        y = self.__y
        errors = y - output_layer.output
        mse = T.mean(T.sqr(errors))
        normalized_mse = mse / 2.0
        normalized_mse.name = note_set_name + " MSE"
        grads = T.concatenate([T.flatten(item) for item in T.grad(normalized_mse, weights_list)])
        grads.name = note_set_name + " Gradients"
        return layers,grads,normalized_mse,weights_list, n_weights, flat_weights

    def __generate_adjusted_weights(self,i,state,factor):
        p = self.search_direction
        if(state is None):
            state = 0
        at = state
        layer = self.layers[i]
        to = at + layer.n_input * layer.n_output
        #adjust by that specific chunk of search direction multiplied by sigma
        weights = layer.weights + factor * p[at:to].reshape((layer.n_input,layer.n_output))
        at = to
        to = at + layer.n_output
        bias = layer.bias + factor * p[at:to]
        return weights, bias, to

    def __build_architecture(self, n_features, n_output, x, y, matlab_lambda_updates = False):
        #x & y are symbolic
        #create layers
        #this set is only used for actual prediction (it has the true weights)
        self.layers, self.grads, self.normalized_mse, weights, n_weights, self.__flat_weights = \
            self.__create_node_set(n_features,n_output,x,"Primary")


        self.reciprocals = r = theano.shared(np.zeros(n_weights,dtype = THEANO_FLOAT_X),name="Reciprocals",borrow=True)
        self.search_direction = p = theano.shared(np.zeros(n_weights,dtype = THEANO_FLOAT_X),name="Search Direction (vector \"p\")",borrow=True)

        #to retain error from previous iteration
        self.__train_error = error = theano.shared(theano_f(0.0),name="Train Error")
        success = theano.shared(np.int32(1),name="Reduction Success")
        lambda_ = theano.shared(theano_f(self.train_params.init_lambda),name="Lambda")

        #successful reductions since last restart
        k = theano.shared(np.int32(n_weights),name="k (# of iteration since last restart)")
        self.k = k

        #scale factor normalizer
        lambda_bar = theano.shared(theano_f(0.0),name="Lambda Bar")

        #============(Step 2)=====================#
        #2nd-order information
        init_sigma = self.train_params.sigma

        mag_p = p.T.dot(p)
        pnorm = T.sqrt(mag_p)
        sigma = theano_f(init_sigma) / pnorm
        sigma.name = "Sigma"

        hes_weight_func = lambda i, state: self.__generate_adjusted_weights(i,state,sigma)

        #create mirror node set for Hessian approximation
        self.__hes_approx_layers, hes_grads = \
            self.__create_node_set(n_features,n_output,x,"Hes-Approx",hes_weight_func)[0:2]
        self.hes_grads = hes_grads

        #for storing / re-using old delta in case a reduction wasn't possible
        delta_cache = theano.shared(theano_f(0.0),name="Old Delta")

        #estimate of products of hessian components with the search direction
        #s ~~ Hp
        #this delta indicator is approx. (p^T)Hp, wich has to be positive for
        #positive-definite H
        s = (hes_grads + r) / sigma
        s.name = "s (~~Hp)"
        calc_delta = p.T.dot(s)
        calc_delta.name = "New Delta"
        

        #scaling to get proper indefiniteness of H
        #calculate new delta if last reduction was successfull, use old one otherwise
        #============(Step 3)=====================#
        delta = ifelse(success, calc_delta, delta_cache) + (lambda_ - lambda_bar)*mag_p
        delta.name = "Scaled Delta"

        #============(Step 4)=====================#
        #if delta < 0, we need to make the Hessian positive definite
        #recompute the scaling factor normalizer
        posdef_scale_factor = 2*(lambda_- (delta / mag_p))
        posdef_scale_factor.name = "Lambda After pos-def Updates"
        posdef_updates = ifelse(T.gt(delta,0),
            [delta,lambda_],
            [-delta + lambda_*mag_p, posdef_scale_factor])
        #note - the delta is still calculated with the old lambda
        delta = posdef_updates[0]

        delta.name = "Pos-def Delta"
        #lambda and lambda bar either stay the same or both change to the same value
        calc_lambda = posdef_updates[1]
        calc_lambda.name = "Pos-def Lambda"

        #============(Step 5)=====================#
        #build step-size function
        mu = p.T.dot(r)
        mu.name = "Mu"
        alpha = mu / delta
        alpha.name = "Alpha (Step Size)"
        self.alpha = alpha
        
        #============(Step 6)=====================#
        #create mirror node set for trying a new step
        try_weight_func = lambda i, state: self.__generate_adjusted_weights(i,state,alpha)
        self.__try_layers, try_grads, try_error, try_weights = \
            self.__create_node_set(n_features,n_output,x,"Try",try_weight_func)[0:4]

        gdelta = 2*delta*(error - try_error)/(mu**2)
        #comparison parameters
        gdelta.name = "Comparison Parameter"

        #====================Step 7 & 8===================#
        #successful updates
        #updates to the weights
        r_new = -try_grads
        r_new.name = "New Reciprocals"

        #search direction updates (step 7a)
        mag_r = try_grads.T.dot(try_grads)
        beta = (mag_r + try_grads.T.dot(r)) / mu
        #beta is the magnitude we need to multiply the previous search_direction by to cancel it out of the new verctor
        beta.name = "Beta"
        p_new = r_new + beta * p

        self.__rnorm = ifelse(T.lt(gdelta,theano_f(0.0)),theano_f(1.0),T.sqrt(mag_r))
        self.__rnorm.name = "New Reciprocal Norm"

        if matlab_lambda_updates:
            grow_lambda = 4 * calc_lambda
            shrink_lambda = 0.5 * calc_lambda
        else:
            grow_lambda = calc_lambda + delta*(1 - gdelta)/mag_p
            shrink_lambda = 0.25 * calc_lambda

        grow_lambda_conditional = ifelse(T.lt(gdelta,theano_f(0.25)),grow_lambda,calc_lambda)
        reduction_lambda = ifelse(
            T.lt(gdelta,theano_f(0.75)),
            grow_lambda_conditional,
            shrink_lambda)
        
        #if the epoch reaches # of weights, restart the algorithm (Step 7a)
        direction_update_ops = ifelse(T.eq(k,n_weights),
            [r_new,np.int32(1),theano_f(self.train_params.init_lambda)],
            [p_new,k+1,reduction_lambda])
        
        #updates to weights, reciprocals, search direction (p), k, lambda, lambda_bar, delta_cache, error, success
        learning_update_ops = ifelse(T.lt(gdelta,theano_f(0.0)),
            weights + [r,p] + [k, grow_lambda, calc_lambda, delta, error, np.int32(0)],
            try_weights + [r_new] + direction_update_ops + [T.zeros_like(lambda_bar), delta, try_error, np.int32(1)])

        vals_to_update = weights + [r,p] + [k, lambda_, lambda_bar, delta_cache, error, success]
        self.success = success

        learning_updates = []
        for val, update in zip (vals_to_update, learning_update_ops):
            learning_updates.append((val,update))
        self.learning_updates = learning_updates

    def get_flat_weights(self):
        return self.__flat_weights.eval()

    def train_iteration(self,x_c, y_c):
        if self.__train_single is None:
            self.__train_single = theano.function(
                inputs=[self.__x, self.__y],
                outputs=self.__rnorm,
                updates=self.learning_updates,
            )
        return self.__train_single(x_c,y_c)

    def build_train_function(self, x_c, y_c):
        x = self.__x
        y = self.__y
        return theano.function(
            inputs=[],
            outputs=theano.Out(self.__rnorm,borrow=True),
            updates=self.learning_updates,
            givens={
                x: x_c,
                y: y_c
            }
        )


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
    def train(self,x,y,randomize=True,
         split_sizes=None,batch_size=None):
        
        #prepare the data
        (x_c,y_c,x_v,y_v,x_t,y_t,
         do_val,do_test,y_ranges,y_mins,
         x_ranges,x_mins) =\
        self.__init_data(x,y, split_sizes, randomize)
        x = self.__x
        y = self.__y
        self.x_ranges = x_ranges
        self.y_ranges = y_ranges
        self.x_mins = x_mins
        self.y_mins = y_mins

        self.__train_single = None

        first_dir_updates=[
            (self.reciprocals,-self.grads),
            (self.search_direction,-self.grads),
            (self.__train_error, self.normalized_mse)
        ]

        initial_run = theano.function(
            inputs=[],
            outputs=[],
            updates=first_dir_updates,
            givens={x: x_c,y: y_c},
        )
        #set initial conditions
        initial_run()

        # set up validation if necessary
        if(do_val):
            validation_func = theano.function(
                inputs=[],
                outputs=self.normalized_mse,
                givens={x: x_v,y: y_v}
            )

        train_error = theano.function(
            inputs=[],
            outputs=self.normalized_mse,
            givens={x: x_c, y: y_c}
        )
        self.train_error = train_error

        train = self.build_train_function(x_c,y_c)
        self.__train = train


        performance_goal = self.train_params.goal
        max_fail = self.train_params.max_fail
        max_epochs = self.train_params.epochs
        min_grad = self.train_params.min_grad
        epoch = 0
        done = False
        success = self.success
        t0 = time.time()
        if max_epochs > 0:
            if(not do_val):
                while not done:
                    grad_norm = train()
                    err = self.__train_error.get_value()
                    epoch+=1
                    print "epoch: %d, train error: %f, gradient norm: %f" % (epoch,err,grad_norm)
                    done = (err <= performance_goal or grad_norm < min_grad or epoch == max_epochs)
            else:
                cur_fails = 0
                last_val_err = float("infinity")
                while not done:
                    grad_norm = train()
                    err = self.__train_error.get_value()
                    val_err = validation_func()
                    epoch+=1
                    done = (err <= performance_goal or grad_norm < min_grad or epoch == max_epochs)
                    if(val_err > last_val_err):
                        cur_fails +=1
                        if cur_fails == max_fail:
                            done = True
                    last_val_err = val_err
                    if(verbose and epoch % report_interval):
                        print "epoch %d, train error %f, validation error %f" % (epoch,err,val_err)
        t1 = time.time()
        print 'Total training time: %f seconds.' % (t1-t0)
        self.last_training_time = t1-t0
        x_in = T.matrix("x_in")
        xrs = theano.shared(x_ranges.astype(THEANO_FLOAT_X), "Feature Ranges")
        yrs = theano.shared(y_ranges.astype(THEANO_FLOAT_X), "Feature Ranges")
        xms = theano.shared(x_mins.astype(THEANO_FLOAT_X), "Feature Ranges")
        self.predict = theano.function(
            inputs = [x_in],
            outputs = (self.layers[-1].output + 1) / (2 * yrs),
            givens = {x: ((x_in - xms) * 2 / xrs) - 1}
        )


        self.prediction_error = theano.function(
            inputs = [x,y],
            outputs = self.normalized_mse
        )

        print "Done training at epoch %d" % epoch
        print "Training MSE: %f" % train_error()
        if do_val:
            print "Validation MSE: %f" % validation_func()
        if do_test:
            test_error = theano.function(
                inputs=[],
                outputs=self.normalized_mse,
                givens={
                    data_in: x_t,
                    data_out: y_t
                }
            )
            print "Testing MSE: %f" % test_error()