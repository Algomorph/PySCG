'''
Created on Jul 4, 2013

@author: Gregory Kramida
'''
import theano
import theano.tensor as T
import numpy as np
import transfer_gradients as tg
import data_prep as dp
from trainscg_params import TrainSCGParams
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

    @type num_input: int
    @param num_input: dimensionality of input

    @type num_output: int
    @param num_output: number of hidden units

    @type link_function: function
    @defualt: np.tanh
    @param link_function: used to make the hidden layer's
        output a continous function, accepts a numpy array and returns
        a numpy array of equal dimensions
    """
    def __init__(self, num_input, num_output,
                 weights=None, bias=None, random_gen = None,
                 link_function = np.tanh, train_params = None):
        self.link_function = link_function
        #initialize weights if initial weights aren't given
        if weights is None:
            if random_gen is not None:
                np.random.set_state(random_gen)
            bound = np.sqrt(0.6 /  num_input + num_output)
            self.weights = np.asarray(np.random.uniform(
                                low=-bound,high=bound,
                                size= (num_input,num_output)),
                                dtype=THEANO_FLOAT_X)
        elif type(weights) is int:
            self.weights = np.zeros((num_input, num_output),dtype=THEANO_FLOAT_X)
            self.weights[:] = weights
        elif type(weights) is np.ndarray:
            self.weights = weights
        else:
            raise ValueError('Bad weight type: %s.' % type(weights))

        if train_params is None:
            #create default parameter set
            train_params = TrainSCGParams();
        self.train_params = train_params

        #initialize bias vector if it's not given
        if bias is None:
            self.bias = np.zeros((num_output,), dtype=THEANO_FLOAT_X)
        elif type(bias) is np.ndarray:
            self.bias = bias
        else:
            raise ValueError('Bad bias type: %s.' % type(bias))
        if link_function is not None:
            self.compute_output = self.transfer_output
        else:
            self.compute_output = self.linear_output

        self.output = None

    def activation(self,x):
        return x.dot(self.weights) + self.bias

    #function to compute linear output (activation)
    #produces output of size (batch_size,num_output),
    def linear_output(self,x):
        o = self.activation(x)
        self.output = o
        return o

    def get_flat_weights(self):
        '''
        Produces a flat (1-D) vector of all the layer's weights and biases
        '''
        return np.append(self.weights.reshape(-1),self.bias)

    def __str__(self):
        st = "Neural Network Layer\nWeights:\n"
        st += str(self.weights) + "\nBias:\n" + str(self.bias) + "\nTransfer function:\n" + str(self.link_function)
        return st

    #differentiable output
    def transfer_output(self,x):
        o = self.activation(x)
        self.output = self.link_function(o)
        return self.output


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

    The hidden laytostringer would produce Wx + b, where
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
        link_function_hidden = np.tanh,
        link_function_output = None):
        if type(hidden_dimensions) is int:
            self.hidden_dimensions = [hidden_dimensions]
        else:
            self.hidden_dimensions = hidden_dimensions
        self.link_function_hidden = link_function_hidden
        self.link_function_output = link_function_output

    def get_flat_weights(self):
        return self.__get_flat_weights(self.layers)

    def __get_flat_weights(self,layers):
        w_set = [layer.get_flat_weights() for layer in layers]
        return np.concatenate(w_set)


    def predict(self,x):
        x = dp.map_to_ranges(x,self.x_ranges,self.x_mins)
        return dp.unmap_y(self.__predict(x,self.layers),self.y_ranges)

    def __predict(self,x,layers):
        layer_output = x
        for layer in layers:
            layer_output = layer.compute_output(layer_output)
        return layer_output

    def get_mse(self,output,y):
        return np.mean((output - y)**2)

    def compute_gradients(self, x, y, layers):
        #output layer
        ol = layers[-1]
        #ow = ol.get_flat_weights()
        #output transfer func derivative
        link_deriv = tg.grad_dict[ol.link_function]
        #output layer gradients
        output = ol.output
        #derivative of error with respect to activation *
        #derivative of activation with respect to the net input input
        deltas = 2 * (output-y) * link_deriv(output)
        next_layer = layers[-2]
        grads = []
        #output layer gradients
        #derivative of net input with respect to the weight is just the input
        #
        grad_weights = ol.gradient = next_layer.output.T.dot(deltas) / len(y)
        grad_bias = np.mean(deltas,axis=0)

        grads = [grad_bias.flatten(),grad_weights.flatten()]

        output = next_layer.output
        layer = next_layer
        prev_layer = ol

        #traverse layers backwards
        for i_layer in range(len(layers)-3,-1,-1):
            next_layer = layers[i_layer]
            next_output = next_layer.output
            link_deriv = tg.grad_dict[layer.link_function]
            deltas = deltas.dot(prev_layer.weights.T) * link_deriv(output)
            grad_weights = next_output.T.dot(deltas) / len(next_output)
            grad_bias = np.mean(deltas,axis=0)
            grads.append(grad_bias.flatten())
            grads.append(grad_weights.flatten())
            prev_layer = layer
            layer = next_layer
            output = next_output
        #the very first layer
        link_deriv = tg.grad_dict[layer.link_function]
        deltas = deltas.dot(prev_layer.weights.T) * link_deriv(output)
        grad_weights = x.T.dot(deltas) / len(x)
        grad_bias = np.mean(deltas,axis=0)
        grads.append(grad_bias.flatten())
        grads.append(grad_weights.flatten())
        return np.hstack(grads[::-1])/ len(y)

    def __init_data(self,x, y,split_sizes,randomize):
        return dp.prep_data_np(x, y,split_sizes,randomize)


    def __create_node_set(self, LayerType, num_features, num_output, weightsFunc = None):
        prev_dim = num_features
        layers = []
        num_weights = 0
        for i_h_layer in range(0,len(self.hidden_dimensions)):
            n_hidden_nodes = self.hidden_dimensions[i_h_layer]
            weights = None
            #weights = np.ones((prev_dim,n_hidden_nodes)) - 0.5
            bias = None
            if weightsFunc is not None:
                weights,bias = weightsFunc(i_h_layer)
            #acutal hidden layer
            hidden_layer = LayerType(num_input=prev_dim,
                                    num_output=n_hidden_nodes,
                                    link_function=self.link_function_hidden,
                                    weights=weights,
                                    bias=bias)

            layers.append(hidden_layer)
            num_weights += (prev_dim+1)*n_hidden_nodes
            prev_out = hidden_layer.output
            prev_dim = n_hidden_nodes


        weights = None
        #weights = np.ones((prev_dim,num_output)) - 0.5
        bias = None
        if weightsFunc is not None:
            weights,bias = weightsFunc(len(self.hidden_dimensions))
        output_layer = LayerType(
            num_input=prev_dim,
            num_output=num_output,
            link_function=self.link_function_output,
            weights=weights,
            bias=bias)
        num_weights += (prev_dim+1)*num_output
        layers.append(output_layer)
        return layers,num_weights

    def adjust_weights(self, layers, adjustments):
        at = 0
        #traverse layers backwards
        for layer in layers:
            to = at + layer.weights.size
            chunk = adjustments[at:to]
            at = to
            layer.weights += chunk.reshape(layer.weights.shape)
            to = at + layer.bias.size
            chunk = adjustments[at:to]
            at = to
            layer.bias += chunk.reshape(layer.bias.shape)

    def __build_architecture(self, num_features, num_output):
        #create layers
        #this set is only used for actual __prediction (it has the true weights)
        self.layers,self.num_params = \
            self.__create_node_set(Layer,num_features,num_output)
        def weight_func(i):
            layer = self.layers[i]
            weights = np.copy(layer.weights)
            bias = np.copy(layer.bias)
            return weights, bias
        self.hes_approx_layers = \
            self.__create_node_set(Layer,num_features,num_output,weight_func)[0]
        self.try_layers = \
            self.__create_node_set(Layer,num_features,num_output,weight_func)[0]



    default_params = {
                'max_epochs': 0,
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

    def copy_weights(self, layers_a, layers_b):
        pass

    def train_iteration(self, x, y):
        #print "======================"
        p = self.search_dir
        r = self.reciprocals
        lambda_ = self.lambda_
        lambda_bar = self.lambda_bar
        pSqNorm = self.pSqNorm
        delta = self.delta
        k = self.k
        error = self.error
        #Find 2nd-order information to approximate
        #p.T.dot(H).dot(p)
        if(self.success):
            pSqNorm = p.T.dot(p)
            #print pSqNorm
            self.pSqNorm = pSqNorm
            pnorm = np.sqrt(pSqNorm)
            sigma = self.sigma / pnorm
            self.adjust_weights(self.hes_approx_layers, sigma*p)
            #needed to calculate grads
            self.__predict(x, self.hes_approx_layers)
            hes_grads = self.compute_gradients(x, y, self.hes_approx_layers)
            #s approxiates H.dot(p)
            s = (hes_grads + r) / sigma
            #delta is an approximation to p.T.dot(H).dot(p),
            #where H is the hessian and is never calculated explicitly
            delta = p.T.dot(s)

        delta = delta + (lambda_ - lambda_bar)*pSqNorm

        #if delta <= 0, make Hessian positive definite
        if (delta <= 0):
            lambda_bar = 2*(lambda_ - delta/pSqNorm)
            delta = -delta + lambda_*pSqNorm
            lambda_ = lambda_bar
        self.delta = delta
        #compute step size
        mu = p.T.dot(r)
        alpha = mu / delta
        self.alpha = alpha
        self.adjust_weights(self.try_layers, alpha*p)
        output = self.__predict(x, self.try_layers)
        new_error = self.get_mse(output,y) / len(y)
        #comparison parameter
        gdelta = 2*delta*(error - new_error)/(mu**2)

        if gdelta >= 0:
            for layer, try_layer, hes_layer in zip(self.layers, self.try_layers, self.hes_approx_layers):
                #apply new weights
                layer.weights = np.copy(try_layer.weights)
                hes_layer.weights = np.copy(try_layer.weights)
                layer.bias = np.copy(try_layer.bias)
                hes_layer.bias = np.copy(try_layer.bias)

            r_new = -self.compute_gradients(x,y,self.try_layers)
            self.lambda_bar = lambda_bar = 0
            self.success = True
            self.error = new_error
            if(k >= self.num_params):
                self.search_dir = np.copy(r_new)
                self.k = 1
                self.lambda_ = self.initial_lambda
            else:
                beta = (r_new.T.dot(r_new) - r_new.T.dot(r)) / mu
                self.search_dir = r_new + beta * p
            self.reciprocals = r_new

            if(gdelta > 0.75):
                self.lambda_ = lambda_/4
        else:
            for layer, try_layer in zip(self.layers, self.try_layers):
                #restore original weights
                try_layer.weights = np.copy(layer.weights)
                try_layer.bias = np.copy(layer.bias)
            self.lambda_bar = lambda_
            self.success = False

        if gdelta < 0.25:
            self.lambda_ += delta * (1 - gdelta) / pSqNorm

        self.k+=1


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

        if params is None:
            self.params = NeuralNet.default_params
        else:
            for param in NeuralNet.default_params:
                if(param not in params):
                    params[param] = NeuralNet.default_params[param]
            self.params = params

        #prepare the data
        (x,y,
         val_in,val_out,
         test_in,test_out,
         do_validation,do_test) =\
        self.__init_data(x, y, split_sizes, randomize)
        self.y_ranges, self.y_mins = dp.get_ranges(y)
        self.x_ranges, self.x_mins = dp.get_ranges(x)
        x_c = dp.map_to_ranges(x,self.x_ranges,self.x_mins)
        y_c = dp.map_to_ranges(y,self.y_ranges,self.y_mins)

        #input size is number of features
        num_features = x.shape[1]
        num_output = y.shape[1]

        #default batch size is the whole train dataset

        #self.ranges_y = d
        n_samples = len(x)
        if(batch_size is None):
            batch_size = n_samples

        self.__build_architecture(num_features,num_output)

        self.sigma = self.params["sigma"]
        self.initial_lambda = self.lambda_ = self.params["lambda"]
        self.lambda_bar = 0.0
        self.max_epochs = self.params["max_epochs"]
        self.goal = self.params["goal"]
        output = self.__predict(x_c, self.layers)
        self.error = self.get_mse(output, y_c) / len(y_c)
        self.reciprocals = -self.compute_gradients(x_c,y_c,self.layers)
        self.search_dir = np.copy(self.reciprocals)
        self.pSqNorm = 0.0
        self.success = True
        self.delta = 0.0
        self.k = self.num_params
        i_it = 0
        while(i_it < self.max_epochs and self.error > self.goal):
            self.train_iteration(x_c,y_c)
            print self.error
            i_it +=1




