'''
Created on Oct 27, 2013

@author: algomorph
'''
import unittest
from neural_net import *
import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse
from theano import ProfileMode
from theano.gof import toolbox
import theano.gof as gof


class Test(unittest.TestCase):
    def setUp(self):
        hw = np.array([[-0.729649214899581,1.734387460896752],[-1.655025343120032,-0.199628277736654]])
        hb = np.array([2.203101900593036, -2.180048963939269])
        ow = np.array([[ -0.851309568790796],[0.295793612123733]])
        ob = np.array([0.74613148705201])
        #self.initial_weights = (None,None, 0, None)
        self.initial_weights=(hw,hb,ow,ob)
        pass


    def tearDown(self):
        pass

    def testSCG(self):
        train_in = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]],dtype='float64')
        train_out = np.array([[0.,1.,1.,0.]],dtype='float64').T
        num_features = 2
        num_hidden = 2
        num_output = 1
        nn = NeuralNet(num_hidden)
        #nn.train((train_in,train_out))

    
    def testSCG_Manual(self):
        train_in = theano.shared(np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]],dtype='float64'),name="train_in")
        train_out = theano.shared(np.array([[0.,1.,1.,0.]],dtype='float64').T,name="train_out")
        data_in = T.matrix('data_in')
        data_out = T.matrix('data_out')
        batch_size = 4
        batch_index = T.lscalar()
        iteration = T.lscalar()

        num_features = 2
        num_hidden = 2
        num_output = 1

        num_params = (num_features+1) * num_hidden + (num_hidden+1) * num_output

        (hw,hb,ow,ob) = self.initial_weights

        ol_link_func = None

        hl = SCGLayer(data_in=data_in,
                   num_input=num_features,
                   num_output=num_hidden,
                   weights = hw, bias = hb)
        ol = SCGLayer(data_in=hl.output,
                   num_input=num_hidden,
                   num_output=num_output,
                   #weights=0,
                   link_function=ol_link_func,
                   weights = ow, bias = ob)


        sigma = 5.0e-5
        lambda_ = theano.shared(5.0e-7,name="lambda")
        lambda_bar = theano.shared(0.0,name="lambda_bar")
        success = theano.shared(np.int8(1),name="reduction_success")

        error = T.mean((data_out - ol.output)**2)
        params = ol.params + hl.params
        reciprocals = ol.reciprocals + hl.reciprocals
        grads = T.grad(error,params)
        search_dirs = ol.search + hl.search

        first_dir_updates = []
        for search_dir,reciprocal, grad in zip(search_dirs,reciprocals,grads):
            first_dir_updates.append((search_dir,-grad))
            first_dir_updates.append((reciprocal,-grad))

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

        pL2norm = T.sum(search_dirs[0]**2)
        for search_dir in search_dirs[1:]:
            pL2norm = pL2norm + T.sum(T.sqr(search_dir))


        pnorm = T.sqrt(pL2norm)
        calc_sigma = sigma / pnorm

        #create mirror node set for Hessian approximation
        alt_hl = Layer(data_in=data_in,
                       num_input=2,
                       num_output=2,
                       weights=hl.weights + calc_sigma * hl.w_search_dir,
                       bias =hl.bias + calc_sigma * hl.b_search_dir)

        alt_ol = Layer(data_in=alt_hl.output,
                       num_input=2,
                       num_output=1,
                       weights=ol.weights + calc_sigma * ol.w_search_dir,
                       bias =ol.bias + calc_sigma * ol.b_search_dir,
                       link_function=ol_link_func,)

        alt_error = T.mean((data_out - alt_ol.output)**2)


        alt_params = alt_ol.params + alt_hl.params
        alt_grads = T.grad(alt_error,alt_params)
        delta_cache = theano.shared(0.0,name="delta_cache")

        #estimate of products of hessian components with the search direction
        #s = ~Hp
        #this delta indicator is approx. (p^T)Hp, wich has to be positive for
        #positive-definite H
        calc_delta = 0

        #note that the reciprocals are negative gradients and are computed in the previous iteration
        for reciprocal, alt_grad, search_dir in zip(reciprocals,alt_grads,search_dirs):
            #the numerator is subtracting the gradients from the modified gradients
            hessian_comp = (alt_grad + reciprocal) 
            #hessian_prod.append(hessian_comp)
            #equivalent to dot product of p with s
            calc_delta += T.sum(hessian_comp*search_dir)
        calc_delta = calc_delta / calc_sigma

        calc_delta.name = "calc_delta"

        #scaling to get effects of a positive-definite H
        delta = ifelse(success, calc_delta, delta_cache) + (lambda_ - lambda_bar)*pL2norm
        delta.name = "delta"

        posdef_lambda = 2*(lambda_- (delta / pL2norm))
        posdef_lambda.name = "posdef lambda"
        posdef_updates = ifelse(T.gt(delta,0),
            [delta,lambda_, lambda_bar],
            [-delta + lambda_*pL2norm, posdef_lambda, posdef_lambda])



        delta = posdef_updates[0]
        calc_lambda = posdef_updates[1]
        calc_lambda_bar = posdef_updates[2]

        #build stepsize function
        mu =  T.sum(search_dirs[0]*reciprocals[0])
        for search_dir,reciprocal in zip(search_dirs[1:],reciprocals[1:]):
            mu += T.sum(search_dir * reciprocal)

        step_size = mu / delta

        #create mirror node set for stepsize testing
        try_hl = Layer(data_in=data_in,
                       num_input=2,
                       num_output=2,
                       weights=hl.weights + step_size * hl.w_search_dir,
                       bias =hl.bias + step_size * hl.b_search_dir)

        try_ol = Layer(data_in=try_hl.output,
                       num_input=2,
                       num_output=1,
                       weights=ol.weights + step_size * ol.w_search_dir,
                       bias =ol.bias + step_size * ol.b_search_dir,
                       link_function=ol_link_func,)

        try_error = T.mean((data_out - try_ol.output)**2)
        try_params = try_ol.params + try_hl.params
        try_grads = T.grad(try_error,try_params)

        #TODO: optimize, try_error becomes error for the next computation if a reduction is made,
        # don't recompute it
        com_param = 2*delta*(error - try_error)/(mu**2)

        #successful updates
        reduction_updates =[]
        for param in try_params:
            reduction_updates.append(param)
        #these will become new reciprocals
        reduction_updates += [-try_grad for try_grad in try_grads]

        #search direction updates
        new_reciprocal_norm = 0
        for try_grad in try_grads:
            new_reciprocal_norm += T.sum(T.sqr(try_grad))

        direction_update_ops = []
        beta = 0
        
        #beta is the magnitude we need to multiply the previous search_direction by to cancel it out of the new vector
        #first aggregate the dot product of the future reciprocal with the current reciprocal
        for try_grad, reciprocal in zip(try_grads, reciprocals):
            beta += T.sum(try_grad * reciprocal)
        #subtract from the 2-norm of the new reciprocal and divide the whole thing by mu
        beta = (new_reciprocal_norm + beta) / mu
        beta.name = "Beta"
        
        for try_grad, search_dir in zip(try_grads, search_dirs):
            direction_update_ops.append(-try_grad + beta*search_dir)

        direction_update_ops = ifelse(T.eq(iteration % num_params,0),[-try_grad for try_grad in try_grads],direction_update_ops)

        reduction_updates += direction_update_ops

        lambda_increase = 4*calc_lambda
        #lambda_increase = calc_lambda + delta*(1.0-com_param)/pL2norm

        reduction_lambda_update_op = ifelse(T.lt(com_param,0.75),
                    ifelse(T.lt(com_param,0.25),lambda_increase,calc_lambda)
                    ,calc_lambda/2)

        #updates to weights, reciprocals, search_dirs, lambda, lambda_bar, success, delta_cache,
        learning_update_ops = ifelse(T.lt(com_param,0.0),
            params + reciprocals + search_dirs + [lambda_increase, calc_lambda, 0, delta],
            reduction_updates +  [reduction_lambda_update_op, T.zeros_like(calc_lambda), 1, delta])

        vals_to_update = params + reciprocals + search_dirs + [lambda_, lambda_bar, success, delta_cache]

        learning_updates = []
        for val, update in zip (vals_to_update, learning_update_ops):
            learning_updates.append((val,update))

        trainSCG = theano.function(
            inputs=[batch_index,iteration],
            outputs=[error,delta,com_param,step_size,calc_delta,calc_sigma, mu],
            updates=learning_updates,
            givens={
                    data_in: train_in[batch_index * batch_size:(batch_index + 1) * batch_size],
                    data_out: train_out[batch_index * batch_size:(batch_index + 1) * batch_size]
                    },
            name='trainSCG'
        )

        #theano.printing.pydotprint(trainSCG,outfile="scg_train.png",var_with_name_simple=True)
        '''
        calc_delta_func = theano.function(
            inputs=[batch_index],
            outputs=delta,
            givens={
                    data_in: train_in[batch_index * batch_size:(batch_index + 1) * batch_size],
                    data_out: train_out[batch_index * batch_size:(batch_index + 1) * batch_size]
                    },
            name='calc_delta'
        )'''
        #theano.printing.pydotprint(calc_delta_func,outfile="scg_calc_delta.png",var_with_name_simple=True)


        print "=======Beginning SCG backpropagation===================="
        epoch = 1
        it = 0
        while(epoch < 100):
            err, delta, com_param, step_size, calc_delta,sig_k, mu = trainSCG(0,epoch)
            if(success.get_value() == True):
                epoch += 1
            if(it % 10 == 0):
                print "===epoch: %d === error: %f" % (epoch,err)
                '''
                print ("iteration %d\nerror %f,\ncalc_sigma: %f,\nlambda: %f,\nlambda_bar: %f,\ndelta: %f,"
                        +"\nmu: %f, \nstep_size: %f,\ncom_param: %f,\nlast reduction successful: %d") %\
                    (it, err, sig_k, lambda_.get_value(), lambda_bar.get_value(), delta, 
                        mu, step_size, com_param, success.get_value(),)
                print "Mod: %d" % (epoch % num_params)
                '''
            it+=1
                
        if(err > 0.003):
            '''
            print hl.weights.get_value();
            print "---"
            print hl.bias.get_value();
            print "---"
            print ol.weights.get_value();
            print "---"
            print ol.bias.get_value();
            '''
            np.save("bad_hidden_weights.npy",hl.weights.get_value())
            np.save("bad_output_weights.npy",ol.weights.get_value())
        self.assertAlmostEqual(err,0.0,3,"Not close enough to true result")
        print "======Passed SCG backpropagation with error %f========" % err



    def testSimpleNetwork(self):
        #column of ones at the end - for Biases
        train_in = theano.shared(np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]],dtype='float64'))
        train_out = theano.shared(np.array([[0.,1.,1.,0.]],dtype='float64').T)
        data_in = T.matrix('data_in')
        data_out = T.matrix('data_out')
        batch_size = 4

        (hw,hb,ow,ob) = self.initial_weights
        

        hl = Layer(data_in=data_in,
                         num_input=2,
                         num_output=2,
                         weights=hw,
                         bias = hb
                         )
        ol = Layer(data_in=hl.output,
                         num_input=2,
                         num_output=1,
                         weights=ow, 
                         bias=ob,
                         link_function=None)


        learning_rate = 0.5
        #weights & biases
        params = ol.params + hl.params
        error = T.mean((data_out - ol.output)**2)
        grads = T.grad(error,params)
        #add weight & bias updates
        learning_updates = []
        for param, grad in zip(params,grads):
            learning_updates.append((param, param - learning_rate*grad))

        batch_index = T.lscalar()

        trainingFunction = theano.function(
            inputs=[batch_index],
            outputs=ol.output,
            updates=learning_updates,
            givens={
                    data_in: train_in[batch_index * batch_size:(batch_index + 1) * batch_size],
                    data_out: train_out[batch_index * batch_size:(batch_index + 1) * batch_size]
                    },
            #mode='DebugMode'
        )

        testFunction = theano.function(
            inputs = [],
            outputs= error,
            givens={
                    data_in: train_in,
                    data_out: train_out
                    }
        )

        batch_count = 1

        print "=======Beginning basic backpropagation================"
        interval = 100
        for iter in xrange(0,10000):
            for b_ix in xrange(0,batch_count):
                trainingFunction(b_ix)
            if iter % interval == 1:
                err = testFunction()
                #print "hidden weights:"
                #print hl.weights.get_value()
                #print "output weights:"
                #print ol.weights.get_value()
                print "===epoch: %d === error: %f" % (iter,err)
            if iter > 1000:
                interval = 1000
        output = testFunction()


        self.assertAlmostEqual(output,0.0,3,"Not close enough to true result")
        print "======Passed basic backpropagation with error %f========" % output


    def testDataPrep(self):
        nn = NeuralNet(20)
        n_samples = 100
        n_features = 50
        frac_train = 0.7
        frac_val = 0.1
        frac_test = 0.2


        dataset1_in = np.ones((n_samples,n_features),dtype=np.float64)
        dataset1_out = np.ones((n_samples),dtype=np.float64)

        (train_in,train_out,
         val_in,val_out,
         test_in,test_out,
         do_validation,do_test)=\
        nn._NeuralNet__init_data((dataset1_in,dataset1_out),(frac_train,frac_val,frac_test),True)

        #check flags
        self.assertTrue(do_validation, "do_validation should be true, was false")
        self.assertTrue(do_test, "do_test should be true, was false")

        #check sizes
        train_internal = train_in.get_value()
        val_internal = val_in.get_value()
        test_internal = test_in.get_value()

        train_out_internal = train_out.get_value()
        val_out_internal = val_out.get_value()
        test_out_internal = test_out.get_value()

        exp_train_size = int(n_samples*frac_train)
        exp_val_size = int(n_samples*frac_val)
        exp_test_size = int(n_samples*frac_test)



        self.assertEqual(train_internal.shape[0], exp_train_size ,
                         "Wrong size for train set: expected %d, got %d"
                         % (train_internal.shape[0], exp_train_size))
        self.assertEqual(val_internal.shape[0], exp_val_size ,
                         "Wrong size for train set: expected %d, got %d"
                         % (val_internal.shape[0], exp_val_size))
        self.assertEqual(test_internal.shape[0], exp_test_size ,
                         "Wrong size for train set: expected %d, got %d"
                         % (test_internal.shape[0], exp_test_size))
        self.assertEqual(train_out_internal.shape[0], exp_train_size ,
                         "Wrong size for train set: expected %d, got %d"
                         % (train_out_internal.shape[0], exp_train_size))
        self.assertEqual(val_out_internal.shape[0], exp_val_size ,
                         "Wrong size for train set: expected %d, got %d"
                         % (val_out_internal.shape[0], exp_val_size))
        self.assertEqual(test_out_internal.shape[0], exp_test_size ,
                         "Wrong size for train set: expected %d, got %d"
                         % (test_out_internal.shape[0], exp_test_size))

        frac_train = .8
        frac_test = .2

        (train_in,train_out,
         val_in,val_out,
         test_in,test_out,
         do_validation,do_test)=\
        nn._NeuralNet__init_data((dataset1_in,dataset1_out),(frac_train,frac_test),True)

        #check flags
        self.assertFalse(do_validation, "do_validation should be false, was true")
        self.assertTrue(do_test, "do_test should be true, was false")

        #check sizes
        train_internal = train_in.get_value()
        test_internal = test_in.get_value()

        train_out_internal = train_out.get_value()
        test_out_internal = test_out.get_value()

        exp_train_size = int(n_samples*frac_train)
        exp_test_size = int(n_samples*frac_test)

        self.assertEqual(train_internal.shape[0], exp_train_size ,
                         "Wrong size for train set: expected %d, got %d"
                         % (train_internal.shape[0], exp_train_size))
        self.assertEqual(test_internal.shape[0], exp_test_size ,
                         "Wrong size for train set: expected %d, got %d"
                         % (test_internal.shape[0], exp_test_size))
        self.assertEqual(train_out_internal.shape[0], exp_train_size ,
                         "Wrong size for train set: expected %d, got %d"
                         % (train_out_internal.shape[0], exp_train_size))
        self.assertEqual(test_out_internal.shape[0], exp_test_size ,
                         "Wrong size for train set: expected %d, got %d"
                         % (test_out_internal.shape[0], exp_test_size))

        frac_train = 1.0
        frac_val = .0
        frac_test = .0

        (train_in,train_out,
         val_in,val_out,
         test_in,test_out,
         do_validation,do_test)=\
        nn._NeuralNet__init_data((dataset1_in,dataset1_out),(frac_train,frac_val,frac_test),True)

        #check flags
        self.assertFalse(do_validation, "do_validation should be false, got true")
        self.assertFalse(do_test, "do_test should be False, got True")

        #check sizes
        train_internal = train_in.get_value()

        train_out_internal = train_out.get_value()

        exp_train_size = int(n_samples*frac_train)

        self.assertEqual(train_internal.shape[0], exp_train_size ,
                         "Wrong size for train set: expected %d, got %d"
                         % (train_internal.shape[0], exp_train_size))
        self.assertEqual(train_out_internal.shape[0], exp_train_size ,
                         "Wrong size for train set: expected %d, got %d"
                         % (train_out_internal.shape[0], exp_train_size))
        print "=======data prep testing passed======="


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testDataSplit']
    unittest.main()