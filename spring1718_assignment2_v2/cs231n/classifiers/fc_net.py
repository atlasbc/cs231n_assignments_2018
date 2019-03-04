from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        self.params['W1'] = np.random.normal(0, weight_scale, size= (input_dim, hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.normal(0, weight_scale, size= (hidden_dim, num_classes))
        self.params['b2'] = np.zeros(num_classes) 
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        fl_out, fl_cache = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        scores, scores_cache = affine_forward(fl_out, self.params['W2'], self.params['b2'])                                                 
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
        #compute loss
        loss, dscores = softmax_loss(scores, y)
        #add regularization
        loss += self.reg*(0.5*np.sum(self.params['W1']*self.params['W1']) \
               + 0.5*np.sum(self.params['W2']*self.params['W2']))
        
        #compute gradients of second layer
        drelu, grads['W2'], grads['b2'] = affine_backward(dscores, scores_cache)
        #add regularization
        grads['W2'] += self.reg*self.params['W2']
        
        #compute gradients of first layer
        dx, grads['W1'], grads['b1'] = affine_relu_backward(drelu, fl_cache)
        grads['W1'] += self.reg*self.params['W1']
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        self.layers = [] # my addition
        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        self.layers.append(input_dim)
        self.layers.extend(hidden_dims)
        self.layers.append(num_classes)
        for i in range(0, len(self.layers)-1):
            
            #initialize weights and biases
            self.params[f'W{i+1}'] = np.random.normal(0, weight_scale, size= (self.layers[i], self.layers[i+1]))
            self.params[f'b{i+1}'] = np.zeros(self.layers[i+1]) 
            
            #initialize batchnorm scale and shift parameters
            if self.normalization == 'batchnorm' or self.normalization == 'layernorm' and  i < self.num_layers-1:
                self.params[f'gamma{i+1}'] = np.ones(self.layers[i+1])
                self.params[f'beta{i+1}'] = np.zeros(self.layers[i+1])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        
        fc_caches = []
        relu_caches = []
        dropout_caches = []
        
        if self.normalization == 'batchnorm':
            caches = []
            
            #first layer
            out, cache = affine_bn_relu_forward(X, self.params['W1'], self.params['b1'],\
                                                self.params['gamma1'], self.params['beta1'], self.bn_params[0])
            caches.append(cache)
            
            #add dropout layer if it exists
            if self.use_dropout:
                out, dropout_cache = dropout_forward(out, self.dropout_param)
                dropout_caches.append(dropout_cache)

            i = 1
            if self.num_layers >= 3:
                for i in range(2, self.num_layers):
                    out, cache = affine_bn_relu_forward(out, self.params[f'W{i}'], self.params[f'b{i}'],\
                                                        self.params[f'gamma{i}'], self.params[f'beta{i}'], self.bn_params[i-1])     
                    caches.append(cache)
                    
                    #add dropout layer if it exists
                    if self.use_dropout:
                        out, dropout_cache = dropout_forward(out, self.dropout_param)
                        dropout_caches.append(dropout_cache)                    
                    
        elif self.normalization == 'layernorm':
            caches = []
            
            #first layer
            out, cache = affine_ln_relu_forward(X, self.params['W1'], self.params['b1'],\
                                                self.params['gamma1'], self.params['beta1'], self.bn_params[0]) #check this again
            caches.append(cache)
            
            #add dropout layer if it exists
            if self.use_dropout:
                out, dropout_cache = dropout_forward(out, self.dropout_param)
                dropout_caches.append(dropout_cache)            

            i = 1
            if self.num_layers >= 3:
                for i in range(2, self.num_layers):
                    out, cache = affine_ln_relu_forward(out, self.params[f'W{i}'], self.params[f'b{i}'],\
                                                        self.params[f'gamma{i}'], self.params[f'beta{i}'], self.bn_params[i-1]) #check     
                    caches.append(cache)
                    
                    #add dropout layer if it exists
                    if self.use_dropout:
                        out, dropout_cache = dropout_forward(out, self.dropout_param)
                        dropout_caches.append(dropout_cache)                     
        
        else:    
            #first layer
            a, fc_cache = affine_forward(X, self.params['W1'], self.params['b1'])
            out, relu_cache = relu_forward(a)
            fc_caches.append(fc_cache)
            relu_caches.append(relu_cache)
            
            #add dropout layer if it exists
            if self.use_dropout:
                out, dropout_cache = dropout_forward(out, self.dropout_param)
                dropout_caches.append(dropout_cache) 
                
            #intermediate layers
            i = 1
            if self.num_layers >= 3:
                for i in range(2, self.num_layers):
                    a, fc_cache = affine_forward(out, self.params[f'W{i}'], self.params[f'b{i}']) #fc_cache gives x, w b
                    out, relu_cache = relu_forward(a) #relu cache gives a
                    fc_caches.append(fc_cache)
                    relu_caches.append(relu_cache)
                    
                    #add dropout layer if it exists
                    if self.use_dropout:
                        out, dropout_cache = dropout_forward(out, self.dropout_param)
                        dropout_caches.append(dropout_cache)                     

        #last layer
        scores, scores_cache = affine_forward(out, self.params[f'W{i+1}'], self.params[f'b{i+1}'])
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dscores = softmax_loss(scores, y)
        
        #add regularization
        for j in range(1, i + 2):
            loss += self.reg*0.5*np.sum(self.params[f'W{j}']*self.params[f'W{j}'])
        
        #with batchnorm
        if self.normalization == 'batchnorm':
            #compute gradients of last layer
            drelu, grads[f'W{j}'], grads[f'b{j}'] = affine_backward(dscores, scores_cache)
            #add regularization
            grads[f'W{j}'] += self.reg*self.params[f'W{j}']

            #gradients of remaining layers
            for i in range(j-1, 0, -1): 
                if self.use_dropout:
                     drelu = dropout_backward(drelu, dropout_caches[i-1])                
                drelu, grads[f'W{i}'], grads[f'b{i}'], grads[f'gamma{i}'], grads[f'beta{i}'] = affine_bn_relu_backward(drelu, caches[i-1])
                grads[f'W{i}'] += self.reg*self.params[f'W{i}']               
        
        #with layernorm
        elif self.normalization == 'layernorm':
            #compute gradients of last layer
            drelu, grads[f'W{j}'], grads[f'b{j}'] = affine_backward(dscores, scores_cache)
            #add regularization
            grads[f'W{j}'] += self.reg*self.params[f'W{j}']

            #gradients of remaining layers
            for i in range(j-1, 0, -1):
                #add dropout layer if it exists
                if self.use_dropout:
                     drelu = dropout_backward(drelu, dropout_caches[i-1])
                drelu, grads[f'W{i}'], grads[f'b{i}'], grads[f'gamma{i}'], grads[f'beta{i}'] = affine_ln_relu_backward(drelu, caches[i-1])
                grads[f'W{i}'] += self.reg*self.params[f'W{i}']   
                
        # without batchnorm
        else: 
            #compute gradients of last layer
            drelu, grads[f'W{j}'], grads[f'b{j}'] = affine_backward(dscores, scores_cache)
            #add regularization
            grads[f'W{j}'] += self.reg*self.params[f'W{j}']

            #gradients of remaining layers
            for i in range(j-1, 0, -1):
                if self.use_dropout:
                     drelu = dropout_backward(drelu, dropout_caches[i-1])                
                d_affine = relu_backward(drelu, relu_caches[i-1])
                drelu, grads[f'W{i}'], grads[f'b{i}'] = affine_backward(d_affine, fc_caches[i-1])
                grads[f'W{i}'] += self.reg*self.params[f'W{i}']    
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
