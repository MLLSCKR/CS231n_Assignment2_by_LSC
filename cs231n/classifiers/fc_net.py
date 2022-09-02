from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

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
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.gb_params = {}
        # ex [H1, H2] = [15, 20]
        # layer1 : 15 layer2 : 20 layer3 : 10

        # first layer's parameter : W1, b1
        # W1's shape : [input_dim, H1]

        # final layer's parameter : Wx, bx
        # Wx's shape : [Hx-1, num_classes]

        for i in range(self.num_layers):
          W_iter, b_iter = None, None
          
          if i == 0: # for the first layer
            W_iter = weight_scale * np.random.randn(input_dim, hidden_dims[i])
            b_iter = np.zeros(hidden_dims[i])
          elif i == self.num_layers - 1: # for the last layer
            W_iter = weight_scale * np.random.randn(hidden_dims[i-1], num_classes)
            b_iter = np.zeros(num_classes)
          else:
            W_iter = weight_scale * np.random.randn(hidden_dims[i-1], hidden_dims[i])
            b_iter = np.zeros(hidden_dims[i])

          self.params['W' + str(i+1)] = W_iter
          self.params['b' + str(i+1)] = b_iter

        if self.normalization == "batchnorm":
          # {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax
          for i in range(self.num_layers - 1):
            g_iter, be_iter = None, None

            g_iter = np.random.randn(hidden_dims[i])
            be_iter = np.random.randn(hidden_dims[i])

            self.gb_params['g' + str(i+1)] = g_iter
            self.gb_params['be' + str(i+1)] = be_iter

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.

        """
        bn_param: Dictionary with the following keys:
        - mode: 'train' or 'test'; required
        - eps: Constant for numeric stability
        - momentum: Constant for running mean / variance.
        - running_mean: Array of shape (D,) giving running mean of features
        - running_var Array of shape (D,) giving running variance of features
        """
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
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
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
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
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        out_list, cache_list = [], []
        batch_cache_list = []
        dropout_cahce_list = []

        # forward operation + batch norm_ not drop out yet
        # if we use dropout -> use forward dropout layer after pass activation layer
        for i in range(self.num_layers):
          out_iter, cache_iter, ba_cache_iter, drop_cache_iter = None, None, None, None
          
          if i == 0:
            if self.normalization == "batchnorm":
              out_iter, cache_iter, ba_cache_iter = affine_norm_relu_forward(X, self.params['W' + str(i+1)], self.params['b' + str(i+1)], self.gb_params['g' + str(i+1)], self.gb_params['be' + str(i+1)], self.bn_params[i])

              if self.use_dropout:
                out_iter, drop_cache_iter = dropout_forward(out_iter, self.dropout_param)
            
            else:
              out_iter, cache_iter = affine_relu_forward(X, self.params['W' + str(i+1)], self.params['b' + str(i+1)])

              if self.use_dropout:
                out_iter, drop_cache_iter = dropout_forward(out_iter, self.dropout_param)

          elif i == self.num_layers - 1: # end of iteration -> no batch + no relu
            out_iter, cache_iter = affine_forward(out_list[i-1], self.params['W' + str(i+1)], self.params['b' + str(i+1)])

          else:
            if self.normalization == "batchnorm":
              out_iter, cache_iter, ba_cache_iter = affine_norm_relu_forward(out_list[i-1], self.params['W' + str(i+1)], self.params['b' + str(i+1)], self.gb_params['g' + str(i+1)], self.gb_params['be' + str(i+1)], self.bn_params[i])
              
              if self.use_dropout:
                out_iter, drop_cache_iter = dropout_forward(out_iter, self.dropout_param)
            
            else:
              out_iter, cache_iter = affine_relu_forward(out_list[i-1], self.params['W' + str(i+1)], self.params['b' + str(i+1)])

              if self.use_dropout:
                out_iter, drop_cache_iter = dropout_forward(out_iter, self.dropout_param)
            
          out_list.append(out_iter)
          cache_list.append(cache_iter)

          if self.normalization == "batchnorm":
            batch_cache_list.append(ba_cache_iter)
          
          if self.use_dropout:
            dropout_cahce_list.append(drop_cache_iter)

        scores = out_list[-1]

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # backprop operation + batch normalization _ not drop out yet
        dx_list, dw_list, db_list = [], [], []
        dg_list, dbe_list = [], []

        loss, dout = softmax_loss(scores, y)

        for i in range(self.num_layers):
          loss += self.reg * np.sum(self.params['W' + str(i+1)] * self.params['W' + str(i+1)]) * 0.5

        iter_cnt = 0

        temp_list = list(range(self.num_layers))
        temp_list.reverse()

        for i in temp_list:
          dx_iter, dw_iter, db_iter, dg_iter, dbe_iter = None, None, None, None, None
          
          if iter_cnt == 0: # the last layer -> only affine layer (no activation layer)
            dx_iter, dw_iter, db_iter = affine_backward(dout, cache_list[i])
            dw_iter += self.reg * self.params['W' + str(i + 1)]
          
          else:
            dout_prev = dx_list[iter_cnt - 1]
            
            if self.use_dropout:
              dout_prev = dropout_backward(dout_prev, dropout_cahce_list[i])

            if self.normalization == "batchnorm":
              dx_iter, dw_iter, db_iter, dg_iter, dbe_iter = affine_norm_relu_backward(dout_prev, cache_list[i], batch_cache_list[i])
              dw_iter += self.reg * self.params['W' + str(i + 1)]
          
            else:
              dx_iter, dw_iter, db_iter = affine_relu_backward(dout_prev, cache_list[i])
              dw_iter += self.reg * self.params['W' + str(i + 1)]

          iter_cnt += 1
          dx_list.append(dx_iter)
          dw_list.append(dw_iter)
          db_list.append(db_iter)

          if self.normalization == "batchnorm":
            dg_list.append(dg_iter)
            dbe_list.append(dbe_iter)
        
        dx_list.reverse()
        dw_list.reverse()
        db_list.reverse()
        dg_list.reverse()
        dbe_list.reverse()

        # define grad
        for i in range(self.num_layers):
          grads['W' + str(i + 1)] = dw_list[i]
          grads['b' + str(i + 1)] = db_list[i]

        
        if self.normalization == "batchnorm": # update gamma beta
          for i in range(self.num_layers - 1):
            self.gb_params['g' + str(i + 1)] -= dg_list[i] * 1e-3
            self.gb_params['be' + str(i + 1)] -= dbe_list[i] * 1e-3
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
