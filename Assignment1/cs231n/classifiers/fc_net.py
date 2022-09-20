from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


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

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
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
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.params["W1"] = np.random.normal(0.0,weight_scale,size =(input_dim,hidden_dim))
        ## In order to get hidden_dim in our outputs after multiplying input with W1 our W1 needs to be in this shape.
        ## Again it is useful to make dimension check
        self.params["W2"] = np.random.normal(0.0,weight_scale,size = (hidden_dim,num_classes))
        ## It is the same thing in order to have score matrix of correct size we need W2 to be in this shape.
        self.params["b1"] = np.zeros(hidden_dim)
        self.params["b2"] = np.zeros(num_classes)
        ## Sizes of bias can be confusing but you can remember as an example in cs231 n documents (linear-classify)
        ## for 3 classes we used a bias of size 3 therefore it is the same thing in here. For b1 it is same logic
        ## bias size should be equal to hidden_dim. Lets'say we have 25 input image and 7 classes. We don't need
        ## (25,7) matrix for that because we will add same values for each input image therefore a size of (7,) works
        ## fine .

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
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
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        out_1, cache_1 = affine_relu_forward(X,self.params["W1"],self.params["b1"])
        ##First we calculate the affine and relu using W1 and b1.
        out_2,cache_2 = affine_forward(out_1,self.params["W2"],self.params["b2"])
        ## Then we use that output to calculate the scors using(out,W2,b2)
        scores = out_2
        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
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
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        loss,dout_2 = softmax_loss(scores,y)
        loss += self.reg * 0.5* (np.sum(self.params["W1"] * self.params["W1"])+np.sum(self.params["W2"] * self.params["W2"]))
        ## to calculate the regularizated loss we use all weights
        dout_1 , grads["W2"], grads["b2"] = affine_backward(dout_2,cache_2)
        grads["W2"] += 0.5 * 2 * self.reg * self.params["W2"]
        dx , grads["W1"] , grads["b1"] = affine_relu_backward(dout_1,cache_1)
        grads["W1"] += 0.5 * 2 * self.reg * self.params["W1"]
        ## We don't do regularization with biases. And for regularization of weights we do the same thing we did before.
        ## (the ones in Softmax.py , linear_svm.py)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
