from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for_true_class_grad = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if (scores[j] - correct_class_score + 1>0):
                dW[:,j]+= X[i]*1
                for_true_class_grad+=-1
            else:
                dW[:, j] += X[i] * 0
            if margin > 0:
                loss += margin
            dW[:,y[i]]+= for_true_class_grad*X[i]
            for_true_class_grad = 0


    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dW = dW/num_train
    dW += 2*reg*W

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    num_train = X.shape[0]

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    scores = X.dot(W) ##Now we have a scores matrix that has (C X N) shape
    correct_class_scores = scores[range(scores.shape[0]),y]
    correct_class_scores = np.reshape(correct_class_scores,(X.shape[0],-1))  ##We turned our array into 2D
    correct_class_scores = np.hstack([correct_class_scores] * W.shape[1])  ## Since we are going to subtract the real
    ## score from all scores in a row . I just changed my correct class score matrix size to be same with our scores
    ## matrix. In each row we only have true value.

    margins = np.maximum(0, scores - correct_class_scores + 1)
    margins[range(scores.shape[0]),y] = 0  ## I need to fix the problem with true score
    loss+= np.sum(margins)/num_train
    loss += reg * np.sum(W * W)






    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    first_step = scores - correct_class_scores + 1
    second_step = first_step>0
    third_step = second_step.astype(int)

    ## All we did until now is we created a matrix that has the shape of scores matrix and values in this matrix are
    ## only 0 (if scores - correct_class_scores + 1<=0) and 1 otherwise.
    third_step[range(scores.shape[0]), y] = 0  ## We use a little bit different operation for calculating the gradient
    ## for true class. Therefore I first make it zero then sum all values for each row.

    fourth_step = np.sum(third_step,axis=1)
    third_step[range(scores.shape[0]), y] -=fourth_step
    dW = X.transpose().dot(third_step)
    dW = dW/num_train
    dW += 2 * reg * W

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
