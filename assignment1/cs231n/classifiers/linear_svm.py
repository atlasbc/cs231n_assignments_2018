import numpy as np
from random import shuffle

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    failed_margin = 0 #mine
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        correct_class = y[i] #mine
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        failed_margin += 1 #mine
        dW[:, j] += X[i] #mine
    dW[:, correct_class] += -failed_margin*X[i] #mine

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  dW /= num_train #mine
  loss /= num_train

  # Add regularization to the loss and dW.
  loss += reg * np.sum(W * W)
  dW += 2*reg*W #mine
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #gradient of true class 
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)
  rows = np.arange(scores.shape[0])
  columns = y
  true_class_scores = scores[rows, columns]
  margin = np.clip(scores - true_class_scores.reshape(true_class_scores.shape[0], 1) + 1, a_min = 0, a_max = None)
  margin[rows, columns] = 0
  loss = np.sum(margin) / X.shape[0] + reg*np.sum(W*W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  dw_margin = np.asarray((margin > 0), dtype=int)
  fmargin_perexample = dw_margin.sum(axis = 1) #total number of failed margins per example
  dw_margin[rows, columns]  = -fmargin_perexample
  dW = X.T.dot(dw_margin) / X.shape[0] 
  dW += 2*reg*W
        
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
