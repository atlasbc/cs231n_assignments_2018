import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.zeros((X.shape[0], W.shape[1]))
  for i in range(0, X.shape[0]):
      scores[i, :] = X[i, :].dot(W) #1
      scores[i, :] -= np.max(scores[i, :]) # for numerical stabilization
      scores_exp = np.exp(scores[i, :]) #softmax num
      sum_scores_exp = np.sum(scores_exp) #softmax denum
      softmax = scores_exp / sum_scores_exp #2 This is softmax function
      loss += -np.log(softmax[y[i]]) # this is loss
      
      for j in range(0, W.shape[1]):
          dW[:, j] += X[i]*(softmax[j] - (j == y[i])) 
#       #derivative with chain rule
#       dloss = 1
#       dtclass_lprob = -1*dloss #7 # shape > 1
#       dtclass_prob = (1/tclass_prob)*dtclass_lprob #6  # shape > 1
      
#       dprobs[i, y[i]] = 1*dtclass_prob
#       dnum = invden*dprobs[i, :] #(1)
#       dinvden = num*dprobs[i, :] #(1,10) #broadcast
    
#       dden = (-1 / den**2)*dinvden #(1,10) #broadcast

      
#       dscores_exp = 1*dnum #(1)
#       dscores_exp += W.shape[1]*dden #(1,10)broadcast
    
#       ddot = scores[i, :]*dscores_exp #(1,10)
#       dW += X[i, :].reshape(-1,1).dot(ddot.reshape(1,-1)) # dscores > (1,10) dW > 3073x10

  loss = loss / X.shape[0]
  loss += reg*np.sum(W*W)
  
  dW = dW / X.shape[0]
  dW += 2*reg*W  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  scores = X.dot(W)
  scores -= np.max(scores, axis = 1).reshape(N, 1) #for numerical stability
  exp = np.exp(scores)
  softmax = exp / exp.sum(axis=1).reshape(N, 1)
  true_class_probs = softmax[range(0, N), y]
  losses = -np.log(true_class_probs)
    
  #total loss
  loss = np.sum(losses) / N  
  # add regularization
  loss += reg*np.sum(W*W)
  
  #dW
  gama = np.zeros(softmax.shape)
  gama[range(0, N), y] = 1
  dW = X.T.dot(softmax - gama)
  dW = dW / X.shape[0]
  dW += 2*reg*W 

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

