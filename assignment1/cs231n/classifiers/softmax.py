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
      for j in range(0, W.shape[1]):
          dot = np.sum(X[i, :]*(W[:, j])) #1 (1)
          scores[i, j] = dot  #(1)
      scores[i, :] -= np.max(scores[i, :]) #avoiding numerical stabilization
      scores_exp = np.exp(scores[i, :]) #2 (1,10)
      num = scores_exp #3 #numerator (1,10)
      scores_sum = scores_exp.sum() #3 (1)
      den = scores_sum #3 (1)
      invden = 1 / den #3.5 #denumerator > (1) 
      probs = num * invden #4 #sigmoid is done (1, 10) #broadcast
      tclass_prob = probs[y[i]] #5 (1)
      tclass_lprob = np.log(tclass_prob) #6 (1)
      loss += -(tclass_lprob) #7 (1)
    
      #derivative with chain rule
      dloss = 1
      dtclass_lprob = -1*dloss #7 # shape > 1
      dtclass_prob = 1/tclass_prob*dtclass_lprob #6  # shape > 1
    
      dnum = invden*dtclass_prob #(1)
      dinvden = num*dtclass_prob #(1,10) #broadcast
    
      dden = (-1 / den**2)*dinvden #(1,10) #broadcast
      dscores_sum = 1*dden #(1,10)
      dscores_exp = 1*dnum #(1)
      dscores_exp += dscores_exp*dscores_sum #(1,10)broadcast
    
      dscores = np.exp(scores[i, :])*dscores_exp #(1,10)
      ddot = dscores #(1,10)
      dW += X[i, :].reshape(-1,1).dot(dscores.reshape(1,-1)) # dscores > (1,10) dW > 3073x10
    #ddot*X_dev[i, :]
    
    #ddot = ((1 - scores[i, y_dev[i]])*scores[i, y_dev[i]])*d_loss
    #dW[:, y_dev[i]] += X_dev[i, :] * ddot
      #scores[i, :] -= np.max(scores[i, :])
      #scores[i, :] = np.exp(scores[i, :])
      #scores[i, :] = scores[i, :] / scores[i, :].sum() #softmax
      #loss += -np.log(scores[i, y[i]])
      #d_loss = -1 / scores[i, y[i]]  
      ##d_loss = -1 / scores[i, :] 
      #ddot = ((1 - scores[i, y[i]])*scores[i, y[i]])*d_loss
      ##ddot = ((1 - scores[i, :])*scores[i, :])*d_loss
      #dW[:, y[i]] += X[i, :] * ddot
      ##dW += X[i, :].reshape(W.shape[0],-1) * ddot.reshape(1, -1)
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
  #for numerical stability
  scores -= np.max(scores, axis = 1).reshape(N, 1)
  exp = np.exp(scores)
  probs = exp / exp.sum(axis=1).reshape(N, 1)
  loss = np.sum(-np.log(probs[range(0,N), y])) / N  
  # add regularization
  loss += reg*np.sum(W*W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

