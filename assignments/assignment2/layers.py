import numpy as np

def softmax(predictions):
    '''
    Computes probabilities from scores
    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO_ implement softmax

    # raise Exception("Not implemented!")
    
    
    single = (predictions.ndim == 1)
    
    if single:
        predictions = predictions.reshape(1, predictions.shape[0])
    
    maxim = np.amax(predictions, axis=1).reshape(predictions.shape[0], 1)
    predictions_= predictions - maxim
    exp_pred = np.exp(predictions_)
    sums_exp = np.sum(exp_pred, axis=1).reshape(exp_pred.shape[0], 1)
    result = exp_pred / sums_exp
    
    if single:
        result = result.reshape(result.size)
    
    return result
    # Your final implementation shouldn't have any loops
    
def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss
    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)
    Returns:
      loss: single value
    '''
    # TODO_ implement cross-entropy
    # raise Exception("Not implemented!")

    single = (probs.ndim == 1)
    if single:
        probs = probs.reshape(1, probs.shape[0])
        target_index = np.array([target_index])
    
    rows = np.arange(target_index.shape[0])
    cols = target_index
        
    result = np.mean(-np.log(probs[rows, cols]))
    return result



def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    loss = reg_strength * np.trace(np.matmul(W.T, W))
    grad = 2 * W * reg_strength
    
    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    
    single = (predictions.ndim == 1)
    
    if single:
        predictions = predictions.reshape(1, predictions.shape[0])
        target_index = np.array([target_index])
    
    probs = softmax(predictions)

    loss = cross_entropy_loss(probs, target_index)

    mask = np.zeros(probs.shape)
    mask[np.arange(probs.shape[0]), target_index] = 1
    
    dprediction = (probs - mask) / predictions.shape[0]
    
    if single:
        dprediction = dprediction.reshape(dprediction.size)


    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.d_out_result = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass

        self.d_out_result = np.greater(X, 0).astype(float)
        return np.maximum(X, 0)
        
        #raise Exception("Not implemented!")

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        #raise Exception("Not implemented!")
        
        d_result = np.multiply(d_out, self.d_out_result)
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        #raise Exception("Not implemented!")
        self.X = X
        result = np.matmul(X, self.W.value) + self.B.value
        return result
        
        
    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment

        #raise Exception("Not implemented!")
        d_input = np.matmul(d_out, self.W.value.T)
        dW = np.matmul(self.X.T, d_out)
        dB = 2 * np.mean(d_out, axis=0)
        
        self.W.grad += dW
        self.B.grad += dB
        
        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}
