import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, softmax


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        #raise Exception("Not implemented!")
        
        self.layer1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu_layer = ReLULayer()
        self.layer2 = FullyConnectedLayer(hidden_layer_size, n_output)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        #raise Exception("Not implemented!")
        
        
        params = self.params()
        W1 = params["W1"]
        B1 = params["B1"]
        W2 = params["W2"]
        B2 = params["B2"]
        
        #cleaning
        W1.grad = np.zeros_like(W1.value)
        B1.grad = np.zeros_like(B1.value)
        W2.grad = np.zeros_like(W2.value)
        B2.grad = np.zeros_like(B2.value)
        
        #forward
        out_layer1 = self.layer1.forward(X)
        out_relu = self.relu_layer.forward(out_layer1)
        out_layer2 = self.layer2.forward(out_relu)
        loss, dpred = softmax_with_cross_entropy(out_layer2, y)
                                    
        #backward
        d_out_layer2 = self.layer2.backward(dpred)
        d_out_relu = self.relu_layer.backward(d_out_layer2)
        d_out_layer1 = self.layer1.backward(d_out_relu)
        
        #regularization
        l2_W1_loss, l2_W1_grad = l2_regularization(W1.value, self.reg)
        l2_B1_loss, l2_B1_grad = l2_regularization(B1.value, self.reg)
        l2_W2_loss, l2_W2_grad = l2_regularization(W2.value, self.reg)
        l2_B2_loss, l2_B2_grad = l2_regularization(B2.value, self.reg)
        
        l2_reg = l2_W1_loss + l2_B1_loss + l2_W2_loss + l2_B2_loss
        loss += l2_reg
        
        W1.grad += l2_W1_grad
        W2.grad += l2_W2_grad
        B1.grad += l2_B1_grad
        B2.grad += l2_B2_grad
        
        
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        #raise Exception("Not implemented!")

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)
        
        out_layer1 = self.layer1.forward(X)
        out_relu = self.relu_layer.forward(out_layer1)
        out_layer2 = self.layer2.forward(out_relu)
        
        probs = softmax(out_layer2)
        pred = np.argmax(probs, axis=1)
        
        #raise Exception("Not implemented!")
        return pred

    def params(self):
        result = {
            "W1": self.layer1.params()["W"],
            "B1": self.layer1.params()["B"],
            "W2": self.layer2.params()["W"],
            "B2": self.layer2.params()["B"]       
        }

        # TODO Implement aggregating all of the params

        #raise Exception("Not implemented!")

        return result
