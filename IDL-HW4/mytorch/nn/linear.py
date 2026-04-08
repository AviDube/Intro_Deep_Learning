import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        """
        Initialize the weights and biases with zeros
        W shape: (out_features, in_features)
        b shape: (out_features,)  # Changed from (out_features, 1) to match PyTorch
        """
        # DO NOT MODIFY
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros(out_features)


    def init_weights(self, W, b):
        """
        Initialize the weights and biases with the given values.
        """
        # DO NOT MODIFY
        self.W = W
        self.b = b

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (*, in_features)
        :return: Output Z with shape (*, out_features)
        
        Handles arbitrary batch dimensions like PyTorch
        """
        # TODO: Implement forward pass
        input_shape = A.shape
        A_flat = A.reshape(-1, self.W.shape[1])
        Z_flat = A_flat @ self.W.T + self.b

        Z = Z_flat.reshape(*input_shape[:-1], self.W.shape[0])
        # Store input for backward pass
        self.A = A
        
        return Z

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        """
        # TODO: Implement backward pass
        dLdZ_flat = dLdZ.reshape(-1, self.W.shape[0])
        A_flat = self.A.reshape(-1, self.W.shape[1])

        # Compute gradients
        self.dLdA = dLdZ_flat @ self.W
        self.dLdW = dLdZ_flat.T @ A_flat
        self.dLdb = dLdZ_flat.sum(axis=0)
        self.dLdA = self.dLdA.reshape(self.A.shape)

        # Return gradient of loss wrt input
        return self.dLdA
