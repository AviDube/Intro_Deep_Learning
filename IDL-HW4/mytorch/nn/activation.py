import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
        # TODO: Implement forward pass
        exp_Z = np.exp(Z - np.max(Z, axis=self.dim, keepdims=True))
        self.A = exp_Z / np.sum(exp_Z, axis=self.dim, keepdims=True)
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # TODO: Implement backward pass
        dim = self.dim

        A = np.moveaxis(self.A, dim, -1)
        dLdA = np.moveaxis(dLdA, dim, -1)

        A_flat = A.reshape(-1, A.shape[-1])
        dLdA_flat = dLdA.reshape(-1, dLdA.shape[-1])

        dLdZ_flat = np.zeros_like(dLdA_flat)
        for i in range(A_flat.shape[0]):
            A_i = A_flat[i].reshape(-1, 1)
            dLdA_i = dLdA_flat[i].reshape(-1, 1)
            jacobian = np.diagflat(A_i) - np.dot(A_i, A_i.T)
            dLdZ_flat[i] = np.dot(jacobian, dLdA_i).flatten()
    
        dLdZ = dLdZ_flat.reshape(dLdA.shape)
        dLdZ = np.moveaxis(dLdZ, -1, dim)
        return dLdZ
 

    