#!/usr/bin/env python
"""
Implement your own version of logistic regression with stochastic
gradient descent.

Author: <your_name>
Email : <your_email>
"""

import collections
import math


class LogisticRegressionSGD:

    def __init__(self, eta, mu, n_feature):
        """
        Initialization of model parameters
        """
        self.eta = eta
        self.weight = [0.0] * n_feature

    def fit(self, X, y):
        """
        Update model using a pair of training sample
        """
        pass

    def predict(self, X):
        return 1 if predict_prob(X) > 0.5 else 0

    def predict_prob(self, X):
        return 1.0 / (1.0 + math.exp(-math.fsum((self.weight[f]*v for f, v in X))))
