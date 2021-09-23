import numpy as np
import logging

class Perceptron:
  def __init__(self, eta, epochs):
    self.weights = np.random.randn(3) * 1e-4
    logging.info(f"Initial Weights before training: {self.weights}")

    self.eta = eta
    self.epochs = epochs
  
  def activationFunction(self, inputs, weights):
    z = np.dot(inputs, weights)
    z_sigmoid = (1)/(1+np.exp(-z)) 
    return np.where(z_sigmoid>0.5, 1, 0) 

  def fit(self, X, y):
    self.X = X
    self.y = y

    X_with_bias = np.c_[self.X, -np.ones((len(self.X), 1))]
    logging.info(f"X with bias: {X_with_bias}")

    for epoch in range(self.epochs):
      logging.info("--"*10)
      logging.info(f"for epoch: {epoch}")
      logging.info("--"*10)

      y_hat = self.activationFunction(X_with_bias, self.weights)
      logging.info(f"output after forward pass: {y_hat}")

      self.error = self.y - y_hat
      logging.info(f"error: {self.error}")

      self.weights = self.weights + self.eta * np.dot(X_with_bias.T, self.error)
      logging.info(f"updated weights after epoch: {epoch}/ {self.epochs} : {self.weights}")
      logging.info("--"*10)

      logging.info("####"*10)
  
  def predict(self, X):
    X_with_bias = np.c_[X, -np.ones((len(X), 1))]
    return self.activationFunction(X_with_bias, self.weights)

  def total_loss(self):
    total_loss = np.sum(self.error)
    logging.info(f"total loss: {total_loss}")
    return total_loss
  