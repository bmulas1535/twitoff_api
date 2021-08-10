from .utils import Sigmoid, BinaryCrossEntropy
import numpy as np


class MLPerceptron(object):
  def __init__(self, input_dim, hidden_dim=10, lr=0.1):
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.lr = lr
    self.loss_function = BinaryCrossEntropy()
    self.hidden_activation = Sigmoid()
    self.output_activation = Sigmoid()
    self.w1 = np.random.uniform(0, 1, (self.input_dim, self.hidden_dim))
    self.w1_b = np.zeros((1, self.hidden_dim))
    self.w2 = np.random.uniform(0, 1, (self.hidden_dim, 1))
    self.w2_b = np.zeros((1, 1))

  def forward(self, X):
    # Complete one forward pass
    self.hidden_sum = X.dot(self.w1) + self.w1_b
    self.hidden_act = self.hidden_activation(self.hidden_sum)
    # Hidden => Output
    self.output_sum = self.hidden_act.dot(self.w2) + self.w2_b
    self.output_act = self.output_activation(self.output_sum)

  def backprop(self, X, y):
    # Complete one backpropagation cycle
    # Calculate gradient output => hidden
    self.dx_loss = self.loss_function.gradient(y, self.output_act) * self.output_activation.gradient(self.output_sum)
    self.dx_w2 = self.hidden_act.T.dot(self.dx_loss)
    self.dx_w2_b = np.sum(self.dx_loss, axis=0, keepdims=True)
    # Calculate gradient hidden => input
    self.dx_hidden = self.dx_loss.dot(self.w2.T) * self.hidden_activation.gradient(self.hidden_sum)
    self.dx_w1 = X.T.dot(self.dx_hidden)
    self.dx_w1_b = np.sum(self.dx_hidden, axis=0, keepdims=True)

    # Update weights with gradients
    self.w1 -= self.lr * self.dx_w1
    self.w1_b -= self.lr * self.dx_w1_b
    self.w2 -= self.lr * self.dx_w2
    self.w2_b -= self.lr * self.dx_w2_b

  def train(self, X, y, epochs=150):
    # Train for a given number of epochs
    self.loss = list()

    for i in range(epochs):
      self.forward(X)
      self.backprop(X, y)
      self.loss.append(np.mean(self.loss_function.loss(y, self.output_act)))

  def predict(self, X):
    # Make a prediction on a single input
    self.forward(X)
    return self.output_act
