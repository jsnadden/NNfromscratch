import numpy as np

class Optimiser_SGD:
	def __init__(self, learning_rate = 1.0, decay=0.0):
		self.learning_rate = learning_rate
		self.current_learning_rate = learning_rate
		self.decay = decay
		self.iterations = 0;

	def pre_update(self):
		if self.decay:
			self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))

	def update_params(self, layer):
		layer.weights += -self.current_learning_rate * layer.dweights
		layer.biases += -self.current_learning_rate * layer.dbiases

	def post_update(self):
		self.iterations += 1
