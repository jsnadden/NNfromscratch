import numpy as np

class Optimiser_SGD:
	def __init__(self, learning_rate = 1.0, decay=0.0, momentum=0.):
		self.learning_rate = learning_rate
		self.current_learning_rate = learning_rate
		self.decay = decay
		self.iterations = 0;
		self.momentum = momentum

	def pre_update(self):
		if self.decay:
			self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))

	def update_params(self, layer):
		if self.momentum:
			# Give the layer momenta, if it doesn't already
			if not hasattr(layer, "weight_momenta"):
				layer.weight_momenta = np.zeros_like(layer.weights)
				layer.bias_momenta = np.zeros_like(layer.biases)

			# Compute parameter deltas, with momentum
			weight_updates = self.momentum * layer.weight_momenta - self.current_learning_rate * layer.dweights
			layer.weight_momenta = weight_updates

			bias_updates = self.momentum * layer.bias_momenta - self.current_learning_rate * layer.dbiases
			layer.bias_momenta = bias_updates
		else:
			# Compute parameter deltas, without momentum
			weight_updates = -self.current_learning_rate * layer.dweights
			bias_updates = -self.current_learning_rate * layer.dbiases

		layer.weights += weight_updates
		layer.biases += bias_updates

	def post_update(self):
		self.iterations += 1
