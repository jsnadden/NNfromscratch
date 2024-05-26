import numpy as np

class Optimiser_SGD:
	def __init__(self, learning_rate = 1.0, decay=0.0, momentum=0.):
		self.learning_rate = learning_rate
		self.current_learning_rate = learning_rate
		self.decay = decay
		self.iterations = 0
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

		# Update parameters
		layer.weights += weight_updates
		layer.biases += bias_updates

	def post_update(self):
		self.iterations += 1



class Optimiser_AdaGrad:
	def __init__(self, learning_rate = 1.0, decay=0.0, epsilon=1e-7):
		self.learning_rate = learning_rate
		self.current_learning_rate = learning_rate
		self.decay = decay
		self.iterations = 0
		self.epsilon = epsilon

	def pre_update(self):
		if self.decay:
			self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))

	def update_params(self, layer):
		# Give the layer a gradient cache, if it doesn't already
		if not hasattr(layer, 'weight_cache'):
			layer.weight_cache = np.zeros_like(layer.weights)
			layer.bias_cache = np.zeros_like(layer.biases)

		# Update cache with squared gradients
		layer.weight_cache += layer.dweights**2
		layer.bias_cache += layer.dbiases**2

		# Update parameters
		layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
		layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)


	def post_update(self):
		self.iterations += 1


class Optimiser_RMSprop:
	def __init__(self, learning_rate = 0.001, decay=0.0, epsilon=1e-7, rho=0.9):
		self.learning_rate = learning_rate
		self.current_learning_rate = learning_rate
		self.decay = decay
		self.iterations = 0
		self.epsilon = epsilon
		self.rho = rho

	def pre_update(self):
		if self.decay:
			self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))

	def update_params(self, layer):
		# Give the layer a gradient cache, if it doesn't already
		if not hasattr(layer, 'weight_cache'):
			layer.weight_cache = np.zeros_like(layer.weights)
			layer.bias_cache = np.zeros_like(layer.biases)

		# Update cache with squared gradients
		weight_cache_updates = (self.rho * layer.weight_cache) + ((1 - self.rho) * layer.dweights**2)
		layer.weight_cache += weight_cache_updates
		bias_cache_updates = (self.rho * layer.bias_cache) + ((1 - self.rho) * layer.dbiases**2)
		layer.bias_cache += bias_cache_updates

		# Update parameters
		layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
		layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)


	def post_update(self):
		self.iterations += 1