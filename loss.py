import numpy as np

from activation import Activation_SoftMax

class Loss:
	def calculate(self, output, y):
		sample_losses = self.forward(output, y)
		batch_loss = np.mean(sample_losses)
		return batch_loss

class Loss_CatCrossEntropy(Loss):
	def forward(self, y_pred, y_target):
		n_samples = len(y_pred)
		y_pred_clipped = np.clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

		if len(y_target.shape) == 1:
			confidences = y_pred_clipped[range(n_samples), y_target]
		elif len(y_target.shape) == 2:
			confidences = np.sum(y_pred_clipped*y_target, axis=1)

		neg_log_likelihoods = -np.log(confidences)
		return neg_log_likelihoods

	def backward(self, dvalues, y_target):
		n_samples = len(dvalues)
		n_labels = len(dvalues[0])

		if len(y_target.shape) == 1:
			y_target = np.eye(labels)[y_target]

		self.dinputs = -y_target / dvalues
		self.dinputs = self.dinputs / n_samples


class Loss_CatCrossEntropy_with_SoftMax(Loss):
	def __init__(self):
		self.activation = Activation_SoftMax()
		self.loss = Loss_CatCrossEntropy()

	def forward(self, inputs, y_target):
		self.activation.forward(inputs)
		self.output = self.activation.output
		return self.loss.calculate(self.output, y_target)

	def backward(self, dvalues, y_target):
		n_samples = len(dvalues)

		if len(y_target.shape) == 2:
			y_target = np.argmax(y_target, axis=1)

		self.dinputs = dvalues.copy()
		self.dinputs[range(n_samples), y_target] -= 1
		self.dinputs = self.dinputs / n_samples
