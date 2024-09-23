import jax
import numpy as np
import jax.numpy as jnp
import flax.linen as nn


class ReluMLP(nn.Module):
	"""
	Multilayer Perceptron with ReLU activation function in the last layer

	@param num_features_list: list of the number of features in the layers (number of nodes); Example: [32, 32, 2] -> two hidden layers with 32 nodes and an output layer with 2 nodes
	"""
	n_features_list: np.ndarray

	def setup(self):
		layers = []
		# add hidden layers
		for n_features in self.n_features_list[:-1]:
			layers.append(nn.Dense(features=n_features, kernel_init=nn.initializers.he_normal(),
                                 bias_init=nn.initializers.zeros))
			layers.append(jax.nn.relu)
			layers.append(nn.LayerNorm())
		# add output layer
		layers.append(nn.Dense(features=self.n_features_list[-1], kernel_init=nn.initializers.he_normal(),
                                 bias_init=nn.initializers.zeros))
		layers.append(jax.nn.relu)
		layers.append(nn.LayerNorm())

		self.mlp = nn.Sequential(layers)

	def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
		"""
		forward pass though MLP
		@param x: input data as jax numpy array
		"""
		return self.mlp(x)


class ProbMLP(nn.Module):
	"""
	Multilayer Perceptron with log softmax on the last layer

	@param num_features_list: list of the number of features in the layers (number of nodes); Example: [32, 32, 2] -> two hidden layers with 32 nodes and an output layer with 2 nodes
	"""
	n_features_list: np.ndarray

	def setup(self):
		layers = []
		# add hidden layers
		for n_features in self.n_features_list[:-1]:
			layers.append(nn.Dense(features=n_features, kernel_init=nn.initializers.he_normal(),
                                 bias_init=nn.initializers.zeros))
			layers.append(jax.nn.relu)
			layers.append(nn.LayerNorm())
		# add output layer
		layers.append(nn.Dense(features=self.n_features_list[-1]))
		layers.append(lambda x: jax.nn.log_softmax(x, axis=-1))

		self.mlp = nn.Sequential(layers)

	def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
		"""
		forward pass though MLP
		@param x: imput data as jax numpy array
		"""
		return self.mlp(x)


class ValueMLP(nn.Module):
	"""
	Multilayer Perceptron with ReLU activation function in the last layer

	@param num_features_list: list of the number of features in the layers (number of nodes); Example: [32, 32, 2] -> two hidden layers with 32 nodes and an output layer with 2 nodes
	"""
	n_features_list: np.ndarray

	def setup(self):
		layers = []
		# add hidden layers
		for n_features in self.n_features_list[:-1]:
			layers.append(nn.Dense(features=n_features, kernel_init=nn.initializers.he_normal(),
                                 bias_init=nn.initializers.zeros))
			layers.append(jax.nn.relu)
			layers.append(nn.LayerNorm())
		# add output layer
		layers.append(nn.Dense(features=self.n_features_list[-1], kernel_init=nn.initializers.xavier_normal(),
                                 bias_init=nn.initializers.zeros))

		self.mlp = nn.Sequential(layers)

	def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
		"""
		forward pass though MLP
		@param x: input data as jax numpy array
		"""
		return self.mlp(x)
