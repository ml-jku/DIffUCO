from flax.core import Scope, Array, init, unfreeze, lift
from flax import linen as nn
import jax
import flax
from jax import random, numpy as jnp
import numpy as np

class MLP(nn.Module):
    n_carry_features: int
    n_hidden: int

    def setup(self):
        layers = []
        self.n_features_list = [self.n_hidden,self.n_hidden]
        # add hidden layers
        for n_features in self.n_features_list:
            layers.append(nn.Dense(features=n_features, kernel_init=nn.initializers.he_normal(),
                                   bias_init=nn.initializers.zeros))
            layers.append(jax.nn.relu)
            layers.append(nn.LayerNorm())
        # add output layer
        layers.append(nn.Dense(features=self.n_carry_features, kernel_init=nn.initializers.he_normal(),
                               bias_init=nn.initializers.zeros))
        layers.append(jax.nn.relu)
        layers.append(nn.LayerNorm())

        self.layers = layers

    @flax.linen.jit
    def __call__(self, c, x: jnp.ndarray) -> jnp.ndarray:
        """
        forward pass though MLP
        @param x: input data as jax numpy array
        """
        for layer in self.layers:
            x = layer(x) + c
            c = layer(x)

        return c, x


class DeepMLP(nn.Module):
    n_layers: int
    n_carry: int
    n_hidden: int

    def setup(self):
        self.deep_mlps = [MLP(self.n_carry, self.n_hidden) for i in range(self.n_layers)]

    @flax.linen.jit
    def __call__(self, c, x):

        for deep_mlp in self.deep_mlps:
            c, x = deep_mlp(c,x)

        return c, x

class JitMLP(nn.Module):
    n_layers: int
    n_carry: int
    n_hidden: int

    def setup(self):
        self.deepMLPs = DeepMLP(self.n_layers, self.n_carry, self.n_hidden)

    @flax.linen.jit
    def __call__(self, c, xs):
        y_list = []
        c_list = []
        for i in range(xs.shape[0]):
            c,y = self.deepMLPs(c,xs[i])
            y_list.append(y)
            c_list.append(c)

        return c_list, y_list

class ScanMLP(nn.Module):
    n_layers: int
    n_carry: int
    n_hidden: int

    def setup(self):
        self.deepMLPs = DeepMLP#(self.n_layers)
        scan = nn.scan(
            self.deepMLPs,
            variable_broadcast="params",
            split_rngs={"params": False},
        )
        self.scan = scan(self.n_layers, self.n_carry, self.n_hidden)

    @flax.linen.jit
    def __call__(self, c, xs):
        return self.scan(c, xs)



if(__name__ == "__main__"):

    import jax
    import os
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    from jax import numpy as jnp
    import time
    import numpy as np
    from matplotlib import pyplot as plt
    log_p_uniform = jnp.log(0.5*jnp.ones((10,5)))

    X_prev = jax.random.categorical(key=    jax.random.PRNGKey(0),
                                    logits= log_p_uniform,
                                    axis=-1,
                                    shape=log_p_uniform.shape[:-1])

    step_list = [2,4,6,8,10,20]
    n_hidden_list = [64, 120]
    for n_hidden in n_hidden_list:

        jit_dict = {"forward": [], "backward": []}
        scan_dict = {"forward": [], "backward": []}
        for n_time_steps in step_list:
            n_layers = 2
            n_carry = n_hidden
            xs = 0.5*jnp.zeros((n_time_steps, n_carry))
            c = 0.5*jnp.zeros((n_carry))


            scan_mlp = ScanMLP(n_layers, n_carry, n_hidden)
            params = scan_mlp.init(random.PRNGKey(1), c, xs)
            print(jax.tree_map(lambda x: x.shape, params))

            measure_time_reps = 20

            #forward_scan(scan_mlp.apply, c, xs)
            cs, ys = scan_mlp.apply(params, c, xs)

            time_list = []
            for t in range(measure_time_reps):
                start_time = time.time()
                cs, ys = scan_mlp.apply(params, c, xs)
                end_time = time.time()
                time_list.append(end_time - start_time)

            print("scan_time", np.mean(time_list), np.std(time_list))
            scan_dict["forward"].append([np.mean(time_list), np.std(time_list)])
            #print(time_list)

            def loss_fn(params, c, xs):
                cs, ys = scan_mlp.apply(params, c, xs)
                loss = jnp.mean(ys[-1]**2)
                return loss, (ys, cs)

            jit_loss_grad = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

            jit_loss_grad(params, c, xs)

            time_list = []
            for t in range(measure_time_reps):
                start_time = time.time()
                jit_loss_grad(params, c, xs)
                end_time = time.time()
                time_list.append(end_time - start_time)

            print("grad scan_time", np.mean(time_list), np.std(time_list))
            #print(time_list)
            scan_dict["backward"].append([np.mean(time_list), np.std(time_list)])

            jit_mlp = JitMLP(n_layers, n_carry, n_hidden)
            params = jit_mlp.init(random.PRNGKey(1), c, xs)

            cs, ys = jit_mlp.apply(params, c, xs)

            time_list = []
            for t in range(measure_time_reps):
                start_time = time.time()
                cs, ys = jit_mlp.apply(params, c, xs)
                end_time = time.time()
                time_list.append(end_time - start_time)

            print("Jit_time", np.mean(time_list), np.std(time_list))
            #print(time_list)
            jit_dict["forward"].append([np.mean(time_list), np.std(time_list)])

            def loss_fn(params, c, xs):
                cs, ys = jit_mlp.apply(params, c, xs)
                loss = jnp.mean(ys[-1] ** 2)
                return loss, (ys, cs)


            jit_loss_grad = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

            jit_loss_grad(params, c, xs)

            time_list = []
            for t in range(measure_time_reps):
                start_time = time.time()
                jit_loss_grad(params, c, xs)
                end_time = time.time()
                time_list.append(end_time - start_time)

            print("grad jit_time", np.mean(time_list), np.std(time_list))
            #print(time_list)

            jit_dict["backward"].append([np.mean(time_list), np.std(time_list)])


        plt.figure()
        plt.title(f"n_hidden_{n_hidden}")
        plt.errorbar(step_list, np.array(scan_dict["forward"])[:,0], yerr = np.array(scan_dict["forward"])[:,1], label = "scan forw")
        plt.errorbar(step_list, np.array(jit_dict["forward"])[:,0], yerr = np.array(jit_dict["forward"])[:,1], label = "fit forw")
        plt.legend()
        plt.show()

        plt.figure()
        plt.title(f"n_hidden_{n_hidden}")
        plt.errorbar(step_list, np.array(scan_dict["backward"])[:,0], yerr = np.array(scan_dict["backward"])[:,1], label = "scan backw")
        plt.errorbar(step_list, np.array(jit_dict["backward"])[:,0], yerr = np.array(jit_dict["backward"])[:,1], label = "jit backw")
        plt.legend()
        plt.show()