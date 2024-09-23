
import jax.numpy as np

def cosine_linear(n, n_steps, amplitude=1, mean_lr=1., factor=0.1):
	cos_periodicity = n_steps - 2 * factor * n_steps
	lin_periodicity = factor * n_steps
	if n < lin_periodicity:
		lr_curr = mean_lr + amplitude * n / lin_periodicity
	elif n_steps - lin_periodicity < n:
		lr_curr = mean_lr - amplitude + amplitude * (n - lin_periodicity - cos_periodicity) / lin_periodicity
	else:
		lr_curr = mean_lr + amplitude * np.cos(np.pi * (n - lin_periodicity) / cos_periodicity)
	return lr_curr


def cos_schedule(epoch, N_anneal, max_lr = 10**-3, min_lr = 10**-4, f_warmup = 0.025):
	start_lr = 10**-10
	new_lr = np.where(epoch < N_anneal*f_warmup, start_lr + (epoch)/(N_anneal*f_warmup)*(max_lr - start_lr), (max_lr-min_lr)*np.cos(np.pi/N_anneal * epoch)/2 + min_lr + (max_lr-min_lr)/2)
	new_lr = np.where(epoch > N_anneal , min_lr, new_lr)

	return new_lr


if(__name__ == "__main__"):

	from matplotlib import pyplot as plt
	N_warmup = 1000
	N_anneal = 1
	max_epochs = N_warmup + N_anneal
	xx = np.arange(0, max_epochs)
	plt.figure()
	plt.plot(xx, [cos_schedule(i, max_epochs) for i in xx])
	plt.show()