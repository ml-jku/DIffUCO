# Official Repository of "A Diffusion Model Framework for Unsupervised Neural Combinatorial Optimization"

https://arxiv.org/abs/2406.01661

## Installing the environment

To install the environment fist run
```
conda env create -f environment.yml
```

Some packages will be installed but the installation of jax will run into an error.
Therefore, continue isntalling all missing packages by following the instructions below:

```
conda activate rayjay_clone
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install tqdm jraph matplotlib tqdm optax
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install flax==0.8.1 igraph unipath wandb==0.15.0
```

For the creation of the TSP dataset `pyconcorde` has to be installed.
For that follow the instructions on:
https://github.com/jvkersch/pyconcorde


## Getting started
- get started by creating a dataset with the DatasetCreator
```
cd /DatasetCreator
```
and run for example
```setup
python prepare_datasets.py --dataset RB_iid_100 --problem MIS
```

For more details read
```
DatasetCreator/README.md
```


## Running experiments

To run an experiment on the created dataset (see above) do the following:
```
python argparse_ray_main.py --lrs 0.002 --GPUs 0 --n_GNN_layers 8 --temps 0.6 --IsingMode RB_iid_100 --EnergyFunction MIS --N_anneal 2000 
--n_diffusion_steps 3 --batch_size 20 --n_basis_states 10 --noise_potential bernoulli --project_name FirstRuns --seed 123 
```

### parameter explanation
`--IsingMode` Dataset to train on. In this case the `RB_iid_100`dataset \
`--EnergyFunction` CO problem to train on. In this case the `MIS`problem \
`--noise_potential` Noise Distribution that is used during training \
`--batch_size` Number of different CO Instances in each batch\
`--n_basis_sates` Number of sampled Diffusion Trajectories in each CO Instance\
`--temps` Starting temperature for annealing\

### DiffUCO - ICML 
You can run experiments as in the DiffUCO paper by setting `--train_mode REINFORCE` when running python `argparse_ray_main.py`.

### DiffUCO - RL
You can run DiffUCO with more steps by setting `--train_mode PPO` when running python `argparse_ray_main.py`.
Then, DiffUCO is combined with RL to reduce memory requirements. \
Here, you have to specify the mini-batch size for the inner loops steps within PPO. \
When running DiffUCO with `--n_diffusion_steps A` and `--n_basis_states B` you have to set `--minib_diff_steps X` and `--minib_basis_states Y` 
so that `A/X` and ` B/Y` are integers.

#### Advanced Settings
When runnign DiffUCO-RL you can also set `--proj_mode CE` to project solutions to feasible solutions during training.

### DiffUCO - Forward KL
Alternatively, you can run DiffUCO with more steps by setting `--train_mode Forward_KL` when running python `argparse_ray_main.py`.
(This has not been tested for a while.)

### To evaluate the model use "ConditionalExpectation.py".

After training, you can evaluate the model on the test set with:
```
python ConditionalExpectation.py --wandb_id kj0bihnz --dataset RB_iid_100 --GPU 0 --evaluation_factor 3 --n_samples 8
```

### parameter explanation
`--wandb_id` is the wandb run id \
`--dataset` is the dataset that will be used for evaluation\
`--GPU` is the GPu that will beused for evaluation \
`--n_samples` is the numer of samples that will be obtained for each graph \
`--evaluation_factor` is the factor by which the number of diffusion steps is increased compared to the number 
of diffusion steps that are used during training. So for example if the model is trained with 5 diffusion steps and 
`--evaluation_factor 3`, then the model will be evaluated with 15 diffusion steps
