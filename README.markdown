
### **Official Repository of the ICML 2024 Paper**
### **[A Diffusion Model Framework for Unsupervised Neural Combinatorial Optimization](https://arxiv.org/abs/2406.01661)** (DiffUCO)

#### **Authors:**  
 **Sebastian Sanokowski** , **Sepp Hochreiter** , **Sebastian Lehner**

### **and of the ICLR 2025 Paper**
### **[Scalable Discrete Diffusion Samplers: Combinatorial Optimization and Statistical Physics](https://openreview.net/forum?id=peNgxpbdxB)** (SDDS)





## Installing the environment

To install the environment fist run
```
conda env create -f environment.yml
```

Continue isntalling all missing packages by following the instructions below:

```
conda activate DiffUCO
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
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
`--temps` Starting temperature for annealing

### DiffUCO
You can run experiments as in the DiffUCO paper by setting `--train_mode REINFORCE` when running python `argparse_ray_main.py`.


### SDDS: rKL w/ RL
You can run DiffUCO with more steps by setting `--train_mode PPO` when running python `argparse_ray_main.py`.
Then, DiffUCO is combined with RL to reduce memory requirements. \
Here, you have to specify the mini-batch size for the inner loops steps within PPO. \
When running DiffUCO with `--n_diffusion_steps A` and `--n_basis_states B` you have to set `--minib_diff_steps X` and `--minib_basis_states Y` 
so that `A/X` and ` B/Y` are integers.


### SDDS: fKL w/ MC
Alternatively, you can run DiffUCO with more steps by setting `--train_mode Forward_KL` when running python `argparse_ray_main.py`.
(This has not been tested for a while.)


### To evaluate the model use "ConditionalExpectation.py".

After training, you can evaluate the model on the test set with:
```
python ConditionalExpectation.py --wandb_id <WANDB_ID> --dataset <DATASET_NAME> --GPU <GPU_ID> --evaluation_factor <EVAL_FACTOR> --n_samples <N_SAMPLES>
```
In the papers `<EVAL_FACTOR>` is set to 3, i.e. the model uses 3 times more diffusion steps during evaluation than during training.


### parameter explanation
`--wandb_id` is the wandb run id \
`--dataset` is the dataset that will be used for evaluation\
`--GPU` is the GPu that will beused for evaluation \
`--n_samples` is the numer of samples that will be obtained for each graph \
`--evaluation_factor` is the factor by which the number of diffusion steps is increased compared to the number 
of diffusion steps that are used during training. So for example if the model is trained with 5 diffusion steps and 
`--evaluation_factor 3`, then the model will be evaluated with 15 diffusion steps

### configs:

All configs from the paper SDDS can be found in argparse/experiments/Ising for experiments in unbiased sampling and in argparse/experiments/UCO for experiments in combinatorial optimization.

### model weights:
The following model weights are made available:

#### Weights on Combinatorial Optimization Problems:
- Weights for DiffUCO from the SDDS paper


| CO Problem Type | Dataset   | Seed 1     | Seed 2     | Seed 3     |
|-----------------|-----------|------------|------------|------------|
| MaxCl           | RB_small  | k1zc0zgg   | yxyr9urj   | l3s6eybg   |
| MIS             | RB_small  | m3h9mz5g   | olqaqfnl   | 08i3m2dl   |
| MIS             | RB_large  | cvv1wla0   | fuu10c4p   | 00qoqw0s   |
| MDS             | BA_small  | ydq3mn05   | xzc9mds3   | gk5s4nar   |
| MDS             | BA_large  | 64dnrg5p   | 107hsfqv   | 0liz28ec   |
| MaxCut          | BA_small  | 114mqmhk   | t2ud5ttf   | icuxbpll   |
| MaxCut          | BA_large  | ubti92kx   | c11rjsun   | c6yoqwmp   |

- Weights for SDDS: rKL w/ RL from the SDDS paper


| CO Problem Type | Dataset   | Seed 1     | Seed 2     | Seed 3     |
|-----------------|-----------|------------|------------|------------|
| MaxCl           | RB_small  |  5uvh7t41  |  l3ricjby  |   flbniwwf |
| MIS             | RB_small  |  zk3wkaap  |  91icd2vu  |  fj1lym7o  |
| MIS             | RB_large  |  rj161hwt  |  eh5td2pi  |  c8d4u3uo  |
| MDS             | BA_small  |   qfrf58hx |   2ll1wcw1 |  4rgz2hck  |
| MDS             | BA_large  |  t95aukxn  |  ukunkwov  |  x2z3k811  |
| MaxCut          | BA_small  |  9wstv9tl  |   m5l9z4j6 |  nka5ez0d  |
| MaxCut          | BA_large  |  oi3fyq7w  |  4qmwye2w  |  6irzwfyk  |


- Weights for SDDS: fKL w/ MC from the SDDS paper

| CO Problem Type | Dataset   | Seed 1     | Seed 2     | Seed 3     |
|-----------------|-----------|------------|------------|------------|
| MaxCl           | RB_small  |   ud133dhi |  yp999f1m  |  thpyvtjx  |
| MIS             | RB_small  |  otpu58r3  |   9xoy68e6 |  w3u4cer6  |
| MIS             | RB_large  |  6rrd7m5b  |  rjcm5oto  | lgo8u2aq   |
| MDS             | BA_small  |  a7zogxmh  |   df9rgk6a |  yjwopr68  |
| MDS             | BA_large  | x3mdgetb   |  cpg13tch  |   05juku5c |
| MaxCut          | BA_small  |  8ah3bsvm  |   c1l3z0d4 |   s2ug6f2y |
| MaxCut          | BA_large  |  r3ils8y0  |  qidpkk4j  |  96u1z4mu  |

#### Weights on Statistical Physics Problems:

| Ising Model 24x24 | Weight Id  | 
|-----------------|-----------|
| fKL w/ MC         |   qkfzunur |
| rKL w/ RL            |  ewmsen06 or sw5qr5e6  |

| Spin Glasses 15x15   | Weight Id     | 
|-----------------|------------|
| fKL w/ MC   |   4hl3jr35 |
| rKL w/ RL   |  sw5qr5e6  | 


#### How to evaluate these models:
evauation can be ran with the following command:

```
python evaluate_unbiased_sampling.py --wandb_id <WANDB_ID>  --GPU 0 --n_sampling_rounds 400 --seeds 9  --n_test_basis_states 1200
```

### parameter explanation
`--wandb_id` wandb_id of the model weights \
`--GPU` ID of the GPU \
`--n_sampling_rounds` number of sampling repetitions\
`--n_test_basis_states` the number of samples in each sampling round\
`--seeds` Number of seeds over which the results will be averaged\

n_sampling_rounds*n_test_basis_states will be the amount of overall samples


