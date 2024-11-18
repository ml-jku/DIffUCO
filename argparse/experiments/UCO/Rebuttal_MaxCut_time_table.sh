#!/bin/bash
#MIS small

#MaxCut small
#rKL w/o RL
python argparse_ray_main.py --lrs 0.0001 --relaxed --GPUs 4 --n_GNN_layers 8 --temps 0.3 --IsingMode BA_small --EnergyFunction MaxCut --mode Diffusion --N_anneal 20 --beta_factor 1. --n_diffusion_steps 7 --batch_size 70 --n_basis_states 8 --noise_potential bernoulli --multi_gpu --project_name final_runs --n_rand_nodes 3 --seed 123 --graph_mode normal --train_mode REINFORCE --inner_loop_steps 1 --diff_schedule exp --minib_diff_steps 7 --minib_basis_states 8 --stop_epochs 1500 --mem_frac 0.92

#rKL w/ RL
python argparse_ray_main.py --train_mode PPO --lrs 0.0005 --temps 0.5 --GPUs 4 --minib_diff_steps 7 --n_diffusion_steps 14 --batch_size 140 --n_basis_states 4 --minib_basis_states 4 --relaxed --n_GNN_layers 8 --N_anneal 20 --IsingMode BA_small --EnergyFunction MaxCut --mode Diffusion --beta_factor 1. --noise_potential bernoulli --multi_gpu --project_name final_runs --mov_average 0.09 --n_rand_nodes 3 --seed 123 --graph_mode normal --inner_loop_steps 1 --diff_schedule exp

#fKL w/ MC
python argparse_ray_main.py --train_mode Forward_KL --lrs 0.002 --temps 0.7 --GPUs 4 --minib_diff_steps 7 --n_diffusion_steps 14 --batch_size 140 --n_basis_states 4 --minib_basis_states 4 --relaxed --n_GNN_layers 8 --N_anneal 20 --IsingMode BA_small --EnergyFunction MaxCut --mode Diffusion --beta_factor 1. --noise_potential bernoulli --multi_gpu --project_name final_runs --n_rand_nodes 3 --seed 123 --graph_mode normal --inner_loop_steps 1 --diff_schedule exp
#chmod +x argparse/experiments/UCO/Rebuttal_MaxCut_time_table.sh
