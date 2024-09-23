# Preparing Datasets

Get an academic gurobi licence at https://www.gurobi.com/.
This is a neccesary step to run prepare_datasets.py, which uses gurobi to obtain optimal results for CO problem instances.

If you do not have a Gurobi licence you can switch off gurobi by setting `--gurobi_solve False` when running `run prepare_datasets.py`.

## Create Datasets

To create datasets run `prepare_datasets.py`.

e.g.
```setup
python prepare_datasets.py --dataset RB_iid_100 --problem MIS
```

All possible datasets and CO problems are listed within prepare_datasets.py.

### parameter details

`--time_limits inf 0.1 1` specifies the time limit used in gurobi for solving the dataset splits specified in `--modes test train val`. 
So in this example the test set is solved until gurobi found the best solution. The train dataset is solved at a time limit of `0.1` seconds.
When the gurobi takes too much time to solve instances, we recommend to reduce the timelimit of the test set. 



## Dataset Path
All data will be saved in the folder
```
DatasetCreator/loadGraphDatasets/DatasetSolutions/
```

