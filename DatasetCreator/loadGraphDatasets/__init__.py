"""
Dataset Generator
"""
from . import GsetDatasetGenerator
from .BADatasetGenerator import BADatasetGenerator
from .RBDatasetGenerator import RBDatasetGenerator
from .TSPDatasetGenerator import TSPDatasetGenerator
from .TSPDatasetGenerator import TSPDatasetGenerator
from .GsetDatasetGenerator import GsetDatasetGenerator
from .IsingModelDatasetGenerator import NxNLattice
from .SpinGlassDatasetGenerator import SpinGlassDataset
from .SpinGlassUniformDatasetGenerator import SpinGlassUniformDataset

dataset_generator_registry = {"BA": BADatasetGenerator, "RB_iid": RBDatasetGenerator, "TSP": TSPDatasetGenerator, 
							  "SpinGlass": SpinGlassDataset, "SpinGlassUniform": SpinGlassUniformDataset, "Gset": GsetDatasetGenerator, "NxNLattice": NxNLattice}


def get_dataset_generator(config):
	"""
	:param config: config dictionary specifying the dataset that should be generated
	:return: dataset generator class
	"""
	dataset_name = config["dataset_name"]

	for dataset in dataset_generator_registry.keys():
		if dataset in dataset_name:
			dataset_generator = dataset_generator_registry[dataset]
			return dataset_generator(config)
	raise ValueError(f"Dataset {dataset_name} is not implemented")



