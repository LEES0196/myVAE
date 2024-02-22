import torch, os, sys, time, importlib
import torchvision
import Model
import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def main() -> None:
	args = sys.argv
	try: 
		dataset_name = args[1]
	except:
		dataset_name = ""

	[train_ds, test_ds] = load_dataset(dataset_name)
	
	

def load_dataset(dataset_name:str) -> tuple(int, list[torch.utils.data.Dataset]):
	dataset_list = []

	if dataset_name == "CIFAR-10":
		data_dimension = 32
		transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
			transforms.RandomHorizontalFlip(p=0.2)
		])

		train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, transform=transform, download=True)
		test_dataset = torchvision.datasets.CIFAR10(root=data_path, train=False, transform=transform, download=True)
		dataset_list.append(train_dataset)
		dataset_list.append(test_dataset)

	elif dataset_name == "CelebA":
		data_dimension = 256
		transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Resize([256, 256]),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
			transforms.RandomHorizontalFlip()
		])

		train_dataset = torch

	return data_dimension, dataset_list	

		
	

if __name__ == "__main__":
	cur_path = os.getcwd()
	data_path = "../Data"
	main()
