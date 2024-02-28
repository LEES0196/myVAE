import torch, os, sys, time, importlib
import torchvision
import Model
import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def main() -> None:
	args = sys.argv
	dataset_name, [hidden_dim, latent_dim] = GetUserInput()
	dimension_size, [train_ds, test_ds] = load_dataset(dataset_name)
	train_loader = DataLoader(dataset=train_ds, batch_size=64, shuffle=True, num_workers=2)
	test_loader = DataLoader(dataset=test_ds, batch_size=64, shuffle=True, num_workers=2)

	


	
	
'''
LOADING DATASET GIVEN INPUT DATASET NAME
INPUT:	Dataset name in string type
OUTPUT:	Dimension for training of dataset, List of dataset in order of [train, test]
'''
def load_dataset(dataset_name="CIFAR-10":str) -> tuple(int, list[torch.utils.data.Dataset]):
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

		train_dataset = torchvision.datasets.CelebA(root=data_path, split="train", transform=transform, download=True)
		test_dataset = torchvision.datasets.CelebA(root=data_path, split="test", transform=transform, download=True)
		dataset_list.append(train_dataset)
		dataset_list.append(test_dataset)

	elif dataset_name == "Kitti":
		data_dimension = 256
		transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.CenterCrop([256, 256]),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
			transforms.RandomHorizontalFlip()
		])

		train_dataset = torchvision.datasets.Kitti(root=data_path, train=True, transform=transform, download=True)
		test_dataset = torchvision.datasets.Kitti(root=data_path, train=False, transform=transform, download=True)
		dataset_list.append(train_dataset)
		dataset_list.append(test_dataset)



	return data_dimension, dataset_list	

'''
KL Divergence loss between the distribution estimated mean and variance 
'''
def KLD_Loss(mean: torch.tensor, log_var: torch.tensor) -> torch.tensor:
	KLD = -0.5* torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
	return KLD

'''
Reconstruction loss using input tensor and generated image tensor
'''
def Rec_Loss(x: torch.tensor, x_hat: torch.tensor) -> torch.tensor:
	L_rec = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
	return L_rec
	

if __name__ == "__main__":
	cur_path = os.getcwd()
	data_path = "../Data"
	main()
