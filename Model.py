import os, time
import torch
import torchvision
import torch.nn as nn

class Encoder(nn.Module):
	def __init__(self, input_dim:int, hidden_dim:int, latent_dim:int) -> None:
		super(Encoder, self).__init__()

		self.CNN = nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(True),
			nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(True),
			nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(True)
		)

		self.flatten = nn.Flatten()

		self.mean = nn.Linear(hidden_dim, latent_dim)
		self.var = nn.Linear(hidden_dim, latent_dim)

	def forward(self, x:torch.tensor) -> tuple(torch.tensor, torch.tensor):
		print(x.shape)
		x = self.CNN(x)
		x = self.flatten(x)
		mean = self.mean(x)
		log_var = self.var(x)
		return mean, log_var


class Decoder(nn.Module):
	def __init__(self, latent_dim:tuple, hidden_dim:tuple, output_dim:tuple) -> None:
		super(Decoder, self).__init__()

		self.MLP = nn.Linear(latent_dim, hidden_dim)

		self.unflatten = nn.Unflatten(dim=1, unflattened_size=(input_dim, input_dim//8, input_dim//8))

		self.DeCNN = nn.Sequential(
			nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(True),
			nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(True),
			nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1),
		)

	def forward(self, x:torch.tensor) -> torch.tensor:
		x = self.MLP(x)
		x = self.unflatten(x)
		x_hat = self.DeCNN(x)
		return x_hat

class Model(nn.Module):
	def __init__(self, Encoder, Decoder):
		super(Model, self).__init__()
		self.encoder = Encoder
		self.decoder = Decoder

	def reparametrization(self, mean, var):
		e = torch.randn_like(var).to(device)
		z = mean + var * e
		return z

	def forward(self, x):
		mean, log_var = self.encoder(x)
		z = self.reparametrization(mean, torch.exp(0.5*log_var))
		x_hat = self.decoder(z)
		return x_hat, mean, log_var


if __name__ == "__main__":
	if torch.cuda.is_available(): device = 'cuda'
	else: device = 'cpu'

	input_dim = 256
	latent_dim = 200

	hidden_dim = input_dim*(input_dim//8)*(input_dim//8)
	encoder = Encoder(input_dim, hidden_dim, latent_dim)
	decoder = Decoder(latent_dim, hidden_dim, input_dim)
	model = Model(encoder, decoder)

