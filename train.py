import argparse
import os
import torch
import numpy as np

from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

from loss import CalculateLoss
from partial_conv_net import PartialConvUNet
from places2_train import Places2Data


# https://github.com/naoto0804/pytorch-inpainting-with-partial-conv/blob/master/train.py
class InfiniteSampler(data.sampler.Sampler):
	def __init__(self, num_samples):
		self.num_samples = num_samples

	def __iter__(self):
		return iter(self.loop())

	def __len__(self):
		return 2 ** 31

	def loop(self):
		i = 0
		order = np.random.permutation(self.num_samples)
		while True:
			yield order[i]
			i += 1
			if i >= self.num_samples:
				np.random.seed()
				order = np.random.permutation(self.num_samples)
				i = 0


def requires_grad(param):
	return param.requires_grad


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--train_path", type=str, default="/data_256")
	parser.add_argument("--mask_path", type=str, default="/mask")
	parser.add_argument("--val_path", type=str, default="/val_256")
	parser.add_argument("--lr", type=float, default=2e-4)
	parser.add_argument("--fine_tune_lr", type=float, default=5e-5)
	parser.add_argument("--batch_size", type=int, default=8)
	parser.add_argument("--max_iter", type=int, default=500000)
	parser.add_argument("--out", type=str, default="result")
	parser.add_argument("--fine_tune", action="store_true")
	parser.add_argument("--gpu", type=int, default=0)
	parser.add_argument("--num_workers", type=int, default=8)
	parser.add_argument("--log_interval", type=int, default=20)

	args = parser.parse_args()

	if args.gpu >= 0:
		device = torch.device("cuda:{}".format(args.gpu))
	else:
		device = torch.device("cpu")

	if not os.path.exists(args.out):
		os.makedirs(args.out)

	data_train = Places2Data(args.train_path, args.mask_path)
	print("Loaded training dataset...")

	data_val = Places2Data(args.val_path, args.mask_path)
	print("Loaded validation dataset...")

	iterator_train = iter(data.DataLoader(data_train, batch_size=args.batch_size, num_workers=args.num_workers, sampler=InfiniteSampler(len(data_train))))
	print("Configured iterator with infinite sampling over training dataset...")

	# Move model to gpu prior to creating optimizer, since parameters become different objects after loading
	model = PartialConvUNet().to(device)
	print("Loaded model to device...")

	# Adam optimizer proposed in: "Adam: A Method for Stochastic Optimization"
	# filters the model parameters for those with requires_grad == True
	optimizer = torch.optim.Adam(filter(requires_grad, model.parameters()), lr=args.lr)
	print("Setup Adam optimizer...")

	# Loss function
	# Moves vgg16 model to gpu, used for feature map in loss function
	loss_func = CalculateLoss().to(device)
	print("Setup loss function...")

	if args.fine_tune:
		lr = args.fine_tune_lr
	else:
		lr = args.lr

	for i in tqdm(range(args.max_iter)):
		# Sets model to train mode
		model.train()

		image, mask, gt = [x.to(device) for x in next(iterator_train)]
		output = model(image, mask)[0]
		log_loss_values = True if i % args.log_interval == 0 else False
		loss = loss_func(image, mask, output, gt, log_loss_values)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
