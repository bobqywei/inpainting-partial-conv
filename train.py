import argparse
import os
import torch
import numpy as np

from torch.utils import data
from tqdm import tqdm
from tensorboardX import SummaryWriter

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

	parser.add_argument("--log_dir", type=str, default="/training_logs")
	parser.add_argument("--save_dir", type=str, default="/model")
	parser.add_argument("--load_model", type=str)

	parser.add_argument("--lr", type=float, default=2e-4)
	parser.add_argument("--fine_tune_lr", type=float, default=5e-5)
	parser.add_argument("--batch_size", type=int, default=8)
	parser.add_argument("--max_iter", type=int, default=500000)
	parser.add_argument("--fine_tune", action="store_true")
	parser.add_argument("--gpu", type=int, default=0)
	parser.add_argument("--num_workers", type=int, default=8)
	parser.add_argument("--log_interval", type=int, default=10)
	parser.add_argument("--save_interval", type=int, default=5)

	args = parser.parse_args()

	if not os.path.exists(args.log_dir):
		os.makedirs(args.log_dir)

	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)

	writer = SummaryWriter(args.log_dir)

	if args.gpu >= 0:
		device = torch.device("cuda:{}".format(args.gpu))
	else:
		device = torch.device("cpu")

	data_train = Places2Data(args.train_path, args.mask_path)
	print("Loaded training dataset...")

	# data_val = Places2Data(args.val_path, args.mask_path)
	# print("Loaded validation dataset...")

	iterator_train = iter(data.DataLoader(data_train, batch_size=args.batch_size, num_workers=args.num_workers, sampler=InfiniteSampler(len(data_train))))
	print("Configured iterator with infinite sampling over training dataset...")

	# Move model to gpu prior to creating optimizer, since parameters become different objects after loading
	model = PartialConvUNet().to(device)
	print("Loaded model to device...")

	# Set the fine tune learning rate if necessary
	if args.fine_tune:
		lr = args.fine_tune_lr
	else:
		lr = args.lr

	# Adam optimizer proposed in: "Adam: A Method for Stochastic Optimization"
	# filters the model parameters for those with requires_grad == True
	optimizer = torch.optim.Adam(filter(requires_grad, model.parameters()), lr=lr)
	print("Setup Adam optimizer...")

	# Loss function
	# Moves vgg16 model to gpu, used for feature map in loss function
	loss_func = CalculateLoss().to(device)
	print("Setup loss function...")

	start_iter = 0

	# Resume training on model
	if args.load_model:
		assert os.path.isfile(args.load_model)
		print("Resume training on model: {}".format(args.load_model))

		filename = args.save_dir + args.load_model
		checkpoint_dict = torch.load(filename)
		start_iter = checkpoint_dict["iteration"]

		model.load_state_dict(checkpoint_dict["model"])
		optimizer.load_state_dict(checkpoint_dict["optimizer"])

		# Load all paramters to gpu
		model = model.to(device)
		for state in optimizer.state.values():
			for key, value in state.items():
				if isinstance(value, torch.Tensor):
					state[key] = value.to(device)

	# TRAINING LOOP
	for i in tqdm(range(start_iter, args.max_iter)):

		# Sets model to train mode
		model.train()

		# Gets the next batch of images
		image, mask, gt = [x.to(device) for x in next(iterator_train)]
		output = model(image, mask)

		loss_dict = loss_func(image, mask, output, gt)
		loss = 0.0

		for key, value in loss_dict.items():
			loss += value
			if i % args.log_interval == 0:
				writer.add_scalar(key, value.item(), i + 1)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if i % args.save_interval == 0 or i + 1 == args.max_iter:
			filename = args.save_dir + "/model{}.pth".format(i + 1)
			state = {"iteration": i + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict()}
			torch.save(state, filename)

	writer.close()
