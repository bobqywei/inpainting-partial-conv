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


class SubsetSampler(data.sampler.Sampler):
	def __init__(self, start_sample, num_samples):
		self.num_samples = num_samples
		self.start_sample = start_sample

	def __iter__(self):
		return iter(range(self.start_sample, self.num_samples))

	def __len__(self):
		return self.num_samples


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
	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--epochs", type=int, default=3)
	parser.add_argument("--fine_tune", action="store_true")
	parser.add_argument("--gpu", type=int, default=0)
	parser.add_argument("--num_workers", type=int, default=16)
	parser.add_argument("--log_interval", type=int, default=100)
	parser.add_argument("--save_interval", type=int, default=100)

	args = parser.parse_args()

	cwd = os.getcwd()

	if not os.path.exists(cwd + args.log_dir):
		os.makedirs(cwd + args.log_dir)

	if not os.path.exists(cwd + args.save_dir):
		os.makedirs(cwd + args.save_dir)

	writer = SummaryWriter(args.log_dir)

	if args.gpu >= 0:
		device = torch.device("cuda:{}".format(args.gpu))
	else:
		device = torch.device("cpu")

	data_train = Places2Data(args.train_path, args.mask_path)
	data_size = len(data_train)
	print("Loaded training dataset with {} samples".format(data_size))

	assert(data_size % args.batch_size == 0)
	iters_per_epoch = data_size // args.batch_size

	# data_val = Places2Data(args.val_path, args.mask_path)
	# print("Loaded validation dataset...")

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
	start_epoch = 0
	# Resume training on model
	if args.load_model:
		assert os.path.isfile(cwd + args.save_dir + args.load_model)

		filename = cwd + args.save_dir + args.load_model
		checkpoint_dict = torch.load(filename)
		start_iter = checkpoint_dict["iteration"]
		start_epoch = checkpoint_dict["epoch"]

		model.load_state_dict(checkpoint_dict["model"])
		optimizer.load_state_dict(checkpoint_dict["optimizer"])

		print("Resume training on model:{} from epoch:{}, iteration:{}".format(args.load_model, start_epoch, start_iter))

		# Load all parameters to gpu
		model = model.to(device)
		for state in optimizer.state.values():
			for key, value in state.items():
				if isinstance(value, torch.Tensor):
					state[key] = value.to(device)

	for epoch in range(start_epoch, args.epochs):

		iterator_train = iter(data.DataLoader(data_train, 
											batch_size=args.batch_size, 
											num_workers=args.num_workers, 
											sampler=SubsetSampler(start_iter * args.batch_size, data_size)))

		# TRAINING LOOP
		print("\nEPOCH:{} of {} - starting training loop from iteration:{} to iteration:{}\n".format(epoch, args.epochs, start_iter, iters_per_epoch))
		for i in tqdm(range(start_iter, iters_per_epoch)):

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

			if i % args.save_interval == 0 or i + 1 == iters_per_epoch:
				filename = cwd + args.save_dir + "/model{}.pth".format(i + 1)
				state = {"epoch": epoch, "iteration": i + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict()}
				torch.save(state, filename)

	writer.close()
