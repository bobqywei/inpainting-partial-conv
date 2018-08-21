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


parser = argparse.ArgumentParser()
parser.add_argument("--train_path", type=str, default="/data_256")
parser.add_argument("--mask_path", type=str, default="/mask")
parser.add_argument("--val_path", type=str, default="/val_256")
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--tune_lr", type=float, default=5e-5)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--max_iter", type=int, default=500000)
parser.add_argument("--out", type=str, default="result")
parser.add_argument("--fine_tune", action="store_true")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--num_workers", type=int, default=8)

args = parser.parse_args()

device = torch.cuda.device(args.gpu)

if not os.path.exists(args.out):
    os.makedirs(args.out)

data_train = Places2Data(args.train_path, args.mask_path)
data_val = Places2Data(args.val_path, args.mask_path)

iterator_train = iter(data.DataLoader(data_train, batch_size=args.batch_size, num_workers=args.num_workers, sampler=InfiniteSampler(len(data_train))))

# Move model to gpu prior to creating optimizer, since parameters become different objects
model = PartialConvUNet().to(device)

# Adam optimizer proposed in: "Adam: A Method for Stochastic Optimization"
# filters the model parameters for those with requires_grad == True
optimizer = torch.optim.Adam(filter(requires_grad, model.parameters()), lr=args.lr)

# Moves vgg16 model to gpu, used for feature map in loss function
loss_func = CalculateLoss().to(device)
