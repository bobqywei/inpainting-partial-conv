import argparse
import torch
import os

from PIL import Image
from torchvision.utils import make_grid
from torchvision.utils import save_image
from torchvision import transforms

from partial_conv_net import PartialConvUNet
from places2_train import unnormalize, MEAN, STDDEV
from loss import CalculateLoss

parser = argparse.ArgumentParser()
parser.add_argument("--img", type=str, default="/val_256/Places365_val_00015100.jpg")
parser.add_argument("--mask", type=str, default="/mask/mask_2.jpg")
parser.add_argument("--model", type=str, default="/model/model_e0_i45000.pth")
parser.add_argument("--size", type=int, default=256)

args = parser.parse_args()

cwd = os.getcwd()
device = torch.device("cpu")

img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STDDEV)])
mask_transform = transforms.ToTensor()

mask = Image.open(cwd + args.mask)
mask = mask_transform(mask.convert("RGB"))

gt_img = Image.open(cwd + args.img)
gt_img = img_transform(gt_img.convert("RGB"))
img = gt_img * mask

img.unsqueeze_(0)
gt_img.unsqueeze_(0)
mask.unsqueeze_(0)

checkpoint_dict = torch.load(cwd + args.model, map_location="cpu")
model = PartialConvUNet()
model.load_state_dict(checkpoint_dict["model"])
model = model.to(device)
model.eval()

with torch.no_grad():
    output = model(img.to(device), mask.to(device))

output = (mask * img) + ((1 - mask) * output)

"""
loss_func = CalculateLoss()
loss_out = loss_func(mask, output, gt_img)
for key, value in loss_out.items():
    print("KEY:{} | VALUE:{}".format(key, value))
"""

grid = make_grid(torch.cat((unnormalize(gt_img), mask, unnormalize(output)), dim=0))
save_image(grid, "test.jpg")
