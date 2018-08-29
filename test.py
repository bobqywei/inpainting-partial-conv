import argparse
import torch
import os

from PIL import Image
from torchvision.utils import make_grid
from torchvision.utils import save_image
from torchvision import transforms

from partial_conv_net import PartialConvUNet
from places2_train import unnormalize, MEAN, STDDEV

parser = argparse.ArgumentParser()
parser.add_argument("--img", type=str, default="/test_256/Places365_test_00000050.jpg")
parser.add_argument("--mask", type=str, default="/mask/mask_555.jpg")
parser.add_argument("--model", type=str, default="/model/model-final.pth")
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

checkpoint_dict = torch.load(cwd + args.model)
model = PartialConvUNet()
model.load_state_dict(checkpoint_dict["model"])
model = model.to(device)
model.eval()

with torch.no_grad():
    output = model(img.to(device), mask.to(device))

output = (mask * img) + ((1 - mask) * output) 

grid = make_grid(torch.cat((unnormalize(gt_img), mask, unnormalize(output)), dim=0))
save_image(grid, "test.jpg")
