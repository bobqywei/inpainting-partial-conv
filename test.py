import argparse
import torch
import os
import random

from PIL import Image
from torchvision.utils import make_grid
from torchvision.utils import save_image
from torchvision import transforms

from partial_conv_net import PartialConvUNet
from places2_train import unnormalize, MEAN, STDDEV
from loss import CalculateLoss

image_num = str(random.randint(1, 328501)).zfill(8)
parser = argparse.ArgumentParser()
parser.add_argument("--img", type=str, default="/test_256/Places365_test_{}.jpg".format(image_num))
parser.add_argument("--mask", type=str, default="/mask/mask_{}.png".format(random.randint(0, 250)))
parser.add_argument("--model", type=str, default="/model_e0_i40000.pth")
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

checkpoint_dict2 = torch.load(cwd + "/model_e0_i56358.pth", map_location="cpu")
model2 = PartialConvUNet()
model2.load_state_dict(checkpoint_dict2["model"])
model2 = model2.to(device)
model2.eval()

with torch.no_grad():
    output2 = model2(img.to(device), mask.to(device))

output2 = (mask * img) + ((1 - mask) * output2)

"""loss_func = CalculateLoss()
loss_out = loss_func(img, mask, output, gt_img)
for key, value in loss_out.items():
    print("KEY:{} | VALUE:{}".format(key, value))"""

grid = make_grid(torch.cat((unnormalize(gt_img), unnormalize(img), unnormalize(output), unnormalize(output2)), dim=0))
save_image(grid, "test.jpg")
