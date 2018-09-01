import random
import torch
import os
import glob
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision import utils

MEAN = [0.485, 0.456, 0.406]
STDDEV = [0.229, 0.224, 0.225]


def unnormalize(x):
	x.transpose_(1, 3)
	x = x * torch.Tensor(STDDEV) + torch.Tensor(MEAN)
	x.transpose_(1, 3)
	return x


class Places2Data (torch.utils.data.Dataset):

	def __init__(self, path_to_data="/data_256", path_to_mask="/mask"):
		super().__init__()

		self.img_paths = glob.glob(os.path.dirname(os.path.abspath(__file__)) + path_to_data + "/**/*.jpg", recursive=True)
		self.mask_paths = glob.glob(os.path.dirname(os.path.abspath(__file__)) + path_to_mask + "/*.png")
		self.num_masks = len(self.mask_paths)
		self.num_imgs = len(self.img_paths)

		self.img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STDDEV)])
		self.mask_transform = transforms.ToTensor()

	def __len__(self):
		return self.num_imgs

	def __getitem__(self, index):
		gt_img = Image.open(self.img_paths[index])
		gt_img = self.img_transform(gt_img.convert('RGB'))

		mask = Image.open(self.mask_paths[random.randint(0, self.num_masks - 1)])
		mask = self.mask_transform(mask.convert('RGB'))

		return gt_img * mask, mask, gt_img


# Unit Test
if __name__ == '__main__':
	places2 = Places2Data()
	print(len(places2))
	img, mask, gt = zip(*[places2[i] for i in range(1)]) # returns tuple of 3x256x256 images
	img = torch.stack(img) # --> i x 3 x 256 x 256
	i = img == 0
	print(i.sum())
	mask = torch.stack(mask)
	gt = torch.stack(gt)

	grid = utils.make_grid(torch.cat((unnormalize(img), mask, unnormalize(gt)), dim=0))
	utils.save_image(grid, "test.jpg")
