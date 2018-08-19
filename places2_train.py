import random
import torch
import os
import glob
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

MEAN = [0.485, 0.456, 0.406]
STDDEV = [0.229, 0.224, 0.225]


class Places2Train (torch.utils.data.Dataset):

	def __init__(self):
		super(Places2Train, self).__init__()

		self.img_paths = glob.glob(os.path.dirname(os.path.abspath(__file__)) + "/test_large/*.jpg")
		self.mask_paths = glob.glob(os.path.dirname(os.path.abspath(__file__)) + "/mask/*.jpg")
		self.num_masks = len(self.mask_paths)
		self.num_imgs = len(self.img_paths)

		self.img_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(MEAN, STDDEV)])
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
	places2 = Places2Train()
	mix, mask, gt = places2[0]

	mix = mix.numpy()
	mix = mix[0, :, :]

	plt.imshow(mix)
	plt.show()
