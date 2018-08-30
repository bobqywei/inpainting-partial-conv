import torch
import os
import torch.nn as nn

from torchvision import models
from torchvision import transforms
from places2_train import Places2Data, MEAN, STDDEV
from PIL import Image

LAMBDAS = {"valid": 1.0, "hole": 6.0, "tv": 0.1, "perceptual": 0.05, "style": 120.0}


def gram_matrix(feature_matrix):
	(batch, channel, h, w) = feature_matrix.size()
	feature_matrix = feature_matrix.view(batch, channel, h * w)
	feature_matrix_t = feature_matrix.transpose(1, 2)

	# batch matrix multiplication * normalization factor K_n
	gram = torch.bmm(feature_matrix, feature_matrix_t) / (channel * h * w)

	return gram


def perceptual_loss(h_comp, h_out, h_gt, l1):
	loss = 0.0

	for i in range(3):
		loss += l1(h_out[i], h_gt[i]) + l1(h_comp[i], h_gt[i])

	return loss


def style_loss(h_comp, h_out, h_gt, l1):
	loss_style_out = 0.0
	loss_style_comp = 0.0

	for i in range(3):
		loss_style_out += l1(gram_matrix(h_out[i]), gram_matrix(h_gt[i]))
		loss_style_comp += l1(gram_matrix(h_comp[i]), gram_matrix(h_gt[i]))

	return loss_style_out + loss_style_comp


def total_variation_loss(I_comp, mask, l1):
	canvas = mask.data
	canvas[:,:,:,:-1] += mask.data[:,:,:,1:] #mask left overlap
	canvas[:,:,:,1:] += mask.data[:,:,:,:-1] #mask right overlap
	canvas[:,:,:-1,:] += mask.data[:,:,1:,:] #mask up overlap
	canvas[:,:,1:,:] += mask.data[:,:,:-1,:] #mask bottom overlap

	mask_not_zero = (canvas != 0)
	P = canvas.masked_fill_(mask_not_zero, 1.0)

	return l1(P[:, :, :, :-1] * I_comp[:, :, :, :-1], P[:, :, :, 1:] * I_comp[:, :, :, 1:]) + \
		l1(P[:, :, :-1, :] * I_comp[:, :, :-1, :], P[:, :, 1:, :] * I_comp[:, :, 1:, :])


class VGG16Extractor(nn.Module):
	def __init__(self):
		super().__init__()
		vgg16 = models.vgg16(pretrained=True)
		self.max_pooling1 = nn.Sequential(vgg16.features[4])
		self.max_pooling2 = nn.Sequential(vgg16.features[9])
		self.max_pooling3 = nn.Sequential(vgg16.features[16])

		for i in range(1, 4):
			for param in getattr(self, 'max_pooling{:d}'.format(i)).parameters():
				param.requires_grad = False

	def forward(self, image):
		results = [image]
		for i in range(1, 4):
			func = getattr(self, 'max_pooling{:d}'.format(i))
			results.append(func(results[-1]))
		return results[1:]


class CalculateLoss(nn.Module):
	def __init__(self):
		super().__init__()
		self.vgg_extract = VGG16Extractor()
		self.l1 = nn.L1Loss()

	def forward(self, mask, output, ground_truth):
		composited_output = (ground_truth * mask) + (output * (1 - mask))

		fs_composited_output = self.vgg_extract(composited_output)
		fs_output = self.vgg_extract(output)
		fs_ground_truth = self.vgg_extract(ground_truth)

		loss_dict = dict()

		loss_dict["hole"] = self.l1((1 - mask) * output, (1 - mask) * ground_truth) * LAMBDAS["hole"]
		loss_dict["valid"] = self.l1(mask * output, mask * ground_truth) * LAMBDAS["valid"]
		loss_dict["perceptual"] = perceptual_loss(fs_composited_output, fs_output, fs_ground_truth, self.l1) * LAMBDAS["perceptual"]
		loss_dict["style"] = style_loss(fs_composited_output, fs_output, fs_ground_truth, self.l1) * LAMBDAS["style"]
		loss_dict["tv"] = total_variation_loss(composited_output, mask, self.l1) * LAMBDAS["tv"]

		return loss_dict


if __name__ == '__main__':
	#places2 = Places2Data()
	cwd = os.getcwd()
	loss_func = CalculateLoss()

	gt = Image.open(cwd + "/test_256/Places365_test_00000050.jpg")
	mask = Image.open(cwd + "/mask/mask_512.jpg")

	img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STDDEV)])
	mask_transform = transforms.ToTensor()

	gt = img_transform(gt.convert("RGB"))
	mask = img_transform(mask.convert("RGB"))
	img = gt * mask

	img.unsqueeze_(0)
	mask.unsqueeze_(0)
	gt.unsqueeze_(0)

	loss_out = loss_func(mask, img, gt)

	for key, value in loss_out.items():
		print("KEY:{} | VALUE:{}".format(key, value))
