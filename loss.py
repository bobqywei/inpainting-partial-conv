import torch
import torch.nn as nn
from torchvision import models
from places2_train import Places2Train

LAMBDAS = {'valid': 1.0, 'hole': 6.0, 'tv': 0.1, 'perceptual': 0.05, 'style': 120.0}


def gram_matrix(feature_matrix):
	(batch, channel, h, w) = feature_matrix.size()
	feature_matrix = feature_matrix.view(batch, channel, h * w)
	feature_matrix_t = feature_matrix.transpose(1, 2)

	# batch matrix multiplication * normalization factor K_n
	gram = torch.bmm(feature_matrix, feature_matrix_t) / (channel * h * w)

	return gram


def perceptual_loss(h_comp, h_out, h_gt):
	loss = 0.0

	for i in range(3):
		loss += nn.L1Loss(h_out[i], h_gt[i]) + nn.L1Loss(h_comp[i], h_gt[i])

	return loss


def style_loss(h_comp, h_out, h_gt):
	loss_style_out = 0.0
	loss_style_comp = 0.0

	for i in range(3):
		loss_style_out += nn.L1Loss(gram_matrix(h_out[i]), gram_matrix(h_gt[i]))
		loss_style_comp += nn.L1Loss(gram_matrix(h_comp[i]), gram_matrix(h_gt[i]))

	return loss_style_out + loss_style_comp


def total_variation_loss(I_comp):
	return nn.L1Loss(I_comp[:, :, :, :-1], I_comp[:, :, :, 1:]) + nn.L1Loss(I_comp[:, :, :-1, :], I_comp[:, :, 1:, :])


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

	def forward(self, input, mask, output, ground_truth, log=False):
		composited_output = (ground_truth * mask) + (output * (1 - mask))

		fs_composited_output = self.vgg_extract(composited_output)
		fs_output = self.vgg_extract(output)
		fs_ground_truth = self.vgg_extract(ground_truth)

		loss_hole = nn.L1Loss((1 - mask) * output, (1 - mask) * ground_truth)
		loss_valid = nn.L1Loss(mask * output, mask * ground_truth)
		loss_perceptual = perceptual_loss(fs_composited_output, fs_output, fs_ground_truth)
		loss_style = style_loss(fs_composited_output, fs_output, fs_ground_truth)
		loss_total_variation = total_variation_loss(fs_composited_output)

		if log:
			format_log = 'hole: {:f} | valid: {:f} | perceptual: {:f} | style: {:f} | tv: {:f}\n'
			print(format_log.format(loss_hole, loss_valid, loss_perceptual, loss_style, loss_total_variation))

		return LAMBDAS['valid'] * loss_valid + LAMBDAS['hole'] * loss_hole + LAMBDAS['perceptual'] * loss_perceptual \
			+ LAMBDAS['style'] * loss_style + LAMBDAS['tv'] * loss_total_variation


if __name__ == '__main__':
	extractor = VGG16Extractor()
	places2 = Places2Train()
	loss = CalculateLoss()
	img = (places2[0])[0]
	result = extractor(img)
	print(result[0].size())
