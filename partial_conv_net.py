import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialConvLayer (nn.Module):
	def __init__(self, in_channels, out_channels, bn=True, bias=True, sample="none-3", activation="relu"):
		super().__init__()
		self.bn = bn

		if sample == "down-7":
			# Kernel Size = 7, Stride = 2, Padding = 3
			self.input_conv = nn.Conv2d(in_channels, out_channels, 7, 2, 3, bias=bias)
			self.mask_conv = nn.Conv2d(in_channels, out_channels, 7, 2, 3, bias=False)

		elif sample == "down-5":
			self.input_conv = nn.Conv2d(in_channels, out_channels, 5, 2, 2, bias=bias)
			self.mask_conv = nn.Conv2d(in_channels, out_channels, 5, 2, 2, bias=False)

		elif sample == "down-3":
			self.input_conv = nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=bias)
			self.mask_conv = nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False)

		else:
			self.input_conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=bias)
			self.mask_conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)

		nn.init.constant_(self.mask_conv.weight, 1.0)

		# "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
		# negative slope of leaky_relu set to 0, same as relu
		# "fan_in" preserved variance from forward pass
		nn.init.kaiming_normal_(self.input_conv.weight, a=0, mode="fan_in")

		for param in self.mask_conv.parameters():
			param.requires_grad = False

		if bn:
			# Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
			# Applying BatchNorm2d layer after Conv will remove the channel mean
			self.batch_normalization = nn.BatchNorm2d(out_channels)

		if activation == "relu":
			# Used between all encoding layers
			self.activation = nn.ReLU()
		elif activation == "leaky_relu":
			# Used between all decoding layers (Leaky RELU with alpha = 0.2)
			self.activation = nn.LeakyReLU(negative_slope=0.2)

	def forward(self, input_x, mask):
		# output = W^T x (X .* M) + b
		output = self.input_conv(input_x * mask)

		# requires_grad = False
		with torch.no_grad():
			output_mask = self.mask_conv(mask)

		if self.mask_conv.bias is not None:
			# spreads existing bias values out along 2nd dimension (channels) and then expands to output size
			output_bias = self.feature_conv.bias.view(1, -1, 1, 1).expand_as(output)
		else:
			output_bias = torch.zeros_like(output)

		# mask_sum is the sum of the binary mask at every partial_conv location
		mask_is_zero = (output_mask == 0)
		mask_sum = output_mask.masked_fill_(mask_is_zero, 1.0)

		# output at each location as follows:
		# output = (W^T x (X .* M)) / M_sum + b ; if M_sum > 0
		# output = 0 ; if M_sum == 0
		output = (output - output_bias) / mask_sum + output_bias
		output = output.masked_fill_(mask_is_zero, 0.0)

		# mask is updated at each location
		new_mask = torch.ones_like(output)
		new_mask = new_mask.masked_fill_(mask_is_zero, 0.0)

		if self.bn:
			output = self.batch_normalization(output)

		if hasattr(self, 'activation'):
			output = self.activation(output)

		return output, new_mask

