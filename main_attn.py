import torch
from torch import hub
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from torch import nn
from torch.nn import functional as F

# TODO: visualize losses

# MODEL_DIR = './models/'
# hub.set_dir(MODEL_DIR)
# if not os.path.exists(MODEL_DIR):
# 	os.makedirs(MODEL_DIR)

if torch.cuda.is_available():
	device = torch.device("cuda:1")
	print('Running on GPU 1')
else:
	device = torch.device("cpu")
	print('Running on CPU')

alpha = 1
beta = 1e5

content_feature_args = {6: 1, 11: 1, 20: 1, 29: 1}
f_tmp = alpha / sum(content_feature_args.values())
content_feature_args = {k: v * f_tmp for k, v in content_feature_args.items()}

style_feature_args = {6: 1, 11: 1, 20: 1, 29: 1}
f_tmp = beta / sum(style_feature_args.values())
style_feature_args = {k: v * f_tmp for k, v in style_feature_args.items()}

feature = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.eval()
feature = feature[:(
    max(*list(content_feature_args.keys()), *list(style_feature_args.keys())) +
    1)]
for mod in feature:
	if hasattr(mod, 'inplace'):
		mod.inplace = False
for param in feature.parameters():
	param.requires_grad = False

feature = feature.to(device)
feature.eval()


def calc_gram_matrix(x):
	# 1 x C x H x W -> C x HW -> C x C
	_, c, h, w = x.shape
	x = x.view(c, h * w)
	x_t = x.transpose(0, 1)
	return torch.matmul(x, x_t) / (h * w)


def get_feature(x):
	style_feature = {}
	content_feature = {}
	feature.eval()
	for i in range(len(feature)):
		x = feature[i](x)
		if i in style_feature_args.keys():
			style_feature[i] = calc_gram_matrix(x)
		if i in content_feature_args.keys():
			content_feature[i] = x
	return content_feature, style_feature


def get_std_mean(x):
	assert x.dim() == 4
	mean = torch.mean(x, dim=[-1, -2], keepdim=True)
	std = torch.std(x, dim=[-1, -2], keepdim=True) + 1e-6
	return mean, std


def InstanceNorm(x):
	mean, std = get_std_mean(x)
	return (x - mean) / std


def get_key(feature):
	key_layer = [6, 11, 20, 29]
	total_layer = [6, 11, 20, 29]
	keys = {}
	for i in key_layer:
		results = []
		for j in total_layer:
			results.append(
			    InstanceNorm(
			        nn.functional.interpolate(feature[j],
			                                  size=feature[i].shape[-2:])))
			if (j == i):
				break
		keys[i] = torch.cat(results, dim=1)
	return keys


total_variation_weight = 1e2


def total_variation_loss(x):
	# 1 x C x H x W
	return 0.5 * torch.mean(
	    torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + 0.5 * torch.mean(
	        torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))


laplacian_filter = torch.tensor([[0, 1., 0], [1, -4, 1], [0, 1, 0]]).to(device)
laplacian_weight = 1e2


def get_laplacian(x, p=4):
	# 1 x 3 x H x W
	x = torch.sum(x, axis=1)
	x = F.avg_pool2d(x, p)
	x = F.conv2d(x, laplacian_filter.view(1, 1, 3, 3)).squeeze()
	return x


def calc_loss(x, content_hat, style_hat, lap_hat):
	# tx = x.clone().detach()
	content_feature, style_feature = get_feature(x)
	lap_feature = get_laplacian(x)
	# assert x.eq(tx).all().item()

	content_loss = torch.zeros(1).to(device)
	for i in content_feature_args.keys():
		content_loss = content_loss + content_feature_args[i] * F.mse_loss(
		    InstanceNorm(content_feature[i]),
		    InstanceNorm(content_hat[i].detach()))

	style_loss = torch.zeros(1).to(device)
	for i in style_feature_args.keys():
		style_loss = style_loss + style_feature_args[i] * torch.mean(
		    (style_feature[i] - style_hat[i].detach())**2)

	tv_loss = total_variation_weight * total_variation_loss(x)

	lap_loss = torch.sum(
	    (lap_feature - lap_hat.detach())**2) * laplacian_weight

	return content_loss + style_loss + tv_loss + lap_loss
	# return style_loss


max_sample = 128 * 128


def AttN(style_hat, content_hat):
	key_style = get_key(style_hat)
	key_content = get_key(content_hat)
	value_style = style_hat
	AttN_hat = {}
	for i in style_feature_args.keys():
		k_s = key_style[i].view(key_style[i].shape[0], -1,
		                        key_style[i].shape[2] * key_style[i].shape[3])
		k_c = key_content[i].view(
		    key_content[i].shape[0], -1,
		    key_content[i].shape[2] * key_content[i].shape[3])
		k_c = k_c.permute(0, 2, 1).contiguous()
		# attention = torch.matmul(k_s, k_c)
		v_s = value_style[i].view(
		    value_style[i].shape[0], -1,
		    value_style[i].shape[2] * value_style[i].shape[3])
		if k_s.shape[-1] > max_sample:
			idx = torch.randperm(k_s.shape[-1]).to(device)[:max_sample]
			k_s = k_s[:, :, idx]
			v_s = v_s[:, :, idx]
			v_s = v_s.permute(0, 2, 1).contiguous()
		else:
			v_s = v_s.permute(0, 2, 1).contiguous()
		attn = torch.matmul(k_c, k_s)
		attn = torch.softmax(attn, dim=-1)
		mean = torch.matmul(attn, v_s)
		std = torch.sqrt(torch.relu(torch.bmm(attn, v_s**2) - mean**2))
		mean = mean.view(1, key_content[i].shape[2], key_content[i].shape[3],
		                 -1).permute(0, 3, 1, 2).contiguous()
		std = std.view(1, key_content[i].shape[2], key_content[i].shape[3],
		               -1).permute(0, 3, 1, 2).contiguous()
		# mean, std = get_std_mean(style_hat[i])
		AttN_hat[i] = std * InstanceNorm(content_hat[i]) + mean
		AttN_hat[i] = calc_gram_matrix(AttN_hat[i])
	return AttN_hat


mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])


def preprocess_image(image, shape=None):
	transform = transforms.Compose([
	    # transforms.Resize(shape if shape else image.size),
	    transforms.ToTensor(),
	    transforms.Normalize(mean=mean, std=std)
	])
	image = transform(image)
	image = image.unsqueeze(0)
	image = image.to(device).detach()
	return image


def deprocess_image(image):
	image = image.clone().detach().squeeze().to(std.device)
	# image = transforms.Normalize(-mean / std, 1 / std)(image)
	image = torch.clamp(image.permute(1, 2, 0) * std + mean, 0, 1)
	image = transforms.ToPILImage()(image.permute(2, 0, 1))
	return image


def load_image(style_image_path, content_image_path, shape=None):
	style_image = Image.open(style_image_path).convert('RGB')
	content_image = Image.open(content_image_path).convert('RGB')
	if shape:
		content_image = content_image.resize(shape, Image.Resampling.BICUBIC)
	style_image = style_image.resize(content_image.size,
	                                 Image.Resampling.BICUBIC)
	image_size = content_image.size
	style_image = preprocess_image(style_image)
	content_image = preprocess_image(content_image)
	return image_size, style_image, content_image


OUTPUT_PATH = './outputs/'
if not os.path.exists(OUTPUT_PATH):
	os.makedirs(OUTPUT_PATH)


def show_save_img(image, path=None, shape=None):
	if shape:
		image = image.resize(shape, Image.Resampling.BICUBIC)
	if path:
		image.save('./outputs/' + path)
	plt.imshow(image)
	plt.show(block=False)


STYLE_IMAGE_PATH = './images/inputs/style/'

CONTENT_IMAGE_PATH = './images/inputs/content/'

STYLE_PATH = STYLE_IMAGE_PATH + 'starry_night.jpg'

CONTENT_PATH = CONTENT_IMAGE_PATH + 'winter-wolf.jpg'

shape = (512, 512)

_, style_image, content_image = load_image(STYLE_PATH,
                                           CONTENT_PATH,
                                           shape=shape)

show_save_img(deprocess_image(content_image), path='content.jpg')
show_save_img(deprocess_image(style_image), path='style.jpg')

content_hat, _ = get_feature(content_image)
style_hat, _ = get_feature(style_image)

style_hat = AttN(style_hat, content_hat)

lap_hat = get_laplacian(content_image)

temp_ = Image.open(CONTENT_PATH)
content_shape = temp_.size


def get_init_image(mode='content'):
	if mode == 'content':
		return content_image.clone().requires_grad_(True).to(device)
	elif mode == 'random':
		return torch.randn_like(content_image).requires_grad_(True).to(device)
	else:
		raise ValueError('Invalid mode')


output_image = get_init_image('content')

max_iter = 20

optimizer = torch.optim.LBFGS([output_image], max_iter=max_iter)
# optimizer = torch.optim.Adam([output_image], lr=1e-3)

max_step = 1000
if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(description='The configs')
	parser.add_argument('--remain_shape', type=bool, default=True)
	args = parser.parse_args()

	for step in tqdm(range(max_step)):

		def closure():
			optimizer.zero_grad()
			loss = calc_loss(output_image, content_hat, style_hat, lap_hat)
			loss.backward()
			return loss

		optimizer.step(closure)
		# optimizer.zero_grad()
		# if (step % 100 == 99):
		with torch.no_grad():
			loss = calc_loss(output_image, content_hat, style_hat, lap_hat)
			print(loss.item())
		# loss = calc_loss(output_image, content_hat, style_hat, lap_hat)
		# loss.backward()
		# optimizer.step()
		if step % 10 == 0:
			if args.remain_shape:
				show_save_img(deprocess_image(output_image),
				              path='output_latest.jpg',
				              shape=content_shape)
			else:
				show_save_img(deprocess_image(output_image),
				              path='output_latest.jpg')
