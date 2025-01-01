import torch
from torch import hub
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# TODO: visualize losses

MODEL_DIR = './models/'
hub.set_dir(MODEL_DIR)
if not os.path.exists(MODEL_DIR):
	os.makedirs(MODEL_DIR)

if torch.cuda.is_available():
	device = torch.device("cuda:0")
	print('Running on GPU 0')
else:
	device = torch.device("cpu")
	print('Running on CPU')

alpha = 1
beta = 1e3

content_feature_args = {25: 1}
f_tmp = alpha / sum(content_feature_args.values())
content_feature_args = {k: v * f_tmp for k, v in content_feature_args.items()}

style_feature_args = {0: 1, 5: 1, 10: 1, 19: 1, 28: 1}
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


total_variation_weight = 1e1


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
	x = F.conv2d(x, laplacian_filter.view(1, 1, 3, 3))
	x = F.avg_pool2d(x, p).squeeze()
	return x


def calc_loss(x, content_hat, style_hat, lap_hat):
	# tx = x.clone().detach()
	content_feature, style_feature = get_feature(x)
	lap_feature = get_laplacian(x)
	# assert x.eq(tx).all().item()

	content_loss = torch.zeros(1).to(device)
	for i in content_feature_args.keys():
		content_loss = content_loss + content_feature_args[i] * torch.mean(
		    (content_feature[i] - content_hat[i].detach())**2)

	style_loss = torch.zeros(1).to(device)
	for i in style_feature_args.keys():
		style_loss = style_loss + style_feature_args[i] * torch.mean(
		    (style_feature[i] - style_hat[i].detach())**2)

	tv_loss = total_variation_weight * total_variation_loss(x)

	lap_loss = torch.sum(
	    (lap_feature - lap_hat.detach())**2) * laplacian_weight

	return content_loss + style_loss + tv_loss + lap_loss


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


def show_save_img(image, path=None):
	if path:
		image.save('./outputs/' + path)
	plt.imshow(image)
	plt.show(block=False)


STYLE_IMAGE_PATH = './images/inputs/style/'

CONTENT_IMAGE_PATH = './images/inputs/content/'

shape = (512, 512)

_, style_image, content_image = load_image(
    STYLE_IMAGE_PATH + 'starry_night.jpg',
    CONTENT_IMAGE_PATH + 'gatys-original.jpg',
    shape=shape)

show_save_img(deprocess_image(content_image), path='content.jpg')
show_save_img(deprocess_image(style_image), path='style.jpg')

content_hat, _ = get_feature(content_image)
_, style_hat = get_feature(style_image)
lap_hat = get_laplacian(content_image)


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

for step in tqdm(range(max_step)):

	def closure():
		optimizer.zero_grad()
		loss = calc_loss(output_image, content_hat, style_hat, lap_hat)
		loss.backward()
		return loss

	optimizer.step(closure)
	# optimizer.zero_grad()
	# loss = calc_loss(output_image, content_hat, style_hat, lap_hat)
	# loss.backward()
	# optimizer.step()
	if step % 10 == 0:
		show_save_img(deprocess_image(output_image), path='output_latest.jpg')
