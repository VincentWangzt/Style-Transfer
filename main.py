import torch
from torch import hub
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from torch import nn
from torch.nn.parallel import DataParallel
from torchvision.utils import save_image
# TODO: visualize losses

# MODEL_DIR = './models/'
# hub.set_dir(MODEL_DIR)
# if not os.path.exists(MODEL_DIR):
# 	os.makedirs(MODEL_DIR)

if torch.cuda.is_available():
	device = torch.device("cuda:0")
	print('Running on GPU 0')
else:
	device = torch.device("cpu")
	print('Running on CPU')

alpha = 1
beta = 1e2

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


def get_std_mean(x):
	assert x.dim() == 4
	mean = torch.mean(x, dim=[-1, -2], keepdim=True)
	std = torch.std(x, dim=[-1, -2], keepdim=True)
	return mean, std


def InstanceNorm(x):
	mean, std = get_std_mean(x)
	return (x - mean) / std


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

CONTENT_PATH = CONTENT_IMAGE_PATH + 'nilang.jpg'

shape = (512, 512)

_, style_image, content_image = load_image(STYLE_PATH,
                                           CONTENT_PATH,
                                           shape=shape)

show_save_img(deprocess_image(content_image), path='content.jpg')
show_save_img(deprocess_image(style_image), path='style.jpg')

content_hat, _ = get_feature(content_image)
_, style_hat = get_feature(style_image)
lap_hat = get_laplacian(content_image)

temp_ = Image.open(CONTENT_PATH)
content_shape = temp_.size


def get_init_image(mode='content'):
	if mode == 'content':
		return content_image.clone().requires_grad_(True).to(device)
	elif mode == 'random':
		return torch.randn_like(content_image).requires_grad_(True).to(device)
	elif mode == 'adain':
		decoder = nn.Sequential(
		    nn.ReflectionPad2d((1, 1, 1, 1)),
		    nn.Conv2d(512, 256, (3, 3)),
		    nn.ReLU(),
		    nn.Upsample(scale_factor=2, mode='nearest'),
		    nn.ReflectionPad2d((1, 1, 1, 1)),
		    nn.Conv2d(256, 256, (3, 3)),
		    nn.ReLU(),
		    nn.ReflectionPad2d((1, 1, 1, 1)),
		    nn.Conv2d(256, 256, (3, 3)),
		    nn.ReLU(),
		    nn.ReflectionPad2d((1, 1, 1, 1)),
		    nn.Conv2d(256, 256, (3, 3)),
		    nn.ReLU(),
		    nn.ReflectionPad2d((1, 1, 1, 1)),
		    nn.Conv2d(256, 128, (3, 3)),
		    nn.ReLU(),
		    nn.Upsample(scale_factor=2, mode='nearest'),
		    nn.ReflectionPad2d((1, 1, 1, 1)),
		    nn.Conv2d(128, 128, (3, 3)),
		    nn.ReLU(),
		    nn.ReflectionPad2d((1, 1, 1, 1)),
		    nn.Conv2d(128, 64, (3, 3)),
		    nn.ReLU(),
		    nn.Upsample(scale_factor=2, mode='nearest'),
		    nn.ReflectionPad2d((1, 1, 1, 1)),
		    nn.Conv2d(64, 64, (3, 3)),
		    nn.ReLU(),
		    nn.ReflectionPad2d((1, 1, 1, 1)),
		    nn.Conv2d(64, 3, (3, 3)),
		)

		class Decoder(nn.Module):

			def __init__(self):
				super(Decoder, self).__init__()

				#4_1
				self.pad1 = nn.ReflectionPad2d(1)
				self.invconv1 = nn.Conv2d(512, 256, kernel_size=3, padding=0)
				self.acctivate1 = nn.ReLU()
				self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")

				#3_4
				self.pad2 = nn.ReflectionPad2d(1)
				self.invconv2 = nn.Conv2d(256, 256, kernel_size=3, padding=0)
				self.acctivate2 = nn.ReLU()
				#3_3
				self.pad3 = nn.ReflectionPad2d(1)
				self.invconv3 = nn.Conv2d(256, 256, kernel_size=3, padding=0)
				self.acctivate3 = nn.ReLU()
				#3_2
				self.pad4 = nn.ReflectionPad2d(1)
				self.invconv4 = nn.Conv2d(256, 256, kernel_size=3, padding=0)
				self.acctivate4 = nn.ReLU()
				#3_1
				self.pad5 = nn.ReflectionPad2d(1)
				self.invconv5 = nn.Conv2d(256, 128, kernel_size=3, padding=0)
				self.acctivate5 = nn.ReLU()
				self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")

				#2_2
				self.pad6 = nn.ReflectionPad2d(1)
				self.invconv6 = nn.Conv2d(128, 128, kernel_size=3, padding=0)
				self.acctivate6 = nn.ReLU()
				#2_1
				self.pad7 = nn.ReflectionPad2d(1)
				self.invconv7 = nn.Conv2d(128, 64, kernel_size=3, padding=0)
				self.acctivate7 = nn.ReLU()
				self.upsample3 = nn.Upsample(scale_factor=2, mode="nearest")

				#1_2
				self.pad8 = nn.ReflectionPad2d(1)
				self.invconv8 = nn.Conv2d(64, 64, kernel_size=3, padding=0)
				self.acctivate8 = nn.ReLU()

				#1_1
				self.pad9 = nn.ReflectionPad2d(1)
				self.invconv9 = nn.Conv2d(64, 3, kernel_size=3, padding=0)
				# self.acctivate9 = nn.Tanh()
				#self.decoder = decoder

			def forward(self, x):
				x = self.pad1(x)
				x = self.invconv1(x)
				x = self.acctivate1(x)
				x = self.upsample1(x)

				x = self.pad2(x)
				x = self.invconv2(x)
				x = self.acctivate2(x)

				x = self.pad3(x)
				x = self.invconv3(x)
				x = self.acctivate3(x)

				x = self.pad4(x)
				x = self.invconv4(x)
				x = self.acctivate4(x)

				x = self.pad5(x)
				x = self.invconv5(x)
				x = self.acctivate5(x)
				x = self.upsample2(x)

				x = self.pad6(x)
				x = self.invconv6(x)
				x = self.acctivate6(x)

				x = self.pad7(x)
				x = self.invconv7(x)
				x = self.acctivate7(x)
				x = self.upsample3(x)

				x = self.pad8(x)
				x = self.invconv8(x)
				x = self.acctivate8(x)

				x = self.pad9(x)
				x = self.invconv9(x)
				# x = self.acctivate9(x)

				#x = self.decoder(x)

				return x

		decoder = Decoder()
		decoder.load_state_dict({
		    k[7:]: v
		    for k, v in torch.load( #如果要用decoder_lambda_10.pth需要修改decoder为 x = self.decoder()剩下都注释掉,定义则只保留self.decoder的定义，剩下的nn函数都注释掉
		        './models/checkpoints/decoder_lambda_5_epoch_16_2025-01-14_08-14-23.pth',
		        map_location=device,
		        weights_only=True).items()
		})
		decoder.to(device)

		class Encoder(nn.Module):

			def __init__(self):
				super(Encoder, self).__init__()
				self.features = nn.Sequential(
				    nn.Conv2d(3, 3, (1, 1)),
				    nn.ReflectionPad2d((1, 1, 1, 1)),
				    nn.Conv2d(3, 64, (3, 3)),
				    nn.ReLU(),  # relu1-1
				    nn.ReflectionPad2d((1, 1, 1, 1)),
				    nn.Conv2d(64, 64, (3, 3)),
				    nn.ReLU(),  # relu1-2
				    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
				    nn.ReflectionPad2d((1, 1, 1, 1)),
				    nn.Conv2d(64, 128, (3, 3)),
				    nn.ReLU(),  # relu2-1
				    nn.ReflectionPad2d((1, 1, 1, 1)),
				    nn.Conv2d(128, 128, (3, 3)),
				    nn.ReLU(),  # relu2-2
				    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
				    nn.ReflectionPad2d((1, 1, 1, 1)),
				    nn.Conv2d(128, 256, (3, 3)),
				    nn.ReLU(),  # relu3-1
				    nn.ReflectionPad2d((1, 1, 1, 1)),
				    nn.Conv2d(256, 256, (3, 3)),
				    nn.ReLU(),  # relu3-2
				    nn.ReflectionPad2d((1, 1, 1, 1)),
				    nn.Conv2d(256, 256, (3, 3)),
				    nn.ReLU(),  # relu3-3
				    nn.ReflectionPad2d((1, 1, 1, 1)),
				    nn.Conv2d(256, 256, (3, 3)),
				    nn.ReLU(),  # relu3-4
				    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
				    nn.ReflectionPad2d((1, 1, 1, 1)),
				    nn.Conv2d(256, 512, (3, 3)),
				    nn.ReLU(),  # relu4-1, this is the last layer used
				    nn.ReflectionPad2d((1, 1, 1, 1)),
				    nn.Conv2d(512, 512, (3, 3)),
				    nn.ReLU(),  # relu4-2
				    nn.ReflectionPad2d((1, 1, 1, 1)),
				    nn.Conv2d(512, 512, (3, 3)),
				    nn.ReLU(),  # relu4-3
				    nn.ReflectionPad2d((1, 1, 1, 1)),
				    nn.Conv2d(512, 512, (3, 3)),
				    nn.ReLU(),  # relu4-4
				    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
				    nn.ReflectionPad2d((1, 1, 1, 1)),
				    nn.Conv2d(512, 512, (3, 3)),
				    nn.ReLU(),  # relu5-1
				    nn.ReflectionPad2d((1, 1, 1, 1)),
				    nn.Conv2d(512, 512, (3, 3)),
				    nn.ReLU(),  # relu5-2
				    nn.ReflectionPad2d((1, 1, 1, 1)),
				    nn.Conv2d(512, 512, (3, 3)),
				    nn.ReLU(),  # relu5-3
				    nn.ReflectionPad2d((1, 1, 1, 1)),
				    nn.Conv2d(512, 512, (3, 3)),
				    nn.ReLU()  # relu5-4
				)
				self.features.load_state_dict(
				    torch.load("./models/checkpoints/vgg_normalised.pth",
				               weights_only=True,
				               map_location='cpu'))

		adain_content_feature_args = {30: 1}
		adain_alpha = 1
		f_tmp = adain_alpha / sum(adain_content_feature_args.values())
		adain_content_feature_args = {
		    k: v * f_tmp
		    for k, v in adain_content_feature_args.items()
		}
		adain_beta = 4
		adain_style_feature_args = {3: 1, 10: 1, 17: 1, 30: 1}
		f_tmp = adain_beta / sum(adain_style_feature_args.values())
		adain_style_feature_args = {
		    k: v * f_tmp
		    for k, v in adain_style_feature_args.items()
		}

		pretrained_vgg = Encoder()
		pretrained_vgg_features = pretrained_vgg.features[:(
		    max(*list(adain_content_feature_args.keys()
		              ), *list(adain_style_feature_args.keys())) + 1)]

		for param in pretrained_vgg_features.parameters():
			param.requires_grad = False
		pretrained_vgg_features = pretrained_vgg_features.to(device)

		def compute_mean_std(x: torch.Tensor):
			if x.dim() == 3:
				x = x.unsqueeze(0)
			size = x.size()
			N, C = size[:2]
			std_x = x.view(N, C, -1).std(dim=2) + 1e-6
			std_x = std_x.view(N, C, 1, 1)
			mean_x = x.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
			return mean_x, std_x

		def AdaIN(x: torch.Tensor, y: torch.Tensor):
			'''
			x : N*C*H*W
			y : N*C*H*W 
			N = 1
			'''
			if x.dim() == 3:
				x = x.unsqueeze(0)
			if y.dim() == 3:
				y = y.unsqueeze(0)

			mean_y, std_y = compute_mean_std(y)
			mean_x, std_x = compute_mean_std(x)
			adain = (x - mean_x) / std_x * std_y + mean_y

			return adain

		def __get_feature(x, feature=pretrained_vgg_features):
			style_feature = {}
			content_feature = {}
			feature.eval()
			for i in range(len(feature)):
				x = feature[i](x)
				if i in adain_style_feature_args.keys():
					style_feature[i] = x
				if i in adain_content_feature_args.keys():
					content_feature[i] = x
			return content_feature[30], style_feature

		_content_image = Image.open(CONTENT_PATH)
		_style_image = Image.open(STYLE_PATH)

		# _content_image = deprocess_image(content_image)
		# _style_image = deprocess_image(style_image)
		_transform = transforms.Compose([
		    transforms.Resize(512),  # 调整图像大小
		    # transforms.RandomCrop((256, 256)),  # 随机裁剪至 (256, 256)
		    transforms.ToTensor(),  # 转换为Tensor
		    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
		    #                                                       0.225])  # 归一化
		])
		_content_image = _transform(_content_image).unsqueeze(0).to(device)
		_style_image = _transform(_style_image).unsqueeze(0).to(device)

		with torch.no_grad():
			_content_image, _content_feature = __get_feature(
			    _content_image, pretrained_vgg_features)
			_style_image, _style_feature = __get_feature(
			    _style_image, pretrained_vgg_features)
			Adain_feature = AdaIN(_content_image, _style_image)
			Adain_image = decoder(Adain_feature)
			save_image(Adain_image, './tmp.jpg')
			#Adain_image = transform_new(Adain_image)
			#Adain_image = deprocess_image(Adain_image)
			# print(Adain_image)
			#Adain_image.save('./tmp.jpg')
			# save_image(Adain_image, './tmp.jpg')
			# plt.imshow(Adain_image)
			# plt.show()
		return load_image(
		    './tmp.jpg', './tmp.jpg',
		    shape=shape)[2].clone().requires_grad_(True).to(device)
		# return preprocess_image(Adain_image).clone().requires_grad_(True).to(
		#     device)
	else:
		raise ValueError('Invalid mode')


output_image = get_init_image('adain')

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
