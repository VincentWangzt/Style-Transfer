import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils import data
# from dataloader import train_loader
from torch.utils.tensorboard import SummaryWriter
from glob import glob
from PIL import Image
from random import randint
from tqdm import tqdm
from datetime import datetime
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from argparse import ArgumentParser
import os
import random
import torch.nn.functional as F

# parser = ArgumentParser()
# parser.add_argument("--local-rank", type=int, default=-1)
# args = parser.parse_args()

alpha = 1  #content
beta = 4  #style

content_feature_args = {3: 1, 10: 1, 17: 1, 30: 1, 43: 1}
f_tmp = alpha / sum(content_feature_args.values())
content_feature_args = {k: v * f_tmp for k, v in content_feature_args.items()}

style_feature_args = {3: 1, 10: 1, 17: 1, 30: 1, 43: 1}
f_tmp = beta / sum(style_feature_args.values())
style_feature_args = {k: v * f_tmp for k, v in style_feature_args.items()}


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


# 构造预训练的 VGG 编码器
# pretrained_vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).eval()
pretrained_vgg = Encoder()
pretrained_vgg_features = pretrained_vgg.features[:(
    max(*list(content_feature_args.keys()), *list(style_feature_args.keys())) +
    1)]

for param in pretrained_vgg_features.parameters():
	param.requires_grad = False

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


# 解码器
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


def calc_gram_matrix(x: torch.Tensor):
	# input_tensor: N x C x W x H
	N, C, W, H = x.size()

	# 展平通道
	input_flattened = x.view(N, C, -1)  # N x C x (W*H)

	# 计算 Gram 矩阵
	gram_matrices = torch.bmm(input_flattened, input_flattened.transpose(
	    1, 2)) / (H * W)  # N x C x C

	return gram_matrices


def compute_mean_std(x: torch.Tensor):
	if x.dim() == 3:
		x = x.unsqueeze(0)
	size = x.size()
	N, C = size[:2]
	std_x = x.view(N, C, -1).std(dim=2) + 1e-6
	std_x = std_x.view(N, C, 1, 1)
	mean_x = x.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
	return mean_x, std_x


def InstanceNorm(x):
	mean, std = compute_mean_std(x)
	return (x - mean) / std


def get_key(feature):
	key_layer = style_feature_args.keys()
	total_layer = style_feature_args.keys()
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


max_sample = 64 * 64


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
			idx = torch.randperm(k_s.shape[-1]).to(k_s.device)[:max_sample]
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
		# AttN_hat[i] = calc_gram_matrix(AttN_hat[i])
	return AttN_hat


class AdaAttN(nn.Module):

	def __init__(self, in_planes, key_planes=None):
		super(AdaAttN, self).__init__()
		if key_planes is None:
			key_planes = in_planes
		self.f = nn.Conv2d(key_planes, key_planes, (1, 1))
		self.g = nn.Conv2d(key_planes, key_planes, (1, 1))
		self.h = nn.Conv2d(in_planes, in_planes, (1, 1))

	def forward(self, content, style, content_key, style_key):
		FF = self.f(content_key)
		G = self.g(style_key)
		H = self.h(style)
		b, _, h_g, w_g = G.size()
		G = G.view(b, -1, w_g * h_g).contiguous()
		if w_g * h_g > max_sample:
			index = torch.randperm(w_g * h_g).to(
			    content.device)[:self.max_sample]
			G = G[:, :, index]
			v_s = H.view(b, -1, w_g * h_g)[:, :, index]
			v_s = v_s.transpose(1, 2).contiguous()
		else:
			v_s = H.view(b, -1, w_g * h_g).transpose(1, 2).contiguous()
		b, _, h, w = FF.size()
		FF = FF.view(b, -1, w * h).permute(0, 2, 1)
		S = torch.matmul(FF, G)
		S = torch.softmax(S, dim=-1)
		mean = torch.matmul(S, v_s)
		std = torch.sqrt(torch.relu(torch.matmul(S, v_s**2) - mean**2))
		mean = mean.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
		std = std.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
		return std * InstanceNorm(content) + mean


class Transformer(nn.Module):

	def __init__(self, in_planes, key_planes=None, shallow_layer=False):
		super(Transformer, self).__init__()
		self.attn4 = AdaAttN(in_planes=in_planes, key_planes=960)
		self.attn5 = AdaAttN(in_planes=in_planes, key_planes=1472)
		self.upsample5 = nn.Upsample(scale_factor=2, mode='nearest')
		self.pad = nn.ReflectionPad2d(1)
		self.conv = nn.Conv2d(in_planes, in_planes, (3, 3))

	def forward(self, content4_1, style4_1, content5_1, style5_1,
	            content4_1_key, style4_1_key, content5_1_key, style5_1_key):
		return self.conv(
		    self.pad(
		        self.attn4(content4_1, style4_1, content4_1_key, style4_1_key)
		        + self.upsample5(
		            self.attn5(content5_1, style5_1, content5_1_key,
		                       style5_1_key))))


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


class FlatFolderDataset(data.Dataset):

	def __init__(self, root, transform):
		super(FlatFolderDataset, self).__init__()
		self.root = root
		self.paths = glob(f'{root}/*')
		self.transform = transform

	def __getitem__(self, index):
		path = self.paths[index]
		img = Image.open(str(path)).convert('RGB')
		img = self.transform(img)
		return img

	def __len__(self):
		return len(self.paths)


class ContentStyleDataset(data.Dataset):

	def __init__(self, content_dataset, style_dataset):
		super(ContentStyleDataset, self).__init__()
		self.content_dataset = content_dataset
		self.style_dataset = style_dataset
		self.style_len = len(style_dataset)

	def __getitem__(self, index):
		content_img = self.content_dataset[index]
		style_img = self.style_dataset[randint(0, self.style_len - 1)]
		return content_img, style_img

	def __len__(self):
		return len(self.content_dataset)


def get_feature(x, feature=pretrained_vgg_features):
	style_feature = {}
	content_feature = {}
	feature.eval()
	for i in range(len(feature)):
		x = feature[i](x)
		if i in style_feature_args.keys():
			style_feature[i] = x
		if i in content_feature_args.keys():
			content_feature[i] = x
	return content_feature[30], style_feature


class Model(nn.Module):

	def __init__(self, encoder, device):
		super(Model, self).__init__()
		self.decoder = Decoder().to(device)
		self.transformer = Transformer(512, 512).to(device)
		self.encoder = encoder
		self.dev = device

	def forward(self, x):
		content_feature, style_feature = get_feature(x)
		content_key = get_key(content_feature)
		style_key = get_key(style_feature)
		Mix_feature = self.transformer(content_feature[30], style_feature[30],
		                               content_feature[43], style_feature[43],
		                               content_key[30], style_key[30])
		Mix_image = self.decoder(Mix_feature)
		return Mix_image

	def generate(self, content, style):
		contents_image, contents_feature = get_feature(content, self.encoder)
		styles_image, styles_feature = get_feature(style, self.encoder)
		content_key = get_key(contents_feature)
		style_key = get_key(styles_feature)

		# Mix_feature = AdaIN(contents_image, styles_image)
		Mix_feature = self.transformer(contents_feature[30],
		                               styles_feature[30],
		                               contents_feature[43],
		                               styles_feature[43], content_key[30],
		                               style_key[30], content_key[43],
		                               style_key[43])

		Mix_image = self.decoder(Mix_feature)

		return Mix_image

	def forward(self, contents, styles):

		# transform = transforms.Compose([
		#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
		#                          std=[0.229, 0.224, 0.225])  # 归一化
		# ])
		contents_image, contents_feature = get_feature(contents, self.encoder)
		styles_image, styles_feature = get_feature(styles, self.encoder)
		content_key = get_key(contents_feature)
		style_key = get_key(styles_feature)

		# Mix_feature = AdaIN(contents_image, styles_image)
		Mix_feature = self.transformer(
		    contents_feature[30],
		    styles_feature[30],
		    contents_feature[43],
		    styles_feature[43],
		    content_key[30],
		    style_key[30],
		    content_key[43],
		    style_key[43],
		)

		Mix_image = self.decoder(Mix_feature)
		#Mix_image = transform(Mix_image)

		Mix_content, Mix_style = get_feature(Mix_image)

		Loss_c = torch.tensor(0.0, requires_grad=True).to(self.dev)

		# Loss_c = criterion(Mix_feature, contents_feature)
		for i in content_feature_args.keys():
			Loss_c = Loss_c + content_feature_args[i] * torch.mean(
			    (InstanceNorm(Mix_style[i]) -
			     InstanceNorm(contents_feature[i]))**2)

		# Loss_c = criterion(AdaIN_content, contents_image)

		Loss_gs = torch.tensor(0.0, requires_grad=True).to(self.dev)

		for i in style_feature_args.keys():
			if i == 43:
				break
			s_mean, s_std = compute_mean_std(styles_feature[i])
			m_mean, m_std = compute_mean_std(Mix_style[i])
			Loss_gs = Loss_gs + F.mse_loss(s_mean, m_mean) + F.mse_loss(
			    s_std, m_std)

		Loss_ls = torch.tensor(0.0, requires_grad=True).to(self.dev)
		attn_hat = AttN(styles_feature, contents_feature)
		for i in style_feature_args.keys():
			if i == 43:
				break
			Loss_ls = Loss_ls + F.mse_loss(Mix_style[i], attn_hat[i])

		return Loss_c, 10 * Loss_gs, 3 * Loss_ls


def train():

	# 构造解码器
	#encoder = Encoder(pretrained_model=pretrained_vgg)

	# if torch.cuda.is_available():
	# 	device = torch.device("cuda:0")
	# 	print('Running on GPU 0')
	# else:
	# 	device = torch.device("cpu")
	# 	print('Running on CPU')

	local_rank = int(os.environ['LOCAL_RANK'])
	os.environ['VISIBLE_DEVICES'] = '0,1,2,3'

	seed = 6666
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

	# torch.cuda.set_device(local_rank)
	device = torch.device('cuda', local_rank)
	torch.distributed.init_process_group(backend='nccl')

	pretrained_vgg_features.to(device)
	model = Model(pretrained_vgg_features, device)
	model = DDP(model, device_ids=[local_rank], output_device=local_rank)

	transform = transforms.Compose([
	    transforms.Resize((512, 512)),  # 调整图像大小
	    transforms.RandomCrop((256, 256)),  # 随机裁剪至 (256, 256)
	    transforms.ToTensor(),  # 转换为Tensor
	    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
	    #                      std=[0.229, 0.224, 0.225])  # 归一化
	])

	content_dataset = FlatFolderDataset(root='./data/content',
	                                    transform=transform)
	style_dataset = FlatFolderDataset(root='./data/style', transform=transform)

	content_style_dataset = ContentStyleDataset(content_dataset, style_dataset)

	train_sampler = DistributedSampler(content_style_dataset)

	train_loader = data.DataLoader(content_style_dataset,
	                               batch_size=8,
	                               num_workers=16,
	                               sampler=train_sampler)

	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-4)

	writer = SummaryWriter('./loss_curve')
	num_epochs = 300
	model.train()  # 设置解码器为训练模式
	model.to(device)
	for epoch in range(num_epochs):
		total_loss = 0
		total_loss_c = 0
		total_loss_g = 0
		total_loss_l = 0
		total_tmp_loss_c = 0
		total_tmp_loss_g = 0
		total_tmp_loss_l = 0
		total_tmp_loss = 0

		for batch_idx, (content_images,
		                style_images) in enumerate(tqdm(train_loader)):

			content_images = content_images.to(device)
			style_images = style_images.to(device)

			# 计算损失
			loss_c, loss_g, loss_l = model(content_images, style_images)
			loss = loss_c + loss_g + loss_l
			total_loss_c += loss_c.item()
			total_loss_g += loss_g.item()
			total_loss_l += loss_l.item()
			optimizer.zero_grad()
			# 反向传播和优化
			loss.backward()
			optimizer.step()
			# scheduler.step()
			# for name, param in decoder.named_parameters():
			#     if param.grad is not None:
			#         print(f"Gradient of {name}: {param.grad.norm()}")
			# 	else:
			# 		print(f"Gradient of {name} is None")

			# 打印日志
			if batch_idx % 100 == 0 and local_rank == 0:
				loss_delta = total_loss - total_tmp_loss
				loss_delta_c = total_loss_c - total_tmp_loss_c
				loss_delta_g = total_loss_g - total_tmp_loss_g
				loss_delta_l = total_loss_l - total_tmp_loss_l
				total_tmp_loss = total_loss
				total_tmp_loss_c = total_loss_c
				total_tmp_loss_g = total_loss_g
				total_tmp_loss_l = total_loss_l
				print(
				    f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss_c: {loss_delta_c/100:.4f}, Loss_g: {loss_delta_g/100:.4f}, Loss_l: {loss_delta_l/100:.4f}'
				)
				writer.add_scalars('Loss', {'train': loss_delta},
				                   epoch * len(train_loader) + batch_idx)

		# total_loss = total_loss / len(train_loader)
		# writer.add_scalars('loss', {'train': total_loss}, epoch)
		# 保存模型
		# if epoch % 0 == 9:
		if local_rank == 0:
			timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
			checkpoints_dir = './checkpoints'
			if not os.path.exists(checkpoints_dir):
				os.makedirs(checkpoints_dir)
			torch.save(
			    decoder.state_dict(),
			    f'{checkpoints_dir}/decoder_epoch_{epoch+1}_{timestamp}.pth')


if __name__ == '__main__':
	train()
