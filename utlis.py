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

# parser = ArgumentParser()
# parser.add_argument("--local-rank", type=int, default=-1)
# args = parser.parse_args()

alpha = 1  #content
beta = 4  #style

content_feature_args = {30: 1}
f_tmp = alpha / sum(content_feature_args.values())
content_feature_args = {k: v * f_tmp for k, v in content_feature_args.items()}

style_feature_args = {3: 1, 10: 1, 17: 1, 30: 1}
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
		               map_location='cpu'))


# 构造预训练的 VGG 编码器
# pretrained_vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).eval()
pretrained_vgg = Encoder()
pretrained_vgg_features = pretrained_vgg.features[:(
    max(*list(content_feature_args.keys()), *list(style_feature_args.keys())) +
    1)]

for param in pretrained_vgg.parameters():
	param.requires_grad = False


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

		return x


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


def calcul_loss(decoder,
                contents,
                styles,
                criterion,
                device,
                lamda=10,
                feature=pretrained_vgg_features):

	transform = transforms.Compose([
	    transforms.Normalize(mean=[0.485, 0.456, 0.406],
	                         std=[0.229, 0.224, 0.225])  # 归一化
	])
	contents_image, contents_feature = get_feature(contents, feature)
	styles_image, styles_feature = get_feature(styles, feature)

	Mix_feature = AdaIN(contents_image, styles_image)

	Mix_image = decoder(Mix_feature)
	#Mix_image = transform(Mix_image)

	AdaIN_content, AdaIN_feature = get_feature(Mix_image)

	Loss_c = criterion(AdaIN_content, Mix_feature)
	# Loss_c = criterion(AdaIN_content, contents_image)

	Loss_s = torch.tensor(0.0, requires_grad=True).to(device)

	for i in style_feature_args.keys():
		# mean_style = torch.mean(styles_feature[i], dim=[-2, -1], keepdim=True)
		# mean_AdaIN = torch.mean(AdaIN_feature[i], dim=[-2, -1], keepdim=True)
		# std_style = torch.std(styles_feature[i], dim=[-2, -1], keepdim=True)
		# std_AdaIN = torch.std(AdaIN_feature[i], dim=[-2, -1], keepdim=True)
		mean_style, std_style = compute_mean_std(styles_feature[i])
		mean_AdaIN, std_AdaIN = compute_mean_std(AdaIN_feature[i])
		Loss_s = Loss_s + style_feature_args[i] * criterion(
		    mean_style, mean_AdaIN) + style_feature_args[i] * criterion(
		        std_style, std_AdaIN)

	return Loss_c, lamda * Loss_s


def train():

	# 构造解码器
	decoder = Decoder()
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

	transform = transforms.Compose([
	    transforms.Resize((512, 512)),  # 调整图像大小
	    transforms.RandomCrop((256, 256)),  # 随机裁剪至 (256, 256)
	    transforms.ToTensor(),  # 转换为Tensor
	    transforms.Normalize(mean=[0.485, 0.456, 0.406],
	                         std=[0.229, 0.224, 0.225])  # 归一化
	])

	content_dataset = FlatFolderDataset(root='/root/autodl-tmp/train2017',
	                                    transform=transform)
	style_dataset = FlatFolderDataset(root='./images/inputs/style',
	                                  transform=transform)

	content_style_dataset = ContentStyleDataset(content_dataset, style_dataset)

	train_sampler = DistributedSampler(content_style_dataset)

	train_loader = data.DataLoader(content_style_dataset,
	                               batch_size=32,
	                               num_workers=16,
	                               sampler=train_sampler)

	criterion = nn.MSELoss()
	optimizer = torch.optim.AdamW(decoder.parameters(),
	                              lr=1e-3,
	                              weight_decay=0.01)
	# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
	#                                                                  T_0=21,
	#                                                                  T_mult=2)
	decoder = DDP(decoder.cuda(local_rank),
	              device_ids=[local_rank],
	              output_device=local_rank)

	writer = SummaryWriter('./loss_curve')
	num_epochs = 300
	decoder.train()  # 设置解码器为训练模式
	decoder.to(device)
	for epoch in range(num_epochs):
		total_loss = 0
		total_loss_c = 0
		total_loss_s = 0
		total_tmp_loss_c = 0
		total_tmp_loss_s = 0
		total_tmp_loss = 0

		for batch_idx, (content_images,
		                style_images) in enumerate(tqdm(train_loader)):

			optimizer.zero_grad()
			content_images = content_images.to(device)
			style_images = style_images.to(device)

			# 计算损失
			loss_c, loss_s = calcul_loss(decoder, content_images, style_images,
			                             criterion, device)
			loss = loss_c + loss_s
			total_loss_c += loss_c.item()
			total_loss_s += loss_s.item()

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
				loss_delta_s = total_loss_s - total_tmp_loss_s
				total_tmp_loss = total_loss
				total_tmp_loss_c = total_loss_c
				total_tmp_loss_s = total_loss_s
				print(
				    f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss_c: {loss_delta_c/100:.4f}, Loss_s: {loss_delta_s/100:.4f}'
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
