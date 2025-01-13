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


# 激活函数
def decoder_activation():
	return nn.ReLU(inplace=True)


# 编码器
class Encoder(nn.Module):

	def __init__(self, pretrained_model, target_layer="relu4_1"):
		super(Encoder, self).__init__()
		self.layers = []

		# 定义层名称到序号的映射
		layer_mapping = {
		    "conv1_1": 0,
		    "relu1_1": 1,
		    "conv1_2": 2,
		    "relu1_2": 3,
		    "pool1": 4,
		    "conv2_1": 5,
		    "relu2_1": 6,
		    "conv2_2": 7,
		    "relu2_2": 8,
		    "pool2": 9,
		    "conv3_1": 10,
		    "relu3_1": 11,
		    "conv3_2": 12,
		    "relu3_2": 13,
		    "conv3_3": 14,
		    "relu3_3": 15,
		    "conv3_4": 16,
		    "relu3_4": 17,
		    "pool3": 18,
		    "conv4_1": 19,
		    "relu4_1": 20,
		}

		# 构造编码器直到目标层
		for name, index in layer_mapping.items():
			self.layers.append(pretrained_model.features[index])
			if name == target_layer:
				break

	def forward(self, x):
		for layer in self.layers:
			x = layer(x)
		return x


# 解码器
class Decoder(nn.Module):

	def __init__(self):
		super(Decoder, self).__init__()

		# self.decoder = nn.Sequential(
		#     # 解码器的每一部分对应VGG-19的一个卷积块
		#     # 第一个卷积块（对应VGG-19的第5个卷积块）

		#     nn.ReflectionPad2d(1),
		#     nn.ConvTranspose2d(512, 256, kernel_size=3, padding=0),
		#     nn.ReLU(inplace=False),
		#     nn.Upsample(scale_factor=2, mode="nearest"),

		#     # 第二个卷积块（对应VGG-19的第4个卷积块）

		#     nn.ReflectionPad2d(1),
		#     nn.ConvTranspose2d(256, 256, kernel_size=3, padding=0),
		#     nn.ReLU(inplace=False),

		#     nn.ReflectionPad2d(1),
		#     nn.ConvTranspose2d(256, 256, kernel_size=3, padding=0),
		#     nn.ReLU(inplace=False),

		#     nn.ReflectionPad2d(1),
		#     nn.ConvTranspose2d(256, 256, kernel_size=3, padding=0),
		#     nn.ReLU(inplace=False),

		#     nn.ReflectionPad2d(1),
		#     nn.ConvTranspose2d(256, 128, kernel_size=3, padding=0),
		#     nn.ReLU(inplace=False),

		#     nn.Upsample(scale_factor=2, mode="nearest"),

		#     # 第三个卷积块（对应VGG-19的第3个卷积块）
		#     nn.ReflectionPad2d(1),
		#     nn.ConvTranspose2d(128, 128, kernel_size=3, padding=0),
		#     nn.ReLU(inplace=True),

		#     nn.ReflectionPad2d(1),
		#     nn.ConvTranspose2d(128, 64, kernel_size=3, padding=0),
		#     nn.ReLU(inplace=True),
		#     nn.Upsample(scale_factor=2, mode="nearest"),

		#     # 第四个卷积块（对应VGG-19的第2个卷积块）
		#     nn.ReflectionPad2d(1),
		#     nn.ConvTranspose2d(64, 64, kernel_size=3, padding=0),
		#     nn.ReLU(inplace=True),

		#     # 第五个卷积块（对应VGG-19的第1个卷积块）
		#     nn.ReflectionPad2d(1),
		#     nn.ConvTranspose2d(64, 3, kernel_size=3, padding=0),  # 上采样
		#     nn.Tanh()  # 输出层使用Tanh激活函数，输出范围[-1, 1]
		# )

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

	mean_y = y.mean(dim=[-2, -1], keepdim=True)
	std_y = y.std(dim=[-2, -1], keepdim=True)
	mean_x = x.mean(dim=[-2, -1], keepdim=True)
	std_x = x.std(dim=[-2, -1], keepdim=True)
	adain = (x - mean_x) / (std_x + 1e-6) * std_y + mean_y

	return adain


alpha = 1  #content
beta = 4  #style

content_feature_args = {20: 1}
f_tmp = alpha / sum(content_feature_args.values())
content_feature_args = {k: v * f_tmp for k, v in content_feature_args.items()}

style_feature_args = {1: 1, 6: 1, 11: 1, 20: 1}
f_tmp = beta / sum(style_feature_args.values())
style_feature_args = {k: v * f_tmp for k, v in style_feature_args.items()}

# 构造预训练的 VGG 编码器
pretrained_vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).eval()
pretrained_vgg_features = pretrained_vgg.features[:(
    max(*list(content_feature_args.keys()), *list(style_feature_args.keys())) +
    1)]

# 构造解码器
decoder = Decoder()
#encoder = Encoder(pretrained_model=pretrained_vgg)

if torch.cuda.is_available():
	device = torch.device("cuda:0")
	print('Running on GPU 0')
else:
	device = torch.device("cpu")
	print('Running on CPU')

pretrained_vgg_features.to(device)


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


transform = transforms.Compose([
    transforms.Resize((512, 512)),  # 调整图像大小
    transforms.RandomCrop((256, 256)),  # 随机裁剪至 (256, 256)
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
                                                          0.225])  # 归一化
])

content_dataset = FlatFolderDataset(root='/root/autodl-tmp/train2017',
                                    transform=transform)
style_dataset = FlatFolderDataset(root='./images/inputs/style',
                                  transform=transform)


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


train_loader = data.DataLoader(ContentStyleDataset(content_dataset,
                                                   style_dataset),
                               batch_size=32,
                               shuffle=True,
                               num_workers=16)


def calc_gram_matrix(x):
	# 1 x C x H x W -> C x HW -> C x C
	_, c, h, w = x.shape
	x = x.view(c, h * w)
	x_t = x.transpose(0, 1)
	return torch.matmul(x, x_t) / (h * w)


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
	return content_feature[20], style_feature


def get_AdaIN(contents, styles, feature=pretrained_vgg_features):
	feature.eval()
	target_dep = 20
	x = contents
	y = styles
	for i in range(target_dep + 1):
		x = feature[i](x)
		y = feature[i](y)

	return AdaIN(x, y)


criterion = nn.MSELoss()


def calcul_loss(contents,
                styles,
                criterion,
                lamda=1e-2,
                feature=pretrained_vgg_features):
	contents_image, contents_feature = get_feature(contents, feature)
	styles_image, styles_feature = get_feature(styles, feature)

	Mix_feature = AdaIN(contents_image, styles_image)
	Mix_image = decoder(Mix_feature)

	AdaIN_content, AdaIN_feature = get_feature(Mix_image)

	Loss_c = criterion(AdaIN_content, contents_image)

	Loss_s = torch.tensor(0.0).to(device)

	for i in style_feature_args.keys():
		mean_style = torch.mean(styles_feature[i], dim=[-2, -1], keepdim=True)
		mean_AdaIN = torch.mean(AdaIN_feature[i], dim=[-2, -1], keepdim=True)
		std_style = torch.std(styles_feature[i], dim=[-2, -1], keepdim=True)
		std_AdaIN = torch.std(AdaIN_feature[i], dim=[-2, -1], keepdim=True)
		Loss_s = Loss_s + style_feature_args[i] * criterion(
		    mean_style, mean_AdaIN) + style_feature_args[i] * criterion(
		        std_style, std_AdaIN)

	Loss = Loss_c + lamda * Loss_s
	return Loss


optimizer = torch.optim.AdamW(decoder.parameters(), lr=1e-3, weight_decay=0.01)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
#                                                                  T_0=21,
#                                                                  T_mult=2)


def train():
	writer = SummaryWriter('./loss_curve')
	num_epochs = 10
	decoder.train()  # 设置解码器为训练模式
	decoder.to(device)
	for epoch in range(num_epochs):
		total_loss = 0
		total_tmp_loss = 0

		for batch_idx, (content_images,
		                style_images) in enumerate(tqdm(train_loader)):

			optimizer.zero_grad()
			content_images = content_images.to(device)
			style_images = style_images.to(device)

			# 计算损失
			loss = calcul_loss(content_images, style_images, criterion)
			total_loss += loss.item()

			# 反向传播和优化
			loss.backward()
			optimizer.step()
			# scheduler.step()

			# 打印日志
			if batch_idx % 100 == 0:
				loss_delta = total_loss - total_tmp_loss
				total_tmp_loss = total_loss
				print(
				    f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss_delta/100:.4f}'
				)
				writer.add_scalar('Loss', {'train': loss_delta.item()},
				                  epoch * len(train_loader) + batch_idx)

		# total_loss = total_loss / len(train_loader)
		# writer.add_scalars('loss', {'train': total_loss}, epoch)
		# 保存模型
		# if epoch % 0 == 9:
		timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
		checkpoints_dir = './checkpoints'
		torch.save(
		    decoder.state_dict(),
		    f'{checkpoints_dir}/decoder_epoch_{epoch+1}_{timestamp}.pth')


if __name__ == '__main__':
	train()
