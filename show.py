import torch
from utlis import Decoder, get_feature, AdaIN
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

if torch.cuda.is_available():
	device = torch.device("cuda:0")
	print('Running on GPU 0')
else:
	device = torch.device("cpu")
	print('Running on CPU')

decoder = Decoder()
decoder.load_state_dict(
    torch.load('./decoder_epoch_3.pth', weights_only=True,
               map_location=device))
decoder = decoder.to(device)
decoder.eval()

content_dir = './images/inputs/content'
style_dir = './images/inputs/style'

content_name = 'gatys-original.jpg'
style_name = 'starry_night.jpg'

content_image = Image.open(f'{content_dir}/{content_name}')
style_image = Image.open(f'{style_dir}/{style_name}')

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图像大小
    # transforms.RandomCrop((256, 256)),  # 随机裁剪至 (256, 256)
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
                                                          0.225])  # 归一化
])

content_image = transform(content_image).unsqueeze(0).to(device)
style_image = transform(style_image).unsqueeze(0).to(device)

mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])


def deprocess_image(image):
	image = image.clone().detach().squeeze().to(std.device)
	# image = transforms.Normalize(-mean / std, 1 / std)(image)
	image = torch.clamp(image.permute(1, 2, 0) * std + mean, 0, 1)
	image = transforms.ToPILImage()(image.permute(2, 0, 1))
	return image


with torch.no_grad():
	content_image, content_feature = get_feature(content_image)
	style_image, style_feature = get_feature(style_image)
	Adain_feature = AdaIN(content_image, style_image)
	Adain_image = decoder(Adain_feature)
	Adain_image = deprocess_image(Adain_image)
	# print(Adain_image)
	Adain_image.save('./tmp.jpg')
	# plt.imshow(Adain_image)
	# plt.show()
