import torch
from utlis import Decoder, get_feature, AdaIN
from PIL import Image
from torchvision import transforms, models
import matplotlib.pyplot as plt

if torch.cuda.is_available():
	device = torch.device("cuda:0")
	print('Running on GPU 0')
else:
	device = torch.device("cpu")
	print('Running on CPU')

device = torch.device("cpu")

decoder = Decoder()
decoder.load_state_dict({
    k[7:]: v
    for k, v in torch.load(
        '/root/Style-Transfer/checkpoints/decoder_epoch_10_2025-01-13_23-01-03.pth',
        weights_only=True,
        map_location=device).items()
})
decoder = decoder.to(device)
decoder.eval()

alpha = 1  #content
beta = 4e2  #style

content_feature_args = {20: 1}
f_tmp = alpha / sum(content_feature_args.values())
content_feature_args = {k: v * f_tmp for k, v in content_feature_args.items()}

style_feature_args = {1: 1, 6: 1, 11: 1, 20: 1}
f_tmp = beta / sum(style_feature_args.values())
style_feature_args = {k: v * f_tmp for k, v in style_feature_args.items()}

pretrained_vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).eval()
pretrained_vgg_features = pretrained_vgg.features[:(
    max(*list(content_feature_args.keys()), *list(style_feature_args.keys())) +
    1)]

pretrained_vgg_features = pretrained_vgg_features.to(device)

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
	content_image, content_feature = get_feature(content_image,
	                                             pretrained_vgg_features)
	style_image, style_feature = get_feature(style_image,
	                                         pretrained_vgg_features)
	Adain_feature = AdaIN(content_image, style_image)
	Adain_image = decoder(Adain_feature)
	Adain_image = deprocess_image(Adain_image)
	# print(Adain_image)
	Adain_image.save('./tmp.jpg')
	# plt.imshow(Adain_image)
	# plt.show()
