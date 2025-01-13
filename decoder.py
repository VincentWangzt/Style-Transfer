import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# 定义激活函数，类似于Lua代码中的 `decoderActivation`
def decoder_activation():
    return nn.ReLU(inplace=False)

# 构造解码器
class Decoder(nn.Module):
    def __init__(self, encoder):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential()
        

        channels = []  # 用于存储每层的通道数
        
        # 遍历编码器以提取通道信息
        for layer in reversed(list(encoder.children())):
            if isinstance(layer, nn.Conv2d):
                channels.append(layer.out_channels)

        # 反转 encoder 的层，并构建解码器
        for layer in reversed(list(encoder.children())):
            if isinstance(layer, nn.Conv2d):
                # 转置卷积：交换输入/输出通道
                n_input_plane = layer.out_channels
                n_output_plane = layer.in_channels
                self.decoder.add_module(
                    "reflection_padding",
                    nn.ReflectionPad2d(1)  # 反射填充，避免边界伪影
                )
                self.decoder.add_module(
                    "transposed_conv",
                    nn.ConvTranspose2d(in_channels= n_input_plane, out_channels= n_output_plane, kernel_size=3, stride=1, padding=0)
                )
                self.decoder.add_module("activation", decoder_activation())
            elif isinstance(layer, nn.MaxPool2d):
                # 使用最近邻上采样替代最大池化
                self.decoder.add_module("upsampling", nn.Upsample(scale_factor=2, mode="nearest"))

    def initialize_weights(self, encoder):
        """使用 encoder 参数初始化 decoder"""
        encoder_layers = [layer for layer in encoder.children() if isinstance(layer, nn.Conv2d)]
        decoder_layers = [layer for layer in self.decoder if isinstance(layer, nn.ConvTranspose2d)]
        
        # 遍历对应层，拷贝权重和偏置
        for enc_layer, dec_layer in zip(reversed(encoder_layers), decoder_layers):
            dec_layer.weight.data = enc_layer.weight.data.permute(1, 0, 2, 3)
            dec_layer.bias.data = enc_layer.bias.data.clone()
    
    def forward(self, x):
        return self.decoder(x)

# 构造编码器
class Encoder(nn.Module):
    def __init__(self, pretrained_model, target_layer="relu4_1"):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential()
        layer_mapping = {
            "conv1_1": 0, "relu1_1": 1,
            "conv1_2": 2, "relu1_2": 3,
            "pool1": 4,
            "conv2_1": 5, "relu2_1": 6,
            "conv2_2": 7, "relu2_2": 8,
            "pool2": 9,
            "conv3_1": 10, "relu3_1": 11,
            "conv3_2": 12, "relu3_2": 13,
            "conv3_3": 14, "relu3_3": 15,
            "conv3_4": 16, "relu3_4": 17,
            "pool3": 18,
            "conv4_1": 19, "relu4_1": 20,
        }
        for name, index in layer_mapping.items():
            self.encoder.add_module(name, pretrained_model.features[index])
            if name == target_layer:
                break
    
    def forward(self, x):
        return self.encoder(x)

# 加载预训练 VGG-19 模型
pretrained_vgg = models.vgg19(weights =models.VGG19_Weights.DEFAULT)
pretrained_vgg_features = pretrained_vgg.features.eval()

# 构造 encoder 和 decoder
encoder = Encoder(pretrained_model=pretrained_vgg, target_layer="relu4_1")
decoder = Decoder(encoder.encoder)

decoder.initialize_weights(encoder.encoder)

# 测试网络
input_image = torch.randn(1, 3, 256, 256)  # 假设输入图像为 256x256
encoded = encoder(input_image)
decoded = decoder(encoded)


print("Input shape:", input_image.shape)
print("Encoded shape:", encoded.shape)
print("Decoded shape:", decoded.shape)
