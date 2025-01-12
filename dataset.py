import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
#from pycocotools.coco import COCO

# 定义预处理步骤
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # 调整图像大小
    transforms.RandomCrop((256, 256)),  # 随机裁剪至 (256, 256)
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

# 创建数据集实例
train_dataset = datasets.CocoDetection(
    root='D:/DataSet/train2014/train2014',  # COCO训练集图像路径
    annFile='D:/DataSet/annotations_trainval2014/annotations',  # COCO训练集注释文件路径
    transform=transform
)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

# 测试数据加载器
for images, targets in train_loader:
    print(images.shape)  # 输出图像的形状
    print(targets)  # 输出标注信息
    break