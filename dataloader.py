import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import CocoDetection
import random
#from pycocotools.coco import COCO

class ContentStyleDataset(Dataset):
    def __init__(self, content_root, style_root, transform=None):
        self.content_dataset = datasets.ImageFolder(root=content_root, transform=transform)
        self.style_dataset = datasets.ImageFolder(root=style_root, transform=transform)
        self.transform = transform
        self.style_length = len(self.style_dataset) - 1

    def __len__(self):
        return len(self.content_dataset)  # 数据集的长度由内容图像决定

    def __getitem__(self, index):
        content_image, _ = self.content_dataset[index]
        style_index = random.randint(0, self.style_length)
        style_image, _ = self.style_dataset[style_index]
        return content_image, style_image

# 定义预处理步骤
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # 调整图像大小
    transforms.RandomCrop((256, 256)),  # 随机裁剪至 (256, 256)
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

# 创建自定义数据集实例
dataset = ContentStyleDataset(
    content_root='/root/D/DataSet/train2014',  # COCO训练集图像路径，注意train2014里面还要有一个文件夹来存储所有图片
    style_root='./images/inputs/style/',  # WikiArt数据集图像路径
    transform=transform
)

# 创建数据加载器
train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# 测试数据加载器
if __name__ == '__main__':
    # 创建自定义数据集实例
    dataset = ContentStyleDataset(
        content_root='/root/D/DataSet/train2014',  # COCO训练集图像路径
        style_root='./images/inputs/style/',  # WikiArt数据集图像路径
        transform=transform
    )

    # 创建数据加载器
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # 测试数据加载器
    for batch_idx, (content_images, style_images) in enumerate(train_loader):
        print(f"Batch {batch_idx}")
        print("Content Images Shape:", content_images.shape)  # 输出内容图像的形状
        print("Style Images Shape:", style_images.shape)  # 输出风格图像的形状
        if batch_idx == 2:
            break

































# import os
# import torch
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
# from pycocotools.coco import COCO

# class CocoTrainDataset(datasets.CocoDetection):
#     def __init__(self, root, annFile, transform=None):
#         super(CocoTrainDataset, self).__init__(root, annFile, transform)
#         self.coco = COCO(annFile)

#     def __getitem__(self, index):
#         img_id = self.ids[index]
#         image = self.coco.loadImgs(img_id)[0]
#         img_path = os.path.join(self.root, image['file_name'])
#         image = datasets.folder.default_loader(img_path)
#         if self.transform is not None:
#             image = self.transform(image)
#         return image

# # 定义预处理步骤
# transform = transforms.Compose([
#     transforms.Resize((512, 512)),  # 调整图像大小
#     transforms.RandomCrop((256, 256)),  # 随机裁剪至 (256, 256)
#     transforms.ToTensor(),  # 转换为Tensor
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
# ])
# # 创建数据集实例
# train_dataset = CocoTrainDataset(
#     root='path/to/train2014',  # COCO训练集图像路径
#     annFile='path/to/annotations/instances_train2014.json',  # COCO训练集注释文件路径
#     transform=transform
# )

# # 创建数据加载器
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

# # 测试数据加载器
# for images in train_loader:
#     print(images.shape)  # 输出图像的形状
#     break