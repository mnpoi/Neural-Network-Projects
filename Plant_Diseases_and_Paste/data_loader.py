import os
import torch
import xml.etree.ElementTree as ET
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset,random_split
import matplotlib.pyplot as plt
from torchvision.models import resnet152
from tqdm import tqdm
from torch import nn, optim

class CustomVocDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}  # 初始化空字典
        self.current_idx = 0    # 当前未分配的索引

        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path):
                xml_file = os.path.join(root_dir, subdir + '.xml')
                if os.path.exists(xml_file):
                    tree = ET.parse(xml_file)
                    root = tree.getroot()
                    class_tag = root.find('.//class')
                    if class_tag is not None:
                        label = class_tag.text
                        # 如果新类别，添加到映射
                        if label not in self.class_to_idx:
                            self.class_to_idx[label] = self.current_idx
                            self.current_idx += 1
                        # 为图像添加样本
                        for file in os.listdir(subdir_path):
                            if file.endswith('.jpg'):
                                image_path = os.path.join(subdir_path, file)
                                self.samples.append((image_path, label))
                else:
                    print(f"XML file not found: {xml_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label_str = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # 使用类别映射将字符串标签转换为整数
        label = self.class_to_idx[label_str]

        # 将标签转换为张量
        label = torch.tensor(label, dtype=torch.long)

        return image, label

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(45),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_datasets(dataset_folders, transform):
    datasets = [CustomVocDataset(root_dir=folder, transform=transform) for folder in dataset_folders]
    combined_dataset = ConcatDataset(datasets)
    return combined_dataset

# 创建数据加载器
def get_data_loader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)