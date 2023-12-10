import torch
from tqdm import tqdm
from resnet_model import get_resnet_model
from data_loader import get_dataset, get_data_loader
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import ConcatDataset, random_split
from SAM import SAMSGD as _SAMSGD
from functools import partial


def SAM(net_parameters, lr=1e-3, momentum=0.0, dampening=0.0,
        weight_decay=0.0, nesterov=False, rho=0.05):
    return _SAMSGD(net_parameters, lr=lr, momentum=momentum,
                   dampening=dampening, weight_decay=weight_decay,
                   nesterov=nesterov, rho=rho)


def main():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(45),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset1 = get_dataset('/home/data/2722', transform=val_transform)
    dataset2 = get_dataset('/home/data/2724', transform=val_transform)
    dataset3 = get_dataset('/home/data/2902', transform=val_transform)

    combined_dataset = ConcatDataset([dataset1, dataset2, dataset3])

    # Calculate sizes for train and val splits
    total_size = len(combined_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    # Split datasets
    train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size])

    # Apply validation transform to val_dataset
    train_dataset.dataset.transform = train_transform

    # Define data loaders
    train_loader = get_data_loader(train_dataset, batch_size=32)
    val_loader = get_data_loader(val_dataset, batch_size=32)

    # 获取模型
    net = get_resnet_model(num_classes=144)
    net.to(device)

    # 定义损失函数和优化器
    loss_function = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(net.parameters(), lr=0.0001)
    optimizer = SAM(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4, rho=5e-2)


    # 训练过程
    best_acc = 0.0
    save_path = '/project/train/models'
    for epoch in range(15):
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for step, data in enumerate(tqdm(train_loader), start=0):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            def closure():
                optimizer.zero_grad()
                outputs = net(images)
                loss = loss_function(outputs, labels)
                loss.backward()
                return loss

            loss = closure()  # Call closure to compute the loss and perform backprop
            optimizer.step(closure)  # Pass the closure to the optimizer's step method

            logits = net(images)  # Calculate logits again for accuracy calculation
            running_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # 计算并打印这个epoch的平均loss和准确率
        print(f"Epoch {epoch + 1}/{15}, Loss: {running_loss / len(train_loader):.4f}, "
              f"Accuracy: {100 * correct / total:.2f}%")

        net.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                loss = loss_function(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        print(f"Validation Loss: {val_loss / len(val_loader):.4f}, "
              f"Accuracy: {100 * val_correct / val_total:.2f}%")


if __name__ == '__main__':
    main()