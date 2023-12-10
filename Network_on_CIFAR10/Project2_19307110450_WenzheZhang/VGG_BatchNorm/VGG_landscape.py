from matplotlib import pyplot as plt
from torch import nn
import numpy as np
import torch
import random
from tqdm.notebook import tqdm as tqdm


from models.vgg import VGG_A
from models.vgg import VGG_A_BN
from data.loaders import get_cifar_loader

def get_accuracy(model, loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            _, y_pred = torch.max(out.data, 1)
            total += y.size(0)
            correct += (y_pred == y).sum()
    return (100 * float(correct) / total)
  
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
def train(model, optimizer, criterion, train_loader, val_loader, epochs_n=100, best_model_path=None):
    model.to(device)
    losses_list = []

    for epoch in tqdm(range(epochs_n), unit='epoch'):
        model.train()
        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            loss.backward()
            optimizer.step()
            losses_list.append(loss.item())

    return losses_list

device=torch.device('cuda')
set_random_seeds(seed_value=2023, device=device)
lrs = [2e-3, 1e-3, 5e-4, 1e-4]
criterion = nn.CrossEntropyLoss()
train_loader = get_cifar_loader(train=True)
val_loader = get_cifar_loader(train=False)

all_losses, all_losses_bn = [], []

for lr in lrs:
    model = VGG_A()
    model_bn = VGG_A_BN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer_bn = torch.optim.Adam(model_bn.parameters(), lr=lr)
    print(f"vgg: lr {lr}")
    losses = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=20)
    print(f"vgg_bn: lr {lr}")
    losses_BN = train(model_bn, optimizer_bn, criterion, train_loader, val_loader, epochs_n=20)
    all_losses.append(losses)
    all_losses_bn.append(losses_BN)

    
a_losses = np.array(all_losses).T.tolist()
min_losses = [min(i) for i in a_losses]
max_losses = [max(i) for i in a_losses]
a_losses_bn = np.array(all_losses_bn).T.tolist()
min_losses_bn = [min(i) for i in a_losses_bn]
max_losses_bn = [max(i) for i in a_losses_bn]

plt.rcParams["figure.figsize"] = (20,10)
plt.plot(range(len(min_losses)-50), min_losses[50:], c='b',linewidth=0.1)
plt.plot(range(len(min_losses)-50), max_losses[50:], c='b',linewidth=0.1)
plt.plot(range(len(min_losses)-50), min_losses_bn[50:], c='r',linewidth=0.1)
plt.plot(range(len(min_losses)-50), max_losses_bn[50:], c='r',linewidth=0.1)
plt.fill_between(range(len(min_losses)-50), max_losses[50:], min_losses[50:], facecolor='dodgerblue', alpha=0.3, label='Standard VGG')
plt.fill_between(range(len(min_losses)-50), max_losses_bn[50:], min_losses_bn[50:], facecolor='orangered', alpha=0.3, label='Standard VGG + Batch')
plt.xlabel('steps')
plt.ylabel('loss')
plt.title('Loss landscape')
plt.legend()
plt.show()


