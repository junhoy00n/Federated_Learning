import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import syft as sy
server = sy.TorchHook(torch)
client1 = sy.VirtualWorker(server, id="client1")
client2 = sy.VirtualWorker(server, id="client2")

class Arguments():
    def __init__(self):
        self.batch_size = 64
        self.test_batch_size = 1000
        self.epochs = 1
        self.lr = 0.01
        self.momentum = 0.5
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 10
        self.save_model = False

args = Arguments()

use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

federated_train_loader = sy.FederatedDataLoader(
    datasets.MNIST('data/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
    .federate((client1, client2)),
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data/', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

class SimpleCNN(nn.Module):
    def __init__(self, conv1_channels=64, conv2_channels=128, linear1_size=256, linear2_size=128, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=conv1_channels, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=conv1_channels, out_channels=conv2_channels, kernel_size=3, stride=1, padding=1)

        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(in_features=conv2_channels*7*7, out_features=linear1_size)
        self.fc2 = nn.Linear(in_features=linear1_size, out_features=linear2_size)
        self.fc3 = nn.Linear(in_features=linear2_size, out_features=10)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool(out)

        out = self.flat(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.drop(out)

        out = self.fc2(out)
        out = self.relu(out)
        out = self.drop(out)

        out = self.fc3(out)

        return out

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(federated_train_loader):
        model.send(data.location)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        model.get()
        if batch_idx % args.log_interval == 0:
            loss = loss.get()
            print(f'Train Epoch: {epoch} [{batch_idx * args.batch_size}/{len(train_loader) * args.batch_size} ({100. * batch_idx / len(train_loader)}%)]\tLoss: {loss.item()}')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(f'Test set: Average loss: {test_loss}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset)}%)')

model = SimpleCNN().to(device)
optimizer = optim.SGD(model.parameters(), lr=args.lr)

for epoch in range(1, args.epochs + 1):
    train(args, model, device, federated_train_loader, optimizer, epoch)
    train(model, device, test_loader)