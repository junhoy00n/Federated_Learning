import flwr as fl
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import json
import random

from torch.utils.data import Dataset, DataLoader
from torch.multiprocessing import Process
from tqdm import tqdm
from collections import OrderedDict

from simple_cnn import FeMnistNetwork

# Dataset Class
class FeMnistDataset(Dataset):
    def __init__(self, dataset, transform):
        self.x = dataset['x']
        self.y = dataset['y']
        self.transform = transform

    def __getitem__(self, index):
        input_data = np.array(self.x[index]).reshape(28, 28, 1)
        if self.transform:
            input_data = self.transform(input_data)
        target_data = self.y[index]
        return input_data, target_data

    def __len__(self):
        return len(self.y)

def main(DEVICE):
    # GPU or cpu-only 설정
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # model import
    model = FeMnistNetwork().to(DEVICE)

    def load_data():
        transform = transforms.Compose(
            [transforms.ToTensor()]
        )

        number = random.randint(0, 35)
        if number == 35:
            subject_number = random.randint(0, 96)
        else:
            subject_number = random.randint(0, 99)
        print('number : {}, subject number : {}'.format(number, subject_number))

        with open('dataset/train/all_data_' + str(number) + '_niid_0_keep_0_train_9.json', 'r') as f:
            train_json = json.load(f)

        train_user = train_json['users'][subject_number]
        train_data = train_json['user_data'][train_user]
        trainset = FeMnistDataset(train_data, transform)
        trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

        with open('dataset/test/all_data_' + str(number) + '_niid_0_keep_0_test_9.json', 'r') as f:
            test_json = json.load(f)

        test_user = test_json['users'][subject_number]
        test_data = test_json['user_data'][test_user]
        testset = FeMnistDataset(test_data, transform)
        testloader = DataLoader(testset, batch_size=64)
        return trainloader, testloader

    class CifarClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return [val.cpu().numpy() for _, val in model.state_dict().items()]

        def set_parameters(self, parameters):
            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            trainloader, _ = load_data()
            train(model, trainloader, epochs=20)
            return self.get_parameters(config={}), len(trainloader.dataset), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            _, testloader = load_data()
            loss, accuracy = test(model, testloader)
            return float(loss), len(testloader.dataset), {'accuracy': accuracy}

    def train(model, trainloader, epochs):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0003)
        model.train()

        for _ in range(epochs):
            for images, labels in trainloader:
                images, labels = images.to(DEVICE).float(), labels.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()

    def test(model, testloader):
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0

        with torch.no_grad():
            for images, labels in tqdm(testloader):
                outputs = model(images.to(DEVICE))
                labels = labels.to(DEVICE)
                loss += criterion(outputs, labels).item()
                total += labels.size(0)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        return loss / len(testloader.dataset), correct / total

    def test(model, testloader):
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        model.eval()
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(DEVICE).float(), data[1].to(DEVICE)
                outputs = model(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return loss, accuracy

    # Start client
    fl.client.start_numpy_client(server_address='127.0.0.1:8080', client=CifarClient())

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    list = [1, 2, 3]

    ps = []
    for i in list:
        p =Process(target=main, args=(i, ))
        ps.append(p)
        p.start()

    for p in ps:
        p.join()