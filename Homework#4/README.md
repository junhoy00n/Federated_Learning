## [Review]
Federated Learning 실습을 위해 다양한 툴이 제공 되고 있음

자주 사용되는 Flower 이외에 다른 툴인 PySyft를 이용한 연합학습 실습을 진행하였으나, 환경 구축 시 버전 차이로 인한 오류 발생이 심함

실습을 위해 가상환경을 사용한다면 gpu 사용은 제한적이였으나, 다음 단계를 따를 시 cpu를 활용한 수행이 가능함

#### 가상환경 구축 방법:

    conda create -n pysyft python==3.8
    conda activate pysyft
    
    pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html
    pip install syft==0.2.6
## [python federated_learning.py]
#### 1. 데이터 소유자(Sever & Client) 생성
    import syft as sy
    server = sy.TorchHook(torch)
    client1 = sy.VirtualWorker(server, id="client1")
    client2 = sy.VirtualWorker(server, id="client2")
#### 2. DataLoader client에게 전송
<img src='https://user-images.githubusercontent.com/59612454/192973414-853c5a0f-8692-49db-b18e-201d1a148556.png' width='50%'></img>
#### 3. Model 생성
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
#### 4. Model Client로 사본 전송
![image](https://user-images.githubusercontent.com/59612454/192975987-dff89c55-4b9c-470c-9c1d-168ff0f6707f.png)

#### 5. 업데이트된 Model Server로 전송
![image](https://user-images.githubusercontent.com/59612454/192975929-4310c13b-9bef-4d06-86d1-9a33ea063b53.png)

#### 6. 모델 학습
    model = SimpleCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, federated_train_loader, optimizer, epoch)
        test(model, device, test_loader)
<img src='https://user-images.githubusercontent.com/59612454/192979104-182ae8c3-eb01-4d04-9fd3-1d559baee8e9.png' width='70%'></img>
