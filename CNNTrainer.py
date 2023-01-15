print("ok")

import torch
import torch.nn.functional as F  # Parameterless activation functions
from torchvision.datasets import ImageFolder  
import torchvision.transforms as transforms  
from torch import optim  
from torch import nn  
from torch.utils.data import (DataLoader,) 


class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=13):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x



in_channels = 1
num_classes = 13
learning_rate = 3e-4 # karpathy's constant
batch_size = 64
num_epochs = 0
Save=False
Transf=transforms.Compose([transforms.Resize((28,28)),transforms.ToTensor(),transforms.Grayscale(num_output_channels=1)])


# Load Data
dataset = ImageFolder("Data/",transform = Transf)

train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


# Initialize network
model = CNN(in_channels=in_channels, num_classes=num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    print("Epoch:"+str(int(epoch)))
    for batch_idx, (data, targets) in enumerate((train_loader)):
        # forward
        outputs = model(data)
        loss = criterion(outputs, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()
if Save:
    torch.save(model.state_dict(),"TrainedModel.pth")
    print("Saved Successfully")

# Check accuracy on training & test to see how good our model is 
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x
            y = y

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples


print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")

