# For testing validation data and finding network recognition accuracy on every symbol



import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from math import sqrt

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

model=CNN(in_channels=1, num_classes=13)
model.load_state_dict(torch.load("TrainedModel.pth"))
model.eval()





# CNN tager en (28x28) matrix og outputter et enkelt symbol: (0-9) eller (+, -, *)
def CNN(data):
    TensorTran=transforms.ToTensor()
    file=np.array(data)
    file=TensorTran(file)
    file=file.type(torch.float)
    Transf=transforms.Compose([transforms.Grayscale(num_output_channels=1)])
    Tens=Transf(file).unsqueeze(0)
    outputs = model(Tens)
    _, predicted = torch.max(outputs.data, 1)


    return outputs.data[0]
    


correct = []

for i in range(10):
    files = os.listdir('valid/'+str(i))
    correct.append([])
    for u in range(len(files)):
        correct[i].append(0)
        correct[i][u] = float(CNN(np.asarray(Image.open('valid/'+str(i)+'/'+files[u])))[i])
    


symbols = ['Minus', 'Multiplication', 'Plus']
for i in range(3):
    files = os.listdir('valid/'+symbols[i])
    correct.append([])
    for u in range(len(files)):
        correct[i+10].append(0)
        correct[i+10][u] = float(CNN(np.asarray(Image.open('valid/'+symbols[i]+'/'+files[u])))[i+10])


mean = []
for i in range(13):
    mean.append(0)
    for u in range(len(correct[i])):
        mean[i] += correct[i][u]
    mean[i] /= len(correct[i])


varians = []

for i in range(13):
    varians.append(0)
    for u in range(len(correct[i])):
        varians[i] += (correct[i][u]-mean[i])**2
    varians[i] /= len(correct[i])


resultup = []
resultdown = []

for i in range(13):
    resultup.append(0)
    resultdown.append(0)

    error = 1.96*sqrt(varians[i])

    resultup[i] = mean[i] + error
    resultdown[i] = mean[i] - error

print(resultup)
print(resultdown)