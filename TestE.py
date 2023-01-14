import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.io import read_image
import torchvision.transforms as transforms
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
print("loaded successfully")
print(model)
   
file = read_image("Data/Minus/Minus1.png")
print(file.dtype,file.shape)
file=file.type(torch.float)
print(file.dtype)
def prediction():
    Transf=transforms.Compose([transforms.Grayscale(num_output_channels=1)])
    #print(Transf)

    Tens=Transf(file).unsqueeze(0)
    #print(Tens,Tens.shape)
    #print(Tens.dtype)
    
    #print(Tens,Tens.shape)
    outputs = model(Tens)
    # max returns (value ,index)
    _, predicted = torch.max(outputs.data, 1)
    print(predicted)
prediction()


