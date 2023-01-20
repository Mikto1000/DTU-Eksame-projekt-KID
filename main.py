#Import CNN dependencies and Current Model
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

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



import numpy as np
from PIL import Image as im
import cv2
from PIL import ImageOps
from imageSegmentation import segmentImageSymbols as segmentImage



# image segmentation af camera feed som returnere et (N,28,28) array, hvor N er antal symboler
# funktion som taler til CNN og returnere string med symboler
def getSymbols(frame):

    # Get processed data from image
    symbolData = segmentImage(frame)

    if symbolData == 'error':
        return 'error'

    # The string to contain handwritten math
    MathString = ''

    Sign={0:"0",1:"1",2:"2",3:"3",4:"4",5:"5",6:"6",7:"7",8:"8",9:"9",10:"-",11:"*",12:"+"}
    for i in symbolData:
        MathString=f"{MathString}{Sign[CNN(i)]}"

    return MathString








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
    
    
    return int(predicted[0])








# Live camera feed hvor billede skal 'fodres' til imageSegmentation og så CNN
def getFrame():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    frame = np.asarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    return frame








#MAIN
while True:
    # Get frame and determin mathString with CNN
    frame = getFrame()
    symbols = getSymbols(frame)
    print(symbols)

    if symbols != 'error':

        # Sorting the string into numbers and operators
        numbers = [0]
        mathOperators = []
        counter = 0
        for i in range(len(symbols)):
            # Trying to convert symbol to number
            try:
                if int(symbols[i]) >= 0 or int(symbols[i]) <= 9:
                    numbers[counter] *= 10
                    numbers[counter] += int(symbols[i])
            
            # If symbol is not a number
            except:
                if symbols[i-1] == '*':
                    pass
                elif symbols[i] != '*' or symbols[i+1] != '*':
                    mathOperators.append(symbols[i])
                    numbers.append(0)
                    counter += 1
                else:
                    mathOperators.append(symbols[i]+symbols[i+1])
                    numbers.append(0)
                    counter += 1
        

        # Converting to Numpy array
        mathOperators = np.asarray(mathOperators)
        numbers = np.asarray(numbers)

        # Calculating ** operator and reducing the expression
        for i in np.where(mathOperators == '**'):
            i = np.flip(i)
            for u in i:
                numbers[u] **= numbers[u+1]
                numbers = np.delete(numbers, u+1)
                mathOperators = np.delete(mathOperators, u)

        # Calculating * operator and reducing the expression
        for i in np.where(mathOperators == '*'):
            i = np.flip(i)
            for u in i:
                numbers[u] *= numbers[u+1]
                numbers = np.delete(numbers, u+1)
                mathOperators = np.delete(mathOperators, u)

        # Calculating + and - operator and getting final result
        result = numbers[0]
        for i in range(1, numbers.size):
            if mathOperators[i-1] == '+':
                result += numbers[i]
            else:
                result-= numbers[i]
        
        # FINAL RESULT
        print(result)
    
    else:
        print('No result')
