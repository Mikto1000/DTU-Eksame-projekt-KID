import numpy as np
import cv2
from PIL import Image as im
from PIL import ImageOps
from imageSegmentation import segmentImageSymbols as segmentImage

import datetime



# image segmentation af camera feed som returnere et (N,28,28) array, hvor N er antal symboler
# funktion som taler til CNN og returnere string med symboler
def getSymbols(frame):

    # Get processed data from image
    symbolData = segmentImage(frame)

    # The string to contain handwritten math
    MathString = ''
    #for i in range(symbolData.shape[0]):
        #MathString += str(CNN(symbolData[i]))
    
    MathString = CNN()
    
    return MathString








# CNN tager en (28x28) matrix og outputter et enkelt symbol: (0-9) eller (+, -, *)
def CNN(data=0):
    # taler til CNN
    return "3-53+2**2*3"








# Live camera feed hvor billede skal 'fodres' til imageSegmentation og så CNN
def getFrame():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    frame = np.asarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    #img = ImageOps.grayscale(im.open('testdata/5plus3.png'))
    #frame = np.asarray(img)

    return frame








#MAIN
while True:
    a = datetime.datetime.now()

    # Gat frame and determin mathString with CNN
    frame = getFrame()
    symbols = getSymbols(frame)

    b = datetime.datetime.now()
    print(b-a)

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
