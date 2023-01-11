import numpy as np
import cv2
from PIL import Image as im
from PIL import ImageOps
from imageSegmentation import segmentImageSymbols as segmentImage



# image segmentation af camera feed som returnere et (N,28,28) array, hvor N er antal symboler
def getSymbolPixelArray(frame):
    data = segmentImage(frame)

    return data



# funktion som taler til CNN og returnere string med symboler
def getSymbols():
    # Get processed data from image
    symbolData = getSymbolPixelArray('billede')

    # The string to contain handwritten math
    MathString = ''
    for i in range(symbolData.shape[0]):
        MathString += str(CNN(symbolData[i]))
    
    return MathString


# CNN tager en (28x28) matrix og outputter et enkelt symbol: (0-9) eller (+, -, *)
def CNN(data):
    # taler til CNN
    pass



# Live camera feed hvor billede skal 'fodres' til imageSegmentation og så CNN
def camera():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    frame = np.asarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    # rå billede
    testbillede = im.fromarray(frame).convert('L')
    testbillede = testbillede.save('hej0.png')


    #img = ImageOps.grayscale(im.open('testdata/5plus3.png'))
    #frame = np.asarray(img)

    data = getSymbolPixelArray(frame)
    if data == 'error':
        print('error. looping again')
        return
    
    for i in range(data.shape[0]):
        img = im.fromarray(data[i]).convert('L')
        img = img.save(str(i)+'.png')
    




# logik som tager en string fra funktion som 'taler' med CNN. beregner simpel matematik.




#MAIN
while True:
    camera()