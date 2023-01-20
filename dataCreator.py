# For creating new data Using a camera
from PIL import Image
import numpy as np
import cv2
from imageSegmentation import segmentImageSymbols as segmentImage



def getFrame():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    frame = np.asarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    return frame




symbols = {
    'Minus': 20,
    'Multiplication': 20
}

for symbol in symbols:
    for i in range(symbols[symbol]):
        input('Enter for next image')
        data = segmentImage(getFrame())
        image = Image.fromarray(data[1]).convert('L')
        image = image.save('valid/'+symbol+'/'+symbol+str(i)+'.png')