import numpy as np
from PIL import Image as im
from PIL import ImageOps

def k_means(data, k, max_iter=100):
    # Initialize the centroids
    centroids = np.array([[0],[255]])
    #centroids = data[np.random.choice(data.shape[0], k, replace=False)] ### old code. maybe still usefull

    
    for _ in range(max_iter):
        # Compute the distance from each point to each centroid
        distances = np.sqrt(((data[:, np.newaxis] - centroids) ** 2).sum(axis=-1))
        
        # Assign each point to the closest centroid
        labels = np.argmin(distances, axis=1)
        
        # Compute the new centroids as the mean of the points assigned to each centroid
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        
        # Check if the centroids have changed
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    return centroids, labels








def segmentImageColor(filepath,resultFilepath=0):
    # Load the image as Grayscale
    image = ImageOps.grayscale(im.open(filepath))

    # Convert image to numpy array and reshaping data to 1D
    data = np.asarray(image).reshape(image.height*image.width,1)

    # Getting means and labels for the pixels
    centroids, labels = k_means(data,2)

    # changing the values of the dark and light averege
    centroids[np.argmax(centroids)] = 0
    centroids[np.argmax(centroids)] = 255

    # mapping new color values for pixels
    newData = np.zeros((image.height*image.width,1))
    for i in range(labels.size):
        newData[i,0] = centroids[labels[i],0]
    newData = newData.reshape(image.height,image.width)

    return newData








def segmentImageSymbols(data,resultFilepath):

    # Getting possitions for pixels in symbols
    yMap,xMap = np.where(data == 255)
    # Cutting image
    data = data[yMap[np.argmin(yMap)]:yMap[np.argmax(yMap)],xMap[np.argmin(xMap)]:xMap[np.argmax(xMap)]]

    # Storing image dimentions in variables
    height,width = data.shape
    # Reshaping data to 1D
    data = data.reshape(height*width,1)

    # Creating empty 1D map for symbol groups
    groupMap = np.zeros((height*width,1))
    # Defining pixel groups horisontally
    group = 1
    for i in range(data.size):
        if i%width == 0:
            group += 1
        if data[i] != 0:
            if groupMap[i-1] == group-1 and group-1 != 0:
                groupMap[i] = groupMap[i-1]
            else:
                groupMap[i] = group
                group += 1
        else:
            groupMap[i] = 0

    # Reshaping groupmap to 2D matrix
    groupMap = groupMap.reshape(height,width)
    # Combining pixelgroups vertically
    for i in range(groupMap.shape[0]):
        for u in range(groupMap.shape[1]):
            if groupMap[i,u] != 0 and groupMap[i-1,u] != 0 and i != 0:
                groupMap[groupMap == groupMap[i,u]] = groupMap[i-1,u]

    # Renaming groups in correct order
    groups = np.delete(np.unique(groupMap), 0).size
    groupMap[:] *= groups

    group = 1
    for i in range(groupMap.shape[1]):
        for u in range(groupMap.shape[0]):
            if groupMap[u,i] != 0 and groupMap[u,i-1] != 0 and i != 0 and groupMap[u,i] >= group:
                groupMap[groupMap == groupMap[u,i]] = group
                group += 1
    
    # Removing groups that are too small
    for i in range(1, np.unique(groupMap).size):
        if groupMap[groupMap == i].size < 250:
            groupMap[groupMap == i] = 0
        else:
            groupMap[groupMap == i] = i *50






    # Creating empry 3D array to hold 2D data from images
    data = np.zeros((np.unique(groupMap).size,28,28))


    for i in range(1, np.unique(groupMap).size):
        print(i)
        yMap,xMap = np.where(groupMap == i*50)
        dummyArray = groupMap[yMap[np.argmin(yMap)]:yMap[np.argmax(yMap)],xMap[np.argmin(xMap)]:xMap[np.argmax(xMap)]]
        dummyimg = im.fromarray(dummyArray).convert('L')
        dummyimg = dummyimg.resize((int((20/height)*width),20))
        img = dummyimg.save(str(i)+'.png')
        dummyArray = np.asarray(dummyimg)
        print(dummyArray)






    # Converting from array to image and resizing
    img = im.fromarray(data[1]).convert('L')
    img = img.resize((int((20/height)*width),20))
    img = img.save(resultFilepath)

    return





# Test area

fileName = '7plus4.png'
data = segmentImageColor('testdata/'+fileName)
segmentImageSymbols(data,'resultData/'+fileName)

