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








def segmentImageColor(filepath):
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








def segmentImageSymbols(filepath):

    # getting data for color segmented image
    data = segmentImageColor(filepath)

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
    group = 1
    for i in range(1, np.unique(groupMap).size):
        if groupMap[groupMap == i].size < 250:
            groupMap[groupMap == i] = 0
        else:
            groupMap[groupMap == i] = group
            group +=1


    # Creating empry 3D array to hold 2D data from symbols
    newData = np.zeros((np.unique(groupMap).size-1,28,28))

    scale = 24/groupMap.shape[0]
    for i in range(1, np.unique(groupMap).size):
        yMap,xMap = np.where(groupMap == i)
        try:
            # Dummy array that only includes a single group
            dummyArray = groupMap[yMap[np.argmin(yMap)]:yMap[np.argmax(yMap)],xMap[np.argmin(xMap)]:xMap[np.argmax(xMap)]]
            dummyArray[dummyArray != i] = 0
            dummyArray[dummyArray == i] = 255

            # Array tom image, downscale and back to array
            dummyimg = im.fromarray(dummyArray).convert('L')
            dummyimg = dummyimg.resize((int(dummyArray.shape[1]*scale),int(dummyArray.shape[0]*scale)))
            dummyArray = np.asarray(dummyimg)

            # getting margin for symbol
            margin = [int((28-dummyArray.shape[0])/2),int((28-dummyArray.shape[1])/2)]
            end1,end2 = 27,27
            if dummyArray.shape[0]%2 == 0:
                end1 += 1
            if dummyArray.shape[1]%2 == 0:
                end2 += 1

            # Filling data array with symbol data in correct possition according to margin
            newData[i-1,margin[0]:end1-margin[0],margin[1]:end2-margin[1]] = dummyArray

        except:
            pass

    return newData

