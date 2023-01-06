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


def segmentImageColor(filepath,resultFilepath):
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

    #Result image
    newImage = im.fromarray(newData).convert("L")
    newImage = newImage.save(resultFilepath)

    return newData


def segmentImageSymbols(data):
    # Storing image data
    height,width = data.shape
    print(height,width)

    # Getting possitions for pixels in symbols
    yMap,xMap = np.where(data == 255)
    
    
    height = yMap[np.argmax(yMap)] - yMap[np.argmin(yMap)]
    width = xMap[np.argmax(xMap)] - xMap[np.argmin(xMap)]
    print(height,width)

    data = data[yMap[np.argmin(yMap)]:]

    newImage = im.fromarray(data).convert('L')
    newImage = newImage.save('test.png')



    return





# Test area
#for i in range(10):
    #segmentImage('testdata/'+str(i)+'.png','SegmentedResults/'+str(i)+'.png')
data = segmentImageColor('testdata/5plus3.png','SegmentedColors/5plus3.png')
segmentImageSymbols(data)





'''
# Testing 1-, 2- and 3-dimensional data
for i in range(1,4):
    data = np.random.rand(50, i)
    data = np.append(data, data+1, axis=0)
    cluster_centers = k_means(data, 2)
    print(cluster_centers)
'''