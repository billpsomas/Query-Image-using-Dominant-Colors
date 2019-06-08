import numpy as np 
import matplotlib
import pandas as pd
import itertools
import os
import glob
from PIL import Image
from scipy.cluster.vq import whiten
from scipy.cluster.vq import kmeans
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from IPython.display import display

starting_directory = os.getcwd()
os.chdir(starting_directory + r'/' + 'graffiti_madrid/')
directory = os.getcwd()

graffiti_titles = glob.glob("*.jpg")

graffiti_images_list = list([])
clusterings = list([])

for i in range(len(graffiti_titles)):
    #Define the full directory of each image
    graffiti_directory = directory + r'/' + graffiti_titles[i]
    #Read each image
    graffiti_image = matplotlib.image.imread(graffiti_directory)
    #Append each image in a list
    graffiti_images_list.append(graffiti_image)
    
    #Define 3 lists to which the RGB channels will be appended
    r = list([])
    g = list([])
    b = list([])
    
    #Append the RGB channels of each image in the respective list
    for line in graffiti_image:
        for pixel in line:
             temp_r, temp_g, temp_b = pixel
             r.append(temp_r)
             g.append(temp_g)
             b.append(temp_b)
    
    #Plot each image of the dataset
    plt.imshow(graffiti_image)
    plt.title(graffiti_titles[i])
    plt.show
    
    #3D plot of the RGB channels of every image. 
    #This sometimes is very helpful in order to find out the number k of the dominant colors.
    #This number k is the number of clusters that will be used later in k-means.
    fig = plt.figure()
    fig.suptitle('3D Plot of the RGB channels of ' + graffiti_titles[i])
    ax = Axes3D(fig)
    ax.scatter(r, g, b)
    plt.show()
    
    #Create a DataFrame that contains the RGB channels of every image and also whitens them.
    #The whitening is used as a preprocessing step for the k-means clustering algorithm.
    df = pd.DataFrame({'red': r,'blue': b,'green': g})
    df['scaled_red'] = whiten(df['red'])
    df['scaled_blue'] = whiten(df['blue'])
    df['scaled_green'] = whiten(df['green'])
    
    #Run the k-means algorithm for the scaled RGB channels of each image, find the centroids and store them in a list.
    #The number k can be choosed randomly or not. In this example k =3. 
    #Don't hesitate to play with it and see what you get when changing it.
    colors = []
    k = 3
    centroids, distortion = kmeans(df[['scaled_red', 'scaled_green', 'scaled_blue']], k)
    print('The centroids of the clusters of', graffiti_titles[i], 'are:\n', centroids, '\n')
    clusterings.append(centroids)
    
    #Plot the k most dominant colors the algorithm found.
    r_std, g_std, b_std = df[['red', 'green', 'blue']].std()
    for centroid in centroids:
        scaled_r, scaled_g, scaled_b = centroid
        colors.append((
                scaled_r * r_std / 255,
                scaled_g * g_std / 255,
                scaled_b * b_std / 255))
    plt.imshow([colors])
    plt.title('The most dominant colors of ' + graffiti_titles[i])
    plt.show()