import glob
import time
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
from VehicleDetect.FeatureExtraction import get_hog_features as get_hog_features
# Read in our vehicles and non-vehicles
car_images = glob.iglob('dataset/vehicles/**/*.png',recursive=True)
notcar_images = glob.iglob('dataset/non-vehicles/**/*.png',recursive=True)
cars = []
notcars = []

for image in car_images:
	cars.append(image)

for image in notcar_images:
	notcars.append(image)

#Define the colorspace
color_space = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
# Define HOG parameters
spatial_feat=True 
hist_feat=True
hog_feat=True
orient = 9
pix_per_cell = 8
cell_per_block = 2
spatial_size = (32,32)
hist_bins = 32	

# Generate a random index to look at a car image
idx = np.random.randint(0, len(cars))
# Read in the image
car_image = mpimg.imread(cars[idx])
#transfer to YUV colorspace
car_yuv_image = cv2.cvtColor(car_image, cv2.COLOR_RGB2YUV)
# Call our function with vis=True to see an image output
features1, car_hog_image1 = get_hog_features(car_yuv_image[:,:,0], orient, 
                        pix_per_cell, cell_per_block, 
                        vis=True, feature_vec=False)
features2, car_hog_image2 = get_hog_features(car_yuv_image[:,:,1], orient, 
                        pix_per_cell, cell_per_block, 
                        vis=True, feature_vec=False)
features3, car_hog_image3 = get_hog_features(car_yuv_image[:,:,2], orient, 
                        pix_per_cell, cell_per_block, 
                        vis=True, feature_vec=False)

# Generate a random index to look at a car image
idx = np.random.randint(0, len(notcars))
# Read in the image
notcar_image = mpimg.imread(notcars[idx])
#transfer to YUV colorspace
notcar_yuv_image = cv2.cvtColor(notcar_image, cv2.COLOR_RGB2YUV)
# Call our function with vis=True to see an image output
features1, notcar_hog_image1 = get_hog_features(notcar_yuv_image[:,:,0], orient, 
                        pix_per_cell, cell_per_block, 
                        vis=True, feature_vec=False)
features2, notcar_hog_image2 = get_hog_features(notcar_yuv_image[:,:,1], orient, 
                        pix_per_cell, cell_per_block, 
                        vis=True, feature_vec=False)
features3, notcar_hog_image3 = get_hog_features(notcar_yuv_image[:,:,2], orient, 
                        pix_per_cell, cell_per_block, 
                        vis=True, feature_vec=False)

#notcar_hist_features = color_hist(notcar_yuv_image, nbins=32)
#car_hist_features = color_hist(car_yuv_image, nbins=32)

#plot car and not car
fig = plt.figure()
plt.subplot(121)
plt.imshow(car_image)
plt.title("Car")
plt.subplot(122)
plt.imshow(notcar_image)
plt.title("Not car")
plt.show()

#plot histogram
fig = plt.figure()
plt.subplot(231)
plt.hist(car_yuv_image[:,:,0],bins=32,facecolor='b')
plt.title("car ch1")
plt.subplot(232)
plt.hist(car_yuv_image[:,:,1],bins=32,facecolor='b')
plt.title("car ch2")
plt.subplot(233)
plt.hist(car_yuv_image[:,:,2],bins=32,facecolor='b')
plt.title("car ch3")
plt.subplot(234)
plt.hist(notcar_yuv_image[:,:,0],bins=32,facecolor='b')
plt.title("not car ch1")
plt.subplot(235)
plt.hist(notcar_yuv_image[:,:,1],bins=32,facecolor='b')
plt.title("not car ch2")
plt.subplot(236)
plt.hist(notcar_yuv_image[:,:,2],bins=32,facecolor='b')
plt.title("not car ch3")
plt.suptitle("Histogram of YUV image")
plt.show()

# Plot the HOG 
#first row
fig = plt.figure()
plt.subplot(341)
plt.imshow(car_yuv_image[:,:,0], cmap='gray')
plt.axis('off')
plt.title('Car ch1')
plt.subplot(342)
plt.imshow(car_hog_image1, cmap='gray')
plt.axis('off')
plt.title('Car ch1 HOG')
plt.subplot(343)
plt.imshow(notcar_yuv_image[:,:,0], cmap='gray')
plt.axis('off')
plt.title('Not car ch1')
plt.subplot(344)
plt.imshow(notcar_hog_image1, cmap='gray')
plt.axis('off')
plt.title('Not car ch1 HOG')
#second row
plt.subplot(345)
plt.imshow(car_yuv_image[:,:,1], cmap='gray')
plt.axis('off')
plt.title('Car ch2')
plt.subplot(346)
plt.imshow(car_hog_image2, cmap='gray')
plt.axis('off')
plt.title('Car ch2 HOG')
plt.subplot(347)
plt.imshow(notcar_yuv_image[:,:,1], cmap='gray')
plt.axis('off')
plt.title('Not car ch2')
plt.subplot(348)
plt.imshow(notcar_hog_image2, cmap='gray')
plt.axis('off')
plt.title('Not car ch2 HOG')
#third row
plt.subplot(349)
plt.imshow(car_yuv_image[:,:,2], cmap='gray')
plt.axis('off')
plt.title('Car ch3')
plt.subplot(3,4,10)
plt.imshow(car_hog_image3, cmap='gray')
plt.axis('off')
plt.title('Car ch3 HOG')
plt.subplot(3,4,11)
plt.imshow(notcar_yuv_image[:,:,2], cmap='gray')
plt.axis('off')
plt.title('Not car ch3')
plt.subplot(3,4,12)
plt.imshow(notcar_hog_image3, cmap='gray')
plt.axis('off')
plt.title('Not car ch3 HOG')
plt.suptitle("YUV image in HOG")
plt.show()