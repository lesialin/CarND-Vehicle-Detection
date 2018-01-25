import glob
import time
import pickle
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import LinearSVC
from VehicleDetect.FeatureExtraction import extract_features as extract_features

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
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32,32)
hist_bins = 32	

t=time.time()
car_features = extract_features(cars, color_space=color_space, spatial_size=spatial_size,
                    hist_bins=hist_bins, orient=orient, 
                    pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                    spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space, spatial_size=spatial_size,
                    hist_bins=hist_bins, orient=orient, 
                    pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                    spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract HOG features...')
# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()
#Randomly parameter training
#parameters = {'kernel':('linear', 'rbf'), 'C': np.arange( 1, 100+1, 1 ).tolist()} 
#svr = svm.SVC()
#svc = RandomizedSearchCV(svr, param_distributions = parameters, random_state = 2017)


# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

dist_pickle = {'svc': svc, 'color_space': color_space, 'spatial_feat': spatial_feat,'hist_feat':hist_feat,
'hog_feat':hog_feat,'orient':orient,'pix_per_cell':pix_per_cell,'cell_per_block':cell_per_block,
'hog_channel':hog_channel,'spatial_size':spatial_size,'hist_bins':hist_bins,'X_scaler':X_scaler}

f = open('model/dist_pickle.p', 'wb')
pickle.dump(dist_pickle, f)
f.close()
