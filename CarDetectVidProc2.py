import pickle
import glob
import random
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from VehicleDetect.CarDetect import find_cars,draw_boxes
from VehicleDetect.CarDetectPipe import add_heat,apply_threshold,draw_labeled_bboxes,BoxHistory
from sklearn.preprocessing import StandardScaler
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label

dist_pickle = pickle.load( open("model/dist_pickle.p", "rb" ) )

svc = dist_pickle["svc"]
color_space = dist_pickle["color_space"]
spatial_feat = dist_pickle["spatial_feat"]
hist_feat = dist_pickle["hist_feat"]
hog_feat = dist_pickle["hog_feat"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
hog_channel = dist_pickle["hog_channel"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]
X_scaler = dist_pickle["X_scaler"]
box_hist = BoxHistory()

def process_image(img):
	heatmap = np.zeros_like(img[:,:,0]).astype(np.float)
	# Top portion of ROI image
	scale = 1.5
	ystart = 400
	ystop = 550
	xstart = 400
	xstop = 1208
	bboxes1 = find_cars(img, xstart, xstop, ystart, ystop, scale,color_space, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,hog_channel)
	# Middle portion of ROI image
	scale = 2.8
	ystart = 450
	ystop = 600
	xstart = 400
	xstop = 1208
	bboxes2 = find_cars(img,  xstart, xstop,ystart, ystop, scale,color_space, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,hog_channel)
	# Bottom portion of ROI image
	scale = 3.2
	ystart = 500
	ystop = 656
	xstart = 400
	xstop = 1208
	bboxes3 = find_cars(img,  xstart, xstop, ystart, ystop, scale,color_space, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,hog_channel)
	#coimbin three result
	bbox_list = bboxes1 + bboxes2 + bboxes3
	
	draw_img = draw_boxes(img, bbox_list, color=(0, 0, 255), thick=6)
	
	return draw_img, bbox_list

clip = VideoFileClip('project_video.mp4').subclip(14,14.4)
count = 0
all_bbox = []
n_row_plot = 5
n_col_plot = 2
fig = plt.figure(figsize = (4*n_col_plot, 12*n_row_plot))
fig.subplots_adjust(hspace = 0.5, wspace = 0.1)
for img in clip.iter_frames():
	if count < n_row_plot*n_col_plot:
		draw_img,bbox_list = process_image(img)
		all_bbox = all_bbox + bbox_list
		ax = fig.add_subplot(n_row_plot, n_col_plot, count+1)
		ax.axis('off')
		plt.imshow(draw_img)
		plt.title("frame_%04d" %count)
		count+=1
plt.suptitle("car detection frames")
plt.show()

heatmap = np.zeros_like(img[:,:,0]).astype(np.float)
heat = apply_threshold(heatmap, 2)
heatmap = add_heat(heat, all_bbox)
# Visualize the heatmap when displaying    
heatmap = np.clip(heatmap, 0, 255)
#return the isolate detected car
labels = label(heatmap)
draw_img = draw_labeled_bboxes(img, labels)

fig = plt.figure()
plt.subplot(121)
plt.imshow(heatmap)
plt.title("cumulated heatmap")
plt.subplot(122)
plt.imshow(labels[0], cmap='gray')
plt.title("apply threshold")
plt.suptitle("heatmap")
plt.show()

