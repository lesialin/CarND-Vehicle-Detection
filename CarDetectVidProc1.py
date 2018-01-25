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
	#store current bbox data into history
	box_hist.push_bbox(bbox_list)
	# get the past bbox data to predect current one
	hist_bbox_list = box_hist.get_bboxs_list()	
	add_heat(heatmap, hist_bbox_list)
	if box_hist.bbox_queue._qsize() < box_hist.max_size:
		threshold = 1
	else:
		threshold = 3
			
	heat = apply_threshold(heatmap, threshold)
	# Visualize the heatmap when displaying    
	heatmap = np.clip(heat, 0, 255)
	#return the isolate detected car
	labels = label(heatmap)
	draw_img = draw_labeled_bboxes(img, labels)
	return draw_img

clip = VideoFileClip ('test_video.mp4')#.subclip(0,20)
out_clip = clip.fl_image(process_image)
out_clip.write_videofile("test_video_out.mp4", audio=False)


