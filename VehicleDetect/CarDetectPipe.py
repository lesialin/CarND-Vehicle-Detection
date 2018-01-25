import queue
import numpy as np
import cv2

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img
#This class is to record boundary box data from past frames
class BoxHistory:
    
    def __init__(self):
        #record past frame's boundary boxs
        self.max_size = 10
        self.bbox_queue = queue.Queue(self.max_size)

    def push_bbox(self,bbox_list):
        if self.bbox_queue._qsize() >= self.max_size:
            self.bbox_queue.get()
            self.bbox_queue.put(bbox_list)
        else:
            self.bbox_queue.put(bbox_list)

    def get_bboxs_list(self):
        all_bbox_list = []
        all_bbox_list_inQ  = list(self.bbox_queue.queue)
        #transfer the queue 2-d list to 1-d list 
        for i in range(self.bbox_queue._qsize()):
            all_bbox_list = all_bbox_list + all_bbox_list_inQ[i]

        return all_bbox_list
            