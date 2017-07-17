'''
Created on Dec 14, 2015

@author: krsna
'''
import numpy as np
import cv2, os, sys, json
#import cv2.cv as cv
from scipy.io import loadmat
from pylab import *

from skimage.io import imread, imshow
import cv2

# Malisiewicz et al.
def non_max_suppression(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
 
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	#if boxes.dtype.kind == "i":
	#	boxes = boxes.astype("float")
 
	# initialize the list of picked indexes	
	pick = []
 	
	# grab the coordinates of the bounding boxes
	x1 = np.array([i[0] for i in boxes])
	y1 = np.array([i[1] for i in boxes])
	x2 = np.array([i[2] for i in boxes])
	y2 = np.array([i[3] for i in boxes])
	
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
 
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
 
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
 
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		#print overlap, '---overlap---' 
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
 
	# return only the bounding boxes that were picked using the
	# integer data type
	return np.array(boxes)[pick]

#ref_res = [384,720]
#ref_area = ref_res[0]*ref_res[1]

movie_path = sys.argv[1]
movie_name = movie_path.split('/')[-1].split('.')[0]
print movie_name, '---------------------'
mov = cv2.VideoCapture(movie_path)
ret,img=mov.read()
ref_area = prod(img.shape)

json_path = sys.argv[2]
dict_list = json.load(open(json_path,'rU'))

proj_dir = '/proj/krishna/animation'
sal_mat_dir = os.path.join(proj_dir, 'multibox_nn', 'key_frames', movie_name)
print sal_mat_dir
#sys.argv[2]#'../multibox_nn/key_frames_HTD'
sal_mat_ls = [os.path.join(sal_mat_dir, i) for i in os.listdir(sal_mat_dir) if i.endswith('.mat')]

print '# objects before saliency threshold= ', len(dict_list)#[j for j in dict_list if j['bbox_conf_nms']])

SAL_THR = float(sys.argv[3])#10.0
VERBOSE = False
for obj_dict in dict_list:
	bbox_list = [i[0] for i in obj_dict['bbox_conf']]
	conf_list = [i[1] for i in obj_dict['bbox_conf']]
	bbox_nms_list = list(non_max_suppression(bbox_list,0.50))
	chosen_boxes = []
	areas = []
	for bbox_nms_i, bbox_nms in enumerate(bbox_nms_list):
		x1,y1,x2,y2 = bbox_nms
		w = x2-x1
		h = y2-y1
		perc_area = (w*h*100.0)/ref_area
		if 1.0<perc_area<99.5:
			chosen_boxes.append(bbox_nms)
			areas.append(perc_area)
	if chosen_boxes:
		max_box_idx = np.argsort(areas)[-1]
		max_box = [ list(chosen_boxes[max_box_idx]), conf_list[max_box_idx] ]

		obj_dict['bbox_conf_nms'] = max_box
		X1,Y1,X2,Y2 = max_box[0]
		frame_count = obj_dict['frame']
		sal_mat_ = [i for i in sal_mat_ls if 'frame_'+str(frame_count)+'.ppm' in i]
		if sal_mat_:
			sal_mat = np.array(loadmat(sal_mat_[0])['sal_im'])
			if len(sal_mat[sal_mat>0])>0:
				sal_subset = sal_mat[Y1:Y2,X1:X2]
				sal_perc = (100.0*len(sal_mat[sal_mat>0]))/prod(sal_mat.shape)
				#print frame_count, sal_perc,
				sal_subset_perc = (100.0*len(sal_subset[sal_subset>0]))/len(sal_mat[sal_mat>0])
				#print sal_subset_perc
				if sal_perc>=1.0 and sal_subset_perc >= SAL_THR:
					if VERBOSE: print frame_count, 'PICK__ME'
					frame_count
				else: 
					if VERBOSE: print frame_count, 'IGNORE'
					obj_dict['bbox_conf_nms'] = []
			else: 
				if VERBOSE: print frame_count, 'IGNORE'
				obj_dict['bbox_conf_nms'] = []

		else:
			frame_count#print frame_count, 'PICK_ME'
	else:
		max_box=[]
	

	#print max_box
print dict_list[0].keys()
dict_list_t =  [j for j in dict_list if 'bbox_conf_nms' in j.keys()]
dict_list_t1 = [j for j in dict_list_t if j['bbox_conf_nms']]
print '# objects before saliency threshold= ', len(dict_list)#[j for j in dict_list if j['bbox_conf_nms']])
print '# objects after saliency threshold= ', len(dict_list_t1)
outfile_name = json_path.split('.json')[0]+'_SAL_%i.json' % (SAL_THR)

with open(outfile_name,'w') as f:
	json.dump(dict_list, f)
