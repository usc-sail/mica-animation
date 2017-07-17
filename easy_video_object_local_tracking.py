'''
Krishna Somandepalli
01/23/2016
Latest update July 2017
Local Tracking algorithm as described in the paper:
Unsupervised Discovery of Character Dictionaries in Animation Movies

'''


from matplotlib import pyplot
import numpy as np
import os,commands
import cv2
import time
from skimage import io, transform
import sys, json
from numpy import empty, nan
import CMT
import util

CMT = CMT.CMT()
def run_CMT(input_path, skip_frames, bbox, SHOW_IMAGES=False, clip_name='main'):
    CMT.estimate_scale = True
    CMT.estimate_rotation = False
    # read video and set delay accordingly
    cap = cv2.VideoCapture(input_path)
    cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, skip_frames)
    
    # something wrong with reading the shot it seems
    if not cap.isOpened():
        print '#Unable to open video input.'
        sys.exit(1)
    
    status, im0 = cap.read()
    im_gray0_ = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
    im_gray0 = cv2.equalizeHist(im_gray0_)
    im_draw = np.copy(im0)
    
    # prepare initial bbox values
    bbox_values = [int(v) for v in bbox]
    bbox = np.array(bbox_values)
    
    # Convert to point representation, adding singleton dimension
    bbox = util.bb2pts(bbox[None, :])

    # Squeeze
    bbox = bbox[0, :]

    tl = bbox[:2]
    br = bbox[2:4]
    
    print '#using', tl, br, 'as init bb'
    CMT_TRACKS=[]
    CMT.initialise(im_gray0, tl, br)
     
    frame = 1
    while True:
        # Read image
        status, im = cap.read()
        if not status:
            break
        im_gray_ = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_gray = cv2.equalizeHist(im_gray_)
        im_draw = np.copy(im)

        tic = time.time()
        CMT.process_frame(im_gray)
        toc = time.time()

        # Display results
        # Draw updated estimate
        if CMT.has_result:
            CMT_rect = [CMT.tl, CMT.tr,CMT.bl,CMT.br]

        if SHOW_IMAGES:
	    if CMT.has_result:
		cv2.line(im_draw, CMT.tl, CMT.tr, (255, 0, 0), 4)
		cv2.line(im_draw, CMT.tr, CMT.br, (255, 0, 0), 4)
		cv2.line(im_draw, CMT.br, CMT.bl, (255, 0, 0), 4)
		cv2.line(im_draw, CMT.bl, CMT.tl, (255, 0, 0), 4)
	    
	    util.draw_keypoints(CMT.tracked_keypoints, im_draw, (255, 255, 255))
	    util.draw_keypoints(CMT.votes[:, :2], im_draw)  # blue
	    util.draw_keypoints(CMT.outliers[:, :2], im_draw, (0, 0, 255))

            cv2.imshow(clip_name, im_draw)
            # Check key input
            k = cv2.cv.WaitKey(10) & 0xff
        
	# Remember image
        im_prev = im_gray

        # Advance frame number
        frame += 1
        if not(frame%50): print frame,
        print_str =  '{5:04d}: center: {0:.2f},{1:.2f} scale: {2:.2f}, active: {3:03d}, {4:04.0f}ms'.format(CMT.center[0], CMT.center[1], CMT.scale_estimate, CMT.active_keypoints.shape[0], 1000 * (toc - tic), frame)
        if CMT.has_result: CMT_TRACKS.append([1,CMT_rect])
        else: 
            print 'tracked for # frames = ', len(CMT_TRACKS)
            break

    cap.release()
    return CMT_TRACKS

movie_path = sys.argv[1]
object_json_path = sys.argv[2]
out_json_name = os.path.basename(object_json_path)

movie_width = int(commands.getoutput("mediainfo '--Inform=Video;%Width%' "+movie_path))

movie_height = int(commands.getoutput("mediainfo '--Inform=Video;%Height%' "+movie_path))
#for load into json first, object detection on every 10 frames
dict_list = json.load(open(object_json_path,'rU'))
# What is the subsampling ration
DOWNSAMPLE = 10

# what is the threshold for confidence in NN object detection
DNN_BOX_CONF = 0.1

RUN_VIDEO=True

##  - version for the chosen boxes     
key_dict_list = [i for i in dict_list if 'bbox_conf_nms' in i.keys()]
key_frames=[i['frame'] for i in key_dict_list if i['bbox_conf_nms']] #getting the frame numbers from the object detection list 

if RUN_VIDEO:
    if os.path.isfile(movie_path):
      movie_name = os.path.basename(movie_path).split('.')[0]
      frame_count = 0
      

      while(len(key_frames)>0):

          if True: #img_bgr is not None:
	  	
              if True:
	      	  frame_count=key_frames[0]
	      	  #cv2.imwrite('./key_frames/frame_%s.ppm' % (str(frame_count)), img_bgr)
	      	  print 'now tracking on --- ', frame_count
		  key_dict = [i for i in dict_list if i['frame']==frame_count]
		  #2. get bbox coordinates - dict['bbox_conf'][0]
		  [x1,y1,x2,y2] = [b for b in key_dict[0]['bbox_conf_nms'][0]]
		  #3. x1,y1,x2,y2 -> x1,y1,x2-x1,y2-y1=bbox
		  w = x2-x1
		  h = y2-y1
		  #--accounting for boundary errors in tracking
		  # changes with resolution - pad with 5 pixels!
		  if w >= movie_width-5:
		  	print 'changing width input.....'
		  	w = w-5
			x1+=5
		  if h >= movie_height-5: 
		  	print 'changing height input.....'
			h = h-5
			y1+=5
		  bbox = [x1,y1,w,h]
		  #print frame_count, bbox, '--->'
		  #4  - add tracking info and update
		  my_dict={}   #tracking result for single object 
		  my_dict['init_frame']=frame_count
		  my_dict['init_bbox'] = bbox
		  #track the detected object with the coordinates     
		  try: 
		  	cmt_status = run_CMT(movie_path, frame_count,bbox)#, SHOW_IMAGES=True)
			my_dict['tracking_info'] = cmt_status
		  	#frame count for the next tracked object
		  	frames_tracked = len(cmt_status)

		  #jump the frame count to the closest frames from object detector
		  	final_frame = frame_count+frames_tracked
		  	my_dict['final_frame'] = final_frame
			
			if frames_tracked==0: final_frame+=1
		  
		  	my_dict_list.append(my_dict) 
		  except:
		  	my_dict['final_frame'] = frame_count
			print 'TOO BAD!!!! - cannot track with the object specified'
			# artificaially make sure to ignore this object - 
			final_frame = key_frames[0]+1
			pass
		  

		  #print frame_count, key_frames[0]

		  eliminated_objects = 0
		  try:#if key_frames:
		  	while key_frames[0] < final_frame:
		      	    print key_frames[0],
		            eliminated_objects+=1
		      	    key_frames.pop(0)
		  except: IndexError
		  print '# objects elimnated due to tracking = ', eliminated_objects-1

                                      
          #write the dictionary to json file
          with open('SAL_track_detected_%s' % (out_json_name),'w') as outfile:
              json.dump(my_dict_list,outfile)
      
      print 'DONEEEEEEEEEEEEEEEEEEEEEEEE'


    else:
        print 'CHECK MOVIE PATH - DOESNOT EXIST', movie_path, '--nothing to do--'
        #break

print 'TOTAL TIME (seconds) = ', time.time()-start_time
