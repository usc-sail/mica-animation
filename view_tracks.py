'''Krishna Somandepalli - 01/23/2016'''
from matplotlib import pyplot
import numpy as np
import os,commands
import cv2
import time
from skimage import io, transform
import sys, json
# Make sure that you set this to the location your caffe2 library lies.



movie_path = sys.argv[1]
object_json_path = sys.argv[2]
movie_width = int(commands.getoutput("mediainfo '--Inform=Video;%Width%' "+movie_path))

movie_height = int(commands.getoutput("mediainfo '--Inform=Video;%Height%' "+movie_path))
#for load into json first, object detection on every 10 frames
dict_list = json.load(open(object_json_path,'rU'))
# What is the subsampling ration
DOWNSAMPLE = 10

# what is the threshold for confidence in NN object detection
DNN_BOX_CONF = 0.1

RUN_VIDEO=True
DISPLAY=False
SAVE=True

##  - version for the chosen boxes     
key_frames=[i['init_frame'] for i in dict_list if (i['final_frame']-i['init_frame'])>1] #getting the frame numbers from the object detection list 

if RUN_VIDEO:
    if os.path.isfile(movie_path):
      movie_name = os.path.basename(movie_path).split('.')[0]
      frame_count = 0
      im_out_dir = './Objects_SAL_tracked/'+movie_name
      if not os.path.isdir(im_out_dir): os.makedirs(im_out_dir) 
      mov = cv2.VideoCapture(movie_path)

      while(mov.isOpened()):
          frame_count+=1
          ret, img_bgr = mov.read()

          if img_bgr is not None:
	  	
          # same as  -- if ret:
              if frame_count==key_frames[0]:
	      	  print 'now showing --- ', frame_count
		  key_dict = [i for i in dict_list if i['init_frame']==frame_count]
		  x1,y1,w,h = key_dict[0]['init_bbox']
		  frames_tracked = key_dict[0]['final_frame']-key_dict[0]['init_frame']
		  print 'tracked for # frames = ', frames_tracked
		  x2 = x1+w
		  y2 = y1+h
		  print x1,y1,w,h
		  if x1<0: x1=0
		  if y1<0: y1=0
		  if DISPLAY:
		  	cv2.imshow( 'object', img_bgr[y1:y2,x1:x2,:] )
		 	cv2.waitKey(0)
		  if SAVE:
		  	cv2.imwrite(im_out_dir+'/'+str(frame_count)+'_'+str(frames_tracked)+'.ppm', img_bgr[y1:y2,x1:x2,:])
		  key_frames.pop(0)


          else:
              mov.release()
              cv2.destroyAllWindows()
                                      

      
      print 'DONEEEEEEEEEEEEEEEEEEEEEEEE'


    else:
        print 'CHECK MOVIE PATH - DOESNOT EXIST', movie_path, '--nothing to do--'
        #break

print 'TOTAL TIME (seconds) = ', time.time()-start_time
