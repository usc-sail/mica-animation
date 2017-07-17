'''
Krishna Somandepalli
Latest update: July 2017
Mutlibox object detection for animation movies:
https://github.com/krsna6/multibox/blob/master/multibox.ipynb
'''

from matplotlib import pyplot
import numpy as np
import os, cv2
from skimage import io, transform
import sys, json
# Make sure that you set this to the location your caffe2 library lies.
caffe2_root = '/opt/caffe2/'
sys.path.insert(0, os.path.join(caffe2_root, 'gen'))

print caffe2_root
# After setting the caffe2 root path, we will import all the caffe2 libraries needed.
from caffe2.proto import caffe2_pb2
from pycaffe2 import core, net_drawer, workspace, visualize

multibox_data_dir = '/opt/caffe2/multibox/'

# net is the network definition.
net = caffe2_pb2.NetDef()
net.ParseFromString(open(os.path.join(multibox_data_dir, 'multibox_net.pb')).read())

# tensors contain all the parameters used in the net.
# The multibox model is relatively large so we have stored the parameters in multiple files.
import glob
file_parts = glob.glob(multibox_data_dir+"multibox_tensors.pb.part*")
file_parts.sort()
tensors = caffe2_pb2.TensorProtos()
tensors.ParseFromString(''.join(open(f).read() for f in file_parts))

# Note that the following line hides the intermediate blobs and only shows the operators.
# If you want to show all the blobs as well, use the commented GetPydotGraph line.
#graph = net_drawer.GetPydotGraphMinimal(net.op, name="multibox", rankdir='TB')
#graph = net_drawer.GetPydotGraph(net.op, name="inception", rankdir='TB')

#print 'Visualizing network:', net.name
#display.Image(graph.create_png(), width=200)

DEVICE_OPTION = caffe2_pb2.DeviceOption()
# Let's use CPU in our example.
DEVICE_OPTION.device_type = caffe2_pb2.CPU

# If you have a GPU and want to run things there, uncomment the below two lines.
# If you have multiple GPUs, you also might want to specify a gpu id.
#DEVICE_OPTION.device_type = caffe2_pb2.CUDA
#DEVICE_OPTION.cuda_gpu_id = 0

# Caffe2 has a concept of "workspace", which is similar to that of Matlab. Each workspace
# is a self-contained set of tensors and networks. In this case, we will just use the default
# workspace, so we won't dive too deep into it.
workspace.SwitchWorkspace('default')
print 'setp 1 done'
# First, we feed all the parameters to the workspace.
for param in tensors.protos:
    workspace.FeedBlob(param.name, param, DEVICE_OPTION)
# The network expects an input blob called "input", which we create here.
# The content of the input blob is going to be fed when we actually do
# classification.
workspace.CreateBlob("input")
# Specify the device option of the network, and then create it.
net.device_option.CopyFrom(DEVICE_OPTION)
workspace.CreateNet(net)
print 'step 2 done'

# location_prior defines the gaussian distribution for each location: it is a 3200x2
# matrix with the first dimension being the std and the second being the mean.
LOCATION_PRIOR = np.loadtxt(os.path.join(multibox_data_dir, 'ipriors800.txt'))

def RunMultiboxOnImage(img, location_prior):
    #img = io.imread(image_file)
    resized_img = transform.resize(img, (224, 224))
    normalized_image = resized_img.reshape((1, 224, 224, 3)).astype(np.float32) - 0.5
    workspace.FeedBlob("input", normalized_image, DEVICE_OPTION)
    workspace.RunNet("multibox")
    location = workspace.FetchBlob("imagenet_location_projection").flatten(),
    # Recover the original locations
    location = location * location_prior[:,0] + location_prior[:,1]
    location = location.reshape((800, 4))
    confidence = workspace.FetchBlob("imagenet_confidence_projection").flatten()
    return location, confidence

def PrintBox(loc, height, width, style='r-'):
    """A utility function to help visualizing boxes."""
    xmin, ymin, xmax, ymax = loc[0] * width, loc[1] * height, loc[2] * width, loc[3] * height 
    #pyplot.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], style)
    return [xmin, ymin, xmax, ymax]



movie_path = sys.argv[1]

# What is the subsampling ration
DOWNSAMPLE = 10

# what is the threshold for confidence in NN object detection
DNN_BOX_CONF = 0.1

RUN_VIDEO=True
CONF_BOXES_ONLY=True
if CONF_BOXES_ONLY:
	if RUN_VIDEO:
		if os.path.isfile(movie_path):
			movie_name = os.path.basename(movie_path).split('.')[0]
			frame_count = 0
			print 'loading..', movie_path	
			mov = cv2.VideoCapture(movie_path)
			my_dict_list=[]

			while(mov.isOpened()):
				frame_count+=1
				ret, img_bgr = mov.read()
				if img_bgr is not None:
					if not (frame_count%1000): print 'Now doing frame #', frame_count
					
					if not(ret):
						print 'skipping frame #', frame_count
						continue;
					
					#Run DNN object detector for only 1 in 10 frames (DOWNSAMPLE = 10)
					if not (frame_count%DOWNSAMPLE):
						img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
						height, width, ch  = img_rgb.shape
						location, confidence = RunMultiboxOnImage(img_rgb, LOCATION_PRIOR)

						# Note that argsort sorts things in increasing order.
						my_dict={}
						bboxes=[]
						sorted_idx = np.argsort(confidence)
						# convert the logit confidence to probilities
						p_conf = 1/(1+np.exp(-1*confidence))

						conf_idx = np.where(p_conf >= DNN_BOX_CONF)
						
						if len(conf_idx[0])>0:
							print 'confident objects found in', frame_count
							my_dict["frame"] = frame_count
							for idx in conf_idx[0]:
							    bbox=PrintBox(location[idx], height, width)
							    bboxes.append([ bbox, np.float(p_conf[idx]) ])
							
							my_dict["bbox_conf"]=bboxes
							my_dict_list.append(my_dict)
				else:
					mov.release()
					cv2.destroyAllWindows()

			with open('deep_multibox_objects_%s_Nby10.json' % (movie_name),'w') as outfile:
				json.dump(my_dict_list, outfile)
			print 'DONEEEEEEEEEEEEEEEEEEEEEEEE'


		else:
			print 'CHECK MOVIE PATH - DOESNOT EXIST', movie_path, '--nothing to do--'
			#break

#print 'TOTAL TIME (seconds) = ', time.time()-start_time
