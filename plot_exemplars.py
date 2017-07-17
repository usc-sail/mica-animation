import json, os, sys
import numpy as np
from pylab import *
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib._png import read_png

from skimage.exposure import equalize_adapthist

SAVE_IM=True
def imscatter(x, y, images, labels, mask, ax=None, zoom=1, prefix_name='XMP'):
    if ax is None:
        ax = gca()
    x, y = np.atleast_1d(x, y)
    artists = []
    for i in range(len(x)):
    	if mask[i]:
		x0=x[i]
		y0=y[i]
		im0 = equalize_adapthist(imread(images[i]))
		out_im_name = str(x0)+'_'+str(y0)+'_'+prefix_name+'_'+os.path.basename(images[i]).split('.')[0]+'_'+str(labels[i])+'.png'
		if SAVE_IM: imsave(out_im_name, im0)
		im = OffsetImage(im0, zoom=zoom)
		ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False, pad=0.5)
		artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    #ax.autoscale()
    return artists


exp_dict_list = json.load(open('MASTER_EXPERIMENT_DICTIONARY_8_movies.json','rU')) 
annot_dict_list = json.load(open('ANNOT_MATCHED_EXPERIMENTS.json','rU'))
exp_key = 'SAL10_T1_'

fpath_key = exp_key+'fpath'
#cluster size
count_key = exp_key+'counts'
# track duration
length_key = exp_key+'lengths'
label_key = exp_key+'mturk'

for exp_dict in exp_dict_list[5:6]:
	movie_name = exp_dict['movie_name']
	print movie_name
	annot_dict = [i for i in annot_dict_list if i['movie_name']==movie_name][0]
	mov_path = os.path.join('/data/GDI/animated', movie_name+'.mkv')
	if not os.path.isfile('ffprobe_%s.json' % (movie_name)):
		commands.getoutput("ffprobe -show_chapters -print_format json %s > ffprobe_%s.json" \
					% (mov_path, movie_name))
	ffprobe_dict = json.load(open('ffprobe_%s.json' % (movie_name),'r'))
	credit_start_frame = int(float(ffprobe_dict['chapters'][-1]['start_time'])*23.98)
	
	im_count = exp_dict[count_key]
	#im_count = [100*i for i in exp_dict[count_key]]
	im_fpath = exp_dict[fpath_key]
	im_frames = [int(os.path.basename(f).split('.')[0]) for f in im_fpath]
	im_choose = array(im_frames)<credit_start_frame
	im_length = [sum(l) for l in exp_dict[length_key]]
	im_labels = annot_dict[label_key]
	#ax = gca()
	fig,ax = subplots()
	imscatter(im_count, im_length, im_fpath, im_labels, im_choose, zoom=0.25, ax=ax, prefix_name = movie_name)
	ax.plot(list(array(im_count)[im_choose]), list(array(im_length)[im_choose]), '.')
	ax.tick_params(axis='both', which='major', pad=15)
	#xticks(im_count, im_counts)
	
	#show()
	#for i in range(len(im_count)):
	#	xy = (im_count[i], im_length[i])
	#	im = imread(im_fpath[i])
	#	im_box = OffsetImage(im, zoom=.1)
	#	im_annot = AnnotationBbox(im_box, xy,xycoords='data',boxcoords="offset points")  
	#	ax.add_artist(im_annot)
	#ax.grid(True)
	#draw()
	#show()

	#plot(im_count, im_length, '*')
	#show()

#def main():
#    x = np.linspace(0, 10, 20)
#    y = np.cos(x)
#    image_path = get_sample_data('ada.png')
#    fig, ax = plt.subplots()
#    imscatter(x, y, image_path, zoom=0.1, ax=ax)
#    ax.plot(x, y)
#    plt.show()
#
#
#main()
