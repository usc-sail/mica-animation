'''

Author Krishna Somandepalli
Latest update: July 2017
email: somandep@usc.edu
This script contains routines to parse the annotations from Mturk and perform majority voting to obtain labels

'''

import json, os, sys
from pylab import *
import numpy as np
from csv_handler import *
from collections import Counter
import itertools

## READ ME!!!
# First make sure, the answers under the column header answe_key in csv file are all clean and seperated by ","
batch_file_name = "8_movies_ALL_objects_batch_results.csv" #'Final_Batch_2869622_batch_results.csv'
batch_dict_list_orig = read_csv(batch_file_name)


#---some info about th keys of interest
movie_key = 'Input.movie_name' # or the category
answer_key = 'Answer.answer' # my annotation
image_key = 'Input.image_url' # what was annotated
cast_key = 'Input.character_url'
num_annot = 3 # no. annotations per image
worker_key = 'WorkerId'
time_key = 'WorkTimeInSeconds'
assign_key = 'AssignmentStatus'
assign_yes = 'Approved'

#--- choose approved items only
batch_dict_list = [i for i in batch_dict_list_orig if i[assign_key]==assign_yes]

print 'keys or headers in csv : ', sorted(batch_dict_list[0].keys())
#--- some overall stats
movie_list = unique([i[movie_key] for i in batch_dict_list])
worker_list = unique([i[worker_key] for i in batch_dict_list])

print 'movies annotated : ', movie_list
print 'number of workers involved : ', len(worker_list)

#---- do for each movie
mturk_dict_list=[]
for movie_name in movie_list:
	mturk_dict={}
	mturk_dict['movie_name'] = movie_name.lower().replace(' ','_')
	print 'examining...', movie_name, 
	movie_dict_list = [i for i in batch_dict_list if i[movie_key]==movie_name]
	num_images = len(movie_dict_list)/num_annot
	movie_image_list = unique([i[image_key] for i in movie_dict_list])
	
	im_answer_list=[]
	im_maj_answer_list=[]

	m_multiple_annots = []
	m_single_annots = []
	m_no_common_annots = []

	for im_name in movie_image_list:
		# im_dict has the original annots
		im_dict = {}
		# im_dict_maj has the majority votes
		im_dict_maj={}
		maj_annots=[]		
		common_annots=[]
		im_answers = [sorted(map(int, i[answer_key].split(','))) \
				for i in movie_dict_list \
				if i[image_key]==im_name]
		#print len(im_answers)
		im_dict[os.path.basename(im_name)] = im_answers
		im_answer_list.append(im_dict)		
		
		# -------------------generating stuff for annotation agreeement--------
		#check how many have more than 1 character - and other agreement metrics
		if sum([len(im_ans) for im_ans in im_answers])>3:
			im_answers_nonzero = [[j for j in i if j>0] for i in im_answers]
			m_multiple_annots.append(im_answers_nonzero)
			#print im_name, im_answers_nonzero,
			A1,A2,A3 = im_answers_nonzero
			common_annots = list(set(A1).intersection(A2).intersection(A3))
			#print common_annots
			if common_annots:
				maj_annots = [common_annots]
				m_single_annots.append([[100],[100],[100]])	
				maj_vote = common_annots
				#print maj_annots,'1'
				print maj_vote
			else:
				im_answers_nonzero_true = [i for i in \
								im_answers_nonzero if i]
				#print len(im_answers_nonzero_true)
				if len(im_answers_nonzero_true)==2:
					m_single_annots.append([[100],[200],[100]])
				
				if len(im_answers_nonzero_true)==1:
					m_single_annots.append([[100],[200],[300]])
				
				m_no_common_annots.append(im_answers_nonzero)
				maj_annots = im_answers_nonzero_true
				if not im_answers_nonzero_true:
					maj_annots = [[-1]]

				#print maj_annots,'2'
		
		else:
			m_single_annots.append(im_answers)
			maj_annots = im_answers
			#print maj_annots,'3'
		
		#-----------------------------
		# meanwhile also do amjority voting
		#print maj_annots
		if not common_annots:
			maj_annots_list = list(itertools.chain.from_iterable(maj_annots))
			maj_vote_counter = Counter(maj_annots_list)
			maj_vote = [maj_vote_counter.most_common()[0][0]]
		im_dict_maj[os.path.basename(im_name)] = maj_vote
		im_maj_answer_list.append(im_dict_maj)


	single_annot_str_ = '\n'.join([','.join(map(str, reduce(lambda x,y: x+y, i))) \
				for i in m_single_annots])+'\n'	
	
	# for ease: replace -1 by 0 since we don't really care about this for agreement
	single_annot_str = single_annot_str_.replace('-1','0')
	
	with open('%s_single_char_annots_all_objects_07_15_17.csv' % movie_name, 'w') as f1:
		f1.write(single_annot_str); f1.close()
	#---------------------------------------------------------------------
	
	mturk_dict['original_annotations'] = im_answer_list
	mturk_dict['majority_annotations'] = im_maj_answer_list

	print ' has #images = ', num_images, 
	worker_sublist = unique([i[worker_key] for i in movie_dict_list])
	print ' with # workers = ', len(worker_sublist),
	image_id_list_ = [i[answer_key] for i in movie_dict_list]
	image_id_list = [map(int, i.split(',')) for i in image_id_list_ if i]
	print ' and has unnique char IDs: ', len(unique(array(image_id_list)))
	missing_annot = [i for i in movie_dict_list if i[answer_key]=="{}"]
	print ' with missing annotations: ', len(missing_annot),
	print '...#multiple character = ', len(m_multiple_annots), 
	print '= ', 100.0*len(m_multiple_annots)/num_images,'%',
	print 'no common annots in multiple cases = ', len(m_no_common_annots) 
	
	mturk_dict_list.append(mturk_dict)


WRITE_JSON=True
if WRITE_JSON:
	with open('MTurk_annotations_8_movies_ALL_objects_07_15_17.json','w') as f:
		json.dump(mturk_dict_list,f)




#---- do some worker-specific QC

