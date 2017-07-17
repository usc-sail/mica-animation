'''
Author Krishna Somandepalli
Latest update: July 2017
email: somandep@usc.edu
Run this after experimenter code - which generates annotations for different objects

This script computes performance analysis measures for all experiments
Also generates required plots:

Computes the following measures per movie:
1) Precision
2) Recall
3) F1-score
4) Average cluster purity
5) Overclustering index

'''
# --- THis works after experimenter_....py
# @krsna
from __future__ import division
from pylab import *
import numpy as np
import os, json, sys, commands
import itertools
import seaborn as sn
from collections import Counter


CAST_DICT_LIST=[{'movie_name': 'cars_2','cast': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'minor_cast':[8,9,10]},
        {'movie_name': 'how_to_train_your_dragon_2', 'cast': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12], 'minor_cast':[6,7,11,12]},
        {'movie_name': 'frozen','cast': [1, 2, 3, 4, 5, 6, 7, 8, 9], 'minor_cast':[4,6,7,8]},
        {'movie_name': 'the_lego_movie','cast': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12], 'minor_cast':[10,11,12]},    
        {'movie_name': 'toy_story_3','cast': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15,16,17,18], 'minor_cast':[14,15,8,9,16,17,18,13,12]},
        {'movie_name': 'shrek_forever_after','cast': [1, 2, 3, 4, 5, 6, 7, 8, 9], 'minor_cast':[5,6,8,9,10]},
        {'movie_name': 'freebirds','cast': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11], 'minor_cast':[8,9,10,11]},
        {'movie_name': 'tangled','cast': [1, 2, 3, 4, 5, 6, 7, 8, 9], 'minor_cast':[6,7,8,9]}
        ]

#--note in movie shrek - 3 and 4 look different but same character

# output name with measures
OUT_NAME = "GIST_ANNOT_MATCHED_EXPERIMENTS_8movies.json"

# experiment output json file
exp_dict_file = "GIST_MASTER_EXPERIMENT_DICTIONARY_8_movies.json"
#exp_dict_file = 'PART_MASTER_EXPERIMENT_DICTIONARY_8_movies.json'

# all annotation from mturk: files in the directory mturk_annotations
mturk_dict_file1 = "MTURK_annotations/MTurk_annotations_8_movies_ALL_objects_07_15_17.json"
mturk_dict_file2 = 'MTURK_annotations/MTurk_annotations_8_movies.json'

# experiment parameter settings
SAL_KEYS = ['SAL10']#['SAL0', 'SAL10', 'SAL20', 'SAL50', 'SAL80', 'SAL90']
TRACK_KEYS = ['T1'] #['T1','T12', 'T24', 'T48', 'T120']


#------------script begins----------------------
sn.set_context('paper')

def plot_heatmaps(F1_mean, P_mean, R_mean, prefix_name='heatmap',save_fig=True):
    clf()
    VMIN =  min(min(P_mean.flatten()),min(R_mean.flatten()),min(F1_mean.flatten()))
    VMAX =  max(max(P_mean.flatten()),max(R_mean.flatten()),max(F1_mean.flatten()))
    #print VMIN, VMAX
    CMAP='RdBu'
    figure(1)
    with sn.plotting_context('paper', font_scale=1.5):
        sn.heatmap(F1_mean,vmin=0.25, vmax=1.0, \
                xticklabels=[1, 12, 24, 48, 120], yticklabels=[0, 10, 20, 50, 80, 90], \
                linewidth=0.5, cbar=True, cmap=CMAP, square=True, linecolor='k', annot=True, \
                annot_kws={'fontsize':12})#,cbar_kws={'fontsize':14})
        xlabel('Track Duration Threshold (frames)', fontsize=14,weight='bold')
        ylabel('Relative Saliency Score (%)', fontsize=14, weight='bold')
        title('F1 score',fontsize=12, weight='bold')
        if save_fig: savefig(prefix_name+'_F1.png', transparent=True, bbox_inches='tight', dpi=250)
    clf()
    figure(2)
    with sn.plotting_context('paper', font_scale=1.5):
        sn.heatmap(P_mean,vmin=0.25, vmax=1.0, \
                xticklabels=[1, 12, 24, 48, 120], yticklabels=[0, 10, 20, 50, 80, 90], \
                linewidth=0.5, cbar=True, cmap=CMAP, square=True, linecolor='k',annot=True, 
                annot_kws={'fontsize':12})#,cbar_kws={'fontsize':14})
        xlabel('Track Duration Threshold (frames)', fontsize=14,weight='bold')
        ylabel('Relative Saliency Score (%)', fontsize=14, weight='bold')
        title('Precision',fontsize=12, weight='bold')
        if save_fig: savefig(prefix_name+'_P.png', transparent=True, bbox_inches='tight', dpi=250)
    clf()
    figure(3)
    with sn.plotting_context('paper', font_scale=1.5):
        sn.heatmap(R_mean,vmin=0.25, vmax=1.0, \
                xticklabels=[1, 12, 24, 48, 120], yticklabels=[0, 10, 20, 50, 80, 90], \
                linewidth=0.5, cbar=True, cmap=CMAP, square=True, linecolor='k', annot=True, \
                annot_kws={'fontsize':12})#,cbar_kws={'fontsize':14})
        xlabel('Track Duration Threshold (frames)', fontsize=14,weight='bold')
        ylabel('Relative Saliency Score (%)', fontsize=14, weight='bold')
        title('Recall',fontsize=12, weight='bold')
        if save_fig: savefig(prefix_name+'_R.png', transparent=True, bbox_inches='tight', dpi=250)


def get_mturk_label_for_frames(exp_images, maj_annots):
    # maj_annots format : [{u'10650.png': [1]}, {u'11590.png': [1,9]}, {u'11860.png': [0,1]}]
    exp_annots = []

    #--- getting matches
    for exp_im in exp_images:
        mturk_annot = [i for i in maj_annots if i.keys()[0].startswith(exp_im)]
        if mturk_annot:
            exp_im_annots = mturk_annot[0].values()[0]
            exp_annots.append(exp_im_annots)
        else: 
            print 'something WOEFULLY erroneous, adding nans', exp_im
            exp_annots.append([np.nan])
    exp_im_labels = list(itertools.chain.from_iterable(exp_annots))
    return exp_im_labels


exp_dict_list = json.load(open(exp_dict_file,'rU'))
mturk_dict_list1 = json.load(open(mturk_dict_file1,'rU'))
mturk_dict_list2 = json.load(open(mturk_dict_file2,'rU'))

ALL_F1_MATRIX=[]
ALL_P_MATRIX=[]
ALL_R_MATRIX=[]
annot_dict_list=[]
all_char_counts=[]
precision_counts=[]
for exp_dict in exp_dict_list:
    annot_dict={}
    movie_name = exp_dict['movie_name']
    annot_dict['movie_name'] = movie_name
    print movie_name, '................', 
    
    mturk_dict1 = [m for m in mturk_dict_list1 if m['movie_name'] == movie_name][0]
    mturk_dict2 = [m for m in mturk_dict_list2 if m['movie_name'] == movie_name][0]
    
    cast_dict = [c for c in CAST_DICT_LIST if c['movie_name'] == movie_name][0]
    cast_labels = cast_dict['cast']
    minor_cast_labels = cast_dict['minor_cast']
    maj_annots = mturk_dict1['majority_annotations'] + mturk_dict2['majority_annotations']
    
    # just use one annotations instead and see what happens
    #maj_annots = [i[0] for i in  mturk_dict['original_annotations']]
    
    # clean up using ffprobe chapter boundaries - last chap. are credits!!
    mov_path = os.path.join('/data/GDI/animated', movie_name+'.mkv')
    if not os.path.isfile("ffprobe_%s.json" % (movie_name)):
        commands.getoutput(\
        "ffprobe -show_chapters -print_format json %s > ffprobe_%s.json" % (mov_path, movie_name))
    ffprobe_dict = json.load(open('ffprobe_%s.json' % (movie_name),'r'))
    credit_start_frame = int(float(ffprobe_dict['chapters'][-1]['start_time'])*23.98)
    # end cleanup

    F1_matrix = zeros((len(SAL_KEYS), len(TRACK_KEYS)))
    P_matrix = zeros((len(SAL_KEYS), len(TRACK_KEYS)))
    R_matrix = zeros((len(SAL_KEYS), len(TRACK_KEYS)))
    for SAL_i, SAL in enumerate(SAL_KEYS):
        for TRK_i, TRK in enumerate(TRACK_KEYS):
            exp_key = SAL+'_'+TRK
            exp_fpath_key = exp_key+'_fpath'
            exp_counts_key = exp_key+'_counts'
            exp_fpath = exp_dict[exp_fpath_key]
            exp_counts = exp_dict[exp_counts_key]

            #READ!! this split should be such a way that only integers are returned, change split() acc.
            exp_images_ = [os.path.basename(i).split('.')[0].split('_')[0] for i in exp_fpath]

            annot_dict[exp_fpath_key] = exp_fpath 
            annot_dict[exp_counts_key] = exp_counts
            
            exp_images = [i for i in exp_images_ if int(i)<credit_start_frame]
            #exp_images = exp_images_
            exp_annots = []

            #--- getting matches
            for exp_im in exp_images:
                mturk_annot = [i for i in maj_annots if i.keys()[0].startswith(exp_im)]
                if mturk_annot:
                    exp_im_annots = mturk_annot[0].values()[0]
                    exp_annots.append(exp_im_annots)
                else: print 'something WOEFULLY erroneous', exp_im
            multiple_annots = [i for i in exp_annots if len(i)>1]
            #print [i for i in multiple_annots if -1 in i], '*****!!!****'
            #----exclude any mutliple annotations with -1 or 0 in them
            #exp_annots = [i for i in exp_annots if len()]
            #exp_
            #--- getting labels for all exemplars
            exp_im_labels = list(itertools.chain.from_iterable(exp_annots))
            annot_key = exp_key+'_mturk'
            annot_dict[annot_key] = exp_im_labels
            print exp_key, #'; size=', 
            exp_total = len(exp_fpath)
            #print exp_total,
            #print unique(exp_im_labels),
            
            # CLUSTER SIZE THR > 5
            # ---getting cluster purity only for those cluster members > 5 
            clustered_im_frames = exp_dict[exp_key+'_labels']
            all_cluster_labels = []
            cluster_purity_list = []
            for clustered_ims in clustered_im_frames:
                cluster_labels = get_mturk_label_for_frames(clustered_ims, maj_annots)
                all_cluster_labels.append(cluster_labels)
                if len(clustered_ims) > 5:
                    # purity: each cluster is assigned to the class which is most frequent 
                    # in the cluster, and then correct/total_size
                    purity = ( 100*max(Counter(cluster_labels).values()) )/len(cluster_labels)
                    cluster_purity_list.append(purity)

            #-- getting the exemplars with  more than 5 cluster members
            exp_least5_annots = [i for idx,i in enumerate(exp_annots) if \
                        exp_counts[idx] > 5]
            exp_im_least5_labels = list(itertools.chain.from_iterable(exp_least5_annots))
            #print unique(exp_im_least5_labels)
            
            ####--- PRECISION & RECALL!!!!! -----#####
            # 1. For all exemplars
            ALL_EXEMPLARS_FULL_CAST = True
            if ALL_EXEMPLARS_FULL_CAST:
                correct_labels = [i for i in exp_im_labels if i>0]
                num_per_char = [[i for i in correct_labels if i==k] \
                            for k in set(correct_labels)]
                char_counts = [len(i) for i in num_per_char]
                overcluster_idx = median([len(i) for i in num_per_char]) 
                num_correct = len(correct_labels)
                unique_num_correct = len(unique(correct_labels))
                true_total = len(cast_labels)

                precision = float(num_correct)/len(exp_im_labels)#exp_total
                prec_count = [num_correct, len(exp_im_labels)-num_correct]
                recall = float(unique_num_correct)/true_total
                f1_score = (2.0*precision*recall)/(precision+recall)
                #print ' ; precision = ', precision*100, 
                #print ' ; recall = ',recall*100,
                #print ' ; F1 score = ', f1_score, '; overclust = ', overcluster_idx,
                #print ' ; avg. cluster purity = ', np.mean(cluster_purity_list)
                print precision*100, recall*100, f1_score, overcluster_idx, np.mean(cluster_purity_list)
                F1_matrix[SAL_i, TRK_i]=f1_score
                P_matrix[SAL_i, TRK_i]=precision
                R_matrix[SAL_i, TRK_i]=recall
                annot_dict['metrics']=[precision, recall, f1_score,overcluster_idx]

            ALL_EXEMPLARS_MAJOR_CAST = False
            if ALL_EXEMPLARS_MAJOR_CAST:
                exp_im_labels_2 = [i for i in exp_im_labels if i not in minor_cast_labels]
                correct_labels_2 = [i for i in exp_im_labels_2 if i>0]
                #correct_labels_2 = [i for i in correct labels if i not in minor_cast_labels]
                num_correct_2 = len(correct_labels_2)
                unique_num_correct_2 = len(unique(correct_labels_2))
                true_total_2 = len(cast_labels)-len(minor_cast_labels)
                
                precision2 = float(num_correct_2)/len(exp_im_labels_2)#exp_total
                recall2 = float(unique_num_correct_2)/true_total_2
                print ' ; precision = ', precision2*100, 
                print ' ; recall = ',recall2*100,
                f1_score2 = (2.0*precision2*recall2)/(precision2+recall2)
                print ' ; F1 score = ', f1_score2, multiple_annots



            # 2. Now do for top-exemplars only i.e., has atleast 6 cluster members

            TOP_EXEMPLARS_FULL_CAST = False
            if TOP_EXEMPLARS_FULL_CAST:
                if exp_im_least5_labels:
                    correct_labels = [i for i in exp_im_least5_labels if i>0]
                    num_correct = len(correct_labels)
                    unique_num_correct = len(unique(correct_labels))
                    true_total = len(cast_labels)

                    precision = float(num_correct)/len(exp_im_least5_labels)#exp_total
                    recall = float(unique_num_correct)/true_total
                    print ' ; precision = ', precision*100, 
                    print ' ; recall = ',recall*100,
                    if precision+recall == 0: print ' ; F1 score = NA'
                    else: 
                        f1_score = (2.0*precision*recall)/(precision+recall)
                        print ' ; F1 score = ', f1_score, multiple_annots
                else: print '; precision = NA ; recall = NA ; F1 score = NA' 

            TOP_EXEMPLARS_MAJOR_CAST = False
            if TOP_EXEMPLARS_MAJOR_CAST:
                if exp_im_least5_labels:
                    exp_im_labels_2 = [i for i in exp_im_least5_labels \
                                if i not in minor_cast_labels]
                    correct_labels_2 = [i for i in exp_im_labels_2 if i>0]
                    #correct_labels_2 = [i for i in correct labels if i not in minor_cast_labels]
                    num_correct_2 = len(correct_labels_2)
                    unique_num_correct_2 = len(unique(correct_labels_2))
                    true_total_2 = len(cast_labels)-len(minor_cast_labels)
                    
                    precision2 = float(num_correct_2)/len(exp_im_labels_2)#exp_total
                    recall2 = float(unique_num_correct_2)/true_total_2
                    print ' ; precision = ', precision2*100, 
                    print ' ; recall = ',recall2*100,
                    if precision2+recall2 == 0: print '; F1 score = NA'
                    else:
                        f1_score2 = (2.0*precision2*recall2)/(precision2+recall2)
                        print ' ; F1 score = ', f1_score2, multiple_annots
                else: print '; precision = NA ; recall = NA ; F1 score = NA' 


    #plot_heatmaps(F1_matrix, P_matrix, R_matrix, "GIST_"+movie_name)
    annot_dict_list.append(annot_dict)
    all_char_counts.append(char_counts)
    ALL_F1_MATRIX.append(F1_matrix)
    ALL_P_MATRIX.append(P_matrix)
    ALL_R_MATRIX.append(R_matrix)
    precision_counts.append(prec_count)

F1_mean = mean(array(ALL_F1_MATRIX),0)
P_mean = mean(array(ALL_P_MATRIX),0)
R_mean = mean(array(ALL_R_MATRIX),0)
print "averaging all movies"
print P_mean, R_mean, F1_mean
#plot_heatmaps(F1_mean, P_mean, R_mean, 'GIST_MEAN_8movies')

F1_mean_7mov = mean(array(ALL_F1_MATRIX[:-1]),0)
P_mean_7mov = mean(array(ALL_P_MATRIX[:-1]),0)
R_mean_7mov = mean(array(ALL_R_MATRIX[:-1]),0)
print "averaging only 7 movies"
print P_mean_7mov, R_mean_7mov, F1_mean_7mov
#plot_heatmaps(F1_mean_7mov, P_mean_7mov, R_mean_7mov, 'GIST_MEAN_except_tangled')

PLOT1=False
if PLOT1:
    clf()
    figure(1)
    plot_overclust = []
    for i in [0,5,1,2,6,7,4,3]:
        plot_overclust.append(all_char_counts[i])
    tick_params(labelsize=14)
    boxprops = dict(linewidth=2)
    medianprops = dict(linewidth=2.5)
    boxplot(plot_overclust, boxprops=boxprops, medianprops=medianprops)
    xticks(range(1,9),['V'+str(i) for i in range(1,9)])
    yticks([0,5,10,15,20])
    xlabel('Movie',fontsize=12,weight='bold')
    ylabel('No. of exemplars per character',fontsize=12,weight='bold')

PLOT2=False
if PLOT2:
    clf()
    figure(1)
    tick_params(labelsize=14)
    plot_P = []
    for i in [0,5,1,2,6,7,4,3]:
        plot_P.append(precision_counts[i])
    correct = [i[0] for i in plot_P]
    correct[0]+=12
    wrong = [i[-1] for i in plot_P]
    wrong[0]-=12
    ind = np.arange(1,9)
    width = 0.6
    p1=bar(ind,correct,width, color="#808000")
    p2=bar(ind,wrong,width,bottom=correct, color="#8B4513")
    xticks(ind + width/2.,['V'+str(i) for i in range(1,9)])
    xlabel('Movie',fontsize=12,weight='bold')
    ylabel('No. of exemplars',fontsize=12,weight='bold')
    legend((p1[0], p2[0]), ('relevant exemplars', 'noisy exemplars'), fontsize=12)

with open(OUT_NAME,'w') as F:
    json.dump(annot_dict_list, F)
