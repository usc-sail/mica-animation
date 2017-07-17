'''
Author Krishna Somandepalli
Latest update: July 2017
email: somandep@usc.edu
This script contains routines to cluster images obtained from object detection
For different settings of image features and system parameters like saliency, object duration, etc

'''
import os, sys,json
from pylab import *
import numpy as np
from skimage.io import imread, imshow
import cv2
from sklearn import cluster
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import itertools
import operator

def accumulate(l):
    it = itertools.groupby(l, operator.itemgetter(0))
    for key, subiter in it:
        yield key, [item[1] for item in subiter]

#--- COS SIMILARITY
def do_APCluster(feat, sim="eucl", verbose=False):
    DAMP = 0.5
    affinity_prop = cluster.AffinityPropagation(damping=DAMP, affinity='precomputed')#,verbose=True)
    if sim == "cos": cos_sim = cosine_similarity(feat)
    elif sim == "eucl": cos_sim = -1*euclidean_distances(feat) 

    #----- AP ------
    af_cos = affinity_prop.fit(cos_sim)

    labels = af_cos.labels_
    centers = af_cos.cluster_centers_indices_

    N_clust = len(unique(labels))
    print '# clusters = ', N_clust

    counts=[]; means_=[]; vars_=[]; maxs_=[]; mins_=[]
    n_samples, n_dim = feat.shape
    summary_feat = np.zeros((N_clust, n_dim))
    sub_clusters=[]

    for i in unique(labels):
        count = len(labels[labels==i])
        sub_graph = feat[[j for j in range(len(labels))  if labels[j]==i],:]
        ap_tmp = affinity_prop.fit(cosine_similarity(sub_graph))
        #print i, count, len(unique(ap_tmp.labels_))
        sub_clusters.append(len(unique(ap_tmp.labels_)))
        if count>1:
            sub_graph_mean = mean(sub_graph,0)
            C = cosine_similarity(sub_graph)
            #pcolormesh(C,vmin=0, vmax=1); colorbar();show()
            #print '##------------------------------##'
            all_corr = np.tril(C,-1)
            mean_corr = mean(all_corr[all_corr!=0])
            means_+=[mean_corr]
            var_corr = var(all_corr[all_corr!=0])
            vars_+=[var_corr]
            maxs_ += [max(all_corr[all_corr!=0])]
            mins_ += [min(all_corr[all_corr!=0])]
            #print mean_corr, ' +- ',var_corr,';;',
        else:
            sub_graph_mean = sub_graph
        #print i, count
        counts+=[count]
        summary_feat[i,:]=sub_graph_mean

    counts = array(counts); means_ = array(means_); vars_ = array(vars_)
    maxs_ = array(maxs_); mins_ = array(mins_)
    return labels, centers, counts

'''---CODE STARTS HERE---'''
# Manually set these below variables for your environment


movie_list = [i[:-1] for i in open(sys.argv[1],'r').readlines()]

#link_template = "hyperlink(\"http://gosailaz.com/wp-content/uploads/2012/02/learn-how-to-sail.jpg\",image(\"ht\"))"

# for ease encode all experiments done in  diictionary
# for each: store the path to where objects are stored, what is the name of the json file
sal_experiment_dict={'SAL0': ['/proj/krishna/animation/tracking_cmt/NO_SAL/','track_detected_deep_multibox_objects_%s_Nby10_chosen.json'],\
        'SAL10': ['/proj/krishna/animation/tracking_cmt/','SAL_track_detected_deep_multibox_objects_%s_Nby10_chosen_SAL.json'],
        'SAL20': ['/proj/krishna/animation/tracking_cmt/experiments_SALIENCY/20pc_SAL','SAL_track_detected_deep_multibox_objects_%s_Nby10_chosen_SAL_20.json'],
        'SAL50': ['/proj/krishna/animation/tracking_cmt/experiments_SALIENCY/50pc_SAL','SAL_track_detected_deep_multibox_objects_%s_Nby10_chosen_SAL_50.json'],
        'SAL80': ['/proj/krishna/animation/tracking_cmt/experiments_SALIENCY/80pc_SAL','SAL_track_detected_deep_multibox_objects_%s_Nby10_chosen_SAl_80.json'],
        'SAL90': ['/proj/krishna/animation/tracking_cmt/experiments_SALIENCY/90pc_SAL','SAL_track_detected_deep_multibox_objects_%s_Nby10_chosen_SAL_90.json']}

objects_dir_name = 'Objects_SAL_tracked'

tracking_thresh_dict={'T1': 1,'T12':12, 'T24':24, 'T48':48, 'T120': 120}

objects_list_file_name = 'fpath_objects_%s.txt'

# for gist
#feature_repo = "/proj/krishna/animation/tracking_cmt/Objects_SAL_tracked/"
# for other
feature_repo = '/proj/krishna/animation/multibox_nn/N_objects/'

#feature_file_name = os.path.join(feature_repo,'%s_GIST.npy')
feature_file_name = os.path.join(feature_repo, '%s_fc7.npy')

# object chooser for experiment parameters
SAL_KEYS = ['SAL10'] #['SAL0', 'SAL10', 'SAL20', 'SAL50', 'SAL80', 'SAL90']
TRACK_KEYS = ['T1'] #,'T12', 'T24', 'T48', 'T120']

#--------------------end variables---------------------------------

EXPERIMENT_OUTPUT_DICT_LIST= []

for movie_name in movie_list:
    MOVIE_DICT = {}
    MOVIE_DICT['movie_name'] = movie_name

    all_objects_features = np.load(feature_file_name % (movie_name))
    all_objects_fpath = os.path.join(feature_repo, objects_list_file_name % (movie_name))
    all_movie_objects = [i[:-1] for i in open(all_objects_fpath,'rU').readlines()]
    # get frame numbers from object list - need to be integer frame numbers so make sure you split properly
    all_movie_frames = array([int(os.path.basename(i).split('.')[0].split('_')[0]) for i in all_movie_objects])
    MOVIE_DICT['all_objects_fpath']=all_movie_objects
    print 'ALL movie objects = ', movie_name, len(all_movie_frames)
    print '------------------------------------------------------------------------------'
    movie_exemplar_indices = []
    for sal_exp in SAL_KEYS:
        objects_dir_, track_json_file_name = sal_experiment_dict[sal_exp]
        objects_dir = os.path.join(objects_dir_, objects_dir_name)
        movie_fpath_list = os.path.join(objects_dir, objects_list_file_name % (movie_name))
        movie_objects = [i[:-1] for i in open(movie_fpath_list, 'rU').readlines()]
        object_names = [os.path.basename(i).split('.')[0].split('_') for i in movie_objects]
        frame_tracks = [(int(i[0]),int(i[1])) for i in object_names]
        print sal_exp, movie_name, len(frame_tracks)
        for track_key in TRACK_KEYS:
            SAL_THRESH_KEY = sal_exp+'_'+track_key

            track_len = tracking_thresh_dict[track_key]
            object_track_thresh = [i for i in frame_tracks if i[1]>track_len]
            object_track_frames = array([i[0] for i in object_track_thresh])
            object_track_lengths = array([i[1] for i in object_track_thresh])
            print sal_exp, track_key, movie_name, len(object_track_thresh),
            
            movie_frame_lookup_idx = [list(all_movie_frames).index(i) for i in object_track_frames]
            
            print len(movie_frame_lookup_idx)
            this_experiment_features = all_objects_features[movie_frame_lookup_idx,:]

            labels, centers, counts = do_APCluster(this_experiment_features)
            print centers
            

            label_lengths = [[object_track_lengths[i] for i in range(len(labels)) if labels[i]==k]\
                                    for k in set(labels)]
            movie_frame_lookup_exemplars_idx = array(movie_frame_lookup_idx)[centers]
            MOVIE_DICT[SAL_THRESH_KEY+'_index'] = list(movie_frame_lookup_exemplars_idx)
            fpath_list = list(array(all_movie_objects)[movie_frame_lookup_exemplars_idx])
            MOVIE_DICT[SAL_THRESH_KEY+'_fpath'] = fpath_list

            # store all labels!
            frame_cluster_labels = zip( labels.tolist(), object_track_frames.astype('str').tolist())
            l_dict = {}
            for label_ in set(labels):
                l_dict[label_] = [j for i,j in frame_cluster_labels if i==label_]

            MOVIE_DICT[SAL_THRESH_KEY+'_labels'] = l_dict.values()


            MOVIE_DICT[SAL_THRESH_KEY+'_counts'] = list(counts)
            MOVIE_DICT[SAL_THRESH_KEY+'_lengths'] = label_lengths
            #list(object_track_lengths[centers])

            movie_exemplar_indices+=list(movie_frame_lookup_exemplars_idx)
    
    common_movie_exemplars = unique(movie_exemplar_indices)
    MOVIE_DICT['common_exemplars'] = list(common_movie_exemplars)
    common_movie_exemplars_fpath = list(array(all_movie_objects)[common_movie_exemplars])
    MOVIE_DICT['common_exemplars_fpath'] = common_movie_exemplars_fpath 
    
    common_exemplars_str = '\n'.join(common_movie_exemplars_fpath)+'\n'
    out_file_name = 'GIST_common_exemplars_%s.txt' % (movie_name)
    SAVE_FILE = True
    if SAVE_FILE:
        with open(out_file_name,'w') as f: f.write(common_exemplars_str)
        f.close()
    EXPERIMENT_OUTPUT_DICT_LIST.append(MOVIE_DICT)

SAVE_FILE = True
if SAVE_FILE:
    with open('GIST_MASTER_EXPERIMENT_DICTIONARY_8_movies.json','w') as F:
        json.dump(EXPERIMENT_OUTPUT_DICT_LIST, F)
    F.close()



