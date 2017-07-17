Author: Krishna Somandepalli
Project Wiki: 

Affilitation: Signal Analysis and Interpretation Laboratory (SAIL)
University Southern California (USC)
Los Angeles, CA, USA
E-mail: somandep@usc.edu

Latest update: July 2017

This repository contains data relevant for publication titled:

"Unsupervised Discovery of Character Dictionaries in Animation Movies"
Krishna Somandepalli, Naveen Kumar, Tanaya Guha, Shrikanth Naryanan
SAIL, USC, Los Angeles, CA, USA

As of July 17 2017, this paper was accepted for publication in the IEEE Transactions in Multimedia pending modifications

Please refer to the paper to understand the different steps in the methodology and the referred scripts
The scripts are documented as per Figure 3 (Overview schematic diagram) image in the paper

1) MultiBox DNN Object Detector:
    -- video_deep_multibox_detect.py
2) Coarse detection of character candidates:
    -- choose_object_candidates.py
3) Saliency Constraints
    -- choose_object_candidates_SAL.py
4) Local Tracking in video
    -- easy_video_object_local_tracking.py
5) Clustering character candidates
    -- experimenter_cluster_candidates.py
6) Peformance Analysis Measures
    -- performance_analysis_measures.py

Mechanical Turk (MTurk) annotations and parsing scripts in directory:
mturk_annotations/
Clustering results in:
experimental_results/

Data and annotation labels:
Please refer to this link for the outputs and annotations obtained