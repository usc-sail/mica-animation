'''
Created on May 23, 2013

@author: Krishna Somandepalli
This script has a few functions that make handling files easy in python
Written originally for sorting through huge phenotypic CSV files

>> my_dict = read_csv(csv_file.csv)
# converts cvs_file.csv to a list of dictionaries : each list element is a dict 
with keys the same as headers in the csv

>> write_csv(my_dict, my_dict_to_csv.csv, ['key1', 'key2', '...'])
# write a dict list to csv with the order of keys or headers specified

>>pick_and_group(my_dict_list, row_ID_list.txt, ['key1', 'key3', '...'],'row_ID', out_file.csv)
# read a huge csv file in first by read_csv and then pick the row_ID you want to match - say a subject ID
# and then pick the variables you need for each of these IDs amd write out a csv file accordingly
'''
import csv
import json
import os
from collections import defaultdict, OrderedDict
import numpy as np
from datetime import datetime, date, time
from pylab import *

def read_csv(csv_input):
    csv_dictlist = []
    if os.path.isfile(csv_input):
        reader = csv.DictReader(open(csv_input, "U"))
        
        for line in reader:
            csv_dict = dict((k, v) for k, v in line.iteritems())
            csv_dictlist.append(csv_dict)
#        keys_ordered = [k for k, v in line.iteritems()]
#        print keys_ordered
    else:
        print csv_input, 'doesnt exist'
    return csv_dictlist

def write_csv(dict_input, out_file_name, key_order):
    if key_order: dict_keys=key_order
    else: dict_keys=dict_input[1].keys() 
    
    ordered_csv_dict_list=[]
    for dict_i in dict_input:
        ordered_csv_dict=OrderedDict([(var, '') for var in dict_keys])
        for var in dict_keys:
            if var in dict_i: ordered_csv_dict[var]=dict_i[var]
            else: ordered_csv_dict[var]=''
        ordered_csv_dict_list.append(ordered_csv_dict)
 
    print dict_keys, 'headers in the output csv'  
    
    dict_list=[]
    dict_list.append(dict_keys)
    for d in ordered_csv_dict_list:
        dict_list.append([d[k] for k in dict_keys])
            
    out_csv_file = open(out_file_name, 'wb')
    wr = csv.writer(out_csv_file, dialect='excel')
    wr.writerows(dict_list)
    return dict_list


def pick_and_group(dump_dict, subs_list, vars_list, match_variable, out_csv_name):
        if not vars_list: vars_list=sorted(dump_dict[0].keys())
        if match_variable=='': match_variable="SCAN_Subject_ID"
        phen_dict=[]
        for sub in subs_list:
            phen_sub_dict={}
            phen_sub_dict['sub']=sub
            dump_d_ = [d for d in dump_dict if d[match_variable]==sub]
            if dump_d_:
                for dump_d in dump_d_:
                    #dump_d=dump_d_[0]
                    for var in vars_list:
                        if var in dump_d.keys():
                            phen_sub_dict[var]=dump_d[var]
                        else:
                            phen_sub_dict[var]='NA_VAR'
            else:
                print sub, '..absent in dump'
                for var in vars_list:
                    phen_sub_dict[var]='NA_DUMP'
            
            phen_dict.append(phen_sub_dict)
            
        write_csv(phen_dict, out_csv_name , ['sub'] + vars_list)
        return phen_dict


def pick_and_group2(dump_dict, subs_list, vars_list, match_variable, out_csv_name):
        if not vars_list: vars_list=sorted(dump_dict[0].keys())
        if match_variable=='': match_variable="SCAN_Subject_ID"
        phen_dict=[]
        for sub in subs_list:
            phen_sub_dict={}
            phen_sub_dict['sub']=sub
            dump_d_=[]
            for d in dump_dict:
                if '/' in d[match_variable]:
                    if sub in d[match_variable].split('/'):
                        dump_d_.append(d)
                else:
                    if d[match_variable]==sub:
                        dump_d_.append(d)
            if dump_d_:
                print len(dump_d_)
                for dump_d in dump_d_:
                    #dump_d=dump_d_[0]
                    for var in vars_list:
                        if var in dump_d.keys():
                            phen_sub_dict[var]=dump_d[var]
                        else:
                            phen_sub_dict[var]='NA_VAR'
                    
                    phen_dict.append(phen_sub_dict)
            
            else:
                print sub, '..absent in dump'
                for var in vars_list:
                    phen_sub_dict[var]='NA_DUMP'
            
            
            
        write_csv(phen_dict, out_csv_name , ['sub'] + vars_list)
        return phen_dict


