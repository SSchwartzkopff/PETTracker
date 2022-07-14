#!/usr/bin/env python
# coding: utf-8

# # Data Preparation

# In[1]:


import numpy as np
import pandas as pd
import SimToDataframe
import re
import os
import math
import random
import multiprocessing
import re
from multiprocessing import Pool
from shutil import copyfile


# In[2]:


def OrderFiles(files, i=0, j=0):
    """
    bring the ".txt" and ".sim" files and their corresponding id together
    i, j are the position of the ids in the list of numbers in the files name
    """
    
    sim_list = []
    vol_list = []
    
    for file_name in files:
        sub_list = []
        
        id = re.findall(r'\d+', file_name)
        
        if file_name[-4:] == '.sim':
            
            sub_list.append(id[i])
            sub_list.append(file_name)
            sim_list.append(sub_list)
        elif file_name[-4:] == '.txt':
            sub_list.append(id[j])
            sub_list.append(file_name)
            vol_list.append(sub_list)
    
    file_list = []
    
    for i in range(len(sim_list)):
        for j in range(len(vol_list)):
            if sim_list[i][0] == vol_list[j][0]:
                file_list.append([sim_list[i][0], sim_list[i][1], vol_list[j][1]])
    return file_list


# In[3]:


def SimToHitTable(directory, cores, ID_pos_in_number_list=0, check_id='ANNI',
                  save_name='Hit-Table', save_folder_name="Data", time_difference=False, cut_time=1e5, E_min=0, E_max=700, E_perc=True, E_mu=[202., 307., 511., 662.], E_sigma=[12.46, 15.41, 21.55, 24.78],
                  E_rounding_base=1., E_prec=1, rand_round_E=True, t_rounding_base=8e-9, t_prec=9, rand_round_t=True, cut_E=True,
                  time_sigma=8e-9, E_sigmaperc=0.12, pos_round=True, pos_round_base=1e-5, pos_round_prec=5, E_thresh=100.0,
                  t_previous_window=200e-9, alpha=1e9/42, t_measure=800e-9, t_dead=7200e-9, done_dir=None, CutLOR=False, radius=1.7):
    """
    creates a hit table for simulation files in a directory, uses n 'cores' to run the preperation in parallel
    (Data preperation for machine learning)
    directory             - String      - directory in which the '.sim' files are saved
    ID                    - Integer     - ID of the simulation that is supposed to be used
    ID_pos_in_number_list - Integer     - position of the ID in the file names numbers (e.g. 0-F10-E10.sim 0 is the ID at position one of three)
    check_id              - Integer     - ID for the annihilation in the simulation file (should probably not be changed)
    save_name             - String      - under which name should the hit table be saved
    save_folder_name      - String      - choose the name of the folder in the current directory in which the data should be saved
    time_difference       - True/False  - should the time column be converted into the time differences between hits
    cut_time              - Float       - hits past that time will be deleted
    E_min                 - Float       - minimum energy of hit caused by an annihilation event
    E_max                 - Float       - maximum energy of hit caused by an annihilation event
    E_perc                - True/False  - should the gaussian noise be a percentage (True) or a linear interpolation between given sigmas for given mus (False)
    E_mu                  - List(Float) - list of mean values for gaussian noise for the energy, between which linear interpolation will be used
    E_sigma               - List(Float) - list of standard deviation values for gaussian noise for the energy, between which linear interpolation will be used
    E_rounding_base       - Float       - rounding base for the energy
    E_prec                - Integer     - rounding precision for the energy
    rand_round_E          - True/False  - should noise be added and the values be rounded for the energy column
    t_rounding_base       - Float       - rounding base for the time
    t_prec                - Integer     - rounding precision for the time
    rand_round_t          - True/False  - should noise be added and the values be rounded for the time column
    cut_E                 - True/False  - should only annihilation hits with energies between E_min and E_max classified as PET events
    time_sigma            - Float       - standard deviation for time noise
    E_sigmaperc           - Float [0,1] - percentage of energy that is used as sigma for the gaussian noise
    pos_round             - True/False  - should the positions be rounded
    pos_round_base        - Float       - rounding base for the x, y, z position
    pos_round_prec        - Integer     - rounding precision for the x, y, z position
    E_thresh              - True/False  - energy threshold, if energy of a hit goes above the hit will be tracked, if not it will be dropped (to disable set to 0)
    t_previous_window     - Float       - how much should be looked back in time to check the energy of the current 
    alpha                 - Float       - decay constant for the signal of a hit
    t_measure             - Float       - measurement time, all hits within this time window starting from the first hit will be combined
    t_dead                - Float       - dead time, all hits within this will be dropped, happens after t_measure
    CutLOR                - True/False  - should events with a line of response that does not cross a sphere of radius "radius" be ignored
    radius                - Float       - radius of the sphere for which line of responses not crossing it will be dropped
    done_dir              - String      - directory in wich already prepered files are stored, IDs of the files in the directory will be skipped
    """
    
    files = [i for i in os.listdir(directory) if i[-4:]==".sim" or i[-4:]==".txt"]
    if not (done_dir is None):
        already_done = [re.findall(r'\d+', i)[0] for i in os.listdir(done_dir) if i[-4:]==".csv"]
        files = [i for i in files if i[:3]!="Idx" and re.findall(r'\d+', i)[0] not in already_done]
    files = OrderFiles(files)
    files_per_core = math.floor(len(files)/cores)
    
    ID_list = []

    x = len(files)%cores
    for i in range(cores):
        ID_list.append([])
        if i+1 <= x:
            len_files = files_per_core+1
        else:
            len_files = files_per_core

        idx = random.sample(range(len(files)), len_files)
        for j in idx:
            ID_list[-1].append(files[j][0])

        for index in sorted(idx, reverse=True):
            del files[index]
        
    hit_tables = []
    if __name__ == '__main__':
        pool = Pool(cores)
        params = [(directory, ID_list[i], ID_pos_in_number_list, check_id,
                   save_name, save_folder_name, time_difference, cut_time, E_min, E_max, E_perc,
                   E_mu, E_sigma, E_rounding_base, E_prec, rand_round_E,
                   t_rounding_base, t_prec, rand_round_t, cut_E, time_sigma,
                   E_sigmaperc, pos_round, pos_round_base, pos_round_prec, E_thresh,
                   t_previous_window, alpha, t_measure, t_dead, CutLOR, radius) for i in range(cores)]
        hit_tables = pool.starmap(SimToDataframe.HitTableForFileList, params)
        pool.close()
        pool.join()
    return hit_tables


# Here variables can be enetered to run the datapreperation for simulation files obtained from MEGAlib. This needs to be done for any simulation files before the data can be inserted into the neural network.<br>
# The "directory" is the folder in which the simulation files ".sim" and the detector ID files ".txt" are saved. The "save_dir" is the folder relative to the programs location in which the simulation will be saved, any files in there will not be considered for the data preperation.<br>
# <br>
# For multiple simulation files 'SimToDataframe.SimToHitTable' will run on multiple cores tackeling multiple simulations in parallesl. The amount of cores can be changed with the value 'cores'.<br>
# For a single file one can use 'SimToDataframe.HitTableForFile' which will run for a single simulation in the 'directory' with an 'ID' in its name.<br>
# <br>
# The resulting '.csv' files can be used for the neural network. Additionally 'xlsx' files will be saved which can be easily checked.

# In[4]:


""" all files ".sim" and corresponding ".txt" files should be within the directory """
directory = "./Example Simulation"
save_folder = "Data"
""" choose the number of cores to use """
cores = 8
ID = 0

hit_tables = SimToHitTable(directory, cores, done_dir=directory+"/"+save_dir, save_folder=save_folder) # for multiple files
# df = SimToDataframe.HitTableForFile(directory, ID, save_folder=save_folder) # for a single ".sim" file

