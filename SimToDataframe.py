#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import re
import os
import math


# In[2]:


def LoadSim(sim_file, directory=os.getcwd(), check_id='ANNI'):
    """loads a simulation file and takes as imput the name of the
    simulation file ('sim_file') and the ID which should be checked for later ('check_id')"""
    file = os.path.join(directory, sim_file)
    
    origin = 'caused by ' + check_id
    ID = 'event id'
    
    # set the column names
    columns = ['key', 'X', 'Y', 'Z', 'E', 'Time', origin, ID] + [i for i in range(0, 15)]
    
    # load and seperate the important colums (first 8) of the sim file
    sim = pd.read_csv(file, names=(columns), sep=";")
    sim['key'] = sim['key'].astype(str)
    # select lines that start with TI, HT, IA or ID (ID is needed to check where a new measurement starts)
    sim = sim.loc[sim['key'].str.startswith('TI') | sim['key'].str.startswith('HT') | sim['key'].str.startswith('IA') | sim['key'].str.startswith('ID')]
    
    # turn the last two columns into integers because they are meant to be id's
    sim[ID] = round(sim[ID]).astype(pd.Int64Dtype())
    # reset the indexing variable
    sim = sim.reset_index(drop=True)
    
    return sim


# In[3]:


def UsableHits(filename, directory=os.getcwd(), string='ANNI', Print=True):
    """print the percentage of "ANNI" events that don't escape ("ESCP") and the resulting percentage of reconstructable events"""
    sim = LoadSim(filename, directory=directory)
    
    # search for event idx's
    idx_anni = sim.loc[(sim['key'].str[3:7]==string) & (sim[14]>510.) & (sim[14]<512.)].index
    
    pop = []
    for i in range(len(idx_anni)):
        if not (idx_anni[i]+1==idx_anni[(i+1)%len(idx_anni)] or idx_anni[i]-1==idx_anni[i-1]):
            pop.append(i)
    
    idx_anni = idx_anni.delete(pop)
    
    col = sim.columns
    av = sim.iloc[idx_anni]
    idx_id = sim.loc[sim['key'].str[:2]=='ID'].index.values

    anni_count = int(len(idx_anni)/2)

    usable_events = 0
    hit_count = 0
    
    # look at sub_sim (sim from hit index to closest smaller ID index) and check for instant escape ('ESCP') and count hits and escapes
    for i in range(anni_count):
        lower_limit = idx_id[idx_id<idx_anni[i*2]][-1]
        try:
            upper_limit = idx_id[idx_id>idx_anni[i*2]][0]
        except:
            upper_limit = len(sim)

        sub_sim = sim.iloc[lower_limit:upper_limit]
        sub_sim = sub_sim.loc[sub_sim['key'].str[:2]=='IA']

        pointer1 = float(av.iloc[2*i, 0][-2:])
        pointer2 = float(av.iloc[2*i+1, 0][-2:])

        b_1 = any(sub_sim.loc[sub_sim['X']==pointer1]['key'].str[3:7].values!='ESCP')
        b_2 = any(sub_sim.loc[sub_sim['X']==pointer2]['key'].str[3:7].values!='ESCP')

        if b_1 and b_2:
            usable_events += 1

        if b_1:
            hit_count += 1

        if b_2:
            hit_count += 1
    
    # print percentages for each file
    usable_events_percent = usable_events/anni_count*100.
    hit_count_percent = hit_count/(anni_count*2)*100.
    if Print:
        print('File {}:'.format(filename))
        print('Usable events:      {:.2f}%'.format(usable_events_percent))
        print('Percentage of Hits: {:.2f}%'.format(hit_count_percent))
    
    return usable_events_percent, hit_count_percent


# In[4]:


def PrintUsableEvents(directory):
    """UsableHits for all sim files in "directory" """
    usable_events_percent = []
    hit_percent = []
    
    # loop over sim files in directory and do "UsableHits" for each
    for filename in os.listdir(directory):
        if filename[-4:]=='.sim':
            uep, hcp = UsableHits(filename, directory)
            usable_events_percent.append(uep)
            hit_percent.append(hcp)
    print('--------------------------------------------------')
    print('Mean usable events percentage: {:.2f}%'.format(sum(usable_events_percent)/len(usable_events_percent)))
    print('Mean hit percentage: {:.2f}%'.format(sum(hit_percent)/len(hit_percent)))

    return hit_percent, usable_events_percent


# In[5]:


def PreviousEvents(sim_file, directory=os.getcwd()):
    """loads the simulation file ('sim_file') and gets the end of each HT line
    which correspond to the id of the previous event
    it returns a Pandas 'Series' with lists of the previous ids"""
    file = os.path.join(directory, sim_file)
    
    # find all previous events that caused 'HTsim' (at the end of HTsim line)
    previous_events = pd.read_csv(file, names=(0,))
    previous_events = previous_events.astype(str)
    # find all lines starting with HT and only take everything after the 64th character (this is where to find the events id's before the hit)
    previous_events = previous_events.loc[previous_events[0].str[:2]=='HT'].loc[:,0].str[64:]
    # some lines start at the 65th character, those will be corrected here
    previous_events.loc[previous_events.str.startswith(';')] = previous_events.loc[previous_events.str.startswith(';')].str[1:]
    previous_events = previous_events.str.split(';').reset_index(drop=True)
    
    return previous_events


# In[6]:


def FindRowsWithKey(sim, key='ID', column='key'):
    """returns a list of the indexs for lines starting with 'key'"""
    return sim.loc[sim[column].str.startswith(key)].index


# In[7]:


def TotalEventsWithKey(sim, check_id='ANNI'):
    """takes a simulation dataframe returns the total number of 'check_id' events ('event_count')
    and a list with all origin positions ('all_event_positions')"""
    
    origin = sim.columns[6]
    
    # get all lines with 'check_id' in the 'key'
    idx = sim.loc[(sim['key'].str[3:7]==check_id) & (sim[14]>510.) & (sim[14]<512.)].index
    
    # check for idx difference, drop everything !=0
    pop = []
    for i in range(len(idx)):
        if not (idx[i]+1==idx[(i+1)%len(idx)] or idx[i]-1==idx[i-1]):
            pop.append(i)
    idx = idx.delete(pop)
    
    event_count = int(len(idx)/2)
    
    # get the x, y, z position of the events
    all_event_positions = sim.loc[idx, ['E', 'Time', origin]].copy()
    all_event_positions.rename(columns={0: 'x_origin', 1: 'y_origin', 2: 'z_origin'})
    
    return (sim, event_count, all_event_positions)


# In[8]:


def ListOfEventIds(events):
    """Creates a list of events where each event is a list of the ANNI ids"""
    event_count = len(events)
    loe = []
    index = events.index.values
    i = 0
    # go through events matrix and group the two hits making up one event into a list
    while i < int(index.shape[0]/2):
        pos1 = events.loc[index[i*2]].values
        pos2 = events.loc[index[i*2+1]].values
        # can now be that only a single one of annihilation photons fulfills the energy condition (510keV<E<512keV) -> sort them out
        if index[i*2+1]-index[i*2]>1 or any(pos1!=pos2):
            index = np.delete(index, i*2)
        else:
            i += 1
    
    # create the list of events from the remaining hits
    for i in range(int(index.shape[0]/2)):
        loe.append([index[i*2], index[i*2+1]])
    return loe


# In[9]:


def LocalToGlobalTime(sim, search_id='HT'):
    """takes a simulation dataframe ('sim') and reads out the global
    time from the 'TI' lines and adds it to the 'HT' lines time"""
    
    # add the global time to the specific times in 'Time'
    for i in range(len(sim)):
        # the classification are the first two letters (TI, HT, IA or ID)
        classification = sim.loc[i, 'key'][:2]
        # get the global time from the 'key' value of TI and convert it from a string to a float
        if classification == 'TI':           
            global_time = pd.to_numeric(sim.loc[i, 'key'][3:])

        # what to do with HT lines:
        elif classification == search_id:
            # add the global time to all HTsim times
            sim.at[i, 'Time'] += global_time
    
    return sim


# In[10]:


def FindOrigin(sim, previous_events, ID_index, check_id="ANNI"): # check again
    """takes a simulation dataframe ('sim'), a list of event id's that caused 'HT' ('previous_events')
    and checks if any of the previous events have been 'check_id' events and tracks them by setting
    'origin' to the idx of the causing annihilations."""
    
    origin = sim.columns[6]
    check_id = origin.split()[-1]
    ID = sim.columns[7]
    
    # get the indexes of all 'HT' lines
    HT_sim_with_check_id = pd.Int64Index([])
    
    # for improved speed locate ANNI lines beforehand and only loop over HT lines with ANNI lines in previous physics (IA)
    x = sim.loc[(sim['key'].str[3:7]==check_id) & (sim[14]>510.) & (sim[14]<512.)].index[::2]
    
    if len(x)>0:
        # check from which index to which index the hits and interactions caused by a photon go
        insert_pos_anni = np.searchsorted(ID_index, x)
        closest_lower_ID_index = ID_index[insert_pos_anni-1]
        if insert_pos_anni[-1] == len(ID_index):
            closest_upper_ID_index = ID_index[insert_pos_anni[:-1]]
            closest_upper_ID_index = closest_upper_ID_index.append(pd.Index([len(sim)]))
        else:
            closest_upper_ID_index = ID_index[insert_pos_anni]
        
        # search for protons that created an annihilation event
        for i in range(len(closest_lower_ID_index)):
            z = sim.loc[closest_lower_ID_index[i]+2:closest_upper_ID_index[i], 'key']
            new_idx = z.loc[z.str.startswith('HT')].index
            HT_sim_with_check_id = HT_sim_with_check_id.append(new_idx)

        HT_sim_with_check_id = pd.Int64Index(set(HT_sim_with_check_id))

        HT_sim = sim.loc[sim['key'].str.startswith('HT')].index
        
        # create a list of all hits that should be checked
        intersect_pos = np.intersect1d(HT_sim, HT_sim_with_check_id, return_indices=True)[1]
        HT_sim = sorted(HT_sim_with_check_id)
        previous_events = previous_events[intersect_pos].reset_index(drop=True)
        insert_pos_HT = ID_index[np.searchsorted(ID_index, HT_sim)-1]

        event_id_index = []
        hits = []
        
        # loop through all hits that have to be checked
        for i in range(len(HT_sim)):
            """in this part we loop backwards through the 'IA' events and
            the cause of each event until we reach the start (origin_event_id = 0)"""
            HT_idx = HT_sim[i]

            # check index where the current measurement starts
            closest_ID_index = insert_pos_HT[i]
            all_event_ids = []
            all_HTcausing_ids = [int(l) for l in previous_events.iloc[i]]

            # loop over all previous events
            for j in all_HTcausing_ids:
                origin_event_id = j
                # load all 'IA' lines into sub_sim
                sub_sim = sim.loc[closest_ID_index+2:HT_idx]
                sub_sim = sub_sim.loc[sub_sim['key'].str.startswith('IA')]
                # loop backwards through the 'IA' block until we reach the start (id = 0)
                while origin_event_id != 1:
                    # get the index of the 'IA' that is being pointed at
                    origin_event_index = sub_sim.loc[sub_sim['key'].str.endswith(' ' + str(origin_event_id))].index
                    # get the event id and append it
                    origin_event_id = int(sub_sim.loc[origin_event_index, 'X'])
                    string = sub_sim.loc[origin_event_index, 'key'].values[0][3:7]
                    
                    # save HT index and origin_event_index
                    if string == check_id and 510.<sub_sim.loc[origin_event_index, 14].values and sub_sim.loc[origin_event_index, 14].values<512.:
                        if HT_idx not in hits:
                            hits.append(HT_idx)
                            event_id_index.append([origin_event_index[0]])
                        else:
                            k = hits.index(HT_idx)
                            event_id_index[k].append(origin_event_index[0])
        # append causing annihilations photon index/ID to each hit
        sim.loc[:, origin] = np.zeros(len(sim), dtype=object)
        for i in range(len(hits)):
            sim.at[hits[i], origin] = event_id_index[i]
    else:
        sim.loc[:, origin] = pd.Series(np.zeros(len(sim), dtype=object))
    
    sim = sim.drop([i for i in range(0, 15)], axis=1)
    # get rid of all IA lines
    sim = sim.drop(sim.loc[sim['key'].str.startswith('IA')].index)
    # delete the no longer needed 'key' column
    del sim['key']
    del sim[ID]
    
    # drop the empty rows
    sim = sim.dropna()
    return sim


# In[11]:


def AppendDetectors(sim, detector_file, directory=os.getcwd(), pos=5):
    """takes a simulation dataframe ('sim') and appends a file of detector id's ('detector_file') at positon ('pos')"""
    file = os.path.join(directory, detector_file)
    # load the list of detectors
    with open(file) as f:
        detector_array = f.read().split("\n")
    
    if len(detector_array[-2])>3:
        detector_array = np.array(detector_array[:-2], dtype=int)
    else:
        detector_array = np.array(detector_array[:-1], dtype=int)
    
    detectors = np.sort(np.unique(detector_array))
    
    # handle cases where simulation accidently stopped earlier than intended and make the detector id vector and dataframe the same length
    if len(detector_array)!=len(sim):
        print("Mismatch between lenght of sim and detector list.", "\ndetector length: m", len(detector_array), "sim length: ", len(sim), "\n", file)
        with open("New Empty File.txt", "a") as myfile:
            myfile.write(file+"   "+str(len(sim))+"   "+str(len(detector_array))+"\n")
        if len(detector_array)>len(sim):
            sim.insert(pos,'Detector', detector_array[:len(sim)], True)
        else:
            sim = sim.iloc[:len(detector_array)]
            sim.insert(pos,'Detector', detector_array, True)
    else:
        # insert the list into the dataframe
        sim.insert(pos,'Detector', detector_array, True)
    
    return (sim, detectors)


# In[123]:


# """Slower version, but more versitile"""
# class signal_decay():
#     """
#     class that models a signal decay function as exponential decay E*exp(-alpha*t+t_hit)
#     can be initialised with set energy 'E', decay parameter 'alpha', and hit time 't_hit' values
#     can be called with numpy array, or float of time values 't'
#     """
#     def __init__(self, E, t_hit, alpha):
#         self.E = E
#         self.t_hit = t_hit
#         self.alpha = alpha
#     def __call__(self, t):
#         if isinstance(t, (float, int)):
#             if t >= self.t_hit:
#                 return self.E*math.exp(self.alpha*(-t+self.t_hit))
#             else:
#                 return 0.0
#         elif type(t).__module__ == np.__name__:
#             r = np.zeros(len(t))
#             past_hit = t>=self.t_hit
#             r[past_hit] = self.E*np.exp(self.alpha*(-t[past_hit]+self.t_hit))
#             return r
#         else:
#             print("Incompatible data type for signal_decay, only takes numpy array or float")
#             return


# In[124]:


class signal_decay():
    """
    class that models a signal decay function as exponential decay E*exp(-alpha*t+t_hit)
    can be initialised with set energy 'E', decay parameter 'alpha', and hit time 't_hit' values
    can be called with numpy array, or float of time values 't'
    """
    def __init__(self, E, t_hit, alpha):
        self.E = E
        self.t_hit = t_hit
        self.alpha = alpha
    def __call__(self, t):
        if t >= self.t_hit:
            return self.E*math.exp(self.alpha*(-t+self.t_hit))
        else:
            return 0.0


# In[125]:


# """Slower version, but more versitile"""
# class multiple_decays():
#     """
#     models multiple signal decays that will overlap using multiple instances of the 'signal_decay' class
#     can be initialized by a list or numpy array of the energies 'E_list', hit times 't_hit_list', and decay parameters 'alpha_list'
#     can be called with numpy array or float of time values 't'
#     """
#     def __init__(self, E_list, t_hit_list, alpha_list):
#         self.E_list = E_list
#         self.t_hit_list = t_hit_list
#         self.alpha_list = alpha_list
#         self.fcts = []
#         self.fct_count = len(E_list)
#         for i in range(self.fct_count):
#             self.fcts.append(signal_decay(E_list[i], t_hit_list[i], alpha_list[i]))
#     def __call__(self, t):
#         if isinstance(t, (float, int)):
#             return sum([i(t) for i in self.fcts])
#         elif type(t).__module__ == np.__name__:
#             number_of_values = len(t)
#             out = np.zeros(number_of_values)
#             for j in range(self.fct_count):
#                 out += self.fcts[j](t)
#             return out
#         else:
#             print("Incompatible data type for multiple_decays, only takes numpy array or float")
#             return


# In[126]:


class multiple_decays():
    """
    models multiple signal decays that will overlap using multiple instances of the 'signal_decay' class
    can be initialized by a list or numpy array of the energies 'E_list', hit times 't_hit_list', and decay parameters 'alpha_list'
    can be called with numpy array or float of time values 't'
    """
    def __init__(self, E_list, t_hit_list, alpha_list):
        self.E_list = E_list
        self.t_hit_list = t_hit_list
        self.alpha_list = alpha_list
        self.fcts = []
        self.fct_count = len(E_list)
        for i in range(self.fct_count):
            self.fcts.append(signal_decay(E_list[i], t_hit_list[i], alpha_list[i]))
    def __call__(self, t):
        return sum([i(t) for i in self.fcts])


# In[127]:


def MeasurementAndDeadTimeEnergyThreshold(sim, detectors=None, E_thresh=100.0, t_previous_window=200e-9, alpha=1e9/42, t_measure=800e-9, t_dead=7200e-9):
    """
    this function takes the simulation file and checks if a hit can cause a measurement by going above a threshold energy (E_thresh)
    after a measurement all hits within the measuremetn time window (t_measure) will be combined, meaning their energies are added up,
    the position is averaged with the energy as a weight and the time of the first hit is used
    after the t_measure there will be a dead time (t_dead) in which no hits will be recorded
    """
    df = pd.DataFrame(columns=list(sim.columns.values)+["tag_comb", "tag_drop", "E_update"])
    
    if detectors is None:
        detectors = sim["Detector"].unique()
    
    len_sim = len(sim)
    
    sim["tag_comb"  ] = np.zeros(len_sim, dtype=int   )
    sim["tag_drop"  ] = np.zeros(len_sim, dtype=int   )
    sim["E_update"  ] = np.zeros(len_sim              )
    sim["xyz_update"] = np.zeros(len_sim, dtype=object)
    
    # loop over all detectors
    for i in detectors:
        sub_sim = sim.loc[sim["Detector"]==i].sort_values("Time")
        
        j = 0
        # loop over all hits that are not tagged to be dropped or combined (already checked)
        for j in range(len(sub_sim)):
            hit_j = sub_sim.iloc[j]
            if hit_j["tag_drop"] == 0 and hit_j["tag_comb"] == 0:
                hit_time = hit_j["Time"]
                
                # check which hits cause a contribution to the final signal
                signal_contributing_hits = sub_sim.loc[(sub_sim["Time"]>=hit_time-t_previous_window) & (sub_sim["Time"]<=hit_time)]
                number_of_hits = len(signal_contributing_hits)
                # get the energy of a hit
                if number_of_hits>1:
                    t_hits = signal_contributing_hits["Time"].values
                    E_hits = signal_contributing_hits["E"].values
                    signal_function = multiple_decays(E_hits, t_hits, [alpha]*number_of_hits)
                    E_hit = signal_function(hit_time)
                else:
                    E_hit = sub_sim.loc[hit_j.name, "E"]
                
                # if higher then energy threshold tag all hits to be combined within detection time and dropped within dead time
                if E_hit >= E_thresh:
                    # combining hits
                    sub_sim.loc[(sub_sim["Time"]>=hit_time          ) & (sub_sim["Time"]<=hit_time+t_measure       ), "tag_comb"] = hit_j.name
                    # dropping hits
                    sub_sim.loc[(sub_sim["Time"]> hit_time+t_measure) & (sub_sim["Time"]<=hit_time+t_measure+t_dead), "tag_drop"] = 1
                    sub_sim.at[hit_j.name, "E_update"] = E_hit
                    # get energies and save measurement causing hit updated position
                    if number_of_hits>1:
                        Energies_at_hit_time = [i(hit_time) for i in signal_function.fcts[:-1]] + [E_hits[-1]]
                        xyz = signal_contributing_hits[["X", "Y", "Z"]].values
                        xyz_update = []
                        for j in range(3):
                            xyz_update.append(sum([xyz[l, j]*Energies_at_hit_time[l] for l in range(len(xyz))])/sum(Energies_at_hit_time))
                        sub_sim.at[hit_j.name, "xyz_update"] = xyz_update
                else:
                    sub_sim.at[hit_j.name, "tag_drop"] = 1
        df = df.append(sub_sim)
    
    # drop tagged hits
    df = df.drop(df.loc[df["tag_drop"]==1].index)
    
    E_update_idx = df.loc[df["E_update"]!=0].index
    df.loc[E_update_idx, "E"] = df.loc[E_update_idx, "E_update"].values
    
    df.drop(["tag_drop", "E_update"], axis=1, inplace=True)
    
    # get locations that have to be updated
    xyz_update_idx = df.loc[df["xyz_update"]!=0].index
    df.loc[df["xyz_update"]!=0, ["X", "Y", "Z"]] = list(df.loc[df["xyz_update"]!=0, "xyz_update"].values)
    df = df.drop(["xyz_update"], axis=1)
    
    drop = []
    d={}
    
    # create look up table
    val_old = df.iat[0, 7]
    for i in range(len(df)):
        val = df.iat[i, 7]
        if not val in d.keys():
            d[val] = [i]
        elif val_old == val:
            d[val].append(i)
        val_old = val
    
    # update measurement causing hit
    for idx in d.values():
        if len(idx)>1:
            E = df.iloc[idx, 3].values
            xyz = df.iloc[idx, [0, 1, 2]].values

            for j in range(3):
                df.iloc[idx[0], j] = sum([xyz[l, j]*E[l] for l in range(len(xyz))])/sum(E)
            # update energy
            df.iloc[idx[0], 3] = sum(E)

            sub_df = df.iloc[idx, 6]
            # update position
            if any(sub_df!=0):
                df.iat[idx[0], 6] = list(set([item for sublist in [i for i in sub_df.values if i!=0] for item in sublist]))
            
            drop += idx[1:]
    # drop all other hits that did not cause a measurement
    df.drop(df.iloc[drop].index, inplace=True)
    df.drop("tag_comb", axis=1, inplace=True)
    df = df.sort_values(by="Time")
    return df


# In[128]:


def TimeResolutionCheck(sim, detectors, time_resolution):
    """"
    in this part activations of the same detector in a time frame that is too small
    will be averaged in x, y, z weighted by the energy
    """
    origin = sim.columns[6]
    
    # create a new table to append the detectors to one after the other
    hit_table = pd.DataFrame(columns=['X', 'Y', 'Z', 'E', 'Time', 'Detector', origin])
    hit_table[origin] = hit_table[origin].astype('object')
    
    # go thrugh each detector
    for detector in detectors:
        drop = []
        
        # find all measurements taken by a single detector and save them to sub_sim and sort them after their time
        sub_sim = sim.loc[sim['Detector']==detector].sort_values('Time')
        # reset the index of sub_sim so the rows can easily be looped over
        sub_sim = sub_sim.reset_index(drop=True)
        
        # go over every activation of one detector
        for j in range(len(sub_sim)-1):
            # check if the time between activations can be resolved
            if sub_sim['Time'].iloc[j+1] - sub_sim['Time'].iloc[j] < time_resolution:
                sub_sim[origin] = sub_sim[origin].astype('object')
                
                # read out the energies
                E1 = sub_sim.loc[j  , 'E']
                E2 = sub_sim.loc[j+1, 'E']

                # calculate the energy weighted average of x, y, z and save them to the second activation (j+1)
                # so that it can be checked with the one after (j+2) in the next go
                for column in ['X', 'Y', 'Z']:
                    sub_sim.at[j+1, column] = (sub_sim.loc[j+1, column]*E2+sub_sim.loc[j, column]*E1)/(E2+E1)

                # calculate the energy sum the detector will see
                sub_sim.at[j+1, 'E'] = E2+E1
                
                val_1 = sub_sim.loc[j, origin]
                val_2 = sub_sim.loc[j+1, origin]
                
                dtype_1 = type(val_1)
                dtype_2 = type(val_2)
                
                if dtype_1 == list and dtype_2 == list:
                    sub_sim.at[j+1, origin] =  val_1  +  val_2
                elif dtype_1 == list and dtype_2 != list:
                    sub_sim.at[j+1, origin] =  val_1  + [val_2]
                elif dtype_1 != list and dtype_2 == list:
                    sub_sim.at[j+1, origin] = [val_1] +  val_2
                else:
                    sub_sim.at[j+1, origin] = [val_1, val_2]
                
                sub_sim.at[j+1, 'Time'] = sub_sim.loc[j, 'Time']
                
                # save all rows that have to be deleted in a list (at position j)
                drop.append(j)

        # append the one checked detector to the new table and repeat for every detector
        hit_table = hit_table.append(sub_sim.drop(drop))

    # sort the final table for time and reset the indexes
    hit_table = hit_table.sort_values(by = ['Time'])
    hit_table = hit_table.reset_index(drop=True)
    
    return hit_table


# In[129]:


def RemoveZero(df, column='caused by ANNI'):
    """removes all zeros from lists in 'column'"""
    for i in df.index:
        value = df.loc[i, column]
        if isinstance(value, list):
            if all([v == 0. for v in value]):
                df.at[i, column] = 0.
            else:
                df.at[i, column] = [j for j in value if j != 0.]
            
    return df


# In[130]:


def RemoveDuplicates(df, column='caused by ANNI'):
    """removes all duplicates from lists in 'column'"""
    idx = df.loc[df[column]!=0.].index.values
    for i in idx:
        value = df.loc[i, column]
        value = list(set(value))
        df.at[i, column] = value
    
    return df


# In[131]:


def CountHits(df, column='caused by ANNI'):
    """marks every hit of a photon as 1/(2**count) starting with count=0"""
    events = []
    df.sort_values(by='Time').reset_index(drop=True)
    idx = df.loc[df[column]!=0.].index.values
    df['mark'] = np.zeros(len(df))
    for i in idx:
        values = df.loc[i, column]
        if len(values) == 1:
            v = values[0]
            count = events.count(v)
            df.loc[i, 'mark'] = 1/(2**count)
            events.append(v)
        else:
            l = []
            for v in values:
                count = events.count(v)
                l.append(1/(2**count))
                events.append(v)
                
            df.loc[i, 'mark'] = max(l)
    
    return df


# In[132]:


def FromIdToEventNumber(df, loe): # check again
    """marks all photon hits from the same event as a number starting from one and increasing by 1"""
    index = df.loc[df['caused by ANNI']!=0.].index
    
    # create a dictionary for the events to quickly look up wich hit belongs to which event
    doe = {}
    for i, events in enumerate(loe):
        doe[events[0]] = i+1
        doe[events[1]] = i+1
    
    df['event number'] = np.zeros(len(df), dtype=object)
    
    for idx in index:
        value = df.loc[idx, 'caused by ANNI']
        l = []
        for v in value:
            if v in doe:
                x = doe[v]
                if not x in l:
                    l.append(x)
        if len(l) == 0:
            l = 0
        df.at[idx, 'event number'] = l
    return df


# In[133]:


def Test(x, i):
    """test if list 'x' contains the value 'i'"""
    if len(x)==1:
        return j==i
    else:
        return any([True for j in x if j == i])


# In[23]:


# def ClassifyPET(df, event_count):
#     """copy mark to 'classification' column if both photons from the same event hit the detectors"""
#     df['classification'] = np.zeros(len(df))
#     sub_df = df.loc[df["event number"]!=0]
#     for i in range(event_count):
#         idx = sub_df.loc[sub_df['event number'].apply(lambda x: i in x)].index #very slow
#         sub_df_2 = sub_df.loc[idx, 'mark']
#         if len(sub_df_2.loc[sub_df_2==1.0]) > 1:
#             df.loc[idx, 'classification'] = sub_df_2.values
            
#     return df


# In[134]:


def ClassifyPET(df, event_count):
    """copy mark to 'Classification' column if both photons from the same event hit the detectors"""
    df['Classification'] = np.zeros(len(df))
    sub_df = df.loc[df["event number"]!=0]
    
    x = sub_df["event number"].values
    
    d = {}
    for i in range(len(x)):
        for j in range(len(x[i])):
            if x[i][j] in d.keys():
                d[x[i][j]].append(sub_df.iloc[i].name)
            else:
                d[x[i][j]] = [sub_df.iloc[i].name]
    
    for i in d.keys():
        idx = d[i]
        sub_df_2 = sub_df.loc[idx, 'mark']
        # if 2 hits from same event both marked with 1.0
        if len(sub_df_2.loc[sub_df_2==1.0]) == 2:
            df.loc[idx, 'Classification'] = sub_df_2.values
            
    return df


# In[135]:


def FindValues(df, column):
    """gets all unique entries from a cloumn with list entries"""
    x = df.loc[df[column]!=0, column].values
    x = list(set([item for sublist in list(x) for item in sublist]))
    return x


# In[38]:


def CutEnergy(df, E_min=0, E_max=700):
    """sets all event hits where one photons energy is not between E_min and E_max to 0.0"""
    
    sub_df = df.loc[df['event number']!=0]
    
    x = sub_df["event number"].values
    
    # create look up table for "event number" and the corresponding index
    d = {}
    for i in range(len(x)):
        for j in range(len(x[i])):
            if x[i][j] in d.keys():
                d[x[i][j]].append(i)
            else:
                d[x[i][j]] = [i]
    
    # check if energies of each photon are within E_min and E_max, if not set classification to zero
    for i in d.keys():
        sub_df2 = sub_df.iloc[d[i]]
        idx = sub_df2.index
        values = sub_df2.loc[sub_df2['Classification']==1.0, 'E'].values
        for i in range(len(values)):
            if values[i]>E_max or values[i]<E_min:
                df.loc[idx, 'Classification'] = 0.
                break
    
    return df


# In[138]:


def CountEvents(df, event_count):
    """prints the different number of events"""
    print('Total number of annihilation events:          {}'.format(event_count)) # events with 2 photons and E=510.999
    print('Total number of annihilation photons created: {}'.format(event_count*2))
    
    photons_tracked_by_detector = len(FindValues(df, column='caused by ANNI'))
    if event_count==0:
        perc_phot = 0
    else:
        perc_phot = photons_tracked_by_detector/(event_count*2)*100
    print('Number of annihilation photons that got detected: {} ({:.2f}% of photons detected)'.format(photons_tracked_by_detector, perc_phot))
    
    tracked_events = int(len(df.loc[df['Classification']==1.])/2)
    if photons_tracked_by_detector==0:
        perc_events = 0
    else:
        perc_events = tracked_events*2/photons_tracked_by_detector*100
    print('Number of classified events:         {} ({:.2f}% of detected photons classified as events)'.format(tracked_events, perc_events))
    return {'event_count': event_count, 'photons_tracked_by_detector': photons_tracked_by_detector, 'tracked_events': tracked_events}


# In[139]:


def SaveNumberOfEvents(dictionary, folder_name='Data', file_name='Events'):
    """takes the number of events ('event_count', 'unique_events_measured',
    'unique_events_classified',) and saves it in the sub
    folder 'folder_name' under the name 'file_name' as .csv and .xlsx"""
    
    directory = os.getcwd()
    path = os.path.join(directory, folder_name)
    
    try:
        os.mkdir(path)
    except OSError:
        pass
    
    # save the numbers of events to a txt file
    f = open(path + '/' + file_name + '.txt','w+')
    f.write('total_events, photons_tracked_by_detector, tracked_events \n{}, {}, {}'
            .format(dictionary['event_count'], dictionary['photons_tracked_by_detector'], dictionary['tracked_events']))
    f.close()


# In[140]:


def NumberOfEvents(hit_table, recorded_events, event_count=None):
    """takes 2 dataframes 'hit_table' and 'recorded_events' which is the total table and all 'check_id'
    events tracked by atleast one detector as well as the total number of events
    this function prints out the number of total events (if given), the number of events measured
    and the number of events classified as 'check_id'
    returns the recorded_events indexes for unique events"""
    
    origin = hit_table.columns[6]
    check_id = origin.split()[-1]
    ID = hit_table.columns[8]
    
    # get all ID columns of rows with 'origin' column value not 0.0
    measured_events = hit_table.loc[hit_table[origin]!=0.][ID]
    unique_events_measured = FindUnique(measured_events, recorded_events)
    
    # get all events that were classified as 'check_id' events
    classified_events = hit_table.loc[hit_table['activation within time window']!=0.][ID]
    unique_events_classified = FindUnique(classified_events, recorded_events)
    
    # print the amount of events
    if event_count:
        print('{} {} events in total' .format(event_count, check_id))
    
    print('{} hits that at some point were {}' .format(len(unique_events_measured), origin))
    print('{} unique hits that were classified as {}' .format(len(unique_events_classified), check_id))
    
    # return the id lists
    return (unique_events_measured, unique_events_classified)


# In[141]:


def SaveTable(hit_table, folder_name='Data', file_name='Hit Table'):
    """takes a dataframe ('hit_table') and saves it in the sub
    folder 'folder_name' under the name 'file_name' as .csv and .xlsx"""
    
    directory = os.getcwd()
    path = os.path.join(directory, folder_name)
    
    if not os.path.isdir(path):
        os.mkdir(path)
    
    # save the table as '.csv' and '.xlsx' in the folder "Table"
    hit_table.to_csv(path + '/' + file_name + '.csv')
    if len(hit_table)<1048576:
        hit_table.to_excel(path + '/' + file_name + '.xlsx')


# In[142]:


def PrintTable(hit_table, start=0, end=100):
    """takes a dataframe, a start value ('start'),
    an end value ('end') and shows the table from
    row 'start' to row 'end'"""
    from IPython.display import display, HTML
    
    display(HTML(hit_table[start:end].to_html()))


# In[143]:


# plot positions of to_plot with check_id
def PlotEvents(to_plot, size=3., zoom=True, z_size=3.):
    """takes a dataframe ('to_plot') with the first
    three columns being the x, y and z coordinates
    of events and plots them as 2d histograms and
    a 3D scatterplot"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    get_ipython().run_line_magic('matplotlib', 'notebook')
    
    col = to_plot.columns[0:3]
    
    if zoom:
        fig1, axs1 = plt.subplots(1, 3, sharex='all', sharey='all', figsize=(size*3, size))
    else:
        fig1, axs1 = plt.subplots(1, 3, figsize=(size*3, size))
    
    for i in range(3):
        if zoom:
            axs1[i].hist2d(to_plot[col[i]], to_plot[col[(i+1)%3]], bins=100,
                           range=[[-z_size, z_size],[-z_size, z_size]], cmap='gray')
        else:
            axs1[i].hist2d(to_plot[col[i]], to_plot[col[(i+1)%3]], bins=100, cmap='gray')
        axs1[i].set_title(col[i]+' | '+col[(i+1)%3])
        axs1[i].set_xlabel(col[i])
        axs1[i].set_ylabel(col[(i+1)%3])
    plt.show()
    
    # plot all 'check_id' to_plot
    fig2 = plt.figure()
    axs2 = Axes3D(fig2)
    axs2.set_xlabel(col[0])
    axs2.set_ylabel(col[1])
    axs2.set_zlabel(col[2])
    if zoom:
        axs2.set_xlim(-z_size, z_size)
        axs2.set_ylim(-z_size, z_size)
        axs2.set_zlim(-z_size, z_size)
    axs2.scatter(to_plot[col[0]], to_plot[col[1]], to_plot[col[2]])


# In[144]:


def RearrangeColumns(df, col_list_should):
    """takes a list 'col_list_should' and rearanges collumns to that order"""
    col_list_is = df.columns.tolist()
    coloumns = [col_list_is[idx] for idx in col_list_should]
    df = df[coloumns]
    return df


# In[145]:


def CutLargeTimes(hit_table, cut_time=1e5):
    """cuts large times that will never actually appear in reality"""
    hit_table = hit_table.loc[hit_table['Time']<cut_time]
    return hit_table


# In[146]:


def TimeToTimeDiff(hit_table):
    """calculates the differents in time between rows and replaces the global time with that"""
    hit_table.iloc[1:-1, hit_table.columns.get_loc('Time')] = hit_table.time[2:].values - hit_table.time[1:-1].values
    hit_table = hit_table.drop(hit_table.tail(1).index)
    return hit_table


# In[147]:


def AddRandomValueToColumn(df, column='E', mu=[202., 307., 511., 662.], sigma=[12.46, 15.41, 21.55, 24.78]): # sigmas need to be increased
    """draws random gaussian value for a 'column' with standard deviation 'sigma' given for different values of the column 'mu'
    ('sigma', 'mu') should be points for the precission at different values ('mu')
    between these points a linear interpolation is used for values between or outside the given ones
    the mean value for the gaussian is the 'column' value and the standard deviation is taken from the linear interpolation"""
    
    f = lambda z, i: (sigma[i+1]-sigma[i])/(mu[i+1]-mu[i])*z+(mu[i+1]*sigma[i]-mu[i]*sigma[i+1])/(mu[i+1]-mu[i])

    for i in range(len(df[column])): 
        x = df[column].iloc[i]

        if x<mu[0]:
            l = 0
        else:
            l = [i for i in range(len(mu)-1) if mu[i]<x][-1]

        df[column].iloc[i] = np.random.normal(x, f(x, l), 1)
    
    df[column].loc[df[column]<0.] = 0.0
    return df


# In[148]:


def GaussianPerc(df, sigmaperc=0.12, column="E"):
    """add gausian noise to a 'column' with the standard deviation being a percentage ('sigmaperc') of the total value"""
    x = df.loc[:, column].values
    df.loc[:, column] = np.random.normal(x, x*sigmaperc)
    df.loc[df[column]<0., column] = 0.0
    return df


# In[149]:


def Gaussian(df, sigma=5e-10, column="Time"):
    """add gausian noise with standard deviation 'sigma' to a 'column'"""
    df.loc[:, column] = df[column].values + np.random.normal(0, sigma, len(df))
    df.loc[df[column]<0., column] = 0.0
    df = df.sort_values(by = [column])
    return df


# In[150]:


def CustomRound(x, prec=2, base=.05):
    """rounds a x to the 'base'"""
    return round(base * round(float(x)/base), prec)


# In[151]:


def RoundColumn(df, column='E', base=10., prec=1):
    """rounds up all valumes int the 'column'"""
    df[column] = df[column].apply(lambda x: CustomRound(x, base=base, prec=prec))
    return df


# In[152]:


def ShuffleEqualTime(table, column):
    """events happening at the same time get shuffled"""
    df = table.copy()
    values = df[column].values
    unique_values = list(set(values))
    for value in unique_values:
        sub_df = df.loc[df[column] == value]
        if len(sub_df)>1:
            idx = sub_df.index
            sub_df = sub_df.sample(len(sub_df))
            sub_df = sub_df.set_index(idx)
            df.loc[idx] = sub_df
    return df


# In[153]:


def ClassicalPET(df, E_max=410., E_min=580., time_window=2e-9): # very simple, no FOV check, no drop if multiple events classify
    """performs the classical PET detection (time window + energy window check)"""
    idx = df.loc[(df['E']<E_max) & (df['E']>E_min)].index
    events = []
    for i in range(len(idx)-1):
        time_diff = df.loc[idx[i+1], 'Time'] - df.loc[idx[i], 'Time']
        if time_diff < time_window and df.loc[idx[i+1], 'Detector'] != df.loc[idx[i], 'Detector']:
            events.append([idx[i], idx[i+1]])
        
    return events

def UpsideDown(df):
    return df.iloc[::-1]


# In[154]:


def DetectorDeadTime(sim, detection_time, dead_time, detectors):
    """throw out hits which happen in the same detector during the dead time of that detector"""
    hit_table = pd.DataFrame(columns=['X', 'Y', 'Z', 'E', 'Time', 'Detector', 'caused by ANNI'])
    for d in detectors:
        sub_sim = sim.loc[sim["Detector"]==d]
        sub_sim = sub_sim.reset_index(drop=True)
        t = sub_sim.loc[0, "Time"]
        drop = []
        for j in range(len(sub_sim)-1):
            t_new = sub_sim.loc[j+1, "Time"]
            if t_new>=t+detection_time+dead_time:
                t = t_new
            elif t_new>=t+detection_time:
                drop.append(j+1)
        hit_table = hit_table.append(sub_sim.drop(drop))

    hit_table = hit_table.sort_values(by = ['Time'])
    hit_table = hit_table.reset_index(drop=True)
    
    return hit_table


# In[155]:


def Split(path, perc):
    """split 'perc' percent data in 'path'+'/TrainData' into a train and 'TestData' folder"""
    import random
    import shutil
    files   = os.listdir(path+"/TrainData")
    numbers = [i.split("_")[1].split(".")[0] for i in files if i[-4:]==".txt"]
    test    = random.sample(numbers, int(len(numbers)*perc))
    for n in test:
        csv = [i for i in files if i[9:10+len(n)]==n+"-" and i[-4:]==".csv"][0]
        file_names = ["Events_"+n+".txt", csv, csv[:-4]+".xlsx"]
        for f in file_names:
            shutil.move(path+"/TrainData/"+f, path+"/TestData/"+f)


# In[ ]:


"""calculate distance between a point 'r' and a line between points 'p' and 'q'
the points should all be numpy arrays"""
def line_vec(p, q):
    x=p-q
    return x/np.linalg.norm(x)
def distance(p, q, r):
    pr = p-r
    n = line_vec(p, q)
    return np.linalg.norm(pr-np.dot(pr, n)*n)


# In[ ]:


def CutLORNotInFOVSphere(df, radius=1.7):
    """set classification to zero for events with an LOR that is not crossing
    a sphere with radius 'radius'"""
    idx = df.loc[df["event number"]!=0].index
    e = {}
    for i in idx:
        event_num = df.at[i, "event number"]
        for e_n in event_num:
            if not e_n in e.keys():
                e[e_n] = [i]
            else:
                e[e_n].append(i)
    
    for key in e:
        if len(e[key])>1:
            sub_df = df.loc[e[key]]
            idx = sub_df.loc[sub_df["Classification"]==1.0].index
            if len(idx)>1:
                p = df.loc[idx[0], df.columns[:3]]
                q = df.loc[idx[1], df.columns[:3]]
                if distance(p, q, np.zeros(3))>radius:
                    df.at[idx[0], "Classification"] = 0.0
                    df.at[idx[1], "Classification"] = 0.0
    
    return df


# In[156]:


def HitTableForFile(directory, ID, ID_pos_in_number_list=0, check_id='ANNI',
                    save_name='Hit-Table', save_folder_name="Data", time_difference=False, cut_time=1e5,
                    E_min=0., E_max=700., E_perc=True, E_mu=[202., 307., 511., 662.], E_sigma=[12.46, 15.41, 21.55, 24.78],
                    E_rounding_base=1., E_prec=1, rand_round_E=True, t_rounding_base=8e-9, t_prec=9,
                    rand_round_t=True, cut_E=True, time_sigma=8e-9, E_sigmaperc=0.12, 
                    pos_round=True, pos_round_base=1e-5, pos_round_prec=5, E_thresh=100.0,
                    t_previous_window=200e-9, alpha=1e9/42, t_measure=800e-9, t_dead=7200e-9,
                    drop_additional_info=False, CutLOR=False, radius=1.7):
    """
    create a hit table for a ".sim" (and "Volume") file with an 'ID' in a 'directory'
    (data preperation for machine learning)
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
    """
    
    sim_list = [str(ID)]
    for f in sorted(os.listdir(directory)):
        id  = re.findall(r'\d+', f)
        file_type = f[-4:]
        if len(id) > ID_pos_in_number_list and f[:3]!="Idx" and id[ID_pos_in_number_list] == str(ID) and (file_type == ".sim" or file_type == ".txt"):
            sim_list.append(f)
    if len(sim_list)>3:
        raise TypeError("More than one file with same ID ({})".format(ID))
    
    print(sim_list[1])
    hit_table = LoadSim(sim_list[1], directory, check_id)
    previous_events = PreviousEvents(sim_list[1], directory)
    ID_index = FindRowsWithKey(hit_table, key='ID')
    hit_table, event_count, events = TotalEventsWithKey(hit_table, check_id)
    loe = ListOfEventIds(events)
    hit_table = LocalToGlobalTime(hit_table, search_id='HT')
    hit_table = FindOrigin(hit_table, previous_events, ID_index)
    hit_table, detectors = AppendDetectors(hit_table, sim_list[2], directory)
    
    if True:
        hit_table = MeasurementAndDeadTimeEnergyThreshold(hit_table, detectors=detectors, E_thresh=E_thresh, t_previous_window=t_previous_window, alpha=alpha, t_measure=t_measure, t_dead=t_dead)
    else:
        hit_table = DetectorDeadTime(hit_table, t_measure, t_dead, detectors)
        hit_table = TimeResolutionCheck(hit_table, detectors, t_measure)
        hit_table = RemoveZero(hit_table)
        hit_table = RemoveDuplicates(hit_table)
    
    if pos_round:
        for col in ["X", "Y", "Z"]:
            hit_table = RoundColumn(hit_table, column=col, base=pos_round_base, prec=pos_round_prec)
    
    hit_table = CountHits(hit_table)
    hit_table = RemoveDuplicates(hit_table)
    hit_table = FromIdToEventNumber(hit_table, loe)
    hit_table = ClassifyPET(hit_table, event_count)
    
    if rand_round_E:
        if E_perc:
            hit_table = GaussianPerc(hit_table, sigmaperc=E_sigmaperc, column="E")
        else:
            hit_table = AddRandomValueToColumn(hit_table, column='E', mu=E_mu, sigma=E_sigma)
        hit_table = RoundColumn(hit_table, column='E', base=E_rounding_base, prec=E_prec)
    
    if cut_E:
        hit_table = CutEnergy(hit_table, E_min=E_min, E_max=E_max)
    d = CountEvents(hit_table, event_count)
    SaveNumberOfEvents(d, folder_name=save_folder_name, file_name=('Events_'+sim_list[0]))
    
    if time_difference:
        hit_table = TimeToTimeDiff(hit_table)
    
    if rand_round_t:
        hit_table = Gaussian(hit_table, sigma=time_sigma)
        hit_table = RoundColumn(hit_table, column='Time', base=t_rounding_base, prec=t_prec)
    
    if drop_additional_info:
        hit_table = hit_table.drop(['caused by ANNI', 'mark', 'event number'], axis=1)
    else:
        cols = hit_table.columns.tolist()
        hit_table = hit_table[cols[:6]+[cols[-1]]+cols[6:9]]
    
    if CutLOR:
        hit_table = CutLORNotInFOVSphere(hit_table, radius=radius)
    
    hit_table = hit_table.reset_index(drop=True)
    file_name = save_name + sim_list[1].split('.')[0]
    SaveTable(hit_table, folder_name=save_folder_name, file_name=file_name)
    table_list = [sim_list[0], hit_table]
    
    print(ID, "done")
    print('-'*92)
    return table_list


# In[ ]:


def HitTableForFileList(directory, ID_list=None, ID_pos_in_number_list=0, check_id='ANNI',
                        save_name='Hit-Table', save_folder_name="Data", time_difference=False, cut_time=1e5, E_min=0, E_max=700, E_perc=True,
                        E_mu=[202., 307., 511., 662.], E_sigma=[12.46, 15.41, 21.55, 24.78], E_rounding_base=1., E_prec=1,
                        rand_round_E=True, t_rounding_base=8e-9, t_prec=9, rand_round_t=True, cut_E=True, time_sigma=5e-10,
                        E_sigmaperc=0.12, pos_round=True, pos_round_base=1e-5, pos_round_prec=5, E_thresh=100.0,
                        t_previous_window=200e-9, alpha=1e9/42, t_measure=800e-9, t_dead=7200e-9, CutLOR=False, radius=1.7):
    """
    can take a directory, or a list of ID's and a directory and produce the corespoding hit tables to all '.sim' & '.txt' files
    (data preperation for machine learning)
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
    """
    
    if ID_list is None:
        ID_list = [re.findall(r'\d+', i)[ID_pos_in_number_list] for i in os.listdir(directory) if i[-4:]==".sim"]
        if len(ID_list)-len(set(ID_list)) != 0:
            raise TypeError("Duplicate id's in directory: ", directory)
    
    hit_tables = []
    for ID in ID_list:
        hit_tables.append(HitTableForFile(directory, ID, ID_pos_in_number_list=ID_pos_in_number_list, check_id=check_id,
                         save_name=save_name, save_folder_name=save_folder_name, time_difference=time_difference, cut_time=cut_time,
                         E_min=E_min, E_max=E_max, E_perc=E_perc, E_mu=E_mu, E_sigma=E_sigma, 
                         E_rounding_base=E_rounding_base, E_prec=E_prec, rand_round_E=rand_round_E,
                         t_rounding_base=t_rounding_base, t_prec=t_prec, rand_round_t=rand_round_t, cut_E=cut_E,
                         time_sigma=time_sigma, E_sigmaperc=E_sigmaperc, pos_round=pos_round, pos_round_base=pos_round_base, pos_round_prec=pos_round_prec,
                         t_previous_window=t_previous_window, alpha=alpha, t_measure=t_measure, t_dead=t_dead, CutLOR=CutLOR, radius=radius))
    
    return hit_tables

