#!/usr/bin/env python
# coding: utf-8

# # PETTracker

# In[1]:


import os
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import Normalize
from openpyxl import load_workbook
from IPython.display import display, HTML
import random
import optuna
import math 
import heapq
import collections
import json
from IPython.display import clear_output
import sys

get_ipython().run_line_magic('matplotlib', 'inline')


# # Variables<br>
# Every variable has a short description of what it does.

# In[2]:


# directiry in which the ".csv" files are saved for the training/validation and the test data set
train_valid_dir = ''
test_dir = './Example Data'

# maximum length of training and validation sequences (everything longer will be a new sequence)
sequence_length = 10000

# percantage of data in "train_valid_dir" to be used for validation
validation_size = 0.1

# set the batch size
batch_size = 16

# does some time pre selection where certain values are being thrown out (if no two hits within +-"cut_value" window of a hit, the hit will be dropped)
time_pre_selection = True
# the cut value is given in seconds
cut_value = 160e-9

# how large each sequence for gradient calculation is (truncated backpropagation through time with k1=k2=back_prop_length)
back_prop_length = 100

# maximum time difference value, used for time difference normalization
time_norm = 1e-7

# how much of the data should be used as percentage [0, 1]
data_perc = 1.0

# name under which the Optuna hyperparameter study should be saved
study_name = "PETTracker-study"

# should the hyperparameters be trained
train_optuna = False

# set the percentage weight of the validation loss for training with optuna (1.0 = only consider validation loss, 0.0 = only consider train loss)
val_loss_weight = 0.9

# hyper parameter tuning ranges
ranges = {
    "epochs"       : [   10,   30],
    "nodes"        : [   50,  300],
    "layers"       : [    2,    4],
    "dropout"      : [0.001,  0.2],
    "lr"           : [ 1e-4, 1e-1],
    "grad_clip"    : [ 1e-5,  1e1],
    "weight_decay" : [1e-10, 1e-5],
    "optimizer"    : ["Adam", "RMSprop", "SGD"]
}

# enqueue the best trial of a previous study
queue = False
study_name_load = ""

# how many trials should the hyperparameter run or should trials be run until the maximum limit (lim_trials=True) is reached
study_length = 1000
lim_trials = False

# hyperparameter tuning for a saved trial
load_model = False
load_model_path = "./Saved Networks"
load_file_name = "PETTracker"

# should the best studies trial be reproduced?
reproduce = False

# gives accuracy for data labelled 1 and 0 (expensive)
hit_acc = False

# should Optunas study stats be plotted
optuna_stats = True

# set seed for all random number generators
seed = 0

# throws out sequences with no labelled hits (PET hits) if "True"
cut_data = False


# # Defining all the functions<br>
# The most important functions are:<br>
# 

# In[3]:


def SetSeed(seed):
    # sets all required seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False # significant performance reduction


# In[4]:


def DataPreperation(save_dir, cut_data=True, cut_perc=1.0, norm=None, normalize=True, time_diff=True, cut_into_sequences=True, sequence_length=200, Print=True, time_pre_selection=True, cut_value=40.001e-9, E_norm=1000., time_norm=1e-7, data_perc=1.0):
    # load all ".csv" file into a list "data_list" ("hit_tables" is a list of all filenames)
    data_list = []
    hit_tables = []
    
    for file_name in sorted([i for i in os.listdir(save_dir) if i[-4:] ==".csv"]):
        df = pd.read_csv(save_dir + '/' + file_name,  index_col=0)
        if time_pre_selection:
            df = CutDownData(df, cut_value)
        data_list.append(df.iloc[:, :7])
        hit_tables.append(file_name)
    
    # print all loaded files
    if Print:
        print('Files loaded: ' + ', '.join(hit_tables))
    
    if time_diff:
        for i in range(len(data_list)):
            data_list[i].iloc[:-1, 4] = data_list[i].iloc[1:, 4].values-data_list[i].iloc[:-1, 4].values
            data_list[i].iloc[-1, 4]  = 0.
            data_list[i].iloc[:, 4]   = data_list[i].iloc[:, 4].round(9)
    
    # split up the "data_list" into a list of sequences of "labels", "measurement_data" and a list of the lengths of these sequences
    measurement_data = []
    labels = []
    
    if cut_into_sequences:
        # go through loaded data frames in "data_list"
        for df in data_list:
            table_length = len(df)
            number_of_sequences = int(np.ceil(table_length/sequence_length))

            # cut each sequence and append it to the list
            for j in range(number_of_sequences):
                idx = j*sequence_length
                # handle the special case when the list is longer than the set sequence length
                if j != number_of_sequences-1:
                    measurement_data.append(df.iloc[idx:idx+sequence_length, 0:-1].values)
                    labels.append(df.iloc[idx:idx+sequence_length, -1].values)
                else:
                    measurement_data.append(df.iloc[idx:, 0:-1].values)
                    len_diff = number_of_sequences*sequence_length-table_length
                    labels.append(df.iloc[idx:, -1].values)
    else:
        for df in data_list:
            measurement_data.append(df.iloc[:, 0:-1].values)
            labels.append(df.iloc[:, -1].values)
    
    if data_perc < 1.0:
        idx = [i for i in range(len(measurement_data)) if random.random()<data_perc]
        measurement_data = [measurement_data[i] for i in idx]
        labels = [labels[i] for i in idx]
    
    if cut_data:
        measurement_data, labels = CutRandomZeroLabel(measurement_data, labels, cut_perc)
    
    if normalize:
        # normalize each column
        measurement_data, norm = NormalizeDiffTimeData(measurement_data, E_norm=E_norm, norm=norm, time_norm=time_norm)
    
    # turn all lists of dataframes into torch tensors
    measurement_data = list(map(lambda x: torch.tensor(x), measurement_data))
    labels = list(map(lambda x: torch.tensor(x), labels))
    
    labels = Only01(labels)
    
    return (measurement_data, labels, norm, hit_tables)


# In[5]:


# fct that cuts out lines if time to previous and following lines both larger value
def CutDownData(df, value):
    drop = []
    columns = df.columns
    times = df.iloc[:, 4].values
    time_diff = times[1:]-times[:-1]
    d={}
    idx = df.loc[df[columns[-1]]!="0"].index
    for i in idx:
        v = df.at[i, columns[-1]]
        if not v in d.keys():
            d[v] = [i]
        else:
            d[v].append(i)
    
    for j in range(len(time_diff)-1):
        if time_diff[j]>value and time_diff[j+1]>value:
            drop.append(j+1)
            if df.iat[j+1, 6]==1.0:
                e_id = df.iat[j+1, -1]
                df.loc[d[e_id], columns[6]] = 0.0
    df = df.drop(drop)
    
    return df


# In[6]:


# normalization of values
def NormalizeDiffTimeData(measurement_data, E_norm=1000., columns=['x', 'y', 'z', 'E', 'time', 'detector'], time_norm=1e-8, norm=None):
    if not norm:
        norm = {}
        use_given_norm = False
    else:
        use_given_norm = True
    
    _, width = measurement_data[0].shape
    l = len(measurement_data)
    
    for i in range(width):
        if i == 0 or i == 1 or i == 2: # x, y, z, detector number
            if use_given_norm:
                minimum = norm[columns[i]+"_min"]
                for j in range(l):
                    measurement_data[j][:, i] -= minimum
                    
                maximum = norm[columns[i]]
                for j in range(l):
                    measurement_data[j][:, i] /= maximum
            else:
                min_value = []
                for j in range(l):
                    min_value.append(measurement_data[j][:, i].min())

                minimum = min(min_value)
                for j in range(l):
                    measurement_data[j][:, i] -= minimum

                max_value = []
                for j in range(l):
                    max_value.append(measurement_data[j][:, i].max())

                maximum = max(max_value)
                for j in range(l):
                    measurement_data[j][:, i] /= maximum

                norm[columns[i]+"_min"] = minimum
                norm[columns[i]] = maximum
        elif i == 3: # Energy
            if use_given_norm:
                E_norm = norm[columns[i]]
                for j in range(l):
                    measurement_data[j][:, i] /= E_norm
                    measurement_data[j][:, i] = np.clip(measurement_data[j][:, i], 0., 1.)
            else:
                for j in range(l):
                    measurement_data[j][:, i] /= E_norm
                    measurement_data[j][:, i] = np.clip(measurement_data[j][:, i], 0., 1.)
                norm[columns[i]] = E_norm
        elif i == 4: # time
            if use_given_norm:
                time_norm = norm[columns[i]]
                for j in range(l):
                    measurement_data[j][:, i] = np.clip(measurement_data[j][:, i], 0., time_norm)
                    measurement_data[j][:, i] /= time_norm
            else:
                for j in range(l):
                    measurement_data[j][:, i] = np.clip(measurement_data[j][:, i], 0., time_norm)
                    measurement_data[j][:, i] /= time_norm
                norm[columns[i]] = time_norm
        elif i == 5: # detector number
            if use_given_norm:
                m = norm[columns[i]]
                for j in range(l):
                    measurement_data[j][:, i] /= m
            else:
                max_value = []
                for j in range(l):
                    max_value.append(measurement_data[j][:, i].max())

                m = max(max_value)
                for j in range(l):
                    measurement_data[j][:, i] /= m

                norm[columns[i]] = m
    return measurement_data, norm


# In[7]:


# function that cuts random sequences without events
def CutRandomZeroLabel(measurement_data, labels, percent=1.0):
    l = []
    for i in range(len(measurement_data)):
        if random.random()<=percent and all(labels[i] == 0.):
            l.append(i)
    
    for j in sorted(l, reverse=True):
        measurement_data.pop(j)
        labels.pop(j)
    return (measurement_data, labels)


# In[8]:


def Only01(labels):
    """removes events labelled with values other than 1.0,
    these are compton scattering events that will not be predicted"""
    for i in range(len(labels)):
        pos = torch.nonzero((labels[i]>0.) & (labels[i]<1.))
        for j in pos:
            labels[i][j] = 0.
    return labels


# In[9]:


# create a dataset class
class MeasurementsDataset(Dataset):
    def __init__(self, measurement_data, labels):
        self.measurement_data = measurement_data
        self.labels = labels
        
    def __len__(self):
        return len(self.measurement_data)
    
    def __getitem__(self, index):
        return (self.measurement_data[index], self.labels[index])
    
# functions to move all data to the GPU if available

# check if a gaphics card is available
def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

# move data to device (graphics card if possible)
def to_device(data, device):
    # tensors can be directly send to the graphics card
    if torch.is_tensor(data):
        return data.to(device)
    
    # deal with lists
    elif isinstance(data, list):
        res = []
        for v in data:
            res.append(to_device(v, device))
        return res
    
    # deal with dictionaries
    elif isinstance(data, dict):
        res = {}
        for k, v in data.items():
            res[k] = to_device(v, device)
        return res
    
    # if all fails raise data type error
    else:
        raise TypeError("Invalid type for to_device")

# create a device data loader to wrap arround the dataloader and send everythin to the graphics card
class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)

# create a custom collate function for the dataloader to create a tuple of lists of tensors -> ([tensors], [tensors])
def custom_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return (data, target)


# In[10]:


class PETTracker(nn.Module):
    """here the class of the model is created this is where the model structure and methods are defined"""
    def __init__(self, batch_size, device, hidden_size, num_layers=2, dropout=0., bidirectional=True, input_dim=6):
        """during initialization one can set teh batch size, the device, the size of the hidden layers within the LSTM,
        the number of layers in the LSTM, the percentage of dropout and if the +LSTM should be a bidirectional one"""
        super().__init__()
        self.device      = device
        self.batch_size  = batch_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.dropout     = dropout
        
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        
        # initiate the LSTM and convolutional layers
        self.lstm        = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.conv        = nn.Conv2d(1, 1, (1, hidden_size*self.num_directions))
        self.dropout     = nn.Dropout(dropout)
        
        self.loss        = nn.MSELoss()
        
        self.init_hidden()
        self.double().to(device)
        
        self.hidden      = torch.zeros((self.num_layers*self.num_directions, self.batch_size, self.hidden_size), device=self.device, dtype=torch.float64)
        self.cell_state  = torch.zeros((self.num_layers*self.num_directions, self.batch_size, self.hidden_size), device=self.device, dtype=torch.float64)
        
    def init_hidden(self):
        # initialisation of the hidden and cell state, these can be overwritten with model.hidden = tensor
        self.hidden      = torch.zeros((self.num_layers*self.num_directions, self.batch_size, self.hidden_size), device=self.device, dtype=torch.float64)
        self.cell_state  = torch.zeros((self.num_layers*self.num_directions, self.batch_size, self.hidden_size), device=self.device, dtype=torch.float64)
    
    def detach_hidden(self):
        self.hidden      = self.hidden.detach()
        self.cell_state  = self.cell_state.detach()
        
    def forward(self, in_list):
        # pad and pack the given sequences (necessisary pre-processing for LSTM)
        lengths = [i.size()[0] for i in in_list]
        out = pad_sequence(in_list, batch_first=True)
        out = pack_padded_sequence(out, lengths, batch_first=True, enforce_sorted=False)
        
        # get the lstm output and all the hidden and cell states (hc)
        out, (self.hidden, self.cell_state) = self.lstm(out, (self.hidden, self.cell_state))
        
        # convert LSTM output back into input shape
        out, _ = pad_packed_sequence(out, batch_first=True)
        
        out = self.dropout(out)
        # get the dimensions correct for the usage of a convolutional layer
        out = torch.unsqueeze(out, 1)
        out = F.relu(out)
        # convolutional layer turns output for each line of size (hidden_size*num_directions) into a singel output
        out = self.conv(out)
        # cut values above 1.0 and below 0.0
        out = torch.clamp(out, 0., 1.)
        # fix dimensions
        out = torch.squeeze(out, 1)
        out = torch.squeeze(out, 2)
        return out
    
    # calculate loss
    def calc_loss(self, events, labels):
        prediction = self(events)
        return self.loss(prediction, pad_sequence(labels, batch_first=True))
    
    # validation for a single batch
    def validation_step(self, batch):
        (events, labels) = batch
        with torch.no_grad():
            prediction = self(events)
        return {'val_loss': self.loss(prediction, pad_sequence(labels, batch_first=True)).item()}
    
    # test validation dataset
    def validation_epoch_end(self, loss_acc):
        batch_losses = [x['val_loss'] for x in loss_acc]
        epoch_loss   = sum(batch_losses)/len(batch_losses)
        return {'val_loss': epoch_loss}
    
    # print current epochs performance
    def epoch_end(self, epoch, result, hit_acc=True):
        if hit_acc:
            print("Epoch [{}], last_lr: {:.2e}, train_loss: {:.5f}, val_loss: {:.5f}  |  [train] hit_acc: {:.4f}, non_hit_acc: {:.4f}  |  [valid] hit_acc: {:.4f}, non_hit_acc: {:.4f}".format(epoch+1, result['lrs'][-1], result['train_loss'], result['val_loss'], result['train_hit_acc'], result['train_non_hit_acc'], result['val_hit_acc'], result['val_non_hit_acc']))
        else:
            print("Epoch [{}], last_lr: {:.2e}, train_loss: {:.5f}, val_loss: {:.5f}".format(epoch+1, result['lrs'][-1], result['train_loss'], result['val_loss']))
        
    # calculate loss on a validation data set
    def evaluate(self, dl):
        self.eval()
        self.init_hidden()
        loss_acc = [self.validation_step(batch) for batch in dl]
        return self.validation_epoch_end(loss_acc)


# In[11]:


# get the current learning rate from the learning rate scheduler
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# In[12]:


# accuracy for label=1.0
def HitAccuracy(model, dl):
    model.eval()
    error = []
    
    for batch in dl:
        hits, labels = batch
        with torch.no_grad():
            prediction = model(hits)
        
        error_positions = torch.nonzero(labels)
        
        if len(error_positions)>0:
            pos = labels==1.
            error.append(torch.mean(torch.abs(labels[pos]-prediction[pos])).item())
            
    return abs(1-sum(error)/len(error))

# accuracy for label=0.0
def NonHitAccuracy(model, dl):
    model.eval()
    error = []
    
    for batch in dl:
        hits, labels = batch
        with torch.no_grad():
            prediction = model(hits)
        
        if len(torch.nonzero(labels))>0:
            pos = labels==0.
            error.append(torch.mean(torch.abs(labels[pos]-prediction[pos])).item())
        
    return abs(1-sum(error)/len(error))


# In[13]:


def FitOneCycle(epochs, model, train_dl, valid_dl, lr, opt="Adam", back_prop_length=200, weight_decay=0, grad_clip=None, eval_test=False, hit_acc=False):
    """
    fits a model for 'epochs' times the dataset
    epochs           - number of times the gradient descent should go over the training dataset
    model            - the PETTracker model instance that should be fitted
    train_dl         - training data-loader
    valid_dl         - validation data-loader
    lr               - maximum learning rate for learning rate scheduler
    opt              - name of the optimizer
    back_prop_length - how large each sequence for gradient calculation is (truncated backpropagation through time with k1=k2=back_prop_length)
    weight_decay     - how much the network is punished for weights and biases
    grad_clip        - sets a maximum gradient, default is no gradient clip
    eval_test        - should the performance on the test dataset be printed
    hit_acc          - how good is the network with hits labelled 1 and hits labelled 0 seperated
    """
    device = get_default_device()
    torch.cuda.empty_cache()
    history = []
    
    optimizer = getattr(optim, opt)(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    num_seq = int(math.ceil(sequence_length/back_prop_length))
    
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs, steps_per_epoch=len(train_dl)*num_seq)
    for epoch in range(epochs):
        model.train()
        train_losses = []
        lrs = []
        for (hits, labels) in train_dl:
            model.init_hidden()
            hits.sort(key=lambda x: x.size()[0], reverse=True)
            labels.sort(key=lambda x: x.size()[0], reverse=True)
            for i in range(num_seq):
                # take a sequence of length 'back_prop_length' from the input data to calculate its gradient
                current_hits = to_device([hits[j][i*back_prop_length:(i+1)*back_prop_length, :]  for j in range(len(hits)  ) if hits[j].size()[0]  >i*back_prop_length], device)
                current_label  = to_device([labels[j][i*back_prop_length:(i+1)*back_prop_length] for j in range(len(labels)) if labels[j].size()[0]>i*back_prop_length], device)
                
                out = model(current_hits) # takes 1/3rd of the time
                loss = model.loss(out, pad_sequence(current_label, batch_first=True))
                loss.backward() # takes 2/3rd of the time
                
                if grad_clip: 
                    nn.utils.clip_grad_value_(model.parameters(), grad_clip)
                
                optimizer.step()
                optimizer.zero_grad()
                
                train_losses.append(loss.detach())
                model.detach_hidden()
                
                lrs.append(get_lr(optimizer))
                sched.step()
        
        result = model.evaluate(valid_dl)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        
        if hit_acc:
            result['train_hit_acc'] = HitAccuracy(model, train_dl)
            result['train_non_hit_acc'] = NonHitAccuracy(model, train_dl)
            result['val_hit_acc'] = HitAccuracy(model, valid_dl)
            result['val_non_hit_acc'] = NonHitAccuracy(model, valid_dl)
            SetSeed(seed)
        
        model.epoch_end(epoch, result, hit_acc)
        history.append(result)
        
        if eval_test:
            test_result = evaluate(model, test_dl)
            print("test_loss: {:.4f} test_acc: {:.4f}".format(result['val_loss'], result['val_acc']))
    return history


# In[14]:


def LoadModelParams(file_name, path="./Saved Networks"):
    """loads model parameters from a txt file named 'file_name.txt' from the 'path' directory"""
    f = open(path+'/'+file_name+'.txt', "r")
    string = f.read()
    string = string.split('\n')[:-1]
    l = [v.split(': ') for v in string]
    d = {}
    for i in l:
        if i[0][0]!="{":
            try:
                d[i[0]]=int(i[1])
            except:
                try:
                    d[i[0]]=float(i[1])
                except:
                    d[i[0]]=str(i[1])
    return d


# In[15]:


def LoadModel(file_name, path="./Saved Networks"):
    """loads a model from file named 'file_name' and 'file_name.txt' from the 'path' directory"""
    device = get_default_device()
    model_path = path + '/' + file_name
    model_parameters = LoadModelParams(file_name, path)
    model = PETTracker(model_parameters['batch_size'], device, model_parameters['nodes'], model_parameters['layers'], dropout=model_parameters['dropout']).to(device).double()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    return model, model_parameters


# In[16]:


def SaveModelParams(d, file_name, batch_size, path="./Saved Networks/"):
    """saves model parameters 'd' to a file named 'file_name.txt' in the 'path' directory"""
    if not os.path.exists(path):
        os.mkdir(path)
    file_path = path+file_name+'.txt'
    f = open(file_path, "w")
    string=""
    for name, value in d.items():
        string += "{}: {}\n".format(name, value)
    string += str(history[-1])
    f.write(string) 
    f.close()


# In[17]:


def SaveModel(file_name, model, parameters, path="./Saved Networks"):
    """saves a 'model' and its 'parameters' to two files named 'file_name' and 'file_name.txt' in the 'path' directory"""
    if not os.path.exists(path):
        os.makedirs(path)
    file_path = path + '/' + file_name
    if os.path.isfile(file_path) or os.path.isfile(file_path+".txt"):
        while True:
            yes_no = input("File already exists, should it be overwritten?\n    yes: y    no: n\n")
            if yes_no == "n":
                write = False
                break
            elif yes_no == "y":
                write = True
                break
    else:
        write = True
        
    if write:
        SaveModelParams(parameters, file_name, batch_size)
        torch.save(model.state_dict(), file_path)
        print("File saved")
    else:
        print("File not saved")


# In[18]:


def RoundAboveThreshold(prediction, value):
    prediction[prediction<=value] = 0.0
    prediction[prediction>value]  = 1.0
    return prediction


# In[19]:


def ShowLabels(model, dl, save=False, round_pred=False, value=None, num_of_sequences=-1, r=None, repeats=4):
    """
    the function "ShowLabels" gets a model and a dataloader and plots the labels and network prediction vectors next to each other in black (0) and white (1)
    this is done for one batch from the dataloader (if save=True the plot will be saved)
    round_pred       - rounds prediction values
    value            - should only values above a certain threshold, if so set 'value' to the threshold
    num_of_sequences - number of sequences from a batch that should be plotted
    r                - index range, a list going from first value to second value (e.g. [0, 1000])
    repeats          - how wide should the bargraph be
    """
    j = 0
    k = random.randrange(0, len(dl))
    l1=[]
    l2=[]
    
    for batch in dl:
        if j == k:
            events, labels = batch
            
            model.eval()

            # get the predictions of the model and give them with the labels to the cpu
            with torch.no_grad():
                pred = model(events)
            labels = [l.cpu().detach().numpy() for l in labels]
            pred   = pred.cpu().detach().numpy()

            if round_pred:
                pred = RoundPredictions(pred)

            if value:
                pred = RoundAboveThreshold(pred, value)

            if num_of_sequences>0:
                batch_length = num_of_sequences
            else:
                batch_length = len(labels)

            size     = 20.*batch_length/8
            fig, axs = plt.subplots(nrows=1, ncols=2*batch_length, sharex='all', sharey='all', figsize=(size, size))
            # create a subplot for each sequence
            for i in range(batch_length):
                if r:
                    new_labels = labels[i][r[0]:r[1]]
                    new_pred = pred[i, r[0]:r[1]]
                    new_labels = np.transpose(np.tile(new_labels, (repeats, 1)))
                    new_pred = np.transpose(np.tile(new_pred, (repeats, 1)))
                else:
                    new_labels = np.transpose(np.tile(labels[i], (repeats, 1)))
                    new_pred = np.transpose(np.tile(pred, (repeats, 1)))
                
                i1 = i*2
                i2 = i*2+1
                axs[i1].matshow(new_labels, cmap='gray', norm=Normalize(0, 1, clip=True))
                axs[i1].set_aspect(aspect=0.2)
                axs[i1].set_title('lables')
                axs[i2].matshow(new_pred, cmap='gray', norm=Normalize(0, 1, clip=True))
                axs[i2].set_aspect(aspect=0.2)
                axs[i2].set_title('predictions')
                l1.append(new_labels)
                l2.append(new_pred)
            plt.tight_layout()

            if save:
                plt.savefig('LSTM Performance.png')
            return (l1, l2)
        j+=1


# In[20]:


def WrongPredictions(model, dl, norm, num_batches=0, value=0.01):
    """prints lines with the value of a hit, its label, the networks prediction and the batch position and"""
    model.eval()
    count = 0
    print('Wrongly classified:\n| Measurements                                             | Label | Prediction | position  |\n|        x        y        z        E       time  Detector |       |            | batch pos |\n|----------------------------------------------------------|-------|------------|-----------|')
    
    j = 0
    k = random.randrange(0, len(test_dl))
    for batch in test_dl:
        if j == k:
            measurements, labels = batch
            with torch.no_grad():
                prediction = model(measurements)
            
            labels = pad_sequence(labels, batch_first=True, padding_value=0.0)
            labels = [entry.unsqueeze(0) for entry in labels]
            tensor = torch.Tensor(len(labels), labels[0].size(1))
            labels = torch.cat(labels, out=tensor)
            errors = (abs(prediction.cpu()-labels)>value).nonzero()


            for pos in errors:
                print('| {:>8.3f} {:>8.3f} {:>8.3f} {:>8.2f} {:>10.2e}     {:>5.0f} |   {:.0f}   |    {:.2f}    | {:>2d}   {:>4d} |'.format(measurements[pos[0]][pos[1], 0]*norm['x'], measurements[pos[0]][pos[1], 1]*norm['y'], 
                                                                                       measurements[pos[0]][pos[1], 2]*norm['z'], measurements[pos[0]][pos[1], 3]*norm['E'], 
                                                                                       measurements[pos[0]][pos[1], 4]*norm['time'], measurements[pos[0]][pos[1], 5]*norm['detector'], 
                                                                                       labels[pos[0]][pos[1]], prediction[pos[0], pos[1]], pos[0], pos[1]))

            if count == num_batches:
                break
            else:
                count += 1
        j+=1


# In[21]:


def PercOfHitsPET(labels):
    number_zeros = 0
    number_ones = 0
    for l in labels:
        number_zeros += len(l[l==0.])
        number_ones += len(l[l==1.])
    print("Total hits: {}\nPercantage of hits: {:.2f}%".format(number_zeros+number_ones, number_ones/(number_zeros+number_ones)*100))


# In[22]:


# print number of events in dataloader
def NumberOfEvents(dl):
    """the function "PrintNumberOfEvents" sums over all labels in a given dataloader"""
    sum = 0
    for batch in dl:
        _, labels = batch
        for tensor in labels:
            sum += torch.sum(tensor[tensor==1.0])
    return sum/2


# In[23]:


def Performance(model, dl, name=None, save_dir="./Saved Networks"):
    """gives performance for a model saved as 'name' in 'save_dir' for dataloader 'dl'"""
    if isinstance(model, str):
        model, _ = LoadModel(model, save_dir)
    eval = model.evaluate(dl)
    hit_acc = HitAccuracy(model, dl)
    non_hit_acc = NonHitAccuracy(model, dl)
    
    print(name, "\n", 
          "loss: ", eval["val_loss"], "acc: ", eval["val_acc"], "\n", 
          "hit_acc: ", hit_acc, "non_hit_acc: ", non_hit_acc, "\n")
    
    return {"fname": name, 
            "loss": eval["val_loss"], "acc": eval["val_acc"], 
            "hit_acc": hit_acc, "non_hit_acc": non_hit_acc}


# In[24]:


# function to be optimized
def objective(trial):
    """here optuna can choose hyperparameters and observe how these
    hyperparameters affect the result and slowly optimize them"""
    # ensure reproducability
    SetSeed(seed)
    
    """let optuna suggest a number of epochs (so optuna has possibility
    to stop training earlier in order to avoid overfitting)"""
    epochs = trial.suggest_int("epochs", ranges["epochs"][0], ranges["epochs"][1])
    
    # create the model by loading a previous model or by getting parameter sugestions
    if load_model:
        # load a saved model
        model, _ = LoadModel(load_file_name, load_model_path)
    else:
        # generate the model
        """let optuna choose the network parameters within some given ranges
        (number of nodes, number of layers, percantage of dropout if the number
        of layers is greater 1)"""
        nodes = trial.suggest_int("nodes", ranges["nodes"][0], ranges["nodes"][1])
#         layers = trial.suggest_int("layers", ranges["layers"][0], ranges["layers"][1])
        
#         nodes = 146
        layers = 2
        
        if layers == 1:
            dropout = 0
        else:
            dropout = trial.suggest_float("dropout", ranges["dropout"][0], ranges["dropout"][1], log=True)
            
        model = PETTracker(batch_size, device, nodes, num_layers=layers, dropout=dropout, bidirectional=True).to(device).double()
    
    # generate the optimizers
    """let optuna suggest the optimizer parameters (which optimizer to
    use, learning rate, how much gradient clipping, how much weight decay)"""
    optimizer_name = trial.suggest_categorical("optimizer", ranges["optimizer"])
#     optimizer_name = "RMSprop"
    lr = trial.suggest_float("lr", ranges["lr"][0], ranges["lr"][1], log=True)
    grad_clip = trial.suggest_float("grad_clip", ranges["grad_clip"][0], ranges["grad_clip"][1], log=True)
    weight_decay = trial.suggest_float("weight_decay", ranges["weight_decay"][0], ranges["weight_decay"][1], log=True)
    
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # training of the model (this is the default learning loop with the option to prune trials)
    torch.cuda.empty_cache()
    
    num_seq = int(math.ceil(sequence_length/back_prop_length))
    
    # initiate a scheduler for the learning rate (at the beginning high lr, at the end a small lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs, steps_per_epoch=len(train_dl)*num_seq)
    
    print("\n Parameters chosen\n", str(study.trials[-1]).split("{")[1].split("}")[0], "\n")
    
    for epoch in range(epochs):
        # training Phase 
        model.train()
        train_losses = []
        lrs = []
        for (hits, labels) in train_dl:
            model.init_hidden()
            hits.sort(key=lambda x: x.size()[0], reverse=True)
            labels.sort(key=lambda x: x.size()[0], reverse=True)
            for i in range(num_seq):
                current_hits  = to_device([hits[j][i*back_prop_length:(i+1)*back_prop_length, :] for j in range(len(hits)) if hits[j].size()[0]>i*back_prop_length], device)
                current_label = to_device([labels[j][i*back_prop_length:(i+1)*back_prop_length] for j in range(len(labels)) if labels[j].size()[0]>i*back_prop_length], device)
                
                out = model(current_hits) # takes 1/3rd of the time
                loss = model.loss(out, pad_sequence(current_label, batch_first=True))
                loss.backward()
                
                if grad_clip:
                    nn.utils.clip_grad_value_(model.parameters(), grad_clip)
                
                optimizer.step()
                optimizer.zero_grad()
                
                train_losses.append(loss.detach())
                lrs.append(get_lr(optimizer))
                
                model.detach_hidden()
                
                sched.step()
        
        # validation phase
        result = model.evaluate(valid_dl)
        result['train_loss'] = sum(train_losses)/len(train_losses)
        result['lrs'] = lrs
        
        model.epoch_end(epoch, result, False)
        
        sys.stdout.flush()
        
        # report the current value and epoch to let optuna decide if it should prune the trial
        current_performance = result['val_loss']*val_loss_weight + result['train_loss']*(1.0 - val_loss_weight)+20*(result['val_loss']-result['train_loss'])**2#+nodes*1e-5+epochs*1e-5
        trial.report(current_performance, epoch)
        
        # handle pruning based on the intermediate value
        if trial.should_prune() and epoch!=epochs:
            raise optuna.exceptions.TrialPruned()

    return current_performance


# In[25]:


def SaveTrainNorm(train_norm, file_name="TrainingNorm.txt", path="./TrainNorm"):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path+"/"+file_name, "w") as f:
        f.write(json.dumps(train_norm))


# In[26]:


def LoadTrainNorm(file_name="TrainingNorm.txt", path="./TrainNorm"):
    with open(path+"/"+file_name, "r") as f:
        train_norm = json.loads(f.read())
    return train_norm


# In[27]:


def line_vec(p, q):
    x=p-q
    return x/np.linalg.norm(x)
def distance(p, q, r):
    pr = p-r
    n = line_vec(p, q)
    return np.linalg.norm(pr-np.dot(pr, n)*n)


# In[28]:


def GetPrediction(directory, model, model_folder="./Saved Networks", train_norm_file="./TrainNorm", batch_size=1, cut_data=False,
                  device=get_default_device(), network_cut_value=0.5, max_time_difference_network=160.01e-9, fov_sphere_radius=1.7,
                  E_min=410., E_max=580., max_time_difference_classic=8.01e-9, save_dir="./Output", columns=None, cut_value=40.001e-9):
    """ 
        GetPrediction will make predictions using the neural network model and the classical method for given input data that is saved in a directory,
        the data is saved in the 'Output' folder
        
        directory:                   directory of test data has to be given
        model:                       either file name to load in model folder or a model
        train_norm_file:             location of the normalization values used for training
        batch_size:                  amount of files put through network at a time
        cut_data:                    should be the same as the trainings option
        device:                      cpu or gpu automatically selected, can be overwritten though
        
        network_cut_value:           network prediction value below which output should be ignored
        max_time_difference_network: network time difference to narrow down which lines can belong to predicted pet event
        E_min/E_max:                 classical energy window
        max_time_difference_classic: classical time window
        save_dir:                    directory where to save files
        columns:                     column names for the saved files
    """
    
    SetSeed(seed)
    if isinstance(model, str):
        model, _ = LoadModel(model, model_folder)
        
    with torch.no_grad():
        train_norm = LoadTrainNorm()
        measurement_data_test, labels_test, _, _ = DataPreperation(directory, cut_data=cut_data, norm=train_norm, cut_into_sequences=False, time_pre_selection=time_pre_selection, Print=False, cut_value=cut_value)
        test_ds = MeasurementsDataset(measurement_data_test, labels_test)
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=custom_collate, num_workers=0, pin_memory=True)
        test_dl = DeviceDataLoader(test_dl, device)

        # create list for file predictions
        prediction = []
        # load not normalized data
        m, _, _, f_names = DataPreperation(directory, cut_data=cut_data, normalize=False, time_diff=False, cut_into_sequences=False, Print=False, time_pre_selection=time_pre_selection, cut_value=cut_value)

        # put data through network
        model.eval()
        j=0
        for batch in test_dl: # memory leak
            clear_output(wait=True)
            print("1/4   Network Prediction Progress:")
            print("{:<3}/{:<3}".format(j+1, len(test_dl)))
            me, la = batch
            model.init_hidden()
            pred = model(me)
            model.detach_hidden()
            attach = pred.detach().cpu()
            for i in range(len(me)):
                prediction.append(torch.cat((m[i+batch_size*j], torch.unsqueeze(attach[i,:(m[i+batch_size*j].size(0))], dim=1)), dim=1))
            j+=1
            torch.cuda.empty_cache()

        """Network Prediction"""
        for i in range(len(prediction)):
            clear_output(wait=True)
            print("2/4   Network Event Classification:")
            print("{:<3}/{:<3}".format(i+1, len(prediction)))
            events = []

            prediction[i][prediction[i][:, 6]<network_cut_value, 6] = 0.0
            hit_lines = np.where(prediction[i][:, 6]>=network_cut_value)[0]
            if len(hit_lines)==0:
                continue
            past_lines = []
            for j in range(len(hit_lines)):
                in_t_range = [j]

                l=1
                if j!=len(hit_lines)-1:
                    while True:
                        v = float(prediction[i][hit_lines[j+l], 4]-prediction[i][hit_lines[j], 4])
                        if v>max_time_difference_network:
                            break
                        in_t_range.append(j+l)
                        l+=1
                        if j+l>len(hit_lines)-1:
                            break

                l=1
                if j!=0:
                    while True:
                        v = float(prediction[i][hit_lines[j], 4]-prediction[i][hit_lines[j-l], 4])
                        if v>max_time_difference_network:
                            break
                        in_t_range.insert(0, j-l)
                        l+=1
                        if j-l<0:
                            break

                lines = [hit_lines[l] for l in in_t_range]

                # check if the exact same list of lines were already checked in the past
                if not lines in past_lines:
                    past_lines.append(lines)
                    length = len(in_t_range)
                    if length < 1:
                        continue
                    elif length%2 == 0:
                        events.append([lines[0], lines[1], (prediction[i][lines[0], 6].item()+prediction[i][lines[1], 6].item())])
                    elif length%2 == 1:
                        x = prediction[i][lines, 6]
                        order = [int(o) for o in np.argsort(x)]
                        for k in range(int((len(order)-1)/2)):
                            events.append([lines[order[k*2+1]], lines[order[(k+1)*2]], (prediction[i][lines[order[k*2+1]], 6].item()+prediction[i][order[(k+1)*2], 6].item())])

            #######################################

            hl = np.array(events)[:, :2].astype(int)
            count = np.unique(hl, return_counts=True)

            duplicates = []
            for j in count[0][np.where(count[1]!=1)[0]]:
                xy = np.where(hl==j)
                duplicates_of_j = []
                for l in range(len(xy[0])):
                    duplicates_of_j.append([xy[0][l], xy[1][l]])
                duplicates.append(duplicates_of_j)

            dropped = []
            for j in range(len(duplicates)):
                lines = [l[0] for l in duplicates[j]]
                if len(lines)>2:
                    """drop if more than two duplicate events and notify user about uncertanty"""
                    print("Can not seperate and will drop the following events:")
                    print(f_names[i])
                    for l in sorted(lines)[::-1]:
                        if not l in dropped:
                            print(events[l][:2])
                            del events[l]
                            dropped.append(l)
                elif not lines[0] in dropped and not lines[1] in dropped:
                    events_to_check = [events[l] for l in lines]
                    if events_to_check[0][2]<events_to_check[1][2]:
                        del events[lines[0]]
                        dropped.append(lines[0])
                    elif events_to_check[0][2]>events_to_check[1][2]:
                        del events[lines[1]]
                        dropped.append(lines[1])
                    else:
                        print("Can not seperate and will drop the following events:")
                        print(f_names[i])
                        for l in sorted(lines)[::-1]:
                            print(events[l][:2])
                            del events[l]
                            dropped.append(l)

            lines_to_keep = []
            for j in events:
                lines_to_keep.append(j[0])
                lines_to_keep.append(j[1])
            lines_to_keep = np.array(lines_to_keep)
            prediction[i][np.setdiff1d(hit_lines, lines_to_keep), 6] = 0.0
            
            # remove network prediction LORs that do not go through sphere at ceneter
            positions = np.array(events, dtype=int)[:, :2]
            for k in range(positions.shape[0]):
                p = np.array(prediction[i][positions[k, 0], :3])
                q = np.array(prediction[i][positions[k, 1], :3])
                if distance(p, q, np.zeros(3))>fov_sphere_radius:
                    prediction[i][positions[k, 0], 6] = 0.
                    prediction[i][positions[k, 1], 6] = 0.
            
            #######################################
        
        """Labels (ground truth)"""
        for i, batch in enumerate(test_dl):
            _, l_ = batch
            for j in range(len(l_)):
                prediction[j+i*batch_size] = torch.cat((prediction[j+i*batch_size], torch.unsqueeze(l_[j].cpu(), dim=1)), dim=1)
        
        
        for i in range(len(prediction)):
            # remove label LORs that do not go through sphere at ceneter
            positions = np.where(prediction[i][:, 7]==1.0)[0]
            for k in range(int(len(positions)/2)):
                p = np.array(prediction[i][positions[2*k  ], :3])
                q = np.array(prediction[i][positions[2*k+1], :3])
                if distance(p, q, np.zeros(3))>fov_sphere_radius:
                    prediction[i][positions[2*k]  , 7] = 0.
                    prediction[i][positions[2*k+1], 7] = 0.
        
        clear_output(wait=True)
        print("3/4   Classical Event Classification")
        """Classical PET"""
        for l, pred in enumerate(prediction):
            idx = [int(i) for i in np.nonzero(np.logical_and(pred[:, 3]>E_min, pred[:, 3]<E_max))]
            a = []
            
            for i in range(len(idx)):
                time_diff = [idx[i]]
                # future hits in range?
                j=1
                if i!=len(idx)-1:
                    while True:
                        dt = pred[idx[i+j], 4]-pred[idx[i], 4]
                        if dt>max_time_difference_classic:
                            break
                        time_diff.append(idx[i+j])
                        j+=1
                        if i+j>len(idx)-1:
                            break
                # past hits in range?
                j=1
                if i!=0:
                    while True:
                        dt = pred[idx[i], 4]-pred[idx[i-j], 4]
                        if dt>max_time_difference_classic:
                            break
                        time_diff.insert(0, idx[i-j])
                        j+=1
                        if i-j<0:
                            break
                a.append(time_diff)
            a.sort()
            a = list(a for a,_ in itertools.groupby(a))
            
            
            duplicates = {}
            pop = []
            for i, k in enumerate(a):
                for g in k:
                    if not g in duplicates:
                        duplicates[g] = [i]
                    else:
                        duplicates[g].append(i)

                        for o in duplicates[g]:
                            pop.append(o)
            pop = list(set(pop))
            pop.sort(reverse=True)
            for p in pop:
                a.pop(p)
                
            
            lengths = [len(x) for x in a]
            c_pred = np.zeros(pred.size(0))
            for i in range(len(a)):
                if lengths[i]==2:
                    p = np.array(pred[a[i][0], :3])
                    q = np.array(pred[a[i][1], :3])
                    if distance(p, q, np.zeros(3))<fov_sphere_radius:
                        for j in a[i]:
                            c_pred[j] = 1.0
                else:
                    continue
            prediction[l] = torch.cat((pred, torch.unsqueeze(torch.tensor(c_pred), dim=1)), dim=1)
        
        if not columns:
            columns = ['X', 'Y', 'Z', 'Energy', 'Time', 'Detector', 'Network Classification', 'Labels', 'Classical PET']

        clear_output(wait=True)
        print("4/4   Saving Files")
        df_ = []
        for i in range(len(prediction)):
            df = pd.DataFrame(prediction[i].numpy(), columns=columns)

            df.to_csv(save_dir + '/' + f_names[i].split(".")[0][9:] + '.csv')
            df.to_excel(save_dir + '/' + f_names[i].split(".")[0][9:] + '.xlsx')
            print("Saved: {:>30}".format(f_names[i]))
            df_.append(df)
    return df_


# In[ ]:


def CreateTrainValidDL(train_valid_dir, cut_data=True, cut_perc=1.0, norm=None, normalize=True, time_diff=True, cut_into_sequences=True, sequence_length=200, Print=True, time_pre_selection=True, cut_value=40.001e-9, E_norm=1000., time_norm=1e-7, data_perc=1.0):
    measurement_data, labels, train_norm, _ = DataPreperation(train_valid_dir, cut_data=cut_data, time_pre_selection=time_pre_selection, sequence_length=sequence_length,
                                                              cut_value=cut_value, time_norm=time_norm, data_perc=data_perc, Print=False)

    valid_dl = None

    # create the dataset from the data
    dataset = MeasurementsDataset(measurement_data, labels)

    # determine the size of training and validation set
    ds_size    = len(dataset)
    valid_size = int(np.floor(ds_size*validation_size))
    train_size = int(ds_size-valid_size)

    # set device to cuda if available
    device = get_default_device()

    if valid_size!=0:
        SetSeed(seed)
        # randomly split the dataset into a train and a validation set
        train_ds, valid_ds = random_split(dataset, [train_size, valid_size])

        # set up the dataloaders for the train and validation set
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=custom_collate, num_workers=0, pin_memory=True)
        valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=True, collate_fn=custom_collate, num_workers=0, pin_memory=True)

        # wrap the device dataloader around the normal dataloader
        valid_dl = DeviceDataLoader(valid_dl, device)
    else:
        train_dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate, num_workers=0, pin_memory=True)
        device = get_default_device()

    # save the normalization so it can be used for the testing data or any other data
    SaveTrainNorm(train_norm)
    
    return train_dl, valid_dl, train_norm


# In[29]:


# ensure reproducability
SetSeed(seed)


# Here one creates a train and validation dataset used for training the network. It uses the files in the train_dir folder.

# In[30]:


# no trainings data is included because of the file size
# measurement_data, labels, train_norm, _ = DataPreperation(train_valid_dir, cut_data=cut_data, time_pre_selection=time_pre_selection, sequence_length=sequence_length,
#                                                           cut_value=cut_value, time_norm=time_norm, data_perc=data_perc, Print=False)

# valid_dl = None
# test_dl = None

# # create the dataset from the data
# dataset = MeasurementsDataset(measurement_data, labels)

# # determine the size of training and validation set
# ds_size    = len(dataset)
# valid_size = int(np.floor(ds_size*validation_size))
# train_size = int(ds_size-valid_size)

# # set device to cuda if available
# device = get_default_device()

# if valid_size!=0:
#     SetSeed(seed)
#     # randomly split the dataset into a train and a validation set
#     train_ds, valid_ds = random_split(dataset, [train_size, valid_size])

#     # set up the dataloaders for the train and validation set
#     train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=custom_collate, num_workers=0, pin_memory=True)
#     valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=True, collate_fn=custom_collate, num_workers=0, pin_memory=True)

#     # wrap the device dataloader around the normal dataloader
#     valid_dl = DeviceDataLoader(valid_dl, device)
# else:
#     train_dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate, num_workers=0, pin_memory=True)
#     device = get_default_device()

# # save the normalization so it can be used for the testing data or any other data
# SaveTrainNorm(train_norm)


# Here one creates a test dataset used for testing the performance. It uses the files in the test_dir folder.

# In[31]:


train_norm = LoadTrainNorm()
# create test dataset and dataloader for test set
measurement_data_test, labels_test, _, _ = DataPreperation(test_dir, cut_data=cut_data, norm=train_norm, cut_into_sequences=True, time_pre_selection=time_pre_selection,
                                                           sequence_length=sequence_length, cut_value=cut_value, time_norm=time_norm)

# create the dataset from the data
test_ds = MeasurementsDataset(measurement_data_test, labels_test)

# set up the dataloaders for the train and validation set
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=custom_collate, num_workers=0, pin_memory=True)

# set device to cuda if available
device = get_default_device()

# wrap the device dataloader around the normal dataloader
test_dl = DeviceDataLoader(test_dl, device)


# In[32]:


try:
    labels
    print('Number of labels = 1 in the train & valid set')
    PercOfHitsPET(labels)
except:
    pass
if test_dl!=None:
    print('Number of labels = 1 in the test set')
    PercOfHitsPET(labels_test)


# In[33]:


# print the number of hits detected as PET events in the validation/training/test set
try:
    print('Events in training set:   ', int(NumberOfEvents(train_dl).item()))
except:
    pass
try:
    print('Events in validation set: ', int(NumberOfEvents(valid_dl).item()))
except:
    pass
if test_dl!=None:
    print('Events in test set:       ', int(NumberOfEvents(test_dl).item()))


# In[34]:


study = optuna.create_study(study_name=study_name, storage='sqlite:///'+study_name+'.db', load_if_exists=True)

if queue and len(study.trials)==0:
    study_load = optuna.create_study(study_name=study_name_load, storage='sqlite:///'+study_name_load+'.db', load_if_exists=True)
    trial = study_load.best_trial
    parameters = trial.params
    study.enqueue_trial(parameters)


# In[35]:


# start a study
if train_optuna:
    if lim_trials:
        while True:
            print("Step: ", len(study.trials))
            assert len(study.trials) <= study_length
            study.optimize(objective, n_trials=1)
    else:
        study.optimize(objective, n_trials=study_length-len(study.trials))


# # Overview of the hyperparameter tuning

# In[36]:


if optuna_stats:
    # show the trials as dataframe
    df = study.trials_dataframe()
    display(HTML(df.loc[df['state']=='COMPLETE'].to_html()))


# In[37]:


if optuna_stats:
    # plot the importance of each hyperparameter
    fig = optuna.visualization.plot_param_importances(study)
    fig.show()


# In[38]:


if optuna_stats:
    # plot the history of the best trial
    fig = optuna.visualization.plot_optimization_history(study)
    fig.show()


# In[39]:


if optuna_stats:
    # make a countour plot of 2 parameters
    param_1 = 'lr'
    param_2 = 'epochs'
    fig = optuna.visualization.plot_contour(study, params=[param_1, param_2])
    fig.show()


# In[40]:


# reproduce and analyze the best trial of an optuna study
if reproduce:
    SetSeed(seed)

    load = False
    load_name = "LSTM - 1e-7 time_norm"

    # nodes = 146
    layers = 2

    change_epoch = 0
    change_lr = 1e0
    change_wdec = 1e0

    if load:
        model, parameters = LoadModel(load_name, "./Saved Networks")
        print("---    Model loaded     ---")
    else:
        trial = study.best_trial
        parameters = trial.params

    #     parameters['nodes'] = nodes
        parameters['layers'] = layers
        parameters['batch_size'] = batch_size

    if parameters['layers'] == 1:
        dropout = 0
    else:
        dropout = parameters['dropout']

    if not load:
        print("---    Model created     ---")
        model = PETTracker(batch_size, device, parameters['nodes'], num_layers=parameters['layers'], dropout=parameters['dropout'], bidirectional=True).to(device).double()
    history = []
    print("--- Starting first epoch ---")
    history += FitOneCycle(parameters['epochs']+change_epoch, model, train_dl, valid_dl, opt=parameters['optimizer'], lr=parameters['lr']*change_lr, weight_decay=parameters['weight_decay']*change_wdec, grad_clip=parameters['grad_clip'], eval_test=False, hit_acc=False, back_prop_length=back_prop_length)


# In[41]:


# SaveModel("", model, parameters)


# # Test the model on the example data

# In[42]:


model, _ = LoadModel("PETTracker")


# In[43]:


"""show a bar graph of the labels next to the predictions for the train dataloader"""
_ = ShowLabels(model, test_dl, num_of_sequences=8, r=[0, 1000])


# In[44]:


# print all hits where the networks prediction was off by atleast 0.5
WrongPredictions(model, test_dl, train_norm, value=0.5)


# In[45]:


"""get predictions for data saved in directory"""
directory = "./Example Data"
df = GetPrediction(directory, "PETTracker", model_folder="./Saved Networks", train_norm_file="./TrainNorm", batch_size=1, cut_data=False,
                   network_cut_value=1.0, max_time_difference_network=160.01e-9, cut_value=cut_value, E_min=410., E_max=580., max_time_difference_classic=90.01e-9)

