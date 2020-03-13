#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import csv
import numpy as np
import pandas as pd
import plotly.express as px

#############################################################################
#Architecture comparison
total_dict = []

path1 = '/home/stormlab/seg/LSTM-UNet-Outputs/Retrained/LSTMUNet/MyRun_SIM/2020-03-07_120900'


path = [path1, ]

for i in len(path):
    with open(os.path.join(path[i], 'kfold_metrics_list.csv')) as csv_file:
        reader = csv.reader(csv_file)
        data_dict = dict(reader)
        data_dict.extend({'Name': 'ConvLSTMEnc Unet'})
        total_dict.append(data_dict)

name = ['ConvULSTMEnc', 'ConvULSTM', 'ConvULSTMAttn']  
mean = {'Accuracy': [], 'Precision': [], 'Recall': [], 'Jaccard': []}
std = {'Accuracy': [], 'Precision': [], 'Recall': [], 'Jaccard': []}
metrics = ['Accuracy', 'Precision', 'Recall', 'Jaccard']
data = []
data_accuracy = []
data_precision = []
data_recall = []
data_jaccard = []
 
for i in len(total_dict):
    data.append(total_dict[i]['Accuracy'])
    data.append(total_dict[i]['Precision'])
    data.append(total_dict[i]['Recall'])
    data.append(total_dict[i]['Jaccard Index'])
    data_accuracy.append(total_dict[i]['Accuracy'])
    data_precision.append(total_dict[i]['Precision'])
    data_recall.append(total_dict[i]['Recall'])
    data_jaccard.append(total_dict[i]['Jaccard Index'])
    mean['Accuracy'].append(np.mean(total_dict[i]['Accuracy']))
    mean['Precision'].append(np.mean(total_dict[i]['Precision']))
    mean['Recall'].append(np.mean(total_dict[i]['Recall']))
    mean['Jaccard'].append(np.mean(total_dict[i]['Jaccard Index']))
    std['Accuracy'].append(np.std(total_dict[i]['Accuracy']))
    std['Precision'].append(np.std(total_dict[i]['Precision']))
    std['Recall'].append(np.std(total_dict[i]['Recall']))
    std['Jaccard'].append(np.std(total_dict[i]['Jaccard Index']))

total_data = pd.DataFrame(data=data, columns=pd.MultiIndex.from_tuples(zip(metrics,name)))
accuracy_data = pd.DataFrame(data=data_accuracy, columns=name)
precision_data = pd.DataFrame(data=data_precision, columns=name)
recall_data = pd.DataFrame(data=data_recall, columns=name)
jaccard_data = pd.DataFrame(data=data_jaccard, columns=name) 

fig = px.box(total_data, x="Metrics", y="test_value (%)", color="Network architectures")
fig.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default
fig.show()

fig = px.box(data_accuracy, x="Network achitectures", y="test_value (%)")
fig.show()

fig = px.box(data_precision, x="Network architectures", y="test_value (%)")
fig.show()

fig = px.box(data_recall, x="Network architectures", y="test_value (%)")
fig.show()

fig = px.box(data_jaccard, x="Network architectures", y="test_value (%)")
fig.show()  


fig = px.scatter_matrix(total_data,
    dimensions=["Accuracy", "Precision", "Recall", "Jaccard"],
    color="Network architectures", symbol="Network architectures",
    title="Scatter matrix of metrics") # remove underscore
fig.update_traces(diagonal_visible=False)
fig.show()

fig = px.parallel_coordinates(total_data, color="Network architectures",
                    color_continuous_scale=px.colors.diverging.Tealrose, color_continuous_midpoint=2)
fig.show()

    
#############################################################################
#Loss comparison

total_dict = []

path1 = '/home/stormlab/seg/LSTM-UNet-Outputs/Retrained/LSTMUNet/MyRun_SIM/2020-03-07_120900'


path = [path1, ]

for i in len(path):
    with open(os.path.join(path[i], 'kfold_metrics_list.csv')) as csv_file:
        reader = csv.reader(csv_file)
        data_dict = dict(reader)
        data_dict.extend({'Name': 'ConvLSTMEnc Unet'})
        total_dict.append(data_dict)

name = ['ComboLoss', 'TverskyLoss']  
mean = {'Accuracy': [], 'Precision': [], 'Recall': [], 'Jaccard': []}
std = {'Accuracy': [], 'Precision': [], 'Recall': [], 'Jaccard': []}
metrics = ['Accuracy', 'Precision', 'Recall', 'Jaccard']
data = []
data_accuracy = []
data_precision = []
data_recall = []
data_jaccard = []
 
for i in len(total_dict):
    data.append(total_dict[i]['Accuracy'])
    data.append(total_dict[i]['Precision'])
    data.append(total_dict[i]['Recall'])
    data.append(total_dict[i]['Jaccard Index'])
    data_accuracy.append(total_dict[i]['Accuracy'])
    data_precision.append(total_dict[i]['Precision'])
    data_recall.append(total_dict[i]['Recall'])
    data_jaccard.append(total_dict[i]['Jaccard'])
    mean['Accuracy'].append(np.mean(total_dict[i]['Accuracy']))
    mean['Precision'].append(np.mean(total_dict[i]['Precision']))
    mean['Recall'].append(np.mean(total_dict[i]['Recall']))
    mean['Jaccard'].append(np.mean(total_dict[i]['Jaccard Index']))
    std['Accuracy'].append(np.std(total_dict[i]['Accuracy']))
    std['Precision'].append(np.std(total_dict[i]['Precision']))
    std['Recall'].append(np.std(total_dict[i]['Recall']))
    std['Jaccard'].append(np.std(total_dict[i]['Jaccard Index']))

total_data = pd.DataFrame(data=data, columns=pd.MultiIndex.from_tuples(zip(metrics,name)))
accuracy_data = pd.DataFrame(data=data_accuracy, columns=name)
precision_data = pd.DataFrame(data=data_precision, columns=name)
recall_data = pd.DataFrame(data=data_recall, columns=name)
jaccard_data = pd.DataFrame(data=data_jaccard, columns=name)   

fig = px.box(total_data, x="Metrics", y="test_value (%)", color="Loss")
fig.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default
fig.show()

fig = px.box(data_accuracy, x="Loss", y="test_value (%)")
fig.show()

fig = px.box(data_precision, x="Loss", y="test_value (%)")
fig.show()

fig = px.box(data_recall, x="Loss", y="test_value (%)")
fig.show()

fig = px.box(data_jaccard, x="Loss", y="test_value (%)")
fig.show()

#############################################################################
#Loss comparison

total_dict = []

path1 = '/home/stormlab/seg/LSTM-UNet-Outputs/Retrained/LSTMUNet/MyRun_SIM/2020-03-07_120900'


path = [path1, ]

for i in len(path):
    with open(os.path.join(path[i], 'kfold_metrics_list.csv')) as csv_file:
        reader = csv.reader(csv_file)
        data_dict = dict(reader)
        data_dict.extend({'Name': 'ConvLSTMEnc Unet'})
        total_dict.append(data_dict)

name = ['Scratch', 'Pretraining']  
mean = {'Accuracy': [], 'Precision': [], 'Recall': [], 'Jaccard': []}
std = {'Accuracy': [], 'Precision': [], 'Recall': [], 'Jaccard': []}
metrics = ['Accuracy', 'Precision', 'Recall', 'Jaccard']
data = []
data_accuracy = []
data_precision = []
data_recall = []
data_jaccard = []
 
for i in len(total_dict):
    data.append(total_dict[i]['Accuracy'])
    data.append(total_dict[i]['Precision'])
    data.append(total_dict[i]['Recall'])
    data.append(total_dict[i]['Jaccard Index'])
    data_accuracy.append(total_dict[i]['Accuracy'])
    data_precision.append(total_dict[i]['Precision'])
    data_recall.append(total_dict[i]['Recall'])
    data_jaccard.append(total_dict[i]['Jaccard'])
    mean['Accuracy'].append(np.mean(total_dict[i]['Accuracy']))
    mean['Precision'].append(np.mean(total_dict[i]['Precision']))
    mean['Recall'].append(np.mean(total_dict[i]['Recall']))
    mean['Jaccard'].append(np.mean(total_dict[i]['Jaccard Index']))
    std['Accuracy'].append(np.std(total_dict[i]['Accuracy']))
    std['Precision'].append(np.std(total_dict[i]['Precision']))
    std['Recall'].append(np.std(total_dict[i]['Recall']))
    std['Jaccard'].append(np.std(total_dict[i]['Jaccard Index']))

total_data = pd.DataFrame(data=data, columns=pd.MultiIndex.from_tuples(zip(metrics,name)))
accuracy_data = pd.DataFrame(data=data_accuracy, columns=name)
precision_data = pd.DataFrame(data=data_precision, columns=name)
recall_data = pd.DataFrame(data=data_recall, columns=name)
jaccard_data = pd.DataFrame(data=data_jaccard, columns=name)   

fig = px.box(total_data, x="Metrics", y="test_value (%)", color="Weight Initialization")
fig.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default
fig.show()

fig = px.box(data_accuracy, x="Weight Initialization", y="test_value (%)")
fig.show()

fig = px.box(data_precision, x="Weight Initialization", y="test_value (%)")
fig.show()

fig = px.box(data_recall, x="Weight Initialization", y="test_value (%)")
fig.show()

fig = px.box(data_jaccard, x="Weight Initialization", y="test_value (%)")
fig.show()
