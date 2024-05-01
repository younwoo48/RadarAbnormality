import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import pickle as pkl
import os


def parse(subject,num):
    folder_path  = os.path.dirname(__file__)
    absolute_folder_path = os.path.abspath(folder_path)
    file_path = absolute_folder_path +'/Filtered_anonymised_data/'+subject+'/'+str(num)+'/Filtered_data.json'
    with open(file_path,'r') as file:
        data = [json.loads(line) for line in file]
    
    return data
def clean_data(data):
    x = 0
    y = 0
    for single_data in data:
        if(single_data['X']>=600 or single_data['Y']>=800 or single_data['X']<=100 or single_data['Y']<=0):
            single_data['X'] = x
            single_data['Y'] = y
            single_data['AccX'] = 0
            single_data['AccY'] = 0
            single_data['VelX'] = 0
            single_data['VelY'] = 0
        else:
            x = single_data['X']
            y = single_data['Y']

def create_delta(data):
    n = len(data)
    prev_x = data[0]['X']
    prev_y = data[0]['Y']
    data[0]['dX'] = 0
    data[0]['dY'] = 0
    for i in range(1,n):
        data[i]['dX'] = data[i]['X'] - prev_x
        data[i]['dY'] = data[i]['Y'] - prev_y
        prev_x = data[i]['X']
        prev_y = data[i]['Y']
    
def create_scalar(data):
    for single_data in data:
        single_data['Sca_Vel'] =  (single_data['VelX']**2 + single_data['VelY']**2)**0.5   
        single_data['Sca_Acc'] =  (single_data['AccX']**2 + single_data['AccY']**2)**0.5        

def create_toilet_frequency(data,subject,test_no):
    folder_path  = os.path.dirname(__file__)
    absolute_folder_path = os.path.abspath(folder_path)
    file_path = absolute_folder_path + '/toilet_visit/' + subject + '_' + str(test_no) + '_toilet'
    with open(file_path, 'rb') as file:
        toilet_time = pkl.load(file)
    start_time = datetime.strptime(data[0]["EventProcessedUtcTime"].split()[1], '%H:%M:%S.%f')
    for d in data:
        timestamp= datetime.strptime(d["EventProcessedUtcTime"].split()[1], '%H:%M:%S.%f')
        toilet_num = 0
        for time_string in toilet_time:
            time_obj = datetime.strptime(time_string, '%H:%M:%S')
            if time_obj<timestamp:
                toilet_num+=1
        time_elapsed = timestamp - start_time
        if(time_elapsed.total_seconds()>0):
            d['Toilet_Freq'] = 3000/time_elapsed.total_seconds() * toilet_num
        else:
            d['Toilet_Freq'] = 0

def rotate_points(X,Y, angle_degrees):
    # Convert angle from degrees to radians
    angle_radians = np.radians(angle_degrees)

    # Rotation matrix components
    cos_theta, sin_theta = np.cos(angle_radians), np.sin(angle_radians)

    # Rotate each point
    rotated_x = []
    rotated_y = []
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        x_rotated = x * cos_theta - y * sin_theta
        y_rotated = x * sin_theta + y * cos_theta
        rotated_x.append(x_rotated)
        rotated_y.append(y_rotated)

    return (rotated_x,rotated_y)

def drop_column(data):
    for single_data in data:
        del single_data['X']
        del single_data['Y']
        del single_data['Z']
    
def cut_time(subject_data, start_time, end_time):
    new_list = []
    for data in subject_data:
        timestamp= datetime.strptime(data["EventProcessedUtcTime"].split()[1], '%H:%M:%S.%f')
        if(timestamp>start_time and timestamp<end_time):
            new_list.append(data)
    return new_list
def cut_walking(subject_data):
    walking_list = []
    walking_series = []
    discontinue = 0
    for data in subject_data:
        if((abs(data['dX']) <=0.5 and abs(data['dY']) <= 0.5)):
            discontinue += 1
        elif(discontinue>=6):
            if len(walking_series)>=40:
                walking_list.append(walking_series)
            walking_series = []
            discontinue = 0
        else:
            discontinue = 0
            walking_series.append(data)   
    return walking_list


def visualise_heatmap(data):
    x = [round(single['X'],3) for single in data]
    y = [round(single['Y'],3) for single in data]

    x,y = rotate_points(x,y,25)
    data = pd.DataFrame({'x': x,
                         'y': y
                         })
    
    joint_plot = sns.jointplot(x='x', y='y', data=data, kind='hist')
    joint_plot.ax_joint.set_xlim(0, 500)
    joint_plot.ax_joint.set_ylim(100, 800)
    joint_plot.ax_joint.invert_xaxis()
    plt.show()

def visualise_lasagna(data):
    x = [round(single['X'],3) for single in data]
    y = [round(single['Y'],3) for single in data]
    x_norm = [(x_i+100)/700 for x_i in x]
    y_norm = [(y_i+100)/1000 for y_i in y]
    # Initialize RGB array with ones for green channel and zeros for red and blue channels
    rgb_array = np.zeros((len(x), 3))
    rgb_array[:, 1] = x_norm # Red channel
    rgb_array[:, 2] = y_norm
    fig, ax = plt.subplots(figsize=(10, 2))
    # Use imshow to show each time slice as a different row, with RGB color depending on 'X' and 'Y'
    ax.imshow(rgb_array[np.newaxis, :, :], aspect='auto', interpolation='nearest')

    # Set the y-axis to be blank as there's only one 'row' in this data
    ax.set_yticks([])
    ax.set_xticks(np.arange(0,30001,6000))
    ax.set_xticklabels([0,10,20,30,40,50])
    ax.set_xlabel('Time in min')
    plt.show()

def walking_pattern(data):
    vel_sca = [(data[i]['VelX']**2 + data[i]['VelY']**2)**0.5 for i in range(len(data))]
    acc_sca = [(data[i]['AccX']**2 + data[i]['AccX']**2)**0.5 for i in range(len(data))]
    vel_sca = [vel for vel in vel_sca if vel>0.1]
    vel_sca = [acc for acc in acc_sca if acc>0]

    plt_data = [vel_sca,acc_sca]
    plt.boxplot(plt_data,showfliers=False)
    plt.xticks([1, 2], ['Velocity Scalar', 'Acceleration Scalar'])
    plt.yticks([0.00,0.04,0.08,0.12,0.16,0.20,0.24,0.28])
    plt.show()


data_D_1 = parse('D',1)
clean_data(data_D_1)
visualise_heatmap(data_D_1)