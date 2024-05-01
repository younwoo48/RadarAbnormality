import numpy as np
import json
import pandas as pd
import random
import visualisation


subjects = ['C','D','E','F','G','H','I']
window_size = 40

def make_same_size(walk_data):
    new_walk_data = []
    for data in walk_data:
        count_data = []
        for i in range(len(data)):
            count_data.append(data[i])
            if(len(count_data)==window_size):
                new_walk_data.append(count_data)
                count_data = []
                if(len(data)-i-1<window_size):
                    i-= window_size - len(data)+i+1              
                else:
                    i -= 10
    return new_walk_data




# Example: Creating a NumPy array

    
def make_train_test_walk_csv(subjects):
    x_traindf = pd.DataFrame(columns =['dX','dY','AccX','AccY','VelX','VelY', 'Sca_Vel','Sca_Acc','Toilet_Freq','Label'])
    y_traindf = pd.DataFrame(columns =['Label','Truth'])
    lab = 0
    for subject in subjects:
        for i in range(1,4,1):
            data = visualisation.parse(subject,i)
            visualisation.clean_data(data)
            visualisation.create_delta(data)
            visualisation.create_scalar(data)
            visualisation.drop_column(data)
            visualisation.create_toilet_frequency(data,subject,i)
            walk_data = make_same_size(visualisation.cut_walking(data))
            for single_walk in walk_data:
                df = pd.DataFrame(single_walk)
                df.drop("EventProcessedUtcTime",axis=1,inplace=True)
                df['Label'] = [lab for j in range(len(single_walk))]
                if(i!=2):
                    new_row = {'Label': lab,'Truth': 'Normal'}
                else:
                    new_row = {'Label': lab,'Truth': 'Abnormal'}
                
                x_traindf = pd.concat([x_traindf,df], ignore_index=True)
                y_traindf = pd.concat([y_traindf, pd.DataFrame([new_row])], ignore_index=True)
                lab+=1
    unique_labels = x_traindf['Label'].unique()
    np.random.shuffle(unique_labels)
    x_traindf = pd.concat([x_traindf[x_traindf['Label'] == label] for label in unique_labels], ignore_index=True)
    x_traindf.to_csv("x_train.csv")
    y_traindf.to_csv("y_train.csv")


# make_train_test_walk_csv(['C','D','E','F','G','H','I'])