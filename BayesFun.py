import numpy as np
from math import pi, sqrt, exp
import pandas as pd
import matplotlib.pyplot as plt
import statistics as stats

mi_1 = [0, 0]
sigma_1 = [[2, -1],[-1,2]]

mi_2 = [2, 2]
sigma_2 = [[1,0],[0,1]]

data_C0 = np.random.multivariate_normal(mi_1,sigma_1,size =100)
class_C0 = np.zeros(100).transpose()
data_C0 = np.column_stack((data_C0,class_C0))
data_C1 = np.random.multivariate_normal(mi_2,sigma_2,size =100)
class_C1 = np.ones(100).transpose()
data_C1 = np.column_stack((data_C1,class_C1))
data = np.concatenate((data_C0,data_C1),axis=0)
np.random.shuffle(data)
train,test = np.split(data,2)
train = pd.DataFrame(train, columns = ["X1","X2","CLASS"])
test = pd.DataFrame(test, columns = ["X1","X2","CLASS"])

def apriori_propability(train):
    propability_C0 = len(train[train["CLASS"] == 0]) / len(train)
    propability_C1 = len(train[train["CLASS"] == 1]) / len(train)
    return propability_C0, propability_C1

training_set = train
test_set = test
data_frame = df_C0
a = 0
b = 1
def multiple_df_rows(data_frame,a,b):
    new_column = np.ones((len(data_frame),1,))
    new_column = pd.DataFrame(new_column)
    for i in range(a,b):
        if i == 1:
            new_column = data_frame.iloc[:,i]
        else:
            new_column *= data_frame.iloc[:,i]

def gaussian_distribution_Bayes(training_set, test_set):
    df_C0 = pd.DataFrame(np.nan, index=np.arange(0,len(test_set)).tolist(),
                         columns = training_set.columns.to_list())
    df_C0["CLASS"] = 0
    df_C1 = pd.DataFrame(np.nan, index=np.arange(0, len(test_set)).tolist(),
                         columns=training_set.columns.to_list())
    df_C1["CLASS"] = 1
    for class_id in range(2):
        feature_id = 0
        for i in training_set:
            if i == "CLASS":
                break
            df_train = training_set[training_set["CLASS"] == class_id][i]
            sd = stats.stdev(df_train)
            mean = stats.mean(df_train)
            df_test = test_set[i]
            f_x = 1/(sd * sqrt(2*pi)) * np.exp((-1*(df_test - mean)**2)/(2*sd**2))
            if class_id == 0:
                df_C0.iloc[:,feature_id] = f_x
            else:
                df_C1.iloc[:, feature_id] = f_x
            feature_id += 1
    return df_C0,df_C1

df_C0,df_C1 = gaussian_distribution_Bayes(training_set, test_set)

def parzen_window_Bayes(training_set, test_set, h):
    # creating empty data frames for each class to fill with density propabilities
    df_C0 = pd.DataFrame(np.nan, index=np.arange(0,len(test_set)).tolist(),
                         columns = training_set.columns.to_list())
    df_C0["CLASS"] = 0
    df_C1 = pd.DataFrame(np.nan, index=np.arange(0, len(test_set)).tolist(),
                         columns=training_set.columns.to_list())
    df_C1["CLASS"] = 1
    for class_id in range(2):
        feature_id = 0
        for i in training_set:
            if i == "CLASS":
                break
            df_train = training_set[training_set["CLASS"] == class_id][i]
            df_test = test_set[i]
            # searching for the h-area neighbours in training set for test point
            observation = 0
            for x_test in df_test:
                min_x_test = x_test - h
                max_x_test = x_test + h
                x_train_subset = df_train.loc[(df_train >= min_x_test) & (df_train <= max_x_test)]
                # searching for the h-area neighbours in training set for training subset
                propability_sum = 0
                for parzen in x_train_subset:
                    # calculating propabilities for h-are neighbours
                    x_train_parzen = df_train.loc[(df_train >= parzen - h) & (df_train <= parzen + h)]
                    if len(x_train_parzen) < 2:
                        next
                    else:
                        sd = stats.stdev(x_train_parzen)
                    mean = stats.mean(x_train_parzen)
                    propability = 1/(sd * sqrt(2*pi)) * np.exp((-1*(x_test - mean)**2)/(2*sd**2))
                    # propiability summation
                    propability_sum += propability
                if class_id == 0:
                    df_C0.iloc[observation, feature_id] = propability_sum
                else:
                    df_C1.iloc[observation, feature_id] = propability_sum
                observation += 1
            feature_id += 1

# Synthetic dataset
samples = 200
features = 8
C0_nfeat = np.zeros((samples, features))
C1_nfeat = np.zeros((samples, features))

for i in range((np.shape(C0_nfeat)[1])):
    C0_nfeat[:,i] = np.random.exponential(scale=1/0.5, size=samples)

for i in range((np.shape(C1_nfeat)[1])):
    C1_nfeat[:, i] = np.random.uniform(low = -1, high=1, size=samples)

import scipy.io
dir = r'C:\Users\barba\Desktop\II STOPIEŃ\I semestr\Classifiers\Bayes'
from scipy.io import loadmat
import os
files = []
for file in os.listdir(dir):
    if file.endswith(".mat"):
        files.append(os.path.join(dir, file))

mat = loadmat(files[0])
data_train = pd.DataFrame(mat.get('dane_train'))
classes_train = pd.DataFrame(mat.get('etykiety_train'))
frames_train= [data_train,classes_train]
microarray_train = pd.concat(frames_train)

data_test = pd.DataFrame(mat.get('dane_test'))
classes_test = pd.DataFrame(mat.get('etykiety_test'))
frames = [data_test,classes_test]
microarray_test = pd.concat(frames)
