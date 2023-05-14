import numpy as np
from math import pi, sqrt, exp
import pandas as pd
import matplotlib.pyplot as plt
import statistics as stats

# dataset generation

mi_1 = [0, 0]
sigma_1 = [[2, -1], [-1, 2]]

mi_2 = [2, 2]
sigma_2 = [[1, 0], [0, 1]]

data_C0 = np.random.multivariate_normal(mi_1, sigma_1, size =100)
class_C0 = np.zeros(100).transpose()
data_C0 = np.column_stack((data_C0, class_C0))
data_C1 = np.random.multivariate_normal(mi_2, sigma_2, size =100)
class_C1 = np.ones(100).transpose()
data_C1 = np.column_stack((data_C1, class_C1))
data = np.concatenate((data_C0, data_C1), axis=0)
np.random.shuffle(data)
train, test = np.split(data,2)
train = pd.DataFrame(train, columns=["X1", "X2", "CLASS"])
test = pd.DataFrame(test, columns=["X1", "X2", "CLASS"])

training_set = train
test_set = test

# main functions

def multiple_df_rows(data_frame, num_of_cols):
    new_column = np.ones((len(data_frame), 1,))
    new_column = pd.DataFrame(new_column)
    for i in range(0, num_of_cols):
        if i == 0:
            new_column = data_frame.iloc[:, i]
        else:
            new_column *= data_frame.iloc[:, i]
    output_df = pd.concat([new_column, data_frame.iloc[:, data_frame.shape[1]-1]], axis=1)
    return output_df

def gaussian_distribution_Bayes(training_set, test_set, number_of_features = 2):

    df_C0 = pd.DataFrame(np.nan, index=np.arange(0, len(test_set)).tolist(),
                         columns=training_set.columns.to_list())
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
                df_C0.iloc[:, feature_id] = f_x
            else:
                df_C1.iloc[:, feature_id] = f_x

            feature_id += 1

    df_C0=multiple_df_rows(df_C0, number_of_features)
    df_C1=multiple_df_rows(df_C1, number_of_features)

    merged_df = pd.concat([df_C0, df_C1], axis=1)
    classified = pd.DataFrame(np.multiply([merged_df.iloc[:, 0] < merged_df.iloc[:, 2]], 1).T)

    return classified

def parzen_window_Bayes(training_set, test_set, h,number_of_features):
    # creating empty data frames for each class to fill with density propabilities
    df_C0 = pd.DataFrame(np.nan, index=np.arange(0, len(test_set)).tolist(),
                         columns=training_set.columns.to_list())
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

    df_C0 = multiple_df_rows(df_C0,number_of_features)
    df_C1 = multiple_df_rows(df_C1,number_of_features)

    merged_df = pd.concat([df_C0,df_C1],axis = 1)
    classified = pd.DataFrame(np.multiply([merged_df.iloc[:,0] < merged_df.iloc[:,2]], 1).T)

    return classified

classified = gaussian_distribution_Bayes(training_set, test_set, 2)
parzen_kernel_classifier = parzen_window_Bayes(training_set, test_set, 1.5,2)

# creating confussion martices

from sklearn import metrics

confusion_matrix_bayes = metrics.confusion_matrix(test_set.iloc[:,2], classified)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_bayes, display_labels = [False, True])
cm_display.plot()
plt.show()

confusion_matrix_parzen = metrics.confusion_matrix(test_set.iloc[:,2], parzen_kernel_classifier)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_parzen, display_labels = [False, True])
cm_display.plot()
plt.show()

ACC_bayes = (confusion_matrix_bayes[0,0]+confusion_matrix_bayes[1,1])/sum(sum(confusion_matrix_parzen))
ACC_parzen_kernel = (confusion_matrix_parzen[0,0]+confusion_matrix_parzen[1,1])/sum(sum(confusion_matrix_parzen))

# DATA CLASSIFICATION AND FUNCTION TESTING

samples = 200
features = 8
C0_nfeat = np.zeros((samples, features))
C1_nfeat = np.zeros((samples, features))


for i in range((np.shape(C0_nfeat)[1])):
    # C0_nfeat[:,i] = np.random.exponential(scale=1/0.5, size=samples)
    C0_nfeat[:, i] = np.random.uniform(low=-1, high=1, size=samples)
for i in range((np.shape(C1_nfeat)[1])):
    C1_nfeat[:, i] = np.random.uniform(low = -1, high=1, size=samples)

C0_nfeat = np.append(C0_nfeat,np.zeros([200,1]), axis=1)
C1_nfeat = np.append(C1_nfeat,np.ones([200,1]), axis=1)
import scipy.io

dir = r'C:\MARCEL\STUDIA\DATA SCIENCE\Sem I\Classifiers\Bayes lab'
from scipy.io import loadmat
import os
files = []
for file in os.listdir(dir):
    if file.endswith(".mat"):
        files.append(os.path.join(dir, file))

#microarray dataset
mat = loadmat(files[0])
data_train = pd.DataFrame(mat.get('dane_train'))
classes_train = pd.DataFrame(mat.get('etykiety_train')).T
classes_train = classes_train.rename(columns={0:"CLASS"}).T
frames_train= [data_train,classes_train]
train_microarray = pd.concat(frames_train).T

data_test = pd.DataFrame(mat.get('dane_test'))
classes_test = pd.DataFrame(mat.get('etykiety_test')).T
classes_test = classes_test.rename(columns={0:"CLASS"}).T
frames = [data_test,classes_test]
test_microarray = pd.concat(frames).T

# synthetic 8 features dataset

data_8ft = np.concatenate((C0_nfeat, C1_nfeat), axis=0)
np.random.shuffle(data_8ft)
train_8ft, test_8ft = np.split(data_8ft,2)
col_names = ["1","2","3","4","5","6","7","8","CLASS"]
train_8ft = pd.DataFrame(train_8ft, columns=col_names)
test_8ft = pd.DataFrame(test_8ft, columns=col_names)

classified_8ft = gaussian_distribution_Bayes(train_8ft, test_8ft, 8)
parzen_kernel_classifier_8ft = parzen_window_Bayes(train_8ft, test_8ft, 1,8)

confusion_matrix_bayes_8ft = metrics.confusion_matrix(test_8ft.iloc[:,8], classified_8ft)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_bayes_8ft, display_labels = [False, True])
cm_display.plot()

confusion_matrix_parzen_8ft = metrics.confusion_matrix(test_8ft.iloc[:,8], parzen_kernel_classifier_8ft)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_parzen_8ft, display_labels = [False, True])
cm_display.plot()

ACC_bayes = (confusion_matrix_bayes_8ft[0,0]+confusion_matrix_bayes_8ft[1,1])/sum(sum(confusion_matrix_parzen_8ft))
ACC_parzen_kernel = (confusion_matrix_parzen_8ft[0,0]+confusion_matrix_parzen_8ft[1,1])/sum(sum(confusion_matrix_parzen_8ft))

# microarray dataset

classified_microarray = gaussian_distribution_Bayes(train_microarray, test_microarray, 300)
parzen_kernel_classifier_microarray = parzen_window_Bayes(train_microarray, test_microarray, 1,300)

confusion_matrix_bayes_microarray = metrics.confusion_matrix(test_microarray.iloc[:,300], classified_microarray)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_bayes_microarray, display_labels = [False, True])
cm_display.plot()

confusion_matrix_parzen_microarray = metrics.confusion_matrix(test_microarray.iloc[:,8], parzen_kernel_classifier_microarray)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_parzen_microarray, display_labels = [False, True])
cm_display.plot()

ACC_bayes = (confusion_matrix_bayes_microarray[0,0]+confusion_matrix_bayes_microarray[1,1])/sum(sum(confusion_matrix_parzen_microarray))
ACC_parzen_kernel = (confusion_matrix_parzen_microarray[0,0]+confusion_matrix_parzen_microarray[1,1])/sum(sum(confusion_matrix_parzen_microarray))