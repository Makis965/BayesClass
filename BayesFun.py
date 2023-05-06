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

            f_x = 1/(sd * sqrt(2*pi)) * np.exp((-1*(df_test - mean)**2)/2*sd**2)
            if class_id == 0:
                df_C0.iloc[:,feature_id] = f_x
            else:
                df_C1.iloc[:, feature_id] = f_x

            feature_id += 1

    return df_C0,df_C1

df_C0,df_C1 = gaussian_distribution_Bayes(training_set, test_set)

# parametry potrzebne: wielkosc okna, krok postepu okna, zakres wartosci po ktorych poruszac sie bedzie okno
# funkcja kernel

# Wybieramy test point i sprawdzamy dla danego okna prawdopoodbienstwo dla kazdej z klas.

def parzen_window_Bayes():
    densities = []
    class_densities = []
    for id_class in range (2):
        for j in range(training_set.shape[1]):
            x = test_set[:, j]
            kernel = np.nan
            density = np.nan
            class_densities.append(density)
        densities.append(class_densities)
    pass
