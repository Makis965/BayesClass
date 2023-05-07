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

            f_x = 1/(sd * sqrt(2*pi)) * np.exp((-1*(df_test - mean)**2)/(2*sd**2))
            if class_id == 0:
                df_C0.iloc[:,feature_id] = f_x
            else:
                df_C1.iloc[:, feature_id] = f_x

            feature_id += 1

    return df_C0,df_C1

df_C0,df_C1 = gaussian_distribution_Bayes(training_set, test_set)

# wyznaczamy parametry dla gaussa dla kazdego punktu treninowego dla zadanego sasiedztwa h
# dla punktu testowego wyznaczamy sume prawdopodobienstwa wystapienia go w rozkładach wszystkich punktow sasiedztwa
# mnozymy prawdopodobienstwa wszystkich zmiennych i wybieramy te klase dla ktorej iloczyn jest wiekszy

def parzen_window_Bayes(training_set, test_set, h):
    # creating empty data frames for each class to fill with density propabilities
    df_C0 = pd.DataFrame(np.nan, index=np.arange(0,len(test_set)).tolist(),
                         columns = training_set.columns.to_list())
    df_C0["CLASS"] = 0
    df_C1 = pd.DataFrame(np.nan, index=np.arange(0, len(test_set)).tolist(),
                         columns=training_set.columns.to_list())
    df_C1["CLASS"] = 1

    for class_id in range(1):
        feature_id = 0
        for i in training_set:
            if i == "CLASS":
                break
            # h = 0.5
            df_train = training_set[training_set["CLASS"] == class_id][i]
            df_test = test_set[i]
            # searching for the h-area neighbours in training set for test point
            for x_test in df_test:
                x_test = df_test[2]
                min_x_test = x_test - h
                max_x_test = x_test + h
                x_train_subset = df_train.loc[(df_train >= min_x_test) & (df_train <= max_x_test)]
                # searching for the h-area neighbours in training set for training subset
                propability_sum = 0
                for parzen in x_train_subset:
                    # calculating propabilities for h-are neighbours
                    x_train_parzen = df_train.loc[(df_train >= parzen - h) & (df_train <= parzen + h)]
                    sd = stats.stdev(x_train_parzen)
                    mean = stats.mean(x_train_parzen)
                    propability = 1/(sd * sqrt(2*pi)) * np.exp((-1*(x_test - mean)**2)/(2*sd**2))
                    # propiability summation
                    propability_sum += propability
                print(propability_sum)
            if class_id == 0:
                df_C0.iloc[:, feature_id] = propability_sum
            else:
                df_C1.iloc[:, feature_id] = propability_sum

            feature_id += 1

