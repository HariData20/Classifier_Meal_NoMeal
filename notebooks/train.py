import datetime
import os
import pickle

import pandas as pd
import numpy as np

# Initializing variables

from scipy.fft import fft
from sklearn import model_selection, svm, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

time_interval = datetime.timedelta(hours=2)
for_30min = datetime.timedelta(minutes=30)


# Extracting timestamps from Insulin data
def meal_nomeal_extraction(df):
    insulin_data = df
    timestamps_meal = []
    timestamps_nomeal = []
    insulin_data = insulin_data.fillna(0)
    insulin_data['Timestamp'] = pd.to_datetime(insulin_data['Date'] + ' ' + insulin_data['Time'])

    insulin_data = insulin_data.rename(columns={"BWZ Carb Input (grams)": "CarbInput"})
    insulin_data = insulin_data.drop(columns={'Time', 'Date'})

    insulin_data = insulin_data.sort_values('Timestamp')
    insulin_list = list(zip(insulin_data['Timestamp'], insulin_data['CarbInput']))
    i = 0
    while i < len(insulin_list):
        # For first carbon intake
        if insulin_list[i][1] != 0:
            timestamp = insulin_list[i][0]
            end_interval = insulin_list[i][0] + time_interval
            # For meal data
            while insulin_list[i][0] <= end_interval and i < len(insulin_list) - 1:
                i += 1
                if insulin_list[i][1] != 0:
                    timestamp = insulin_list[i][0]
                    end_interval = insulin_list[i][0] + time_interval
                elif insulin_list[i][0] == end_interval and insulin_list[i][1] != 0:
                    timestamps_meal.append(insulin_list[i][0] - for_30min)
                    timestamp = insulin_list[i][0]
                    end_interval = insulin_list[i][0] + time_interval
            timestamps_meal.append(timestamp - for_30min)
            # For No meal data
            start_nomeal = timestamp + time_interval
            end_nomeal = start_nomeal + time_interval
            while insulin_list[i][0] <= end_nomeal and i < len(insulin_list) - 1:
                i += 1
                if insulin_list[i][1] != 0:
                    if timestamps_nomeal:
                        time_diff = abs(insulin_list[i][0] - timestamps_nomeal[-1]) / 3600
                        if time_diff > time_interval:
                            while insulin_list[i][0] - timestamps_nomeal[-1] < time_interval:
                                timestamps_nomeal[-1] = timestamps_nomeal[-1] + time_interval
                    i -= 1
                    break
                elif insulin_list[i][0] >= end_nomeal:
                    timestamps_nomeal.append(start_nomeal)
                    start_nomeal = end_nomeal
                    end_nomeal = start_nomeal + time_interval
        i += 1
    return timestamps_meal, timestamps_nomeal


def glucose_collect(df1, meal_data_time, nomeal_data_time):
    cgm_data = df1
    timestamps_meal = meal_data_time
    timestamps_nomeal = nomeal_data_time
    cgm_data['Timestamp'] = pd.to_datetime(cgm_data['Date'] + ' ' + cgm_data['Time'])
    cgm_data = cgm_data.drop(columns=['Date', 'Time'])
    cgm_data = cgm_data.sort_values('Timestamp')
    cgm_data.reset_index(drop=True, inplace=True)
    meal_data = pd.DataFrame()
    meal_data['dummy'] = [i for i in range(1, 31)]
    counter = 0
    for i in range(len(timestamps_meal)):
        r = timestamps_meal[i]
        mask = (cgm_data['Timestamp'] > r) & (cgm_data['Timestamp'] <= (r + time_interval + for_30min))
        df = cgm_data['Sensor Glucose (mg/dL)'][mask].values
        meal_data[counter] = pd.Series(df)
        counter += 1
    meal_data = meal_data.drop(columns='dummy')
    meal_data = meal_data.T
    nomeal_data = pd.DataFrame()
    nomeal_data['dummy'] = [i for i in range(1, 25)]
    counter1 = 0
    for i in range(len(timestamps_nomeal)):
        r = timestamps_nomeal[i]
        mask = (cgm_data['Timestamp'] > r) & (cgm_data['Timestamp'] <= (r + time_interval + for_30min))
        df = cgm_data['Sensor Glucose (mg/dL)'][mask].values
        nomeal_data[counter1] = pd.Series(df)
        counter1 += 1
    nomeal_data = nomeal_data.drop(columns='dummy')
    nomeal_data = nomeal_data.T
    return meal_data, nomeal_data

def transform(r):
    r= r.to_numpy()
    transformed = abs(fft(r))
    #y = [i for i in range(r.size)]
    #plot(y,transformed[0])
    #print(list(zip(y,transformed[0]))[1:3])
    return transformed[1:3]


def feature_extraction(df, flag=0):
    df = df.dropna(axis=0)  # ,thresh=meal_data.shape[1]*0.95)
    df.reset_index(drop=True, inplace=True)
    features_extracted = pd.DataFrame()
    features_extracted['RootMean'] = df.apply(lambda r: np.sqrt((r ** 2).sum() / r.size), axis=1)
    if flag:
        list_ = [i for i in range(5, 25)]
    else:
        list_ = [i for i in range(23)]

    features_extracted['time_diff'] = df[list_].idxmax(axis=1, skipna=True) * 5 - 30
    features_extracted['cgm_diffNorm'] = (df[list_].max(axis=1, skipna=True) - df[5]) / df[5]
    df_FFT = pd.DataFrame()
    df_FFT['FFT'] = df.apply(lambda x: transform(x), axis=1)
    df_FFT = df_FFT['FFT'].apply(pd.Series)
    features_extracted['Fpeak1'] = df_FFT[0]
    features_extracted['Fpeak2'] = df_FFT[1]
    features_extracted['class'] = flag
    return features_extracted


# Loading data into Dataframes
insulin_data1 = pd.read_csv('InsulinData.csv', usecols=['Date', 'Time', 'BWZ Carb Input (grams)'])
insulin_data2 = pd.read_csv('Insulin_patient2.csv', usecols=['Date', 'Time', 'BWZ Carb Input (grams)'])
cgm_data1 = pd.read_csv('CGMData.csv', usecols=['Date', 'Time', 'Sensor Glucose (mg/dL)'])
cgm_data2 = pd.read_csv('CGM_patient2.csv', usecols=['Date', 'Time', 'Sensor Glucose (mg/dL)'])
meal_data1_time, nomeal_data1_time = meal_nomeal_extraction(insulin_data1)
meal_data2_time, nomeal_data2_time = meal_nomeal_extraction(insulin_data2)
# Processing CGM Data and extracting data for the meal and no meal data
meal_data1, nomeal_data1 = glucose_collect(cgm_data1, meal_data1_time, nomeal_data1_time)
meal_data2, nomeal_data2 = glucose_collect(cgm_data2, meal_data2_time, nomeal_data2_time)

meal_data_all = meal_data1.append(meal_data2)
nomeal_data_all = nomeal_data1.append(nomeal_data2)

features_meal = feature_extraction(meal_data_all, 1)
features_nomeal = feature_extraction(nomeal_data_all, 0)
features_extracted = features_meal.append(features_nomeal,ignore_index=True)
features = ['RootMean', 'time_diff', 'cgm_diffNorm','Fpeak1','Fpeak2']
X = features_extracted[features].copy()
y= features_extracted['class']
X = StandardScaler().fit_transform(X)
# Train
scoring = {'precision', 'recall', 'f1', 'accuracy'}

k_fold = KFold(n_splits=5, shuffle=True, random_state=0)

# clf = BernoulliNB()
# # clf.fit(X, y)
# print ('BNB: ', cross_val_score(clf, X, y, cv=k_fold, n_jobs=1))
# print (model_selection.cross_validate(estimator=clf, X=X, y=y, cv=k_fold, scoring=scoring))
#

model_svm = svm.SVC(kernel='poly', degree = 7,C=100)
cross_val_score(model_svm, X, y, cv=k_fold, n_jobs=1)
model_selection.cross_validate(estimator=model_svm, X=X, y=y, cv=k_fold, scoring=scoring)

#
# clf3 = LogisticRegression(random_state=0)
# print ('LR: ', cross_val_score(clf3, X, y, cv=k_fold, n_jobs=1))
# print (model_selection.cross_validate(estimator=clf3, X=X, y=y, cv=k_fold, scoring=scoring))
#
# clf4 = GaussianNB()
# print ('GNB: ', cross_val_score(clf4, X, y, cv=k_fold, n_jobs=1))
# print (model_selection.cross_validate(estimator=clf4, X=X, y=y, cv=k_fold, scoring=scoring))
#
# clf5 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(4, 3), random_state=1)
# print ('MLP: ', cross_val_score(clf5, X, y, cv=k_fold, n_jobs=1))
# print (model_selection.cross_validate(estimator=clf5, X=X, y=y, cv=k_fold, scoring=scoring))
#
# clf6 = AdaBoostClassifier(n_estimators=50, random_state=0)
# print ('Ada: ', cross_val_score(clf6, X, y, cv=k_fold, n_jobs=1))
# print (model_selection.cross_validate(estimator=clf6, X=X, y=y, cv=k_fold, scoring=scoring))
"""
model_RF = RandomForestClassifier(n_estimators=50)
cross_val_score(model_RF, X, y, cv=k_fold, n_jobs=1)
model_selection.cross_validate(estimator=model_RF, X=X, y=y, cv=k_fold, scoring=scoring)
"""
"""
clf8 = tree.DecisionTreeClassifier()
print ('DT: ', cross_val_score(clf8, X, y, cv=k_fold, n_jobs=1))
print('Tree')
print (model_selection.cross_validate(estimator=clf8, X=X, y=y, cv=k_fold, scoring=scoring))
"""
#
# clf9 = KNeighborsClassifier(n_neighbors=5)
# print ('kNN: ', cross_val_score(clf9, X, y, cv=k_fold, n_jobs=1))
# print (model_selection.cross_validate(estimator=clf9, X=X, y=y, cv=k_fold, scoring=scoring))


finalClassifier = model_svm
finalClassifier.fit(X, y)
pickle.dump(finalClassifier, open('final.pkl', 'wb'))

