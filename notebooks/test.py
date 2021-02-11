import datetime
import os
import pickle

import pandas as pd
import numpy as np

# Initializing variables

from scipy.fft import fft


def transform(r):
    r = r.to_numpy()
    transformed = abs(fft(r))
    # y = [i for i in range(r.size)]
    # plot(y,transformed[0])
    # print(list(zip(y,transformed[0]))[1:3])
    return transformed[1:3]


def feature_extraction(df):
    df = df.dropna(axis=0)
    df.reset_index(drop=True, inplace=True)
    features_extracted = pd.DataFrame()
    features_extracted['RootMean'] = df.apply(lambda r: np.sqrt((r ** 2).sum() / r.size), axis=1)
    list_ = [i for i in range(23)]
    features_extracted['time_diff'] = df[list_].idxmax(axis=1, skipna=True) * 5 - 30
    features_extracted['cgm_diffNorm'] = (df[list_].max(axis=1, skipna=True) - df[5]) / df[5]
    df_FFT = pd.DataFrame()
    df_FFT['FFT'] = df.apply(lambda x: transform(x), axis=1)
    df_FFT = df_FFT['FFT'].apply(pd.Series)
    features_extracted['Fpeak1'] = df_FFT[0]
    features_extracted['Fpeak2'] = df_FFT[1]
    return features_extracted


test_data = pd.read_csv('test.csv', usecols=[*range(0, 24)], header=None)
features = feature_extraction(test_data)
finalClassifier = pickle.load(open('final.pkl', 'rb'))
result = finalClassifier.predict(features)
pd.DataFrame(result).to_csv('Result.csv',index=False,header=False)
