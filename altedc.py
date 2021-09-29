import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os

import tensorflow as tf
import keras
from keras.models import Input, Sequential, load_model, Model
from keras.layers import Dense, LSTM, Dropout
from keras.layers import Conv1D, Flatten, MaxPooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from numpy.random import seed
# import embeddings_gensim as embeddings_gensim

#variables needed to create features
day = 24*60*60
year = (365.2425)*day
month = 30*day
quarter = 3*month
week=7*day
twowk=14*day
semester=6*month
twomonth=2*month
fourmonth=4*month
fivemonth=5*month
sevenmonth=7*month
eightmonth=8*month
ninemonth=9*month
tenmonth=10*month
elevenmonth=11*month
thirteenmonth=13*month
fourteenmonth=14*month
fifteenmonth=15*month
sixteenmonth=16*month
seventeenmonth=17*month
eighteenmonth=18*month
twoday=2*day


#functions needed to create features

#switch direction degrees wd to numeric cardinal points 
def degrees_to_cardinal(d):
    #dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    dirs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,17]
    ix = round(d / (360. / len(dirs)))
    return dirs[ix % len(dirs)]


def create_features_train(train):
    #make new features on copy of train df
    df_copy = train.copy()
    df_copy['uv_dir']=(270-np.rad2deg(np.arctan2(df_copy['v'],df_copy['u'])))%360
    df_copy['wva']=(180/np.pi)*(np.arctan2(df_copy['u'],df_copy['v']))
    df_copy['mwd']=(180/np.pi)*(np.arctan2(-df_copy['u'],-df_copy['v']))
    df_copy['vh']=np.sqrt((df_copy['u'])**2+(df_copy['v'])**2)
    df_copy['vh3']=df_copy['vh']**3
    df_copy['hr']=df_copy.date.dt.hour
    df_copy['mo']=df_copy.date.dt.month
    df_copy['dy']=df_copy.date.dt.day
    df_copy['wk']=df_copy.date.dt.week
    df_copy['qr']=df_copy.date.dt.quarter
    df_copy['doy']=df_copy.date.dt.dayofyear
    df_copy['dow']=df_copy.date.dt.dayofweek
    df_copy['yr']=df_copy.date.dt.year
    df_copy['days']=df_copy.doy-df_copy.doy.min()
    df_copy['u_z']=(df_copy['u']-df_copy['u'].mean())/df_copy['u'].std()
    df_copy['v_z']=(df_copy['v']-df_copy['v'].mean())/df_copy['v'].std()
    df_copy['ws_z']=(df_copy['ws']-df_copy['ws'].mean())/df_copy['ws'].std()
    df_copy['wd_z']=(df_copy['wd']-df_copy['wd'].mean())/df_copy['wd'].std()
    df_copy['wsq']=pd.qcut(df_copy['ws'],q=4,labels=False)
    df_copy['wdc']=df_copy['wd'].map(lambda x: degrees_to_cardinal(x))
    df_copy['wvac']=df_copy['wva'].map(lambda x: degrees_to_cardinal(x))
    df_copy['mwdc']=df_copy['mwd'].map(lambda x: degrees_to_cardinal(x))
    df_copy['u*v']=df_copy['u']*df_copy['v']
    df_copy['rwd']=df_copy['mwd']-df_copy['wva']
    df_copy['wsr']=df_copy['ws']/df_copy['ws'].mean()
    df_copy['ws100m']=df_copy['ws']*1.25
    df_copy['ws3']=df_copy['ws']**3
    df_copy['ws3']=(df_copy['ws3']-df_copy['ws3'].mean())/df_copy['ws3'].std()
    df_copy['wpe']=((df_copy['ws']**3)*(2089.77))
    df_copy['wpe']=(df_copy['wpe']-df_copy['wpe'].mean())/df_copy['wpe'].std()
    df_copy['pa']=((df_copy['vh']**3)*(2089.77))
    df_copy['pa']=(df_copy['pa']-df_copy['pa'].mean())/df_copy['pa'].std()
    di=df_copy.groupby([df_copy['date'].dt.hour])['vh'].mean().to_dict()
    df_copy['ahvh']=df_copy['hr'].replace(di)
    tic=df_copy.groupby([df_copy['date'].dt.hour])['ws'].std().to_dict()
    df_copy['ti']=df_copy['hr'].replace(tic)/df_copy['ws']
    tiv=df_copy.groupby([df_copy['date'].dt.hour])['ws'].var().to_dict()
    df_copy['tiv']=df_copy['hr'].replace(tiv)/df_copy['ws']

    wsh_m=df_copy.groupby([df_copy['date'].dt.hour])['ws'].max().to_dict()
    df_copy['wsh_max']=df_copy['hr'].replace(wsh_m)/df_copy['ws']


    wdic=df_copy.groupby([df_copy['date'].dt.hour])['wd'].std().to_dict()
    df_copy['wdi']=df_copy['hr'].replace(wdic)/df_copy['wd']

    vhic=df_copy.groupby([df_copy['date'].dt.hour])['vh'].std().to_dict()
    df_copy['vhi']=df_copy['hr'].replace(vhic)/df_copy['vh']


    vhim=df_copy.groupby([df_copy['date'].dt.hour])['vh'].max().to_dict()
    df_copy['vhi_max']=df_copy['hr'].replace(vhim)/df_copy['vh']


    vhivc=df_copy.groupby([df_copy['date'].dt.hour])['vh'].var().to_dict()
    df_copy['vhiv']=df_copy['hr'].replace(vhivc)/df_copy['vh']

    paic=df_copy.groupby([df_copy['date'].dt.hour])['pa'].std().to_dict()
    df_copy['pai']=df_copy['hr'].replace(paic)/df_copy['pa']

    tid=df_copy.groupby([df_copy['date'].dt.day])['ws'].std().to_dict()
    df_copy['tid']=df_copy['dy'].replace(tid)/df_copy['ws']
    vhid=df_copy.groupby([df_copy['date'].dt.day])['vh'].std().to_dict()
    df_copy['vhid']=df_copy['dy'].replace(vhid)/df_copy['vh']
    paid=df_copy.groupby([df_copy['date'].dt.day])['pa'].std().to_dict()
    df_copy['paid']=df_copy['dy'].replace(paid)/df_copy['pa']

    from statsmodels.tsa.seasonal import seasonal_decompose
    result = seasonal_decompose(df_copy['vh'], model='multiplicative', period=365)
    df_copy['vhs']=result.seasonal
    davh_copy=df_copy.groupby([df_copy['date'].dt.date])['vh'].mean().to_dict()
    df_copy['davh']=df_copy['date'].dt.date.replace(davh_copy)
    wavh_copy=df_copy.groupby([df_copy['date'].dt.week])['vh'].mean().to_dict()
    df_copy['wavh']=df_copy['date'].dt.week.replace(wavh_copy)
    daws_copy=df_copy.groupby([df_copy['date'].dt.date])['ws'].mean().to_dict()
    df_copy['daws']=df_copy['date'].dt.date.replace(daws_copy)
    haws_copy=df_copy.groupby([df_copy['date'].dt.hour])['ws'].mean().to_dict()
    df_copy['haws']=df_copy['date'].dt.hour.replace(haws_copy)
    maws_copy=df_copy.groupby([df_copy['date'].dt.month])['ws'].mean().to_dict()
    df_copy['maws']=df_copy['date'].dt.month.replace(maws_copy)
    waws_copy=df_copy.groupby([df_copy['date'].dt.week])['ws'].mean().to_dict()
    df_copy['waws']=df_copy['date'].dt.week.replace(waws_copy)
    mavh_copy=df_copy.groupby([df_copy['date'].dt.month])['vh'].mean().to_dict()
    df_copy['mavh']=df_copy['date'].dt.month.replace(mavh_copy)
    dmvh_copy=df_copy.groupby([df_copy['date'].dt.date])['vh'].max().to_dict()
    df_copy['dmvh']=df_copy['date'].dt.date.replace(dmvh_copy)
    dmws_copy=df_copy.groupby([df_copy['date'].dt.date])['ws'].max().to_dict()
    df_copy['dmws']=df_copy['date'].dt.date.replace(dmws_copy)

    hmws_copy=df_copy.groupby([df_copy['date'].dt.hour])['ws'].max().to_dict()
    df_copy['hmws']=df_copy['date'].dt.hour.replace(hmws_copy)

    df_copy['vhpa']=df_copy['vh']/df_copy['pa']
    pac=df_copy.groupby([df_copy['date'].dt.month])['pa'].sum().to_dict()
    df_copy['pam']=df_copy['mo'].replace(pac)
    df_copy['pam']=(df_copy['pam']-df_copy['pam'].mean())/df_copy['pam'].std()
    df_copy['gust']=np.array([0]+list(df_copy['ws'][1:].values-df_copy['ws'][:-1].values))
    df_copy['dvh']=np.array([0]+list(df_copy['vh'][1:].values-df_copy['vh'][:-1].values))
    df_copy['dpa']=np.array([0]+list(df_copy['pa'][1:].values-df_copy['pa'][:-1].values))
    df_copy['dti']=np.array([0]+list(df_copy['ti'][1:].values-df_copy['ti'][:-1].values))
    df_copy['dwd']=np.array([0]+list(df_copy['wd'][1:].values-df_copy['wd'][:-1].values))

    gusthrc=df_copy.groupby([df_copy['date'].dt.hour])['gust'].std().to_dict()
    df_copy['gusti']=df_copy['hr'].replace(gusthrc)/(df_copy['gust']+0.1)
    
    spring = range(80, 172)
    summer = range(172, 264)
    fall = range(264, 355)
    daje = []
    for i in df_copy['doy']:
        if i in spring:
            season = 'spring'
        elif i in summer:
            season = 'summer'
        elif i in fall:
            season = 'fall'
        else:
            season = 'winter'
        daje.append(season)   
    #add the resulting column to the dataframe (after transforming it as a Series)
    df_copy['season']= pd.Series(daje)
    df_copy=pd.get_dummies(df_copy, columns= ['season'])
    
    timestamp_s=df_copy.date.map(dt.datetime.timestamp)
    df_copy['day_sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df_copy['day_cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df_copy['twoday_sin'] = np.sin(timestamp_s * (2 * np.pi / twoday))
    df_copy['twoday_cos'] = np.cos(timestamp_s * (2 * np.pi / twoday))
    df_copy['year_sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df_copy['year_cos'] = np.cos(timestamp_s * (2 * np.pi / year))
    df_copy['month_sin'] = np.sin(timestamp_s * (2 * np.pi / month))
    df_copy['month_cos'] = np.cos(timestamp_s * (2 * np.pi / month))
    df_copy['quarter_sin'] = np.sin(timestamp_s * (2 * np.pi / quarter))
    df_copy['quarter_cos'] = np.cos(timestamp_s * (2 * np.pi / quarter))
    df_copy['week_sin'] = np.sin(timestamp_s * (2 * np.pi / week))
    df_copy['week_cos'] = np.cos(timestamp_s * (2 * np.pi / week))
    df_copy['twowk_sin'] = np.sin(timestamp_s * (2 * np.pi / twowk))
    df_copy['twowk_cos'] = np.cos(timestamp_s * (2 * np.pi / twowk))
    df_copy['semester_sin'] = np.sin(timestamp_s * (2 * np.pi / semester))
    df_copy['semester_cos'] = np.cos(timestamp_s * (2 * np.pi / semester))
    df_copy['twomonth_sin'] = np.sin(timestamp_s * (2 * np.pi / twomonth))
    df_copy['twomonth_cos'] = np.cos(timestamp_s * (2 * np.pi / twomonth))
    df_copy['fourmonth_sin'] = np.sin(timestamp_s * (2 * np.pi / fourmonth))
    df_copy['fourmonth_cos'] = np.cos(timestamp_s * (2 * np.pi / fourmonth))
    df_copy['fivemonth_sin'] = np.sin(timestamp_s * (2 * np.pi / fivemonth))
    df_copy['fivemonth_cos'] = np.cos(timestamp_s * (2 * np.pi / fivemonth))
    df_copy['sevenmonth_sin'] = np.sin(timestamp_s * (2 * np.pi / sevenmonth))
    df_copy['sevenmonth_cos'] = np.cos(timestamp_s * (2 * np.pi / sevenmonth))
    df_copy['eightmonth_sin'] = np.sin(timestamp_s * (2 * np.pi / eightmonth))
    df_copy['eightmonth_cos'] = np.cos(timestamp_s * (2 * np.pi / eightmonth))
    df_copy['ninemonth_sin'] = np.sin(timestamp_s * (2 * np.pi / ninemonth))
    df_copy['ninemonth_cos'] = np.cos(timestamp_s * (2 * np.pi / ninemonth))
    df_copy['tenmonth_sin'] = np.sin(timestamp_s * (2 * np.pi / tenmonth))
    df_copy['tenmonth_cos'] = np.cos(timestamp_s * (2 * np.pi / tenmonth))
    df_copy['elevenmonth_sin'] = np.sin(timestamp_s * (2 * np.pi / elevenmonth))
    df_copy['elevenmonth_cos'] = np.cos(timestamp_s * (2 * np.pi / elevenmonth))
    df_copy['thirteenmonth_sin'] = np.sin(timestamp_s * (2 * np.pi / thirteenmonth))
    df_copy['thirteenmonth_cos'] = np.cos(timestamp_s * (2 * np.pi / thirteenmonth))
    df_copy['fourteenmonth_sin'] = np.sin(timestamp_s * (2 * np.pi / fourteenmonth))
    df_copy['fourteenmonth_cos'] = np.cos(timestamp_s * (2 * np.pi / fourteenmonth))
    df_copy['fifteenmonth_sin'] = np.sin(timestamp_s * (2 * np.pi / fifteenmonth))
    df_copy['fifteenmonth_cos'] = np.cos(timestamp_s * (2 * np.pi / fifteenmonth))
    df_copy['sixteenmonth_sin'] = np.sin(timestamp_s * (2 * np.pi / sixteenmonth))
    df_copy['sixteenmonth_cos'] = np.cos(timestamp_s * (2 * np.pi / sixteenmonth))
    df_copy['seventeenmonth_sin'] = np.sin(timestamp_s * (2 * np.pi / seventeenmonth))
    df_copy['seventeenmonth_cos'] = np.cos(timestamp_s * (2 * np.pi / seventeenmonth))
    df_copy['eighteenmonth_sin'] = np.sin(timestamp_s * (2 * np.pi / eighteenmonth))
    df_copy['eighteenmonth_cos'] = np.cos(timestamp_s * (2 * np.pi / eighteenmonth))
    df_copy['wswdsin'] = np.sin(df_copy.wd*df_copy.ws)
    df_copy['wswdcos'] = np.cos(df_copy.wd*df_copy.ws)
    
    df_copy['wdr']=df_copy['wd']*np.pi / 180
    df_copy['wx'] = df_copy['ws']*np.cos(df_copy['wdr'])
    df_copy['wy'] = df_copy['ws']*np.sin(df_copy['wdr'])
    
    df_copy['vhi_diff']= df_copy['vhi'].diff() #.diff(12)
    df_copy['vhi_diff'].fillna((df_copy['vhi_diff'].mean()), inplace=True)
    
    df_copy['ws_diff']= df_copy['ws'].diff() #.diff(12)
    df_copy['ws_diff'].fillna((df_copy['ws_diff'].mean()), inplace=True)
    
    df_copy['ti_diff']= df_copy['ti'].diff() #.diff(12)
    df_copy['ti_diff'].fillna((df_copy['ti_diff'].mean()), inplace=True)
    
    df_copy['wd_ns'] = np.cos(np.deg2rad(df_copy['wd']))*df_copy['ws']
    df_copy['wd_ew'] = np.sin(np.deg2rad(df_copy['wd']))*df_copy['ws']
    df_copy['wd_avg'] = np.arctan(df_copy['wd_ew']/df_copy['wd_ns'])

    
    return df_copy

def create_features_val(val):
    #make new features on copy of validation df
    df_copy_val=val.copy()
    df_copy_val['uv_dir']=(270-np.rad2deg(np.arctan2(df_copy_val['v'],df_copy_val['u'])))%360
    df_copy_val['wva']=(180/np.pi)*(np.arctan2(df_copy_val['u'],df_copy_val['v']))
    df_copy_val['mwd']=(180/np.pi)*(np.arctan2(-df_copy_val['u'],-df_copy_val['v']))
    df_copy_val['vh']=np.sqrt((df_copy_val['u'])**2+(df_copy_val['v'])**2)
    df_copy_val['vh3']=df_copy_val['vh']**3
    df_copy_val['hr']=df_copy_val.date.dt.hour
    df_copy_val['mo']=df_copy_val.date.dt.month
    df_copy_val['dy']=df_copy_val.date.dt.day
    df_copy_val['wk']=df_copy_val.date.dt.week
    df_copy_val['qr']=df_copy_val.date.dt.quarter
    df_copy_val['doy']=df_copy_val.date.dt.dayofyear
    df_copy_val['dow']=df_copy_val.date.dt.dayofweek
    df_copy_val['yr']=df_copy_val.date.dt.year
    df_copy_val['days']=df_copy_val.doy-df_copy_val.doy.min()
    df_copy_val['u_z']=(df_copy_val['u']-df_copy_val['u'].mean())/df_copy_val['u'].std()
    df_copy_val['v_z']=(df_copy_val['v']-df_copy_val['v'].mean())/df_copy_val['v'].std()
    df_copy_val['ws_z']=(df_copy_val['ws']-df_copy_val['ws'].mean())/df_copy_val['ws'].std()
    df_copy_val['wd_z']=(df_copy_val['wd']-df_copy_val['wd'].mean())/df_copy_val['wd'].std()
    df_copy_val['wsq']=pd.qcut(df_copy_val['ws'],q=4, labels=False)
    df_copy_val['wdc']=df_copy_val['wd'].map(lambda x: degrees_to_cardinal(x))
    df_copy_val['wvac']=df_copy_val['wva'].map(lambda x: degrees_to_cardinal(x))
    df_copy_val['mwdc']=df_copy_val['mwd'].map(lambda x: degrees_to_cardinal(x))
    df_copy_val['u*v']=df_copy_val['u']*df_copy_val['v']
    df_copy_val['rwd']=df_copy_val['mwd']-df_copy_val['wva']
    df_copy_val['wsr']=df_copy_val['ws']/df_copy_val['ws'].mean()
    df_copy_val['ws100m']=df_copy_val['ws']*1.25
    df_copy_val['ws3']=df_copy_val['ws']**3
    df_copy_val['ws3']=(df_copy_val['ws3']-df_copy_val['ws3'].mean())/df_copy_val['ws3'].std()
    df_copy_val['wpe']=((df_copy_val['ws']**3)*(2089.77))
    df_copy_val['wpe']=(df_copy_val['wpe']-df_copy_val['wpe'].mean())/df_copy_val['wpe'].std()
    df_copy_val['pa']=((df_copy_val['vh']**3)*(2089.77))
    df_copy_val['pa']=(df_copy_val['pa']-df_copy_val['pa'].mean())/df_copy_val['pa'].std()
    di_val=df_copy_val.groupby([df_copy_val['date'].dt.hour])['vh'].mean().to_dict()
    df_copy_val['ahvh']=df_copy_val['hr'].replace(di_val)
    tiv=df_copy_val.groupby([df_copy_val['date'].dt.hour])['ws'].std().to_dict()
    df_copy_val['ti']=df_copy_val['hr'].replace(tiv)/df_copy_val['ws']


    wsh_mv=df_copy_val.groupby([df_copy_val['date'].dt.hour])['ws'].max().to_dict()
    df_copy_val['wsh_max']=df_copy_val['hr'].replace(wsh_mv)/df_copy_val['ws']


    tival=df_copy_val.groupby([df_copy_val['date'].dt.hour])['ws'].var().to_dict()
    df_copy_val['tiv']=df_copy_val['hr'].replace(tival)/df_copy_val['ws']

    wdiv=df_copy_val.groupby([df_copy_val['date'].dt.hour])['wd'].std().to_dict()
    df_copy_val['wdi']=df_copy_val['hr'].replace(wdiv)/df_copy_val['wd']

    vhiv=df_copy_val.groupby([df_copy_val['date'].dt.hour])['vh'].std().to_dict()
    df_copy_val['vhi']=df_copy_val['hr'].replace(vhiv)/df_copy_val['vh']


    vhimv=df_copy_val.groupby([df_copy_val['date'].dt.hour])['vh'].max().to_dict()
    df_copy_val['vhi_max']=df_copy_val['hr'].replace(vhimv)/df_copy_val['vh']


    vhivval=df_copy_val.groupby([df_copy_val['date'].dt.hour])['vh'].var().to_dict()
    df_copy_val['vhiv']=df_copy_val['hr'].replace(vhivval)/df_copy_val['vh']

    paiv=df_copy_val.groupby([df_copy_val['date'].dt.hour])['pa'].std().to_dict()
    df_copy_val['pai']=df_copy_val['hr'].replace(paiv)/df_copy_val['pa']

    tidv=df_copy_val.groupby([df_copy_val['date'].dt.day])['ws'].std().to_dict()
    df_copy_val['tid']=df_copy_val['dy'].replace(tidv)/df_copy_val['ws']
    vhidv=df_copy_val.groupby([df_copy_val['date'].dt.day])['vh'].std().to_dict()
    df_copy_val['vhid']=df_copy_val['dy'].replace(vhidv)/df_copy_val['vh']
    paidv=df_copy_val.groupby([df_copy_val['date'].dt.day])['pa'].std().to_dict()
    df_copy_val['paid']=df_copy_val['dy'].replace(paidv)/df_copy_val['pa']

    from statsmodels.tsa.seasonal import seasonal_decompose
    result_val = seasonal_decompose(df_copy_val['vh'], model='multiplicative', period=365)
    df_copy_val['vhs']=result_val.seasonal
    davh_val=df_copy_val.groupby([df_copy_val['date'].dt.date])['vh'].mean().to_dict()
    df_copy_val['davh']=df_copy_val['date'].dt.date.replace(davh_val)
    wavh_val=df_copy_val.groupby([df_copy_val['date'].dt.week])['vh'].mean().to_dict()
    df_copy_val['wavh']=df_copy_val['date'].dt.week.replace(wavh_val)
    daws_copy_val=df_copy_val.groupby([df_copy_val['date'].dt.date])['ws'].mean().to_dict()
    df_copy_val['daws']=df_copy_val['date'].dt.date.replace(daws_copy_val)
    haws_copy_val=df_copy_val.groupby([df_copy_val['date'].dt.hour])['ws'].mean().to_dict()
    df_copy_val['haws']=df_copy_val['date'].dt.hour.replace(haws_copy_val)
    maws_copy_val=df_copy_val.groupby([df_copy_val['date'].dt.month])['ws'].mean().to_dict()
    df_copy_val['maws']=df_copy_val['date'].dt.month.replace(maws_copy_val)
    waws_copy_val=df_copy_val.groupby([df_copy_val['date'].dt.week])['ws'].mean().to_dict()
    df_copy_val['waws']=df_copy_val['date'].dt.week.replace(waws_copy_val)
    mavh_copy_val=df_copy_val.groupby([df_copy_val['date'].dt.month])['vh'].mean().to_dict()
    df_copy_val['mavh']=df_copy_val['date'].dt.month.replace(mavh_copy_val)
    dmvh_val=df_copy_val.groupby([df_copy_val['date'].dt.date])['vh'].max().to_dict()
    df_copy_val['dmvh']=df_copy_val['date'].dt.date.replace(dmvh_val)
    dmws_copy_v=df_copy_val.groupby([df_copy_val['date'].dt.date])['ws'].max().to_dict()
    df_copy_val['dmws']=df_copy_val['date'].dt.date.replace(dmws_copy_v)

    hmws_copy_v=df_copy_val.groupby([df_copy_val['date'].dt.hour])['ws'].max().to_dict()
    df_copy_val['hmws']=df_copy_val['date'].dt.hour.replace(hmws_copy_v)

    df_copy_val['vhpa']=df_copy_val['vh']/df_copy_val['pa']
    pac_val=df_copy_val.groupby([df_copy_val['date'].dt.month])['pa'].sum().to_dict()
    df_copy_val['pam']=df_copy_val['mo'].replace(pac_val)
    df_copy_val['pam']=(df_copy_val['pam']-df_copy_val['pam'].mean())/df_copy_val['pam'].std()
    df_copy_val['gust']=np.array([0]+list(df_copy_val['ws'][1:].values-df_copy_val['ws'][:-1].values))
    df_copy_val['dvh']=np.array([0]+list(df_copy_val['vh'][1:].values-df_copy_val['vh'][:-1].values))
    df_copy_val['dpa']=np.array([0]+list(df_copy_val['pa'][1:].values-df_copy_val['pa'][:-1].values))
    df_copy_val['dti']=np.array([0]+list(df_copy_val['ti'][1:].values-df_copy_val['ti'][:-1].values))
    df_copy_val['dwd']=np.array([0]+list(df_copy_val['wd'][1:].values-df_copy_val['wd'][:-1].values))
    gusthrv=df_copy_val.groupby([df_copy_val['date'].dt.hour])['gust'].std().to_dict()
    df_copy_val['gusti']=df_copy_val['hr'].replace(gusthrv)/(df_copy_val['gust']+0.1)
    
    spring = range(80, 172)
    summer = range(172, 264)
    fall = range(264, 355)
    daje_val = []
    for i in df_copy_val['doy']:
        if i in spring:
            season = 'spring'
        elif i in summer:
            season = 'summer'
        elif i in fall:
            season = 'fall'
        else:
            season = 'winter'
        daje_val.append(season)   
    #add the resulting column to the dataframe (after transforming it as a Series)
    df_copy_val['season']= pd.Series(daje_val)
    df_copy_val=pd.get_dummies(df_copy_val, columns= ['season'])
    
    timestamp_sv=df_copy_val.date.map(dt.datetime.timestamp)
    df_copy_val['day_sin'] = np.sin(timestamp_sv * (2 * np.pi / day))
    df_copy_val['day_cos'] = np.cos(timestamp_sv * (2 * np.pi / day))
    df_copy_val['twoday_sin'] = np.sin(timestamp_sv * (2 * np.pi / twoday))
    df_copy_val['twoday_cos'] = np.cos(timestamp_sv * (2 * np.pi / twoday))
    df_copy_val['year_sin'] = np.sin(timestamp_sv * (2 * np.pi / year))
    df_copy_val['year_cos'] = np.cos(timestamp_sv * (2 * np.pi / year))
    df_copy_val['month_sin'] = np.sin(timestamp_sv * (2 * np.pi / month))
    df_copy_val['month_cos'] = np.cos(timestamp_sv * (2 * np.pi / month))
    df_copy_val['quarter_sin'] = np.sin(timestamp_sv * (2 * np.pi / quarter))
    df_copy_val['quarter_cos'] = np.cos(timestamp_sv * (2 * np.pi / quarter))
    df_copy_val['week_sin'] = np.sin(timestamp_sv * (2 * np.pi / week))
    df_copy_val['week_cos'] = np.cos(timestamp_sv * (2 * np.pi / week))
    df_copy_val['twowk_sin'] = np.sin(timestamp_sv * (2 * np.pi / twowk))
    df_copy_val['twowk_cos'] = np.cos(timestamp_sv * (2 * np.pi / twowk))
    df_copy_val['semester_sin'] = np.sin(timestamp_sv * (2 * np.pi / semester))
    df_copy_val['semester_cos'] = np.cos(timestamp_sv * (2 * np.pi / semester))
    df_copy_val['twomonth_sin'] = np.sin(timestamp_sv * (2 * np.pi / twomonth))
    df_copy_val['twomonth_cos'] = np.cos(timestamp_sv * (2 * np.pi / twomonth))
    df_copy_val['fourmonth_sin'] = np.sin(timestamp_sv * (2 * np.pi / fourmonth))
    df_copy_val['fourmonth_cos'] = np.cos(timestamp_sv * (2 * np.pi / fourmonth))
    df_copy_val['fivemonth_sin'] = np.sin(timestamp_sv * (2 * np.pi / fivemonth))
    df_copy_val['fivemonth_cos'] = np.cos(timestamp_sv * (2 * np.pi / fivemonth))
    df_copy_val['sevenmonth_sin'] = np.sin(timestamp_sv * (2 * np.pi / sevenmonth))
    df_copy_val['sevenmonth_cos'] = np.cos(timestamp_sv * (2 * np.pi / sevenmonth))
    df_copy_val['eightmonth_sin'] = np.sin(timestamp_sv * (2 * np.pi / eightmonth))
    df_copy_val['eightmonth_cos'] = np.cos(timestamp_sv * (2 * np.pi / eightmonth))
    df_copy_val['ninemonth_sin'] = np.sin(timestamp_sv * (2 * np.pi / ninemonth))
    df_copy_val['ninemonth_cos'] = np.cos(timestamp_sv * (2 * np.pi / ninemonth))
    df_copy_val['tenmonth_sin'] = np.sin(timestamp_sv * (2 * np.pi / tenmonth))
    df_copy_val['tenmonth_cos'] = np.cos(timestamp_sv * (2 * np.pi / tenmonth))
    df_copy_val['elevenmonth_sin'] = np.sin(timestamp_sv * (2 * np.pi / elevenmonth))
    df_copy_val['elevenmonth_cos'] = np.cos(timestamp_sv * (2 * np.pi / elevenmonth))
    df_copy_val['thirteenmonth_sin'] = np.sin(timestamp_sv * (2 * np.pi / thirteenmonth))
    df_copy_val['thirteenmonth_cos'] = np.cos(timestamp_sv * (2 * np.pi / thirteenmonth))
    df_copy_val['fourteenmonth_sin'] = np.sin(timestamp_sv * (2 * np.pi / fourteenmonth))
    df_copy_val['fourteenmonth_cos'] = np.cos(timestamp_sv * (2 * np.pi / fourteenmonth))
    df_copy_val['fifteenmonth_sin'] = np.sin(timestamp_sv * (2 * np.pi / fifteenmonth))
    df_copy_val['fifteenmonth_cos'] = np.cos(timestamp_sv * (2 * np.pi / fifteenmonth))
    df_copy_val['sixteenmonth_sin'] = np.sin(timestamp_sv * (2 * np.pi / sixteenmonth))
    df_copy_val['sixteenmonth_cos'] = np.cos(timestamp_sv * (2 * np.pi / sixteenmonth))
    df_copy_val['seventeenmonth_sin'] = np.sin(timestamp_sv * (2 * np.pi / seventeenmonth))
    df_copy_val['seventeenmonth_cos'] = np.cos(timestamp_sv * (2 * np.pi / seventeenmonth))
    df_copy_val['eighteenmonth_sin'] = np.sin(timestamp_sv * (2 * np.pi / eighteenmonth))
    df_copy_val['eighteenmonth_cos'] = np.cos(timestamp_sv * (2 * np.pi / eighteenmonth))
    df_copy_val['wswdsin'] = np.sin(df_copy_val.wd*df_copy_val.ws)
    df_copy_val['wswdcos'] = np.cos(df_copy_val.wd*df_copy_val.ws)
    
    df_copy_val['wdr']=df_copy_val['wd']*np.pi / 180
    df_copy_val['wx'] = df_copy_val['ws']*np.cos(df_copy_val['wdr'])
    df_copy_val['wy'] = df_copy_val['ws']*np.sin(df_copy_val['wdr'])
    
    df_copy_val['vhi_diff']= df_copy_val['vhi'].diff() #.diff(12)
    df_copy_val['vhi_diff'].fillna((df_copy_val['vhi_diff'].mean()), inplace=True)
    
    df_copy_val['ws_diff']= df_copy_val['ws'].diff() #.diff(12)
    df_copy_val['ws_diff'].fillna((df_copy_val['ws_diff'].mean()), inplace=True)
    
    df_copy_val['ti_diff']= df_copy_val['ti'].diff() #.diff(12)
    df_copy_val['ti_diff'].fillna((df_copy_val['ti_diff'].mean()), inplace=True)
    
    df_copy_val['wd_ns'] = np.cos(np.deg2rad(df_copy_val['wd']))*df_copy_val['ws']
    df_copy_val['wd_ew'] = np.sin(np.deg2rad(df_copy_val['wd']))*df_copy_val['ws']
    df_copy_val['wd_avg'] = np.arctan(df_copy_val['wd_ew']/df_copy_val['wd_ns'])

    
    return df_copy_val

def create_features_test(test):
    #make new features on copy of test df
    df_copy_test=test.copy()
    df_copy_test['uv_dir']=(270-np.rad2deg(np.arctan2(df_copy_test['v'],df_copy_test['u'])))%360
    df_copy_test['mwd']=(180/np.pi)*(np.arctan2(-df_copy_test['u'],-df_copy_test['v']))
    df_copy_test['wva']=(180/np.pi)*(np.arctan2(df_copy_test['u'],df_copy_test['v']))
    df_copy_test['vh']=np.sqrt((df_copy_test['u'])**2+(df_copy_test['v'])**2)
    df_copy_test['vh3']=df_copy_test['vh']**3
    df_copy_test['hr']=df_copy_test.date.dt.hour
    df_copy_test['mo']=df_copy_test.date.dt.month
    df_copy_test['dy']=df_copy_test.date.dt.day
    df_copy_test['wk']=df_copy_test.date.dt.week
    df_copy_test['qr']=df_copy_test.date.dt.quarter
    df_copy_test['doy']=df_copy_test.date.dt.dayofyear
    df_copy_test['dow']=df_copy_test.date.dt.dayofweek
    df_copy_test['yr']=df_copy_test.date.dt.year
    df_copy_test['days']=df_copy_test.doy-df_copy_test.doy.min()
    df_copy_test['u_z']=(df_copy_test['u']-df_copy_test['u'].mean())/df_copy_test['u'].std()
    df_copy_test['v_z']=(df_copy_test['v']-df_copy_test['v'].mean())/df_copy_test['v'].std()
    df_copy_test['ws_z']=(df_copy_test['ws']-df_copy_test['ws'].mean())/df_copy_test['ws'].std()
    df_copy_test['wd_z']=(df_copy_test['wd']-df_copy_test['wd'].mean())/df_copy_test['wd'].std()
    df_copy_test['wsq']=pd.qcut(df_copy_test['ws'],q=4,labels=False)
    df_copy_test['wdc']=df_copy_test['wd'].map(lambda x: degrees_to_cardinal(x))
    df_copy_test['wvac']=df_copy_test['wva'].map(lambda x: degrees_to_cardinal(x))
    df_copy_test['mwdc']=df_copy_test['mwd'].map(lambda x: degrees_to_cardinal(x))
    df_copy_test['u*v']=df_copy_test['u']*df_copy_test['v']
    df_copy_test['rwd']=df_copy_test['mwd']-df_copy_test['wva']
    df_copy_test['wsr']=df_copy_test['ws']/df_copy_test['ws'].mean()
    df_copy_test['ws100m']=df_copy_test['ws']*1.25
    df_copy_test['ws3']=df_copy_test['ws']**3
    df_copy_test['ws3']=(df_copy_test['ws3']-df_copy_test['ws3'].mean())/df_copy_test['ws3'].std()
    df_copy_test['wpe']=((df_copy_test['ws']**3)*(2089.77))
    df_copy_test['wpe']=(df_copy_test['wpe']-df_copy_test['wpe'].mean())/df_copy_test['wpe'].std()
    df_copy_test['pa']=((df_copy_test['vh']**3)*(2089.77))
    df_copy_test['pa']=(df_copy_test['pa']-df_copy_test['pa'].mean())/df_copy_test['pa'].std()
    di_test=df_copy_test.groupby([df_copy_test['date'].dt.hour])['vh'].mean().to_dict()
    df_copy_test['ahvh']=df_copy_test['hr'].replace(di_test)
    tit=df_copy_test.groupby([df_copy_test['date'].dt.hour])['ws'].std().to_dict()
    df_copy_test['ti']=df_copy_test['hr'].replace(tit)/df_copy_test['ws']


    wsh_mt=df_copy_test.groupby([df_copy_test['date'].dt.hour])['ws'].max().to_dict()
    df_copy_test['wsh_max']=df_copy_test['hr'].replace(wsh_mt)/df_copy_test['ws']


    tivtest=df_copy_test.groupby([df_copy_test['date'].dt.hour])['ws'].var().to_dict()
    df_copy_test['tiv']=df_copy_test['hr'].replace(tivtest)/df_copy_test['ws']

    wdi=df_copy_test.groupby([df_copy_test['date'].dt.hour])['wd'].std().to_dict()
    df_copy_test['wdi']=df_copy_test['hr'].replace(wdi)/df_copy_test['wd']

    vhit=df_copy_test.groupby([df_copy_test['date'].dt.hour])['vh'].std().to_dict()
    df_copy_test['vhi']=df_copy_test['hr'].replace(vhit)/df_copy_test['vh']


    vhimt=df_copy_test.groupby([df_copy_test['date'].dt.hour])['vh'].max().to_dict()
    df_copy_test['vhi_max']=df_copy_test['hr'].replace(vhimt)/df_copy_test['vh']



    vhivtest=df_copy_test.groupby([df_copy_test['date'].dt.hour])['vh'].var().to_dict()
    df_copy_test['vhiv']=df_copy_test['hr'].replace(vhivtest)/df_copy_test['vh']

    pait=df_copy_test.groupby([df_copy_test['date'].dt.hour])['pa'].std().to_dict()
    df_copy_test['pai']=df_copy_test['hr'].replace(pait)/df_copy_test['pa']

    tidt=df_copy_test.groupby([df_copy_test['date'].dt.day])['ws'].std().to_dict()
    df_copy_test['tid']=df_copy_test['dy'].replace(tidt)/df_copy_test['ws']
    vhidt=df_copy_test.groupby([df_copy_test['date'].dt.day])['vh'].std().to_dict()
    df_copy_test['vhid']=df_copy_test['dy'].replace(vhidt)/df_copy_test['vh']
    paidt=df_copy_test.groupby([df_copy_test['date'].dt.day])['pa'].std().to_dict()
    df_copy_test['paid']=df_copy_test['dy'].replace(paidt)/df_copy_test['pa']

    from statsmodels.tsa.seasonal import seasonal_decompose
    result_test = seasonal_decompose(df_copy_test['vh'], model='multiplicative', period=365)
    df_copy_test['vhs']=result_test.seasonal
    davh_test=df_copy_test.groupby([df_copy_test['date'].dt.date])['vh'].mean().to_dict()
    df_copy_test['davh']=df_copy_test['date'].dt.date.replace(davh_test)
    wavh_test=df_copy_test.groupby([df_copy_test['date'].dt.week])['vh'].mean().to_dict()
    df_copy_test['wavh']=df_copy_test['date'].dt.week.replace(wavh_test)
    daws_copy_test=df_copy_test.groupby([df_copy_test['date'].dt.date])['ws'].mean().to_dict()
    df_copy_test['daws']=df_copy_test['date'].dt.date.replace(daws_copy_test)
    haws_copy_test=df_copy_test.groupby([df_copy_test['date'].dt.hour])['ws'].mean().to_dict()
    df_copy_test['haws']=df_copy_test['date'].dt.hour.replace(haws_copy_test)
    maws_copy_test=df_copy_test.groupby([df_copy_test['date'].dt.month])['ws'].mean().to_dict()
    df_copy_test['maws']=df_copy_test['date'].dt.month.replace(maws_copy_test)
    waws_copy_test=df_copy_test.groupby([df_copy_test['date'].dt.week])['ws'].mean().to_dict()
    df_copy_test['waws']=df_copy_test['date'].dt.week.replace(waws_copy_test)
    mavh_copy_test=df_copy_test.groupby([df_copy_test['date'].dt.month])['vh'].mean().to_dict()
    df_copy_test['mavh']=df_copy_test['date'].dt.month.replace(mavh_copy_test)
    dmvh_test=df_copy_test.groupby([df_copy_test['date'].dt.date])['vh'].max().to_dict()
    df_copy_test['dmvh']=df_copy_test['date'].dt.date.replace(dmvh_test)
    dmws_copy_test=df_copy_test.groupby([df_copy_test['date'].dt.date])['ws'].max().to_dict()
    df_copy_test['dmws']=df_copy_test['date'].dt.date.replace(dmws_copy_test)

    hmws_copy_test=df_copy_test.groupby([df_copy_test['date'].dt.hour])['ws'].max().to_dict()
    df_copy_test['hmws']=df_copy_test['date'].dt.hour.replace(hmws_copy_test)

    df_copy_test['vhpa']=df_copy_test['vh']/df_copy_test['pa']
    pac_test=df_copy_test.groupby([df_copy_test['date'].dt.month])['pa'].sum().to_dict()
    df_copy_test['pam']=df_copy_test['mo'].replace(pac_test)
    df_copy_test['pam']=(df_copy_test['pam']-df_copy_test['pam'].mean())/df_copy_test['pam'].std()
    df_copy_test['gust']=np.array([0]+list(df_copy_test['ws'][1:].values-df_copy_test['ws'][:-1].values))
    df_copy_test['dvh']=np.array([0]+list(df_copy_test['vh'][1:].values-df_copy_test['vh'][:-1].values))
    df_copy_test['dpa']=np.array([0]+list(df_copy_test['pa'][1:].values-df_copy_test['pa'][:-1].values))
    df_copy_test['dti']=np.array([0]+list(df_copy_test['ti'][1:].values-df_copy_test['ti'][:-1].values))
    df_copy_test['dwd']=np.array([0]+list(df_copy_test['wd'][1:].values-df_copy_test['wd'][:-1].values))
    gusthrt=df_copy_test.groupby([df_copy_test['date'].dt.hour])['gust'].std().to_dict()
    df_copy_test['gusti']=df_copy_test['hr'].replace(gusthrt)/(df_copy_test['gust']+0.1)
    
    spring = range(80, 172)
    summer = range(172, 264)
    fall = range(264, 355)
    daje_test = []
    for i in df_copy_test['doy']:
        if i in spring:
            season = 'spring'
        elif i in summer:
            season = 'summer'
        elif i in fall:
            season = 'fall'
        else:
            season = 'winter'
        daje_test.append(season)   
    #add the resulting column to the dataframe (after transforming it as a Series)
    df_copy_test['season']= pd.Series(daje_test)
    df_copy_test=pd.get_dummies(df_copy_test, columns= ['season'])
    
    timestamp_st=df_copy_test.date.map(dt.datetime.timestamp)
    df_copy_test['day_sin'] = np.sin(timestamp_st * (2 * np.pi / day))
    df_copy_test['day_cos'] = np.cos(timestamp_st * (2 * np.pi / day))
    df_copy_test['twoday_sin'] = np.sin(timestamp_st * (2 * np.pi / twoday))
    df_copy_test['twoday_cos'] = np.cos(timestamp_st * (2 * np.pi / twoday))
    df_copy_test['year_sin'] = np.sin(timestamp_st * (2 * np.pi / year))
    df_copy_test['year_cos'] = np.cos(timestamp_st * (2 * np.pi / year))
    df_copy_test['month_sin'] = np.sin(timestamp_st * (2 * np.pi / month))
    df_copy_test['month_cos'] = np.cos(timestamp_st * (2 * np.pi / month))
    df_copy_test['quarter_sin'] = np.sin(timestamp_st * (2 * np.pi / quarter))
    df_copy_test['quarter_cos'] = np.cos(timestamp_st * (2 * np.pi / quarter))
    df_copy_test['week_sin'] = np.sin(timestamp_st * (2 * np.pi / week))
    df_copy_test['week_cos'] = np.cos(timestamp_st * (2 * np.pi / week))
    df_copy_test['twowk_sin'] = np.sin(timestamp_st * (2 * np.pi / twowk))
    df_copy_test['twowk_cos'] = np.cos(timestamp_st * (2 * np.pi / twowk))
    df_copy_test['semester_sin'] = np.sin(timestamp_st * (2 * np.pi / semester))
    df_copy_test['semester_cos'] = np.cos(timestamp_st * (2 * np.pi / semester))
    df_copy_test['twomonth_sin'] = np.sin(timestamp_st * (2 * np.pi / twomonth))
    df_copy_test['twomonth_cos'] = np.cos(timestamp_st * (2 * np.pi / twomonth))
    df_copy_test['fourmonth_sin'] = np.sin(timestamp_st * (2 * np.pi / fourmonth))
    df_copy_test['fourmonth_cos'] = np.cos(timestamp_st * (2 * np.pi / fourmonth))
    df_copy_test['fivemonth_sin'] = np.sin(timestamp_st * (2 * np.pi / fivemonth))
    df_copy_test['fivemonth_cos'] = np.cos(timestamp_st * (2 * np.pi / fivemonth))
    df_copy_test['sevenmonth_sin'] = np.sin(timestamp_st * (2 * np.pi / sevenmonth))
    df_copy_test['sevenmonth_cos'] = np.cos(timestamp_st * (2 * np.pi / sevenmonth))
    df_copy_test['eightmonth_sin'] = np.sin(timestamp_st * (2 * np.pi / eightmonth))
    df_copy_test['eightmonth_cos'] = np.cos(timestamp_st * (2 * np.pi / eightmonth))
    df_copy_test['ninemonth_sin'] = np.sin(timestamp_st * (2 * np.pi / ninemonth))
    df_copy_test['ninemonth_cos'] = np.cos(timestamp_st * (2 * np.pi / ninemonth))
    df_copy_test['tenmonth_sin'] = np.sin(timestamp_st * (2 * np.pi / tenmonth))
    df_copy_test['tenmonth_cos'] = np.cos(timestamp_st * (2 * np.pi / tenmonth))
    df_copy_test['elevenmonth_sin'] = np.sin(timestamp_st * (2 * np.pi / elevenmonth))
    df_copy_test['elevenmonth_cos'] = np.cos(timestamp_st * (2 * np.pi / elevenmonth))
    df_copy_test['thirteenmonth_sin'] = np.sin(timestamp_st * (2 * np.pi / thirteenmonth))
    df_copy_test['thirteenmonth_cos'] = np.cos(timestamp_st * (2 * np.pi / thirteenmonth))
    df_copy_test['fourteenmonth_sin'] = np.sin(timestamp_st * (2 * np.pi / fourteenmonth))
    df_copy_test['fourteenmonth_cos'] = np.cos(timestamp_st * (2 * np.pi / fourteenmonth))
    df_copy_test['fifteenmonth_sin'] = np.sin(timestamp_st * (2 * np.pi / fifteenmonth))
    df_copy_test['fifteenmonth_cos'] = np.cos(timestamp_st * (2 * np.pi / fifteenmonth))
    df_copy_test['sixteenmonth_sin'] = np.sin(timestamp_st * (2 * np.pi / sixteenmonth))
    df_copy_test['sixteenmonth_cos'] = np.cos(timestamp_st * (2 * np.pi / sixteenmonth))
    df_copy_test['seventeenmonth_sin'] = np.sin(timestamp_st * (2 * np.pi / seventeenmonth))
    df_copy_test['seventeenmonth_cos'] = np.cos(timestamp_st * (2 * np.pi / seventeenmonth))
    df_copy_test['eighteenmonth_sin'] = np.sin(timestamp_st * (2 * np.pi / eighteenmonth))
    df_copy_test['eighteenmonth_cos'] = np.cos(timestamp_st * (2 * np.pi / eighteenmonth))
    df_copy_test['wswdsin'] = np.sin(df_copy_test.wd*df_copy_test.ws)
    df_copy_test['wswdcos'] = np.cos(df_copy_test.wd*df_copy_test.ws)
    
    df_copy_test['wdr']=df_copy_test['wd']*np.pi / 180
    df_copy_test['wx'] = df_copy_test['ws']*np.cos(df_copy_test['wdr'])
    df_copy_test['wy'] = df_copy_test['ws']*np.sin(df_copy_test['wdr'])
    
    df_copy_test['vhi_diff']= df_copy_test['vhi'].diff() #.diff(12)
    df_copy_test['vhi_diff'].fillna((df_copy_test['vhi_diff'].mean()), inplace=True)
    
    df_copy_test['ws_diff']= df_copy_test['ws'].diff() #.diff(12)
    df_copy_test['ws_diff'].fillna((df_copy_test['ws_diff'].mean()), inplace=True)
    
    df_copy_test['ti_diff']= df_copy_test['ti'].diff() #.diff(12)
    df_copy_test['ti_diff'].fillna((df_copy_test['ti_diff'].mean()), inplace=True)
    
    df_copy_test['wd_ns'] = np.cos(np.deg2rad(df_copy_test['wd']))*df_copy_test['ws']
    df_copy_test['wd_ew'] = np.sin(np.deg2rad(df_copy_test['wd']))*df_copy_test['ws']
    
    df_copy_test['wd_avg'] = np.arctan(df_copy_test['wd_ew']/df_copy_test['wd_ns'])
    
    
    return df_copy_test

def split_data(df):
    times = sorted(df.index.values)
    last_10pct = sorted(df.index.values)[-int(0.1*len(times))]  # Last 10% of series
    last_20pct = sorted(df.index.values)[-int(0.2*len(times))]  # Last 20% of series
    
    df_train = df[(df.index < last_20pct)]  # Training data are 80% of total data
    df_val = df[(df.index >= last_20pct) & (df.index < last_10pct)]
    df_test = df[(df.index >= last_10pct)]

    return df_train, df_val, df_test

# The function below creates a dataset of samples with 'seq_len' time-steps as history 
# and 1-step ahead value as target 
# Input: data (array), seq_len (int, defining the number of time-steps that are used as history to create each sample), 
# ahead (int, defining the number of time-steps that are used as targets
# Output: 
# X - array with shape (samples, seq_len, len(train_columns)), contains historical features per sample
# y - array with shape (samples,), contains the target, 
# that matches the index of feature that we want to predict in the future

def create_supervised_data(data, seq_len=20, ahead=1):
    X = []
    y = []
        
    for i in range(seq_len, len(data)-ahead+1):
    #append historical inputs in X
      X.append(data[i-seq_len:i,:])
      y.append(data[i:i+ahead,0]) #:
    #append targets in y
    X, y = np.array(X), np.array(y)
    return X,y

def compile_and_fit(model, X, y, X_val, y_val, patience=2, max_epochs=100):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])

  history = model.fit(X, y, epochs=max_epochs,
                      validation_data=(X_val, y_val),
                      callbacks=[early_stopping])
  return history

def create_model(seq_len, features_dimension):
    # Adding LSTM layers and Dropout regularisation
    # Fill your code here
    model = Sequential()
    model.add(LSTM(units = 50, input_shape = (seq_len, features_dimension)))
    model.add(Dropout(0.2))

    model.add(Dense(units = 1))
    return model

def plot_loss(history):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    return 

def print_metrics_model(X_train, y_train, X_val, y_val, X_test, y_test):
    print('Evaluation metrics')
    print(
        'Training Data - MSE Loss: {:.8f}, MAE Loss: {:.8f}'.format(
                                                    model.evaluate(X_train, y_train, verbose=0)[0], 
                                                    model.evaluate(X_train, y_train, verbose=0)[1]))
    print(
        'Validation Data - MSE Loss: {:.8f}, MAE Loss: {:.8f}'.format(
                                                    model.evaluate(X_val, y_val, verbose=0)[0],
                                                    model.evaluate(X_val, y_val, verbose=0)[1]))
    print(
        'Test Data - MSE Loss: {:.8f}, MAE Loss: {:.8f}'.format(
                                                    model.evaluate(X_test, y_test, verbose=0)[0],
                                                    model.evaluate(X_test, y_test, verbose=0)[1]))
    return

# Visualize the results
def plot_predictions_test(y_test, predictions):
    plt.plot(np.arange(len(y_test)),y_test, color = 'red', label = 'Real')
    plt.plot(np.arange(len(predictions)),predictions, color = 'green', label = 'Predicted')
    plt.xticks(np.arange(0,len(predictions),10))
    plt.title('wp1 Prediction')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
    return


    
