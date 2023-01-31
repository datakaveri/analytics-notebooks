import math
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta

import h3
import h3pandas

import json 

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate, BatchNormalization, Reshape
from tensorflow.keras.optimizers import Adam

from spektral.utils import gcn_filter
from spektral.layers import GCNConv
import tensorflow
from tensorflow.keras.layers import Flatten

# The idea behind calculating the adjacency matrix is: All the hexagons that have the same parent hexagon (in our case the resolution of parent hexagon is 8) are neighbours and will be marked as 1. This idea will help in correlating the non-neighbouring dumping hexes.
def get_adjacency_matrix(hex_loc):
  adj_mat=[]
  hex_loc= np.array(hex_loc)
  for hex1 in hex_loc:
    row=[]
    for hex2 in hex_loc:
      if (h3.h3_to_parent(hex2[0], 8) == h3.h3_to_parent(hex1[0], 8)):
         row.append(1)
      else:
         row.append(0)
    adj_mat.append(row)
  adj_mat=np.array(adj_mat)
  adj_mat=pd.DataFrame(adj_mat)
  # For converting the file into parquet format, below line of code is required as parquet accepts column names of sting type only.
  adj_mat.columns = adj_mat.columns.astype(str)
  return adj_mat
 
def data_preprocess(data,initial_date,final_date,latitude_boundary,longitude_boundary):
   data['latitude']=data['location.coordinates'].apply(lambda x:float(x.split(",")[1][:-1].strip()))
   data['longitude']=data['location.coordinates'].apply(lambda x:float(x.split(",")[0][1:].strip()))
   data.drop(["location.coordinates", "location.type","id"], axis=1, inplace=True)
   data["observationDateTime"]=data['observationDateTime'].apply(lambda x:pd.Timestamp(x).tz_convert("Asia/Kolkata"))
   data["observationDateTime"]=data['observationDateTime'].apply(lambda x:datetime.strptime(str(x),'%Y-%m-%d %H:%M:%S%z'))
   data["observationDateTime"]=data['observationDateTime'].apply(lambda x:x.tz_localize(None))
   data = data[data["latitude"]<=latitude_boundary]
   data = data[data["longitude"]<=longitude_boundary]
   data.reset_index(inplace=True)
   data_update=data[(data['observationDateTime']>=initial_date) &   (data["observationDateTime"]<= final_date)]
   return pd.DataFrame(data_update)
   
# Calculating and Adding a column for h3 id in the data
def add_column_h3id(data_update,resolution=9):
   h3_column=[]
   for i in range(0,len(data_update)):
      h3_column.append(h3.geo_to_h3(data_update['latitude'].iloc[i],data_update['longitude'].iloc[i], resolution=resolution)) 
   data_update['h3_ids_9']=h3_column
   return pd.DataFrame(data_update)

# Only the hex ids with the suspected dumping locations (stored in hexes) are considered and extracted from the original data   
def data_filtering(data_update,hex_loc):
   filtered_data=pd.DataFrame()
   for i in np.array(hex_loc):
      temp_data=data_update[(data_update['h3_ids_9']==i[0])]
      filtered_data=filtered_data.append(temp_data,ignore_index=True)
   return filtered_data

def datespan(startDate, endDate, delta=timedelta(days=1)):
    currentDate = startDate
    while currentDate < endDate:
        yield currentDate
        currentDate += delta

# Splitting the data on hourly basis
def data_split_hourlybasis(filtered_data,hex_loc,initial_date,final_date):
   hourly_data=[]
   for i in np.array(hex_loc):
      temp=[]
      for timestamp in datespan(datetime(initial_date[0],initial_date[1],initial_date[2],initial_date[3],initial_date[4]), datetime(final_date[0], final_date[1], final_date[2],final_date[3],final_date[4] ),delta=timedelta(hours=1)):
        single_loc_hourly=filtered_data[(filtered_data['h3_ids_9']==i[0]) &(filtered_data["observationDateTime"]>=  str(timestamp)) &     (filtered_data["observationDateTime"]<= str(timestamp+timedelta(minutes=59)))]
        no_of_vehicles=len(np.unique(single_loc_hourly['license_plate']))
        temp.append([str(timestamp),no_of_vehicles]) 
      hourly_data.append([i[0],temp])
   hourly_data=pd.DataFrame(hourly_data)
   # The hourly_data variable has to be reframed to be used as a dataset
   for j in range(0,len(temp)):
    temp2=[]
    for i in range(0,len(hex_loc)):
      temp2.append(hourly_data.iloc[i][1][j][1])
    hourly_data[str(hourly_data.iloc[i][1][j][0])]=temp2
   hourly_data.drop(1,inplace=True,axis=1)
   hourly_data=hourly_data.set_index(0)
   hourly_data.index.names=[None] 
   return hourly_data

def train_test_split(hourly_data, train_portion=0.85):
    time_len = hourly_data.shape[1]
    train_size = int(time_len * train_portion)
    train_data = pd.DataFrame(np.array(hourly_data.iloc[:, :train_size]))
    test_data = pd.DataFrame(np.array(hourly_data.iloc[:, train_size:]))
    train_data.columns = train_data.columns.astype(str)
    test_data.columns = test_data.columns.astype(str)
    return train_data, test_data

def scale_data(train_data, test_data):
    max_count = 61
    min_count = 0
    train_scaled = (train_data - min_count) / (max_count - min_count)
    test_scaled = (test_data - min_count) / (max_count - min_count)
    return train_scaled, test_scaled

def unscale_data(data_scaled):
  max_count=61
  min_count=0
  data_unscaled=(data_scaled*(max_count-min_count))+min_count
  return data_unscaled

def sequence_data_preparation1(seq_len, pre_len, train_data, test_data):
    trainX, trainY, testX, testY = [], [], [], []
    train_data=np.array(train_data)
    test_data=np.array(test_data)
    for i in range(train_data.shape[1] - int(seq_len + pre_len - 1)):
        a = train_data[:, i : i + seq_len + pre_len]
        trainX.append(a[:, :seq_len])
        trainY.append(a[:, -1])

    for i in range(test_data.shape[1] - int(seq_len + pre_len - 1)):
        b = test_data[:, i : i + seq_len + pre_len]
        testX.append(b[:, :seq_len])
        testY.append(b[:, -1])

    trainX = pd.DataFrame(np.array(trainX).reshape(1497,13*12))
    trainY = pd.DataFrame(np.array(trainY))
    testX = pd.DataFrame(np.array(testX).reshape(255,13*12))
    testY = pd.DataFrame(np.array(testY))
    trainX.columns = trainX.columns.astype(str)
    trainY.columns = trainY.columns.astype(str)
    testX.columns = testX.columns.astype(str)
    testY.columns = testY.columns.astype(str)
    return trainX,trainY,testX,testY
    
def sequence_data_preparation2(seq_len, pre_len, train_data, test_data):
    trainX, trainY, testX, testY = [], [], [], []
    trainX_nf, testX_nf = [], []

    for i in range(train_data.shape[1] - int(seq_len + pre_len - 1)):
        a = train_data[:, i : i + seq_len + pre_len]
        
        trainX_nf.append([[np.average(x[:seq_len])] for x in a])
        trainX.append(np.reshape(a[:, :seq_len],(seq_len,a.shape[0])))
        trainY.append(a[:, -1])

    for i in range(test_data.shape[1] - int(seq_len + pre_len - 1)):
        b = test_data[:, i : i + seq_len + pre_len]
        testX_nf.append([[np.average(x[:seq_len])] for x in b])
        testX.append(np.reshape(b[:, :seq_len],(seq_len,b.shape[0])))
        testY.append(b[:, -1])

    trainX = np.array(trainX)
    trainX_nf = np.array(trainX_nf)
    trainY = np.array(trainY)
    testX = np.array(testX)
    testX_nf = np.array(testX_nf)
    testY = np.array(testY)

    return trainX, trainX_nf, trainY, testX, testX_nf, testY

def clean_prediction(prediction):
  for i in range(prediction.shape[0]):
    for j in range(prediction.shape[1]):
      if prediction[i,j]<0:
        prediction[i,j]=-1
  return prediction

# Model training and evaluation  
def model_initialize(adj_mat,hex_loc,learning_rate,trainX,trainY,testX,testY):
  hex_loc=np.array(hex_loc)
  adj_mat= np.array(adj_mat)
  adj_lap=gcn_filter(adj_mat, symmetric=True)
  N=len(hex_loc)
  opt = Adam(learning_rate)
  trainX=np.array(trainX).reshape(1497,13,12)
  trainY=np.array(trainY).reshape(1497,13)
  testX=np.array(testX).reshape(255,13,12)
  testY=np.array(testY).reshape(255,13)
  inp_feat = Input((N, trainX.shape[-1]))

  x = GCNConv(32, activation='relu')([inp_feat, adj_lap])
  x = GCNConv(16, activation='relu')([x, adj_lap])
  x = Reshape((16,N))(x)
  x = LSTM(128, activation='relu', return_sequences=True)(x)
  x = LSTM(32, activation='relu')(x)

  # x = Concatenate()([x,xx])
  x = Dense(128, activation='relu')(x)
  x = Dropout(0.3)(x)
  out = Dense(N)(x)

  model = Model(inp_feat, out)
  model.compile(optimizer=opt, loss='mse', 
              metrics=[tensorflow.keras.metrics.RootMeanSquaredError()])
  print(model.summary())
  callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                            patience=100,
                                            restore_best_weights=True)
  history=model.fit(
    trainX,
    trainY,
    epochs=100,
    batch_size=32,
    shuffle=True,
    verbose=1,
    validation_data=(testX, testY),
    callbacks=[callback]
  )
  y_true=unscale_data(testY)


  y_pred=clean_prediction(unscale_data(model.predict(testX)))

  print(f'MSE:{mean_squared_error(y_true, y_pred)}')
  print(f'RMSE:{math.sqrt(mean_squared_error(y_true, y_pred))}')
  print(f'R-squared:{r2_score(y_true, y_pred)}')
  print(f'MAE:{mean_absolute_error(y_true, y_pred)}')
  
  y_pred=pd.DataFrame(y_pred)
  y_pred.columns = y_pred.columns.astype(str)
  return y_pred

def compare_passenger_capacity(data_plot: pd.DataFrame):
    return data_plot.groupby(["h3_ids_9"]).mean().reset_index()
    
    
