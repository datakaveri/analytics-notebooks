# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 13:07:47 2023

@author: A ARUN JOSEPHRAJ
"""
#Code to find Distance Matrix using the OSRM Api
import pandas as pd
import ast
M = pd.read_csv('Location_Info.csv')

latlong = []

for i in range(len(M)):
    s = str(M['latitude'][i]) + " " + str(M['longitude'][i])
    latlong.append(s)        

D = [[0 for col in range(4270)] for row in range(4270)]

import time
import requests

#Function to get Road Distance between two coordinates
def get_route(pickup_lon, pickup_lat, dropoff_lon, dropoff_lat):
    
    loc = "{},{};{},{}".format(pickup_lon, pickup_lat, dropoff_lon, dropoff_lat)
    url = "http://router.project-osrm.org/route/v1/driving/"
    r = requests.get(url + loc) 
    if r.status_code!= 200:
        return {}
    
    res = r.json()   
    start_point = [res['waypoints'][0]['location'][1], res['waypoints'][0]['location'][0]]
    end_point = [res['waypoints'][1]['location'][1], res['waypoints'][1]['location'][0]]
    distance = res['routes'][0]['distance']
    
    out = {
           'start_point':start_point,
           'end_point':end_point,
           'distance':distance
          }

    return out

l = len(latlong)**2
c = 1
d_dict = {}
for i in range(len(latlong)):
    for j in range(len(latlong)):
        print(c/l*100)
        c = c+1
        s = latlong[i] + " " + latlong[j]
        
        if(latlong[i] == latlong[j]):
            D[i][j] = 0
            print("Equal")
            
        elif(s in d_dict.keys()):
            D[i][j] = d_dict[s]
            print(D[i][j])
                        
        else:
            D[i][j] = get_route(float(latlong[i].split(' ')[1]),float(latlong[i].split(' ')[0]),float(latlong[j].split(' ')[1]),float(latlong[j].split(' ')[0]))['distance']
            d_dict[s] = D[i][j]
            print(D[i][j])


dist = pd.DataFrame(d_dict, index=[0])
dist.to_csv('Found_Dict_Final.csv')

matrix = pd.DataFrame(D)
matrix.to_csv('Matrix_IUDX.csv')
