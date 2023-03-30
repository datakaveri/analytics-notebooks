import json
import pandas as pd
import networkx as nx
import osmnx as ox
import requests

place     = 'Varanasi, Uttar Pradesh, India'
# find shortest route based on the mode of travel
mode      = 'drive'        # 'drive', 'bike', 'walk'
# find shortest path based on distance or time
optimizer = 'time'        # 'length','time'
# create graph from OSM within the boundaries of some 
# geocodable place(s)
graph = ox.graph_from_place(place, network_type = mode)
# find the nearest node to the start location

#Function to get Road Distance between two coordinates from Graph
def get_shortest_route(start_latlng, end_latlng):
    orig_node = ox.distance.nearest_nodes(graph, float(start_latlng[1]), float(start_latlng[0]))
    # find the nearest node to the end location
    dest_node = ox.distance.nearest_nodes(graph, float(end_latlng[1]), float(end_latlng[0]))
    #  find the shortest path
    shortest_route = nx.shortest_path_length(graph,
                                      orig_node,
                                  dest_node,
                                  weight=optimizer)
    #print(shortest_route[:1])
    return shortest_route

#Function to get Road Distance between two coordinates from OSRM API
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

    return out['distance']

#Finding the number of vehicles used, garbage collected and distance travelled for Solution_Dicitonary
f = open('Solution_Dictionary.json')
og = json.load(f)

keys = list(og.keys())
print(keys)

lat = []
long = []
name = keys[9]
loc = og[name]['Node']

t = 0
for k in og.keys():
    print(k)
    loc = og[k]['Node']
    print(len(loc))
    latlong = []
    for j in loc:
        latlong.append(str(j[0])+" "+str(j[1]))
    latlong = list(set(latlong))
    print(len(latlong))
    for i in range(len(latlong)-1):
        #print(latlong[i].split(' ')[1],latlong[i].split(' ')[0],latlong[i+1].split(' ')[1],latlong[i+1].split(' ')[0])
        x = get_route(latlong[i].split(' ')[1],latlong[i].split(' ')[0],latlong[i+1].split(' ')[1],latlong[i+1].split(' ')[0])
        t = t + x
    print(t)
    
#Total Bins here: 867 / 931
#Total distance: 102,28,193
#1/1/2022 Distance - 58,68,268

latlong = list(latlong)
t = 0;
for i in range(len(latlong)-1):
    print(i) 
    x = get_route(latlong[i].split(' ')[1],latlong[i].split(' ')[0],latlong[i+1].split(' ')[1],latlong[i+1].split(' ')[0])
    print(x)
    t = t + x

#Solution Dictionary from Updated Bin Location
f = open('Solution_Dictionary_MatrixIUDXFinal_updatedBin.json')
distance = json.load(f)
d = 0
n_v = 0
n_b = 0
for i in distance.keys():
    if len(distance[i]['Distance'])!=1:
        n_v = n_v + 1
    else:
        print("nope")
    for j in distance[i]['Distance']:
        d = d + j
    for j in distance[i]['Type']:
        if j == 'Bin':
            n_b = n_b + 1

for i in distance.keys():
    if len(distance[i]['Distance'])!=1:
        for j in distance[i]['Node']:
            p = graph.nodes[j]
            print(j['x'],j['y'])

#10 Vehicles not needed
#Total distance = 9,64,263
#Total Bins visited = 816
#Max volume collected = 6*126
#Matrix IUDX Results

d = pd.read_csv('Notebooks/Location Info.csv')

bins = []
for i in range(len(d)):
    t = []
    t.append(d['latitude'][i])
    t.append(d['longitude'][i])
    bins.append(t)

N = []
D = []
O = []

N.append(n_v)
D.append(d/1000)
O.append(t/1000)

#Vehicle analysis from large dataset
df = pd.read_csv('Notebooks/data.csv')
df = df[df['date']=='01-01-2022']

vehicle = df['license_plate'].to_list()
vehicle = set(vehicle)

t = df['vehicleType'].to_list()
t = set(t)

#Capacity and Number of vehicles
capacity = {
    'JCB' : 0,
    'Pashu Bandi': 0.2,
    'QRT': 0,
    'Refused Compactor': 14,
    'Tractor': 3.4,
    'Dumper Placer': 4.5,
    'Dumper': 8,
    'Hopper': 1.8,
    'Auto Tipper': 1.2,
    'Hydraulic Lifter': 14,
    'Sweeping_Machine': 0
    }

quantity = {
    'JCB' : 1,
    'Pashu Bandi': 2,
    'QRT': 9,
    'Refused Compactor': 17,
    'Tractor': 6,
    'Dumper Placer': 6,
    'Dumper': 11,
    'Hopper': 4,
    'Auto Tipper': 84,
    'Hydraulic Lifter': 2,
    'Sweeping_Machine': 2
    }

whole = {}
v = 0

for j in vehicle:
    d = 0
    t = df[df['license_plate'] == j]
    loc = t['location.coordinates'].to_list()
    
    for i in range(len(loc)):
        loc[i] = json.loads(loc[i])
        
    for i in range(len(loc)-1):
        d = d + get_route(loc[i][0], loc[i][1], loc[i+1][0], loc[i+1][1])
        
    whole[j] = t['vehicleType'].to_list()[0]
    v = v + capacity[t['vehicleType'].to_list()[0]]
    print(j)
    print(v)    
#v = 666.7999999999995

with open("Vehicle_Distance_1-1-2022.json", "w") as outfile:
    json.dump(whole, outfile, sort_keys=True, indent=4)

i = list(whole.keys())
d = 0
for j in i:
    d = d + whole[j]
