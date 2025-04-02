# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:36:26 2025

@author: daanv
"""

import osmnx as ox
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import numpy as np

# Load client locations

client_data = gpd.read_file("data_input/clients_nopoly.gpkg")
print('Client number from raw file:',len(client_data))        

# Load street network to assign the closest node on the street network (streetnode) to each client
street_network = ox.io.load_graphml("data_input/study_case_strongly.graphml")

x = [a for a in client_data['geometry'].x]
y = [b for b in client_data['geometry'].y]

streetnodes = ox.distance.nearest_nodes(street_network, x, y)
client_data['streetnode'] = streetnodes

districtlist = []
client_data['district'] = pd.NA

gdf_nodes, gdf_edges = ox.graph_to_gdfs(street_network)

# print(len(gdf_edges))
# print(gdf_edges.info())

### REMOVE CONTAINERS LOCATED ON THE HIGHWAY
filter = ['motorway', 'motorway_link']
gdf_edges = gdf_edges[gdf_edges['highway'].isin(filter)]
gdf_edges = gdf_edges.reset_index()
# print(len(gdf_edges))

u = gdf_edges['u'].tolist()
v = gdf_edges['v'].tolist()

highwaynodes = u + v

for i in streetnodes:
    if i in highwaynodes:
        client_data.drop(client_data[client_data['streetnode'] == i].index, inplace = True)

# reset streetnodes list
streetnodes = client_data['streetnode'].tolist()

# Add coordinate columns for 'streetnodes'
streetnodes_x = []
streetnodes_y = []

for i in range(0, len(client_data)):
    streetnodes_x.append(street_network.nodes[streetnodes[i]]['x'])
    streetnodes_y.append(street_network.nodes[streetnodes[i]]['y'])
    
client_data['streetnode_x'] = streetnodes_x
client_data['streetnode_y'] = streetnodes_y

clients = 1550 # set number of clients

container_size = [240,360,660,770,1100]
cont_percentage = [0.30,0.10,0.21,0.17,0.22]  # distribution of container size over clients

waste_collector = ['1L', '2L', '3M', '4M', '5M', '6S', '7S', '8S', '9S', '10S']
wc_percentage = [0.323,0.323,0.0645,0.0645,0.0645,0.0321,0.0321,0.0321,0.0321,0.0321] # distribution of waste collectors over clients

if clients != 0:
    remove_n = int(len(client_data)-clients)
    drop_indices = np.random.choice(client_data.index, remove_n, replace=False)
    client_data = client_data.drop(drop_indices)
    client_data = client_data.reset_index()
    print('Client number after reduction:',len(client_data)) 
    
# Add container column
client_data['container'] = pd.NA
client_data['container'] = client_data['container'].apply(lambda x: np.random.choice(container_size, p=cont_percentage))

# Verify
print() 
print('Container size and count:')
print(client_data['container'].value_counts())
count = list(client_data['container'].value_counts())
perc = []
for i in count:
    perc.append(i/sum(count)*100)
print() 
print('Container size percentage:')
print(perc)

# Add waste collector column
client_data['collector'] = pd.NA
client_data['collector'] = client_data['collector'].apply(lambda x: np.random.choice(waste_collector, p=wc_percentage))

# Verify
print() 
print('Waste collector name and count:')
print(client_data['collector'].value_counts())
count = list(client_data['collector'].value_counts())
perc = []
for i in count:
    perc.append(i/sum(count)*100)
print() 
print('Waste collector percentage:')
print(perc)

# Add service time per container
stime = [70,  70, 140, 140, 150]

def add_time(cont):
    if cont == 240:
        return stime[0]
    elif cont == 360:
        return stime[1]
    elif cont == 660:
        return stime[2]
    elif cont == 770:
        return stime[3]
    elif cont == 1100:
        return stime[4]

client_data['service_time'] = client_data['container'].apply(add_time)

# Add weight per container
weight = [20, 30, 50, 60, 80]

def add_weight(cont):
    if cont == 240:
        return weight[0]
    elif cont == 360:
        return weight[1]
    elif cont == 660:
        return weight[2]
    elif cont == 770:
        return weight[3]
    elif cont == 1100:
        return weight[4]
    
client_data['weight_container'] = client_data['container'].apply(add_weight)

fig, ax = plt.subplots(figsize=(10,10),dpi=140) 
ax.scatter(streetnodes_x,streetnodes_y,color = 'blue',marker ="o",  s=3, zorder=20)

ox.plot.plot_graph(street_network, ax=ax, node_color='#696969', node_size = 1, edge_color='#A9A9A9', edge_linewidth = 0.8, show=False, close=False)  
plt.show()
plt.close()

# Save to file
# client_data.to_file('data_generated/clients_complete_v3.gpkg')

