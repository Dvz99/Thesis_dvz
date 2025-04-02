# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 14:29:15 2024

@author: daanv
"""
import osmnx as ox
import matplotlib.pyplot as plt
import geopandas as gpd

distr = [
           "Noord, Rotterdam, Netherlands",
            "Kralingen-Crooswijk, Rotterdam, Netherlands",
            "Delfshaven, Rotterdam, Netherlands",
           "Centrum, Rotterdam, Netherlands",
            "Charlois, Rotterdam, Netherlands",
            "Feijenoord, Rotterdam, Netherlands",
            "IJsselmonde, Rotterdam, Netherlands"
          ]

def create_studycase():
    ### Load and save complete study case area  
    # load rough map of the city
    C = ox.io.load_graphml(filepath='data_input/study_case_uncut.graphml')
    
    poly = gpd.read_file('data_input/study_casev2.gpkg')
    poly = poly.iloc[0]['geometry']
    
    # cut off edges
    C = ox.truncate.truncate_graph_polygon(C, poly, truncate_by_edge=False)
    
    # remove unreachable nodes (mostly on edges of map due to the cut off)
    C = ox.truncate.largest_component(C, strongly=True)
    C = ox.project_graph(C, to_crs='epsg:4326')
    
    hwy_speeds = {"residential": 30, "secondary": 50, "secondary_link": 50, "primary": 50, "primary_link": 50, "tertiary": 60, "motorway": 100, "motorway_link": 100}
    C = ox.add_edge_speeds(C, hwy_speeds=hwy_speeds)
    C = ox.add_edge_travel_times(C)
    edges = ox.graph_to_gdfs(C, nodes=False)
    edges["highway"] = edges["highway"].astype(str)

    for edge in C.edges:
        if C[edge[0]][edge[1]][edge[2]]['speed_kph'] == 30:
            C[edge[0]][edge[1]][edge[2]]['speed_kph'] = 14.5
        if C[edge[0]][edge[1]][edge[2]]['speed_kph'] == 50:
            C[edge[0]][edge[1]][edge[2]]['speed_kph'] = 23.5        

    C = ox.add_edge_travel_times(C)
    edges = ox.graph_to_gdfs(C, nodes=False)
    edges["highway"] = edges["highway"].astype(str)

    # ox.save_graphml(C, filepath='data_input/study_case_strongly.graphml')
    fig, ax = plt.subplots(figsize=(10,10),dpi=120) 
    ox.plot.plot_graph(C, ax=ax, node_color='#696969', node_size = 2, edge_color='#A9A9A9', show=False, close=False)
    plt.show()
    plt.close()

def create_clientlocations():
    osmtags = {'office':True,'shop':True, 'amenity':['restaurant','bar','cafe','fast_food','food_court','pub','school']}
    # osmtags = {'tourism':['guest_house','hostel','hotel','motel']}
    # osmtags = {'healthcare':True}
    clients = ox.features.features_from_place(distr, tags = osmtags)
    print(len(clients))
    
    # remove polygons, only keep points
    mask = clients['geometry'].geom_type != 'Point'
    clients = clients[~mask]
    print(len(clients))
    
    # remove clients outside polygon 
    poly = gpd.read_file('data_input/geheelpolygon.gpkg')
    poly = poly.iloc[0]['geometry']
    
    mask = clients['geometry'].within(poly) == False
    clients = clients[~mask]
    print(len(clients))
    # clients.to_file('data_input/clients_nopoly.gpkg')

# create_studycase()
create_clientlocations()