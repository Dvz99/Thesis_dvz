# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 20:19:52 2024

@author: daanv
"""
import osmnx as ox
import networkx as nx
from itertools import product
import numpy as np
from datetime import datetime
from multiprocessing import Pool
import tqdm
import os
import sys

def init_worker(input1,input2):
    # declare scope of a new global variable
    global network
    global path_weight
    # store argument in the global variable for this process
    network = input1
    path_weight = input2

def load():
    G = ox.io.load_graphml("data_input/study_case_strongly.graphml")
    # G = ox.project_graph(G)

    return G

def test_map():
    global result
    pool = Pool(4,initializer=init_worker,initargs=(G,path_weight))
    res = tqdm.tqdm(pool.imap(shortest_path, pairs), total=len(pairs))
    # res = pool.map(shortest_path, pairs)
    result.extend(res)
    pool.close()
    pool.join()

def shortest_path(pair):
    return int(100*nx.shortest_path_length(network, source=pair[0], target=pair[1], weight=path_weight, method='dijkstra'))

if __name__ == '__main__':
    # Prevent Cannot find header.dxf (GDAL_DATA is not defined) (not sure what it does)
    os.environ['GDAL_DATA'] = os.path.join(f'{os.sep}'.join(sys.executable.split(os.sep)[:-1]), 'Library', 'share', 'gdal')
    
    G = load()

    pairs = []
    pairsize = []
    
    CWC = sys.argv[1]               # list of client and depot nodes (need to be part of the street network)
    path_weight = sys.argv[2]       # path weight to use for calculating matrix (length or travel_time)
    
    nodedata = (np.load('data_generated/'+CWC+'_nodes.npy')).tolist()
    
    for j in product(nodedata, repeat=2):
        pairs.append(j)
    
    pairsize.append((len(nodedata))**2) # calculate number of pairs per collector, add this number to list
    result = []
    start = datetime.now()
    test_map()
    distances = np.array(result)   # convert to numpy array
    size = int(np.sqrt(len(distances)))    
    matrix = distances.reshape(size,size)  # convert to numpy matrix
    # np.save('data_generated/'+CWC+'_matrix_'+path_weight, matrix)
    
    print("End Time Map:", (datetime.now() - start).total_seconds())


