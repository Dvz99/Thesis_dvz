# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 20:19:52 2024

@author: daanvz
"""
from itertools import groupby 
import osmnx as ox
import networkx as nx
import os
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# Prevent Cannot find header.dxf (GDAL_DATA is not defined)
import sys
os.environ['GDAL_DATA'] = os.path.join(f'{os.sep}'.join(sys.executable.split(os.sep)[:-1]), 'Library', 'share', 'gdal')

# Set objective (time or distance)
# objective = 'time' 
objective = 'distance'

waste_collector = ['1L', '2L', '3M', '4M', '5M', '6S', '7S', '8S', '9S', '10S'] # available waste collectors
collector = ['8S']              # waste collector to be simulated

vehicle_list = [1,1,2,2]      # available vehicles (0 = MOLENVLIET, 1 = SOUTH, 2 = NORTH)
capacity = 11000               # capacity of truck (kg)
max_shift_duration = 8          # shift duration in hours (excluding break)
break_duration = 45             # break duration in minutes

time_limit_meta = 20            # time limit for the VRP solver (seconds)
vehicle_penalty = 999999        # set vehicle penalty to minimize number of vehicles used in solution

clients = gpd.read_file('data_generated/clients_complete_v3.gpkg')              # clients
nodes = np.load('data_generated/allCWC_entryv3_nodes.npy')                      # nodes

matrix_dist = np.load('data_generated/allCWC_entryv3_matrix_length.npy')        # duration matrix
matrix_time = np.load('data_generated/allCWC_entryv3_matrix_travel_time.npy')   # distance matrix

G = ox.io.load_graphml("data_input/study_case_strongly.graphml")                # street network

remove_c = []                   # clients to be removed
keep_c = []                     # clients to be kept

# Filter relevant clients from dataframe
clients_selected = clients.loc[clients['collector'].isin(collector)]
keep_c.extend((clients_selected.index.tolist()))                                # add to keep list

c_all_indices = (clients.index.values).tolist()
remove_c = [item for item in c_all_indices if item not in keep_c]

# Shift list by 3 because of the three depots at index 0,1,2
remove_c = [x+3 for x in remove_c]

depots_total = [0,1,2]
depots_used  = list(set(vehicle_list))

remove_depot = []

for i in depots_total:
    if i not in depots_used:
        remove_depot.append(i)

# Add unused depots to remove list
for i in remove_depot:
    remove_c.insert(0, i)

nodes = np.delete(nodes,remove_c)
nodes = nodes.tolist()

# Delete unused clients and depots from the rows and columns of the matrices
matrix_dist = np.delete(matrix_dist, remove_c, 0)
matrix_dist = np.delete(matrix_dist, remove_c, 1)

matrix_time = np.delete(matrix_time, remove_c, 0)
matrix_time = np.delete(matrix_time, remove_c, 1)

DISTANCE_MATRIX = matrix_dist
TIME_MATRIX = matrix_time

### SOLVE VRP PROBLEM
n_vehicles = len(vehicle_list)
capacities = [capacity*100] * n_vehicles

max_shift_duration_seconds = int(max_shift_duration*3600*100)
service_time_list = clients_selected['service_time'].tolist()

average_service = sum(service_time_list) / len(service_time_list)
print(average_service)

service_time_list = [ x*100 for x in service_time_list ]
service_time_list = [ int(x) for x in service_time_list ]

income_list = clients_selected['fee'].tolist()
income_list = [ x*1000 for x in income_list ]
income_list = [ int(x) for x in income_list ]

weight_list = clients_selected['weight_container'].tolist()

average_weight = sum(weight_list) / len(weight_list)
print(average_weight)

weight_list = [ x*100 for x in weight_list ]
weight_list = [ int(x) for x in weight_list ]

demands_count = [1] * len(clients_selected)

capacities_count = [999999999] * n_vehicles

# Add zero demand for depots
for i in range(0,len(set(depots_used))):
    service_time_list.insert(0,0)
    weight_list.insert(0,0)
    demands_count.insert(0,0)
    income_list.insert(0,0)

arc_cost = []

from time import localtime, strftime
timecreated = strftime("%Y-%m-%d-%H_%M_%S", localtime())  

if len(collector) == 1:
    f = open(('results/'+collector[0]+'_'+str(time_limit_meta)+'_'+objective+'_'+timecreated+'.txt'), "x")
    f.write(collector[0]+'_'+str(time_limit_meta)+'_'+objective+ "\n")

else:
    collector2 = '-'.join(collector)
    f = open(('results/'+collector2+'_'+str(time_limit_meta)+'_'+objective+'_'+timecreated+'.txt'), "x")
    f.write(collector2+'_'+str(time_limit_meta)+'_'+objective+ "\n")

f.write(timecreated+ "\n")
f.write(str(average_service)+ "\n")
f.write(str(average_weight)+ "\n")

# Correct depot indices
if list(set(vehicle_list)) == [0,2]:
    vehicle_list = [1 if x==2 else x for x in vehicle_list]
else:
    min_depot = min(vehicle_list)
    vehicle_list = [x - min_depot for x in vehicle_list]
print(vehicle_list)

def create_data_model():
    """Stores the data for the problem."""
    
    data = {}
    data["distance_matrix"] = DISTANCE_MATRIX
    data["time_matrix"] = TIME_MATRIX
    data["num_vehicles"] = n_vehicles
    data["demands"]  = weight_list
    data["demands_count"]  = demands_count
    data["vehicle_capacities"] = capacities
    data["vehicle_shift"] = [max_shift_duration_seconds]*n_vehicles
    data["service_time"] = service_time_list
    data["income"] = income_list
    data['starts'] = vehicle_list
    data['ends'] = vehicle_list
    return data

def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    cost = solution.ObjectiveValue()/100
    print(f"Objective: {cost}")
    f.write(f"Objective: {cost}"+ "\n")
    max_route_distance = 0
    total_distance = 0
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle {vehicle_id}:\n"
        route_distance = 0
        while not routing.IsEnd(index):
            plan_output += f" {manager.IndexToNode(index)} -> "
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )
        if route_distance != 0:
            route_distance -= vehicle_penalty
        plan_output += f"{manager.IndexToNode(index)}\n"
        
        if objective == 'distance':
            plan_output += f"Distance of the route: {route_distance/100} m\n"
        else:
            plan_output += f"Duration of the route: {route_distance/100} seconds\n"
        arc_cost.append(route_distance)
        
        print(plan_output)
        f.write(plan_output+ "\n")
        max_route_distance = max(route_distance, max_route_distance)
        total_distance += route_distance
    
    if objective == 'distance':
        print(f"Total of the route distances: {total_distance/100}m")
        f.write(f"Total of the route distances: {total_distance/100}m"+ "\n")
    else:
        print(f"Total of the route durations: {total_distance/100} seconds or {total_distance/100/3600} hours (excluding breaks)")
        f.write(f"Total of the route durations: {total_distance/100} seconds or {total_distance/100/3600} hours (excluding breaks)"+ "\n")

def get_cumul_data(solution, routing, dimension):
    """Get cumulative data from a dimension and store it in an array."""

    cumul_data = []
    for route_nbr in range(routing.vehicles()):
        route_data = []
        index = routing.Start(route_nbr)
        dim_var = dimension.CumulVar(index)
        route_data.append(solution.Value(dim_var))
        while not routing.IsEnd(index):
            index = solution.Value(routing.NextVar(index))
            dim_var = dimension.CumulVar(index)
            route_data.append(solution.Value(dim_var))
        cumul_data.append(route_data[-1]/100)
        
    print(cumul_data)
    for i in cumul_data:
        f.write(str(i)+ "\n")
    return cumul_data

def main():
    """Entry point of the program."""
    # Instantiate the data problem.
    data = create_data_model()

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]), data["num_vehicles"], data["starts"], data["ends"], 

    )
    num_nodes = len(service_time_list)
    modelParameters = pywrapcp.DefaultRoutingModelParameters()
    modelParameters.max_callback_cache_size = 2*num_nodes*num_nodes
    
    # Create Routing Model
    routing = pywrapcp.RoutingModel(manager, modelParameters)
    
    # Create and register a transit callback
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]
    
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data["demands"][from_node]
    
    def demand_callback_count(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data["demands_count"][from_node]
    
    def income_callback_count(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data["income"][from_node]
    
    def time_callback(from_index, to_index):
        """Returns the travel time between the two nodes."""
        # Convert from routing variable Index to time matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["time_matrix"][from_node][to_node]+data["service_time"][from_node]
    
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    demand_callback_count_index = routing.RegisterUnaryTransitCallback(demand_callback_count)
    income_callback_count_index = routing.RegisterUnaryTransitCallback(income_callback_count)
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    time_callback_index = routing.RegisterTransitCallback(time_callback)
    
    # add capacity constraint
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data["vehicle_capacities"],  # vehicle maximum capacities
        True,  # start cumul to zero
        "Capacity",
    )
    
    # Add client balance dimension
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_count_index,
        0,  # null capacity slack
        [9999999999]*n_vehicles,        
        True,  # start cumul to zero
        "Capacity_count",
    )
    
    # Add revenue balance dimension
    routing.AddDimensionWithVehicleCapacity(
        income_callback_count_index,
        0,  # null capacity slack
        [999999999999]*n_vehicles,
        True,  # start cumul to zero
        "Income",
    )

    # Add time dimension
    routing.AddDimensionWithVehicleCapacity(
        time_callback_index,
        0,  # null capacity slack
        # data["vehicle_shift"],  # vehicle maximum time
        [9999999999]*n_vehicles, # when using soft bound with penalty
        True,  # start cumul to zero
        "Time",
    )
    
    # Add penalty to minimize vehicles used
    routing.SetFixedCostOfAllVehicles(vehicle_penalty)
    
    weight_dim = routing.GetDimensionOrDie("Capacity")
    time_dim = routing.GetDimensionOrDie("Time")

    count_dimension = routing.GetDimensionOrDie("Capacity_count")
    income_dimension = routing.GetDimensionOrDie("Income")
        
    # Define cost of each arc.
    if objective == 'time':
        routing.SetArcCostEvaluatorOfAllVehicles(time_callback_index)
    elif objective == 'distance':
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # CLIENT BALANCING CONSTRAINT
    # countlist = [47,55,47,54,48]

    # vehicle_penalty = 0 # set vehicle penalty to zero 
    # softlowerbound_penalty = 50000
    # for vehicle_id in range(data["num_vehicles"]):        
    #     index_end = routing.End(vehicle_id)
        
    #     count_dimension.SetCumulVarSoftLowerBound(index_end, countlist[vehicle_id], softlowerbound_penalty)
    #     count_dimension.SetCumulVarSoftUpperBound(index_end, countlist[vehicle_id], softlowerbound_penalty)

    # REVENUE BALANCING CONSTRAINT
    # countlist = [5089737,6321177,5007297,5743746,5325984]
    # countlist = [1164.43,1047.49,977.41]
    
    # countlist = [1164430,1047490,977410]
    # softlowerbound_penalty = 1
    # # vehicle_penalty = 0
    # for vehicle_id in range(data["num_vehicles"]):        
    #     index_end = routing.End(vehicle_id)
        
    #     income_dimension.SetCumulVarSoftLowerBound(index_end, countlist[vehicle_id]-50000, softlowerbound_penalty)
    #     income_dimension.SetCumulVarSoftUpperBound(index_end, countlist[vehicle_id]+50000, softlowerbound_penalty)
  
    # time soft bound (max shift time)
    softlowerbound_penalty = 50000
    for vehicle_id in range(data["num_vehicles"]):        
        index_end = routing.End(vehicle_id)
        time_dim.SetCumulVarSoftUpperBound(index_end, max_shift_duration_seconds, softlowerbound_penalty)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    )
    search_parameters.local_search_metaheuristic = (
    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = time_limit_meta
    # search_parameters.log_search = True

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        print_solution(data, manager, routing, solution)
        print('Weight per vehicle:')
        f.write('Weight per vehicle:'+ "\n")
        get_cumul_data(solution,routing,weight_dim)
        print('Time per vehicle (according to time matrix, no breaks):')
        f.write('Time per vehicle (according to time matrix, no breaks):'+ "\n")
        get_cumul_data(solution,routing,time_dim)
        print('Income per vehicle:')
        f.write('Income per vehicle:'+ "\n")
        get_cumul_data(solution,routing,income_dimension)

    else:
        print("No solution found !")

    indexlist_total=[]
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        route_distance = 0
        indexlist_truck = []
        while not routing.IsEnd(index):
            indexlist_truck.append(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )
        indexlist_total.append(indexlist_truck) 

    global pass_index_list_total
    pass_index_list_total = indexlist_total

if __name__ == "__main__":
    main()
    
### CREATE ROUTE LISTS FOR PLOTTING
# Do not display unused vehicles
client_routelist = [item for item in pass_index_list_total if len(item) != 1]
arc_cost = [item for item in arc_cost if item != 0]

# Add depot (starting node) as ending node in every route
for i in range(0, len(client_routelist)):
    start_depot = client_routelist[i][0]
    end_depot = start_depot

    client_routelist[i].insert(len(client_routelist[i]), end_depot)

# Create list with routes (route = list of client nodes and all intermediate network nodes)
node_routelist = []

tot_dist = 0
tot_time = 0
tot_break = 0

for i in range(0,len(client_routelist)):
    templist=[]
    routelengte=0
    routeduur=0
    service_times_sum = 0
    if objective == 'time':
        for j in range(0, len(client_routelist[i])-1):
            templist.extend(ox.routing.shortest_path(G, nodes[client_routelist[i][j]], nodes[client_routelist[i][j+1]], weight='travel_time'))
            templist = [k[0] for k in groupby(templist)]
        routeduur = arc_cost[i]/100
        routelengte = nx.path_weight(G,templist,'length')
    elif objective == 'distance':
        for j in range(0, len(client_routelist[i])-1):
            templist.extend(ox.routing.shortest_path(G, nodes[client_routelist[i][j]], nodes[client_routelist[i][j+1]], weight='length'))
            templist = [k[0] for k in groupby(templist)]  
            service_times_sum += service_time_list[client_routelist[i][j]]/100

        routedur = nx.path_weight(G,templist,'travel_time')+service_times_sum
        routelengte = arc_cost[i]/100
    
    node_routelist.append(templist)

    vehicle_number = i+1
    client_amount = len(client_routelist[i])-2
    routedur_service = routedur
    routedur_break = routedur_service+break_duration*60
    
    tot_dist += routelengte
    tot_time += routedur_service

    print('----------------------------------------------')
    print('Vehicle '+str(vehicle_number)+' ('+str(client_amount)+' clients)')
    print('Length of route (meters) = '+str(routelengte))
    print('Route duration (seconds) + service time = '+str(routedur_service))
    
    f.write('----------------------------------------------'+ "\n")
    f.write('Vehicle '+str(vehicle_number)+' ('+str(client_amount)+' clients)'+ "\n")
    f.write('Length of route (meters) = '+str(routelengte)+ "\n")
    f.write('Route duration (seconds) + service time = '+str(routedur_service)+ "\n")
    

    if routedur_service/3600 < 4.5:
        print('No break (route shorter than 4.5 hours)')
        hour_duration = round((routedur_service/3600),2)
        print('Route duration (hours) = '+str(hour_duration))
        f.write('No break (route shorter than 4.5 hours)'+ "\n")
        f.write('Route duration (hours) = '+str(hour_duration)+ "\n")
    else:
        print('Route duration (seconds) + break = '+str(routedur_service+break_duration*60))
        hour_duration = round((routedur_break/3600),2)
        print('Route duration (hours) = '+str(hour_duration))
        tot_break+= break_duration*60
        f.write('Route duration (seconds) + break = '+str(routedur_service+break_duration*60)+ "\n")
        f.write('Route duration (hours) = '+str(hour_duration)+ "\n")

tot_time_hours = round((tot_time/3600),2)
tot_time_break = tot_time+tot_break
tot_time_break_hours = round(((tot_time+tot_break)/3600),2)

print('----------------------------------------------')   
print('Total:')
print('Distance: '+str(tot_dist))
print('Time without break: '+str(tot_time)+'= '+str(tot_time_hours))
print('Time with break: '+str(tot_time_break)+'= '+str(tot_time_break_hours))

f.write('----------------------------------------------'+ "\n")   
f.write('Total:'+ "\n")
f.write('Distance: '+str(tot_dist)+ "\n")
f.write('Time without break: '+str(tot_time)+' = '+str(tot_time_hours)+ "\n")
f.write('Time with break: '+str(tot_time_break)+' = '+str(tot_time_break_hours)+ "\n")
f.close()

### VISUALISATION
colorchoice = [plt.cm.tab10(i) for i in range(10)]

colors = []
for i in range(0, len(node_routelist)):
    colors.append(colorchoice[i])

# Get x, y coordinates of nodes for scatter plot
nodesx, nodesy = [], []
for i in range(0,len(nodes)):
    nodesx.append(G.nodes[nodes[i]]['x'])
    nodesy.append(G.nodes[nodes[i]]['y'])

fig, ax = plt.subplots(figsize=(40,40),dpi=220) 

# Plot street network
ox.plot.plot_graph(G, ax=ax, node_color='#696969', node_size = 1, edge_color='#A9A9A9', edge_linewidth=3, show=False, close=False)

# Plot routes
if len(node_routelist) == 1:
    ox.plot.plot_graph_route(G, node_routelist[0], route_color='b', route_linewidth=5, route_alpha=1, ax=ax,show=False, close=False, zorder=0)
else:
    ox.plot.plot_graph_routes(G, node_routelist, route_colors=colors, route_linewidths=5, route_alpha=1, ax=ax,show=False, close=False, zorder=0)

# Plot visited nodes
ax.scatter(nodesx,nodesy,color = 'red',marker ="o", s=4,zorder=20)

### Plot node visit order
for i in range(0,len(client_routelist)):  # remove last item of every route (depot)
    del client_routelist[i][-1]

node_routelist_flat = np.concatenate(node_routelist).tolist()
client_routelist_flat = np.concatenate(client_routelist).tolist()

visit_dict={}
for i in range(0, len(client_routelist)):
    for j in client_routelist[i]:
        visit_dict[j] = client_routelist[i].index(j)

offset = 5
bbox = dict(boxstyle ="round", fc ="0.8", edgecolor='red') 
arrowprops = dict( 
    arrowstyle = "-", 
    connectionstyle = "angle3, angleA = 0, angleB = 90") 

for i in range(0,len(client_routelist)):
    for j in client_routelist[i]:
        ax.annotate(visit_dict[j], (nodesx[j], nodesy[j]), color='k', xytext=(-2 * offset, offset), textcoords='offset points',bbox = bbox, arrowprops = arrowprops,zorder=21, fontsize=5.0)

if len(collector) == 1:
    plt.savefig('results/'+collector[0]+'_'+str(time_limit_meta)+'_'+objective+'_'+timecreated+'.jpg',bbox_inches='tight',pad_inches=0.0)
    np.save('results/'+collector[0]+'_'+str(time_limit_meta)+'_'+objective+'_'+timecreated+'_nodelist', node_routelist_flat)
    np.save('results/'+collector[0]+'_'+str(time_limit_meta)+'_'+objective+'_'+timecreated+'_routelist', client_routelist_flat)
else:
    collector2 = '-'.join(collector)
    plt.savefig('results/'+collector2+'_'+str(time_limit_meta)+'_'+objective+'_'+timecreated+'.jpg',bbox_inches='tight',pad_inches=0.0)
    np.save('results/'+collector2+'_'+str(time_limit_meta)+'_'+objective+'_'+timecreated+'_nodelist', node_routelist_flat)
    np.save('results/'+collector2+'_'+str(time_limit_meta)+'_'+objective+'_'+timecreated+'_routelist', client_routelist_flat)

plt.close()