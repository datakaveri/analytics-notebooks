"""
This is a boilerplate pipeline 'route_planning'
generated using Kedro 0.18.4
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
import folium
import scipy
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from ast import literal_eval

import networkx as nx
import osmnx as ox
import json
import random
import math
import csv
import sys

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


def data_collection(dump: pd.DataFrame, bin: pd.DataFrame, starting: pd.DataFrame) -> pd.DataFrame:
    """
    Collects location data for optimization.

    Args:
        dump (pandas.core.frame.DataFrame): Dumping locations data
        bin (pandas.core.frame.DataFrame): Garbage bins location data
        starting (pandas.core.frame.DataFrame): Parking locations data

    Returns:
        location_info (pandas.core.frame.DataFrame): A aggregated DataFrame contains all the location data.

    Notes:
        Added a Set of all Dumping Locations for Every Vehicle to facilitate each vehicle to visit any of the available Dumping locations (Each node is allowed only one visit from all the available vehicles, hence created duplicate nodes). All availble dumping locations are duplicated by number_of_vehicles times Number_of_dumping_locations (Number_of_dumping_locations * Number of Vehicles) and each vehicle is assigned a set of all available Duplicated Dumping locations.
    """

    # Get Number of Dumping Locations
    dump_count = len(dump)

    # Get Number of Bin Locations
    bin_count = len(bin)

    # Get Number of Starting Locations
    start_count = len(starting)

    # Get Number of Available Locations
    vehicle_count = starting['vehicles'].sum()#int(np.ceil(vehicle_allocation/start_scale).sum())

    # Get Bin Occupancies
    bin_occupancy = np.full((bin_count), 1)

    # Get Vehicle Capacities
    vehicle_capacity = np.full((vehicle_count), np.ceil(bin_occupancy.sum()/vehicle_count)+5)


    location_info = pd.DataFrame(columns = ['latitude', 'longitude', 'type', 'occupancy'])

    # Add Dumping Locations Data
    for i in range(vehicle_count):
        for ind, row in dump.iterrows():
            # Concatenate Latitude, Longitude, Type, and Occupancy Data of Dumping locations with location_info DataFrame
            location_info = pd.concat([location_info, pd.DataFrame({'latitude': row['latitude'],'longitude': row['longitude'],'type': 'Dump'+' '+str(ind), 'occupancy': 0},index= [ind])])

    # Add Starting Locations Data
    for ind, row in starting.iterrows():
        # Concatenate Latitude, Longitude, Type, and Occupancy Data of Starting locations with location_info DataFrame
        location_info = pd.concat([location_info, pd.DataFrame({'latitude': row['latitude'],'longitude': row['longitude'],'type': 'Parking', 'occupancy': 0},index= [ind])])


    # Add Bin Locations Data
    for ind, row in bin.iterrows():
        # Concatenate Latitude, Longitude, Type, and Occupancy Data of Starting locations with location_info DataFrame
        location_info = pd.concat([location_info, pd.DataFrame({'latitude': row['latitude'],'longitude': row['longitude'],'type': 'Bin', 'occupancy': bin_occupancy[ind]},index= [ind + dump.shape[0]])])


    # Get Number of Total Locations
    location_count = location_info.shape[0]

    # Reset DataFrame Index
    location_info = location_info.reset_index().drop('index', axis=1)

    # Return location_info DataFrame
    return location_info


def get_pairwise_distance(location_info: pd.DataFrame):
    """
    Calculates the pairwise distance between the locations.

    Args:
        location_info (pandas.core.frame.DataFrame): A aggregated DataFrame contains all the location data.

    Returns:
        pairwise_distance (numpy.ndarray): Matrix contains pairwise distances.

    """

    # Compute Pairwise Distance Between all available Locations
    pairwise_distance = scipy.spatial.distance.cdist(location_info[['latitude','longitude']].to_numpy(),location_info[['latitude','longitude']].to_numpy(),lambda u, v: geodist(u, v).m*100)

    # Convert Distance matrix to type integers
    pairwise_distance = pairwise_distance.astype(int)

    # Return pairwise distance matrix
    return pairwise_distance


def assign_vehicles(starting: pd.DataFrame, location_info: pd.DataFrame):
    """
    Assign vehicles to the starting locations.

    Args:
        location_info (pandas.core.frame.DataFrame): A aggregated DataFrame contains all the location data.
        starting (pandas.core.frame.DataFrame): Parking locations data

    Returns:
        vehicle_allocation (numpy.ndarray): assigned vehicle list to starting loations
    """
    # Get Assigned Vehicles to Each Starting Location
    assigned_vehicles = list(starting['vehicles'].values)

    # Get Index of all Starting Locations and Convert into list
    starting_index = list(location_info[location_info['type'] == 'Parking'].index)

    # Create Empty list for Vehicles starting indices
    vehicle_allocation = []
    for i in range(len(assigned_vehicles)):
        # Duplicate each starting location into number of assigned vehicles and add this list to created eampty list
        vehicle_allocation = vehicle_allocation + [starting_index[i]]*assigned_vehicles[i]

    # Return assigned vehicle list to dumping loations
    return vehicle_allocation


def create_data_model(dump: pd.DataFrame, starting: pd.DataFrame, location_info: pd.DataFrame, pairwise_distance: pd.DataFrame):
    """
    Creates and Manipulates Data for the Problem.

    Args:
        location_info (pandas.core.frame.DataFrame): A aggregated DataFrame contains all the location data.
        starting (pandas.core.frame.DataFrame): Parking locations data
        pairwise_distance (numpy.ndarray): Matrix contains pairwise distances.

    Returns:
        None
        Saves the data (Dictionary type) to model input data folder.
    """

    # Get distance matrix thorugh computation not through loading
    #pairwise_distance = get_pairwise_distance(location_info)

    # Convert Distance DataFrame to numpy array
    pairwise_distance = pairwise_distance.iloc[:,1:].to_numpy()*100
    pairwise_distance = pairwise_distance.astype(int)

    # Get vehicle assignments to starting locations
    vehicle_allocation = assign_vehicles(starting, location_info)

    # Get Number of Starting Locations
    start_count = len(starting)

    # Get Number of Available Locations
    vehicle_count = int(starting['vehicles'].sum())

    # Get Number of Dumping Locations
    dump_count = len(dump)

    # Create an Empty Dictionary to Store Data
    data = {}

    # Assign Location Data to 'locations' key in data dictionary
    data['locations'] = list(zip([*range(len(location_info))], location_info['type'].values))

    # Assign High Penality for Travelling Between Dumping and Bin Locations
    pairwise_distance[:vehicle_count*dump_count, vehicle_count*dump_count+start_count:] = 9999999

    # Assign High Penality for Travelling Between Bin and Starting Locations
    pairwise_distance[vehicle_count*dump_count+start_count:, vehicle_count*dump_count : vehicle_count*dump_count+start_count] = 9999999

    # Convert Pairwise Distance Matrix to list and Assign it to 'distance_matrix' key in data dictionary
    data['distance_matrix'] = pairwise_distance.tolist()

    # Get Occupancy data, convert to list, and Assign it to 'demands' key in data dictionary
    data['demands'] = location_info['occupancy'].values.tolist()

    # Assign Vehicle Capacities to 'vehicle_capacities' key in data Dictionary
    data['vehicle_capacities'] = [6]*vehicle_count #list(vehicle_capacity.astype(int))

    # Assign Number of Vehicles to 'num_vehicles' key in data Dictionary
    data['num_vehicles'] = vehicle_count

    # Assign starting locations of vehicles to 'starts' key in data Dictionary
    data['starts'] = vehicle_allocation

    # Assign ending locations of vehicles to 'ends' key in data Dictionary
    data['ends'] = vehicle_allocation

    # Return data Dictionary
    return data


def run_route_planner(dump: pd.DataFrame, starting: pd.DataFrame, location_info: pd.DataFrame, pairwise_distance, parameters: Dict):
    """

    """

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]


    # Get Number of Starting Locations
    start_count = len(starting)

    # Get Number of Available Locations
    vehicle_count = int(starting['vehicles'].sum())

    # Get Number of Dumping Locations
    dump_count = len(dump)

    # Get the Required Data
    data = create_data_model(dump, starting, location_info, pairwise_distance)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['starts'], data['ends'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Get list of bin location indices
    bin_list = [manager.NodeToIndex(x[0]) for x in data['locations'] if x[1] == 'Bin']

    # Get list of Dumping locatin indices
    dump_list = list(range(vehicle_count*dump_count))

    # Get list of Starting location indices
    start_list = [manager.NodeToIndex(x[0]) for x in data['locations'] if x[1] == 'Parking']

    # Get list of Ending location indices
    end_list = [routing.End(i) for i in range(manager.GetNumberOfVehicles())]

    # Add Constraint that Vehicles from staring locations can only go to Bin and End locations.
    for node, node_type in data['locations']:
        index = manager.NodeToIndex(node)
        if node_type == "Parking":
            routing.NextVar(index).SetValues(bin_list + end_list)

    for item in location_info['type'][:dump_count]:
        # For all the available Vehicles
        for i in range(vehicle_count):
            # Get the index range of the current route or vehicles
            dumpi_list = list(range(dump_count*i, dump_count*(i+1)))
            # For all the Dumping Locations
            for j in range(dump_count):
                if item == "Dump"+' '+str(j):
                    # Remove the index already visited
                    dumpi_list.remove(dump_count*i+j)
                    # Delete the remaining nodes assigned to this route or vehicle using AddDisjuction method.
                    routing.AddDisjunction(dumpi_list, 99999)

    # Add Capacity constraint.
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(demand_callback_index, 0, data['vehicle_capacities'], True, 'Capacity')

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.FromSeconds(5)

    """
    search_parameters.savings_neighbors_ratio = 1
    search_parameters.savings_max_memory_usage_bytes = 6000000000
    search_parameters.savings_arc_coefficient = 1
    search_parameters.cheapest_insertion_first_solution_neighbors_ratio = 1
    search_parameters.cheapest_insertion_first_solution_min_neighbors = 1
    search_parameters.cheapest_insertion_ls_operator_neighbors_ratio = 1
    search_parameters.cheapest_insertion_ls_operator_min_neighbors = 1
    search_parameters.local_cheapest_insertion_evaluate_pickup_delivery_costs_independently = True

    search_parameters.local_search_operators.use_relocate = 'BOOL_TRUE'
    search_parameters.local_search_operators.use_relocate_pair = 'BOOL_TRUE'
    search_parameters.local_search_operators.use_light_relocate_pair = 'BOOL_TRUE'
    search_parameters.local_search_operators.use_relocate_neighbors = 'BOOL_FALSE'
    search_parameters.local_search_operators.use_relocate_subtrip = 'BOOL_TRUE'
    search_parameters.local_search_operators.use_exchange = 'BOOL_TRUE'
    search_parameters.local_search_operators.use_exchange_pair = 'BOOL_TRUE'
    search_parameters.local_search_operators.use_exchange_subtrip = 'BOOL_TRUE'
    search_parameters.local_search_operators.use_cross = 'BOOL_TRUE'
    search_parameters.local_search_operators.use_cross_exchange = 'BOOL_FALSE'
    search_parameters.local_search_operators.use_relocate_expensive_chain = 'BOOL_TRUE'
    search_parameters.local_search_operators.use_two_opt = 'BOOL_TRUE'
    search_parameters.local_search_operators.use_or_opt = 'BOOL_TRUE'
    search_parameters.local_search_operators.use_lin_kernighan = 'BOOL_TRUE'
    search_parameters.local_search_operators.use_tsp_opt = 'BOOL_FALSE'
    search_parameters.local_search_operators.use_make_active = 'BOOL_TRUE'
    search_parameters.local_search_operators.use_relocate_and_make_active = 'BOOL_FALSE'
    search_parameters.local_search_operators.use_make_inactive = 'BOOL_TRUE'
    search_parameters.local_search_operators.use_make_chain_inactive = 'BOOL_TRUE'
    search_parameters.local_search_operators.use_swap_active = 'BOOL_TRUE'
    search_parameters.local_search_operators.use_extended_swap_active = 'BOOL_FALSE'
    search_parameters.local_search_operators.use_node_pair_swap_active = 'BOOL_FALSE'
    search_parameters.local_search_operators.use_path_lns = 'BOOL_FALSE'
    search_parameters.local_search_operators.use_full_path_lns = 'BOOL_FALSE'
    search_parameters.local_search_operators.use_tsp_lns = 'BOOL_FALSE'
    search_parameters.local_search_operators.use_inactive_lns = 'BOOL_FALSE'
    search_parameters.local_search_operators.use_global_cheapest_insertion_path_lns = 'BOOL_TRUE'
    search_parameters.local_search_operators.use_local_cheapest_insertion_path_lns = 'BOOL_TRUE'
    search_parameters.local_search_operators.use_relocate_path_global_cheapest_insertion_insert_unperformed = 'BOOL_TRUE'
    search_parameters.local_search_operators.use_global_cheapest_insertion_expensive_chain_lns = 'BOOL_FALSE'
    search_parameters.local_search_operators.use_local_cheapest_insertion_expensive_chain_lns = 'BOOL_FALSE'
    search_parameters.local_search_operators.use_global_cheapest_insertion_close_nodes_lns = 'BOOL_FALSE'
    search_parameters.local_search_operators.use_local_cheapest_insertion_close_nodes_lns = 'BOOL_FALSE'

    search_parameters.multi_armed_bandit_compound_operator_memory_coefficient = 0.04
    search_parameters.multi_armed_bandit_compound_operator_exploration_coefficient = 1000000000000
    search_parameters.relocate_expensive_chain_num_arcs_to_consider = 4
    search_parameters.heuristic_expensive_chain_lns_num_arcs_to_consider = 4
    search_parameters.heuristic_close_nodes_lns_num_nodes = 5
    #search_parameters.local_search_metaheuristic = AUTOMATIC
    search_parameters.guided_local_search_lambda_coefficient = 0.1
    search_parameters.use_cp = 'BOOL_TRUE'
    search_parameters.use_cp_sat = 'BOOL_FALSE'
    search_parameters.use_generalized_cp_sat = 'BOOL_FALSE'

    search_parameters.sat_parameters.num_search_workers = 1
    search_parameters.sat_parameters.linearization_level = 2

    search_parameters.continuous_scheduling_solver = 'SCHEDULING_GLOP'
    search_parameters.mixed_integer_scheduling_solver = 'SCHEDULING_CP_SAT'
    search_parameters.disable_scheduling_beware_this_may_degrade_performance = False
    search_parameters.number_of_solutions_to_collect = 1
    search_parameters.solution_limit = 9223372036854775807
    search_parameters.lns_time_limit.nanos = 100000000

    search_parameters.log_cost_scaling_factor = 1"""

    #print(search_parameters)
    #"""
    # Solve the problem.
    #routing.Solve()
    #faulthandler.enable()
    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        print('>>>>>> Solution Found <<<<<<<')
    else:
        print('!!!! XXX No Solution Found XXX !!!!!')
    #return solution"""
