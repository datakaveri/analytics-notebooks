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


def vehicle_info(data: pd.DataFrame):
    """
    Get the count of each vehicle type

    Args:
        data (pandas.core.frame.DataFrame): Real-time data from the SWM servers.

    Returns:
        vehicle_capacities (list): List of all the vehicle capacities
    """

    # Get each vehicle data from license_plate
    vehicles_data = data.groupby('license_plate')

    # Initialize each type of available vehicle
    Dumper = 0
    Refused_Compactor = 0
    Auto_Tipper = 0
    Dumper_Placer = 0
    Hydraulic_Lifter = 0
    Tractor = 0
    QRT = 0
    Hopper = 0
    Pashu_Bandi = 0
    JCB = 0
    Sweeping_Machine = 0

    # For each license_plate plate in the data, get what type of vehicles it is
    for item in data['license_plate'].unique():
        vehicle_data = vehicles_data.get_group(item).reset_index()
        type = vehicle_data['vehicleType'][0]

        if type == 'Dumper':                # If type of vehicle is Dumper
            Dumper += 1
        elif type == 'Refused Compactor':   # If type of vehicle is Refused Compactor
            Refused_Compactor += 1
        elif type == 'Auto Tipper':         # If type of vehicle is Auto Tipper
            Auto_Tipper += 1
        elif type == 'Dumper Placer':       # If type of vehicle is Dumper Placer
            Dumper_Placer += 1
        elif type == 'Hydraulic Lifter':    # If type of vehicle is Hydraulic Lifter
            Hydraulic_Lifter += 1
        elif type == 'Tractor':             # If type of vehicle is Tractor
            Tractor += 1
        elif type == 'QRT':                 # If type of vehicle is QRT
            QRT += 1
        elif type == 'Hopper':              # If type of vehicle is Hopper
            Hopper += 1
        elif type == 'Pashu Bandi':         # If type of vehicle is Pashu Bandi
            Pashu_Bandi += 1
        elif type == 'JCB':                 # If type of vehicle is JCB
            JCB += 1
        else:
            Sweeping_Machine += 1           # If type of vehicle is Sweeping Machine

    # Combine extracted count of each type of vehicles

    # Get all the type of available vehicles
    vehicle_type = data['vehicleType'].unique()
    # Combine all type of vehicles into a list
    type_count = [Dumper, Refused_Compactor, Auto_Tipper, Dumper_Placer, Hydraulic_Lifter, Tractor, QRT, Hopper, Pashu_Bandi, JCB, Sweeping_Machine]
    # Initialize all vehicle capacities
    vehicle_capacity = [8, 14, 1.2, 4.5, 14, 3.4, 0, 1.8, 0.2, 0, 0]

    # Remove unnecessary non garbage collecting vehicles
    vehicle_type = np.delete(vehicle_type, [6,9,10])
    # Combine all type of vehicles into a list
    type_count = [Dumper, Refused_Compactor, Auto_Tipper, Dumper_Placer, Hydraulic_Lifter, Tractor, Hopper, Pashu_Bandi]
    # Double the count of each vehicle to account for multiple routes
    type_count = [i * 2 for i in type_count]
    # Initialize all vehicle capacities
    vehicle_capacity = [8, 14, 1.2, 4.5, 14, 3.4, 1.8, 0.2]


    # Assign capacities for all the vehicles
    capacities = []
    for i in range(len(type_count)):
        capacities = capacities+[vehicle_capacity[i]]*type_count[i]

    # Get Vehicle Capacities
    capacities = random.sample(capacities, len(capacities))
    vehicle_capacities = list(map(lambda x: int(x*100), capacities))

    # Return vehicle_capacities
    return vehicle_capacities

def create_data_model(dump: pd.DataFrame, starting: pd.DataFrame, location_info: pd.DataFrame, pairwise_distance: pd.DataFrame, swm_data: pd.DataFrame):
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
    #pairwise_distance = pairwise_distance.astype(int)

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
    data['demands'] = (location_info['occupancy']*100).astype(int).tolist()

    # Assign Vehicle Capacities to 'vehicle_capacities' key in data Dictionary
    data['vehicle_capacities'] = vehicle_info(swm_data)

    # Assign Number of Vehicles to 'num_vehicles' key in data Dictionary
    data['num_vehicles'] = vehicle_count

    # Assign starting locations of vehicles to 'starts' key in data Dictionary
    data['starts'] = vehicle_allocation

    # Assign ending locations of vehicles to 'ends' key in data Dictionary
    data['ends'] = vehicle_allocation

    # Return data Dictionary
    return data



def export_print_solution(solution, data, routing, manager, location_info):
    """
    Print the solution of CVRP problem.

    Args:
        solution: Output of the optimization solved using OR-Tools

    Output:
        Solution_Dictionary (Dict): A dictionary containing information about optimized routes.
        Prints the Solution in a readable way.

    """

    # Create an empty Dictionary for all vehicles
    Solution_Dictionary = {}
    print(f'Objective: {solution.ObjectiveValue()}')

    # Variable for total distance
    total_distance = 0
    # Variable for total load
    total_load = 0

    # For all the given vehicles
    for vehicle_id in range(data['num_vehicles']):
        # Get index of staring node of current vehicle
        index = routing.Start(vehicle_id)
        # Variable for update and printing the output
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        # Variable for route distance
        route_distance = 0
        # Variable for route load
        route_load = 0
        # Create a name for current vehicle: Format: Vehicle Number
        Name = 'Vehicle'+' '+str(vehicle_id)
        # Create an empty ditionary for current vehicle
        Solution_Dictionary[Name] = {}
        # Create an empty list for current vehicle load
        load = []
        # Create an empty list for current vehicle distance covered
        distance = []
        # Create an empty list for all the nodes visisted current vehicle path
        node_number = []
        # Create an empty list for type of nodes visited current vehicle
        node_type = []

        # While the current index is not the end index
        while not routing.IsEnd(index):
            # Convert current index to node index
            node_index = manager.IndexToNode(index)
            # Get load of the current node index
            route_load += data['demands'][node_index]
            # Assign current index to previous index
            previous_index = index
            # Get next node index of the vehicle
            index = solution.Value(routing.NextVar(index))
            # Get distance between curent and previous nodes
            route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id)
            # Format the output: NodeIndex(NodeType) Load(Value) Distance(Value) ->
            plan_output += ' {0}({2}) Load({1}) Distance({3}) -> '.format(node_index, route_load, location_info['type'][node_index],routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id))
            # Update load of the current node visited by the current vehicle or route
            load += [data['demands'][node_index]]
            # Update distance from previous node to current node visited by current vehicle or route
            distance += [routing.GetArcCostForVehicle(previous_index, index, vehicle_id)]
            # Update nodes visited by the current vehicle or route
            node_number += [node_index]
            # Update type of nodes visited by the current vehicle or route
            node_type += [location_info['type'][node_index]]
        # Format the output last visited node: NodeIndex(NodeType) Load(Value) Distance(Value) ->
        plan_output += ' {0}({2}) Load({1}) Distance({3})\n'.format(manager.IndexToNode(index),
                            route_load, location_info['type'][manager.IndexToNode(index)], routing.GetArcCostForVehicle(
                            previous_index, index, vehicle_id))
        # Update load of the last node visited by the current vehicle or route
        load += [data['demands'][manager.IndexToNode(index)]]
        # Update distance from previous node to last node visited by current vehicle or route
        #distance += [routing.GetArcCostForVehicle(previous_index, index, vehicle_id)]
        # Update last node visited by the current vehicle or route
        node_number += [manager.IndexToNode(index)]
        # Update type of last node visited by the current vehicle or route
        node_type += [location_info['type'][manager.IndexToNode(index)]]

        # Assign loads to current vehicle dictionary
        Solution_Dictionary[Name]['Load'] = load
        # Assign distance to current vehicle dictionary
        Solution_Dictionary[Name]['Distance'] = distance
        # Assign all nodes to current vehicle dictionary
        Solution_Dictionary[Name]['Node'] = node_number
        # Assign node types visited by to current vehicle dictionary
        Solution_Dictionary[Name]['Type'] = node_type

        # Format the output: Add total distance travelled by the current vehicle
        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        # Format the output: Add total load collected by the current vehicle
        plan_output += 'Load of the route: {}\n'.format(route_load)

        # Print Output of the current vehicle
        print(plan_output)
        #print(Solution_Dictionary[Name])

        # Total distance travelled by all the vehicles
        total_distance += route_distance
        # Total Load collected by all the vehicles
        total_load += route_load

    # Print total distance travelled and total load collected by all the vehicles
    print('Total distance of all routes: {}m'.format(total_distance))
    print('Total load of all routes: {}'.format(total_load))

    return Solution_Dictionary




def run_route_planner(dump: pd.DataFrame, starting: pd.DataFrame, location_info: pd.DataFrame, pairwise_distance, data: pd.DataFrame, parameters: Dict):
    """

    """


    # Double the number of vehicles available
    starting['vehicles'] = starting['vehicles']*2
    # Reduce the number of vehicles available at final starting location
    starting.iloc[-1,-1] = 132


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
    data = create_data_model(dump, starting, location_info, pairwise_distance, data)

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

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        print('>>>>>> Solution Found <<<<<<<')
        Solution_Dictionary = export_print_solution(solution, data, routing, manager, location_info)
    else:
        print('!!!! XXX No Solution Found XXX !!!!!')

    return Solution_Dictionary
