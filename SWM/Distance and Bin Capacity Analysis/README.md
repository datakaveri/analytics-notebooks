# Distance and Bin Capacity Analysis

This folder contains files for the Distance and Bin Capacity Analysis of the Solid Waste Management dataset.

## Files and their description:

- `distance_matrix.py`: This file uses the OSRM API to find the distance matrix of the given latitude and longitude. The obtained Distance Matrix can be plugged in and run in the SWM Route Optimization Notebook

- `joblib-trial.py`: This file uses the Joblib library to run the distance matrix calculation faster.

- `comparison.py`: This file compares the different results with the actual data. It is able to calculate the distance traveled, number of bins visited, total volume of garbage collected, and vehicles used. Input files required are Solution Dictionary generated from Route Optimization

- `SWM_Route_Optimization_OR_Tools.ipynb`: Main driver file to run the Optimization code for the given data and vehicles. Data required: Location Info.csv and Distance Matrix. 