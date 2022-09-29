## Data Synchronization

This notebook provides consistency between source and target data stores, and the continuous harmonization of the data over time. This repository perform synchronization on PM2.5 data but can be easily modified for any other pollutant.


![title](Data_Time_Synchronization.png)


- Divide time series data into 15mins time windows.
- Aggregate data for every 15mins.
- If there are multiple observations in the given time period, mean will be taken.
- If there are no observations in the given time period, NaN will be taken.

### Dataset
Original Air Quality Dataset from Indian city of Pune.

### Dependencies
- Numpy
- Pandas
- Statsmodels
- scikit-learn
- Seaborn
- Matplotlib
- Datetime

### Files
`Date_Time_Synchronization.ipynb` file: Jupyter notebook peforms date time synchronization on unstructured dataset.\
`Date_Time_Synchronization.png` file: Image file illustrating data synchronization.\
`Data_Preprocessing.yml` file: Configuration file for outlier detection.
