# Detect Outliers in Time Series Data

Notebook in this directory uses open source facebook prophet package to detect outliers in the sensor data. Prophet is a forecasting procedure where non-linear trends in time series data are fits in yearly, weekly, and daily seasonality.Prophet is accurate and fast, fully automatic, and forecasts are tunable. More details are about Prophet package is found [here](https://github.com/facebook/prophet).

### Dataset
Data is derived from Air Quality Dataset from Indian city of Pune. Takes time synchronized data obtained from [Data Synchronization directory](https://github.com/datakaveri/analytics-notebooks/tree/main/Air-Quality/Data%20Synchronization) as input and performs outliers detection.

### Dependencies
- Numpy
- Pandas
- Matplotlib
- Datetime
- Statsmodels
- Scikit-learn
- Seaborn
- Prophet

### Files
`Outlier_Detection_Prophet.ipynb` file: Jupyter notebook contains outlier detection algorithm.
`pune_aqm_2020Oct_2022July_TimeSynched.csv` file: Input data obtained after performing time synchronization on original data.
`Data_Preprocessing.yml` file: Configuration file for outlier detection.
