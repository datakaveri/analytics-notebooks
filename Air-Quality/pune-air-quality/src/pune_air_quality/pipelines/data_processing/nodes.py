"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.4
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, Tuple

from prophet import Prophet
import altair as alt
#alt.renderers.enable('notebook')

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, Ridge

import logging
logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)

import warnings
warnings.filterwarnings('ignore')

#class data_preprocessing:
        
def fit_predict_model(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Performs model fitting on the given DataFrame.

    Args:
        dataframe (pandas.core.frame.DataFrame): DataFrame with two columns: ds and y.

        ds (datestamp): column should be of a format expected by Pandas, ideally YYYY-MM-DD for a date or YYYY-MM-DD HH:MM:SS for a timestamp.
        y: Column must be numeric, and represents the measurement we wish to forecast.

        interval width: Data Width to be included, typically ranges from 0 to 1. Example: intervel width 9.9 means 99% of original data will be included.

    Returns:
        forecast (pandas.core.frame.DataFrame): The forecast object contains predicted values (yhat) with the forecast, as well as columns for components and uncertainty intervals with actual data.

    """
    
    interval_width = 1
    changepoint_range = 1
    # Model instatiation with Prophet object enforcing daily, weekly, and yearly seasonalities.
    m = Prophet(daily_seasonality = True, yearly_seasonality = True, weekly_seasonality = True,
                seasonality_mode = 'additive',
                interval_width = interval_width,
                changepoint_range = changepoint_range)

    # Call fit method of Prophet and pass in the historical dataframe.
    m = m.fit(dataframe)

    # Predicting values using predict method.
    # The forecast object here is a new dataframe that includes a predicted values column (yhat) with the forecast,
    # as well as columns for components and uncertainty intervals.
    forecast = m.predict(dataframe)

    # Add observed data in a new column (fact) to forecast object.
    forecast['fact'] = dataframe['y'].reset_index(drop = True)

    # Return forecast DataFrame object
    return forecast
    
def detect_anomalies(forecast: pd.DataFrame) -> pd.DataFrame:
    """
    Detects anamolies in the in the observed data using forecasted data.

    Args:
        forecast (pandas.core.frame.DataFrame): The forecast object contains predicted values (yhat) with the forecast, as well as columns for components and uncertainty intervals with actual data.
                                            
    Returns:
        forecasted (pandas.core.frame.DataFrame): A modified forecast DataFrame with additional an 'anamoly' column.

    """

    forecasted = forecast[['ds','trend', 'yhat', 'yhat_lower', 'yhat_upper', 'fact']].copy()

    forecasted['anomaly'] = 0
    forecasted.loc[forecasted['fact'] > forecasted['yhat_upper'], 'anomaly'] = 1
    forecasted.loc[forecasted['fact'] < forecasted['yhat_lower'], 'anomaly'] = -1

    # Anomaly importances
    forecasted['importance'] = 0
    forecasted.loc[forecasted['anomaly'] ==1, 'importance'] = \
        (forecasted['fact'] - forecasted['yhat_upper'])/forecast['fact']
    forecasted.loc[forecasted['anomaly'] ==-1, 'importance'] = \
        (forecasted['yhat_lower'] - forecasted['fact'])/forecast['fact']

    # Return forecasted DataFrame
    return forecasted


def datetime_preprocessing(raw_data: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    """
    Removes Not a Time (NaT) entires and formats Date-Time column to datetime object column.

    Args:
        raw_data (pandas.core.frame.DataFrame): raw sensor data dowloaded from the data pipeline with sampled data with respect to various pollutants and atmospheric conditions.

    Returns:
        raw_data (pandas.core.frame.DataFrame): Modified file with formated Date-Time column.

    """
    
    # Remove missing Date-Time etries (NaT: Not a Time) from the observationDateTime column.
    preprocessed_raw_data = raw_data.dropna(subset=['observationDateTime'])
    
    # Convert observationDateTime column to datetime object.
    preprocessed_raw_data.observationDateTime = pd.to_datetime(preprocessed_raw_data.observationDateTime, format='%Y-%m-%d %H:%M:%S%Z')
    
    preprocessed_raw_data = preprocessed_raw_data[parameters["column_order"]]
    # Return modified data.
    return preprocessed_raw_data
    

def run_outlier_detection(preprocessed_raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Detects outliers in the date-time formated raw sensor DataFrame using fbpropht.
    
    Args:
        raw_data (pandas.core.frame.DataFrame): raw sensor data dowloaded from the data pipeline with sampled data with respect to various pollutants and atmospheric conditions.
    
    Returns:
        processed_data (pandas.core.frame.DataFrame): DataFrame after removing outliers from raw sensor data.
    
    """
    
    # Disables condition on maximum allowed rows to be processed through fbprophet.
    alt.data_transformers.disable_max_rows()
    
    # Create an empty DataFrame to store processed data.
    processed_data = pd.DataFrame(columns = preprocessed_raw_data.columns)
    
    # Get all sensor ids
    ids = preprocessed_raw_data.id.unique()
    
    # For loop over all the ids
    for idi in ids:
        
        # Extract single sensor data using its id.
        single_sensor_data = preprocessed_raw_data[preprocessed_raw_data['id'] == idi].reset_index().drop(['index'],axis=1)
        
        # Create an empty DataFrame to pass in to fbprophet model
        column_data = pd.DataFrame()
        
        # Assign Date Time data to 'ds' column in column_data
        column_data['ds'] = single_sensor_data['observationDateTime'].values
        
        # For each column related to pollutant and atmospheric conditions in the DataFrame
        
        for column in preprocessed_raw_data.columns[5:]:
            
            # Assign column data to column 'y' in column_data
            column_data['y'] = single_sensor_data[column].values
            
            # If number of numeric entries in column 'y' is more than 20% of total entries
            if (len(column_data) - column_data['y'].isna().sum()) > 0.2*len(column_data):
                
                # Fit data prediction model to column_data
                pred = fit_predict_model(column_data)
                
                # Detect anamolies
                pred = detect_anomalies(pred)
                
                # Replace outlier entries in column_data with 'NaN'
                single_sensor_data.iloc[pred[pred['anomaly'] != 0].index][column] = np.nan
                
        # Concatenate current modified sensor DataFrame to processed DataFrame
        processed_data = pd.concat([processed_data, single_sensor_data])
        
    # Return processed DataFrame with corrected outliers
    return processed_data


def run_datetime_synchronization(sensor_ids: pd.DataFrame, processed_data: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    """
    Generates time synchronized observation data by performing upsample or down sample operations, given start time and end time of the data.
    
    Args:
        processed_data (pandas.core.frame.DataFrame): processed raw sensor data after removing outliers - Output of run_outlier_detection() function.
        start_time (string): Date and time string with format YYYY-MM-DD hh:mm:ss+hh:mm or %Y-%m-%d %H:%M:%S%Z
        parameters (Dict): initializations for date time synchronization defined in parameters/data_processing.yml.
    
    Returns:
        time_synchronized_data (pandas.core.frame.DataFrame): time synchronized data after preprocessing, and time synchronization
    
    """
    #'2020-10-01 00:00:00+05:00'
    #'2023-01-10 23:59:59+05:00'
    
    start_time = parameters["start_time"]
    end_time = parameters["end_time"]
    
    # Get date time index from start time to end time with 15 minute interval
    datetime_index = pd.DatetimeIndex(pd.date_range(start=start_time, end=end_time, freq='15T'))
    
    # Create an empty DataFrame to store time synchronized data.
    time_synchronized_data = pd.DataFrame(columns = processed_data.columns)
    
    # Get all sensor ids
    ids = sensor_ids.id.unique()
    
    # For loop over all the ids
    for idi in ids:
        
        # Create an empty DataFrame to store time synchronized data.
        time_sync_data = pd.DataFrame(columns = processed_data.columns)
        
        # Extract single sensor data using its id.
        single_sensor_data = processed_data[processed_data['id'] == idi].reset_index().drop(['index'],axis=1)
        
        # Make 'observationDateTime' column as index column
        single_sensor_data = single_sensor_data.set_index('observationDateTime')
        
        # Assign datetime_index to DataFrame
        time_sync_data['observationDateTime'] = datetime_index
        
        # Assign sensor id to column 'id'
        time_sync_data['id'] = [idi]*len(datetime_index)
        
        # Assign device status as 'ACTIVE' for all entries
        time_sync_data['deviceStatus'] = ['ACTIVE']*len(datetime_index)
        
        # Assign airQualityLevel as 'NaN' for all entries
        time_sync_data['airQualityLevel'] = [np.nan]*len(datetime_index)
        
        # Assign aqiMajorPolltant as 'NaN' for all entries
        time_sync_data['aqiMajorPollutant'] = [np.nan]*len(datetime_index)
        
        # Perform up or down sampling using resammple() method in pandas. For down sampling uses mean() of the values
        # and save synchronized data to a DataFrame
        # resample('15T') - Resample given data, If there are more than one sample during the period, use mean(), reindex the data frame with datatime object,
        # reset the index, and then drop the index column
        time_sync_data[processed_data.columns[5:]] = single_sensor_data[processed_data.columns[5:]].resample('15T').mean().reindex(datetime_index).reset_index(level=0).drop('index',axis=1)
        
        # Concatenate current modified sensor DataFrame to processed DataFrame
        time_synchronized_data = pd.concat([time_synchronized_data, time_sync_data])
    
    # Return time synchronized data
    return time_synchronized_data
    
def get_imputated_pollutant_data(sensor_ids: pd.DataFrame, time_synchronized_data, parameters: Dict):
    """
    Perform imputation on time synchronized data. Extract and update each pollutant data with historic data.
    
    Args:
        time_synchronized_data (pandas.core.frame.DataFrame): time synchronized data after preprocessing, and time synchronization - Output of run_time_synchronization() function.
        parameters (Dict): Contains imputer options for data imputation defined in parameters/data_processing.yml.
    
    """
    # Extract number of observations
    num_of_observations = int(len(time_synchronized_data)/len(time_synchronized_data.id.unique()))
    
    # Extract datetime index
    datetime_index = time_synchronized_data['observationDateTime'][:num_of_observations]
    
    # For loop over all columns related to the pollutants and atmospheric conditions
    for column in time_synchronized_data.columns[5:]:
        
        # Define (instantiate) IterativeImputer model object with BayesianRidge Estimator
        imp_br_model = IterativeImputer(random_state=0, estimator=BayesianRidge(), max_iter=parameters["maximum_iterations"], tol=parameters["tolerance"])
        
        # Perform model fitting and prediction
        imputated_data = imp_br_model.fit_transform(time_synchronized_data[column].values.reshape((num_of_observations, len(time_synchronized_data.id.unique()))))
        
        # Create DataFrame for imputated pollutant data
        pollutant_data = pd.DataFrame(data = imputated_data,
                                      columns = list(['sensor_'+str(i) for i in range(len(sensor_ids.id.unique()))]))
        
        # Add 'observationDateTime' column to imputated pollutant
        pollutant_data.insert(0, 'observationDateTime', datetime_index, True)
        
        # Update historical data with current data
        # If the current column is not airQualityIndex
        #if column != 'airQualityIndex':
            
        # Split column name separated by '.'
        col = column.split('.')
        
        # Create name to load historical data
        historical_data = pd.read_parquet('data/05_model_input/'+col[0]+'_imputated_data.parquet')
        
        # Update pollutant data
        pollutant_data = pd.concat([historical_data, pollutant_data])
        
        # Delete Duplicate rows if any, reset the index, and drop extra index column
        pollutant_data = pollutant_data.drop_duplicates(subset='observationDateTime', keep="first").reset_index().drop(['index'],axis=1)
        
        # Save updated pollutant data
        pollutant_data.to_parquet('data/05_model_input/'+col[0]+'_imputated_data.parquet')

