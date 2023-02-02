"""
This is a boilerplate pipeline 'deep_learning_interpolation'
generated using Kedro 0.18.4
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.legacy import Nadam
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import initializers
from keras.callbacks import EarlyStopping
import tensorflow as tf
from keras import Input
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error as mae
from sklearn.decomposition import PCA
from typing import Dict, Tuple
import numpy as np
import pandas as pd

    
def get_location_data(sensor_data: pd.DataFrame, sensor_ids: pd.DataFrame, parameters: Dict) -> Tuple:
    """
    Gets location data for training (known) and prediction (unknown).
    
    Args:
        sensor_data (pandas.core.frame.DataFrame): DataFrame contains sensor data - sensor id, sensor location data, sensor address.
        sensor_ids (pandas.core.frame.DataFrame): DataFrame contains ordered sensor ids.
        parameters (Dict): Perameters for sensor location data defined in Parameters/deep_learning_interpolation.yml
        
    Returns:
        known_latlon (pandas.core.frame.DataFrame): DataFrame contains latitude and longitude of known sensors.
        unknown_latlon (pandas.core.frame.DataFrame): DataFrame contains latitude and longitude of unknown sensors - locations of interest.
        latlon (pandas.core.frame.DataFrame): DataFrame contains latitude and longitude of both known unknown sensors.
    """
    
    # Get latitude and longitude of known sensors in the order given in sensor_ids DataFrame
        # First set 'id' column as the index column using set_index
        # Second reindex the DataFrame in the order given in sensor_ids
        # Third extract 'latitude' and 'longitude' columns data from reordered DataFrame
    known_latlon = pd.DataFrame(data=sensor_data.set_index('id').reindex(sensor_ids.id.values)[['latitude', 'longitude']].values, columns=['latitude', 'longitude'])
    
    # Define boundaries of the interpolation field
    # Minimum latitude value
    lat_min = min(known_latlon['latitude'])
    # Maximum latitude value
    lat_max = max(known_latlon['latitude'])
    # Minimum longitude value
    lng_min = min(known_latlon['longitude'])
    # Maximum longitude value
    lng_max = max(known_latlon['longitude'])
    # Value of extension to the grid
    ext = parameters["ext"]
    # Grid Length
    grid_length = parameters["grid_length"]
    
    # Get 100 longitude values between maximum and minimum longitude values
    xx = np.linspace(lng_min-ext, lng_max+ext, grid_length)
    # Get 100 latitude values between maximum and minimum latitudes values
    yy = np.linspace(lat_min-ext, lat_max+ext, grid_length)

    # Get grid with arrays xx and yy
    unknown_set = np.meshgrid(xx, yy)
    
    # Array of all longitude values in the grid
    gridx = np.reshape(unknown_set[0], (1, grid_length*grid_length))
    # Array of all latitude values in the grid
    gridy = np.reshape(unknown_set[1], (1, grid_length*grid_length))

    # Concatenate latitude and longitude arrays
    gridyx = np.concatenate((gridy.T, gridx.T), axis=1)
    
    # Define an empty DataFrame for unknown latitude and longitude values
    unknown_latlon = pd.DataFrame(columns = ['latitude', 'longitude'])
    # Assign Concatenated latitude and longitude array to unknown latitude longitude DataFrame
    unknown_latlon[['latitude', 'longitude']] = gridyx

    # Concatenate both known and Unknown lat lon DataFrame with new index
    latlon = pd.concat([known_latlon, unknown_latlon], ignore_index = True)
    
    # Returns known, unknown, and combined location DataFrames
    return known_latlon, unknown_latlon, latlon


def pollutant_data_preprocessing(data: pd.DataFrame, past_data_range):
    """
    Prepare pollutant time series data for PCA analysis and Data training.
    
    Args:
        data (pandas.core.frame.DataFrame): Pollutant data obtained through data processing pipeline.
        past_data_range: Past data to be used in in the model training.
        
    Returns:
        spatio_temporal_observations (numpy.array): Spatio-Temporal observations of a pollutant
        data_scaler: StandardScalar object for data scaling.
    """

    # Data Selection
    time_series_data = data.iloc[data.shape[0]-(past_data_range*96):,1:].to_numpy().T
    
    # Select the data range [0,T] for manipulation
    spatio_temporal_observations_data = time_series_data
    
    # Define StandardScaler instantiation for data scaling
    data_scaler = StandardScaler()
    
    # Fit Spatio-Temporal data using fit() method of StandardScaler
    data_scaler.fit(spatio_temporal_observations_data)
    
    # Transform the Spatio-Temporal data using transform() method of StandardScaler
    spatio_temporal_observations = data_scaler.transform(spatio_temporal_observations_data)
    
    # Return preprocessed data and data scaler object
    return spatio_temporal_observations, data_scaler
    

def run_pca(spatio_temporal_observations, exp_var):
    """
    Run PCA analysis on Data training.
    
    Args:
        spatio_temporal_observations (numpy.array): Pollutant data obtained after  data preprocessing.
        exp_var: explained variance threshold level
        
    Returns:
        alpha_k (numpy.array): Spatio coefficients of the spatio-temporal observations of a pollutant.
        xmin[0]: Number of components to be used in PCA analysis.
        coeff_scaler: StandardScalar object for spatial coefficients.
        pca: PCA object
    """
    # Number of Components in the data
    K = min(spatio_temporal_observations.shape[1], spatio_temporal_observations.shape[0]-1)
    
    # PCA with K components
    pca = PCA(n_components=K)
    
    # Fit PCA to Spatio-Temporal data
    pca.fit(spatio_temporal_observations)
    
    # Compute explained variance
    explained_variance = pca.explained_variance_ratio_.cumsum()
    explained_variance[explained_variance >= exp_var]
    
    # Minimum components required for analysis
    xmin = np.where(explained_variance >= exp_var)[0]
    
    

    # Spatial Coefficients = dot product of first K PCA components and Spatiotemporal Observations
    alpha_k_ = np.dot(pca.components_[:xmin[0]], spatio_temporal_observations.T)
    
    # Apply Standard Scaler to the Coefficients
    coff_scaler = StandardScaler()
    
    # Fit spatial coefficients to standard scaler
    coff_scaler.fit(alpha_k_.T)
    
    # Transform Spatial Coefficients through standard scaler
    alpha_k = coff_scaler.transform(alpha_k_.T)
    
    # Return Spatial Coefficients, Min Components, and Coefficient Scaler
    return alpha_k, xmin[0], coff_scaler, pca
    

def deep_learning_model(data: pd.DataFrame, known_latlon: pd.DataFrame, unknown_latlon: pd.DataFrame, parameters: Dict, pollutant_name):
    """
    Deep learning based spatio-tempotal data interpolation model.
    
    Args:
        data (pandas.core.frame.DataFrame): Pollutant data obtained after  data preprocessing.
        known_latlon (pandas.core.frame.DataFrame): DataFrame contains latitude and longitude of known sensors.
        unknown_latlon (pandas.core.frame.DataFrame): DataFrame contains latitude and longitude of unknown sensors - locations of interest.
        parameters (Dict): parameters for the deep learning model.
        pollutant_name (Dict): Name of the pollutant data passed to the model.
        
    Returns:
        None
    """

    # Load column oreder from parameters
    #column_order = parameters["column_order"]
    
    # For each column in column order
    #for column in column_order[5:6]:
    # Split column name separated by '.'
    #col = column.split('.')
        
    # Load Pollutant data
    #data = pd.read_parquet('data/05_model_input/'+col[0]+'_imputated_data.parquet')
        
    # Perform data preprocessing on pollutant data.
    spatio_temporal_observations, data_scaler = pollutant_data_preprocessing(data, parameters["past_data_range"])
    
    # Run Principle Component Analysis on pollutant data.
    alpha_k, components, coff_scaler, pca = run_pca(spatio_temporal_observations, parameters["exp_var"])
    print(alpha_k.shape)
    # Initializer
    initializer = initializers.HeNormal()
    # Early Stopping criterion
    es = EarlyStopping(monitor=parameters["monitor"], mode=parameters["mode"], verbose=0)
    # Loss function
    loss_fn = losses.MeanSquaredError()
    # Activation function
    act_fn = parameters["activation_function"]

    # Sequential Model with one input, one output and 6 Dense hidden layers
    model = Sequential() # Model
    model.add(Input(shape=(2,), name='Input-Layer')) # Input Layer - need to speicfy the shape of inputs
    model.add(BatchNormalization())
    model.add(Dense(512, activation= act_fn, kernel_initializer=initializer, name='Hidden-Layer-1')) # Hidden Layer, softplus(x) = log(exp(x) + 1)
    model.add(BatchNormalization())
    model.add(Dense(256, activation= act_fn, kernel_initializer=initializer, name='Hidden-Layer-2')) # Hidden Layer, softplus(x) = log(exp(x) + 1)
    model.add(BatchNormalization())
    model.add(Dense(128, activation= act_fn, kernel_initializer=initializer, name='Hidden-Layer-3')) # Hidden Layer, softplus(x) = log(exp(x) + 1)
    model.add(BatchNormalization())
    model.add(Dense(64, activation= act_fn, kernel_initializer=initializer, name='Hidden-Layer_4')) # Hidden Layer, softplus(x) = log(exp(x) + 1)
    model.add(BatchNormalization())
    model.add(Dense(32, activation= act_fn, kernel_initializer=initializer, name='Hidden-Layer-5')) # Hidden Layer, softplus(x) = log(exp(x) + 1)
    model.add(BatchNormalization())
    model.add(Dense(16, activation= act_fn, kernel_initializer=initializer, name='Hidden-Layer-6')) # Hidden Layer, softplus(x) = log(exp(x) + 1)
    model.add(BatchNormalization())
    model.add(Dense(components, activation= act_fn, name='Output-Layer')) # Output Layer, sigmoid(x) = 1 / (1 + exp(-x))
    
    # Compile model
    model.compile(optimizer = Nadam(
                                learning_rate=parameters["learning_rate"], # Learning rate
                                beta_1=parameters["beta_1"],
                                beta_2=parameters["beta_2"],
                                epsilon=parameters["epsilon"],
                                name=parameters["name"]),
                  loss=loss_fn, # Loss function
                  metrics=[parameters["metric"]]) # Performance metric
    
    # Fit training data
    model.fit(known_latlon.to_numpy(), # input data
              alpha_k, # target data
              batch_size=parameters["batch_size"], # Number of samples per gradient update. If unspecified, batch_size will default to 32.
              epochs=parameters["epochs"]) # Number of epochs
    
    # Perform Interpolation at unknown locations
    pred_intr_op = model.predict(unknown_latlon.to_numpy(), batch_size = parameters["intr_batch_size"])
    #pred_train_op = model.predict(known_latlon.to_numpy(), batch_size = 128)     # Perform Prediction on Training data
    
    # Perform Inverse Tranform on spatial coefficient data
    pred_intr = coff_scaler.inverse_transform(pred_intr_op)
    
    # Get spatio-temporal data from spatial and temporal coefficients thorugh dot product
    pred_intr_coeff = np.dot(pred_intr, pca.components_[:components])
    
    # Perform Inverse Tranform on spatio-temporal data to get interpolated data
    Interpolated_Data = data_scaler.inverse_transform(pred_intr_coeff)
    
    pd.DataFrame(Interpolated_Data).to_parquet('data/07_model_output/'+pollutant_name+'_interpolated_data.parquet')

