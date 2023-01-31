"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import datetime_preprocessing, run_outlier_detection, run_datetime_synchronization, get_imputated_pollutant_data#, get_location_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func = datetime_preprocessing,
                inputs = ["raw_data", "params:initializations"],
                outputs = "preprocessed_raw_data",
                name = "datetime_preprocessing_node",
            ),
            node(
                func = run_outlier_detection,
                inputs = "preprocessed_raw_data",
                outputs = "processed_data",
                name = "run_outlier_detection_node",
            ),
            node(
                func = run_datetime_synchronization,
                inputs = ["sensor_ids", "processed_data", "params:initializations"],
                outputs = "time_synchronized_data",
                name = "run_datetime_synchronization_node",
            ),
            node(
                func = get_imputated_pollutant_data,
                inputs = ["sensor_ids", "time_synchronized_data", "params:imputer_options"],
                outputs = None,
                name = "get_imputated_pollutant_data_node",
            ),
        ]
    )
