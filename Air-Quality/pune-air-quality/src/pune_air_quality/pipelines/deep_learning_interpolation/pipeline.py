"""
This is a boilerplate pipeline 'deep_learning_interpolation'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import get_location_data, deep_learning_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func = get_location_data,
                inputs = ["sensor_data", "sensor_ids", "params:deep_learning_model"],
                outputs = ["known_latlon", "unknown_latlon", "latlon"],
                name = "get_location_data_node",
            ),
            node(
                func = deep_learning_model,
                inputs = ["known_latlon", "unknown_latlon", "params:deep_learning_model"],
                outputs = None,
                name = "deep_learning_model_node",
            ),
        ]
    )
