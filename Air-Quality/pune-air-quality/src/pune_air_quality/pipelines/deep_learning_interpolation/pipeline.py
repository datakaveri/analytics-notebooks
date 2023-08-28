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
                inputs = ["airQualityIndex", "known_latlon", "unknown_latlon", "params:deep_learning_model", "params:airQualityIndex"],
                outputs = None,
                name = "air_quality_index_deep_learning_model_node",
            ),
            node(
                func = deep_learning_model,
                inputs = ["airTemperature", "known_latlon", "unknown_latlon", "params:deep_learning_model", "params:airTemperature"],
                outputs = None,
                name = "air_temperature_deep_learning_model_node",
            ),
            node(
                func = deep_learning_model,
                inputs = ["ambientNoise", "known_latlon", "unknown_latlon", "params:deep_learning_model", "params:ambientNoise"],
                outputs = None,
                name = "ambient_noise_deep_learning_model_node",
            ),
            node(
                func = deep_learning_model,
                inputs = ["atmosphericPressure", "known_latlon", "unknown_latlon", "params:deep_learning_model", "params:atmosphericPressure"],
                outputs = None,
                name = "air_pressure_deep_learning_model_node",
            ),
            node(
                func = deep_learning_model,
                inputs = ["co", "known_latlon", "unknown_latlon", "params:deep_learning_model", "params:co"],
                outputs = None,
                name = "co_deep_learning_model_node",
            ),
            node(
                func = deep_learning_model,
                inputs = ["co2", "known_latlon", "unknown_latlon", "params:deep_learning_model", "params:co2"],
                outputs = None,
                name = "co2_deep_learning_model_node",
            ),
            node(
                func = deep_learning_model,
                inputs = ["illuminance", "known_latlon", "unknown_latlon", "params:deep_learning_model", "params:illuminance"],
                outputs = None,
                name = "illuminance_deep_learning_model_node",
            ),
            node(
                func = deep_learning_model,
                inputs = ["no2", "known_latlon", "unknown_latlon", "params:deep_learning_model", "params:no2"],
                outputs = None,
                name = "no2_deep_learning_model_node",
            ),
            node(
                func = deep_learning_model,
                inputs = ["o3", "known_latlon", "unknown_latlon", "params:deep_learning_model", "params:o3"],
                outputs = None,
                name = "o3_deep_learning_model_node",
            ),
            node(
                func = deep_learning_model,
                inputs = ["pm2p5", "known_latlon", "unknown_latlon", "params:deep_learning_model", "params:pm2p5"],
                outputs = None,
                name = "pm2p5_deep_learning_model_node",
            ),
            node(
                func = deep_learning_model,
                inputs = ["pm10", "known_latlon", "unknown_latlon", "params:deep_learning_model", "params:pm10"],
                outputs = None,
                name = "pm10_deep_learning_model_node",
            ),
            node(
                func = deep_learning_model,
                inputs = ["so2", "known_latlon", "unknown_latlon", "params:deep_learning_model", "params:so2"],
                outputs = None,
                name = "so2_deep_learning_model_node",
            ),
            node(
                func = deep_learning_model,
                inputs = ["uv", "known_latlon", "unknown_latlon", "params:deep_learning_model", "params:uv"],
                outputs = None,
                name = "uv_deep_learning_model_node",
            ),
            node(
                func = deep_learning_model,
                inputs = ["relativeHumidity", "known_latlon", "unknown_latlon", "params:deep_learning_model", "params:relativeHumidity"],
                outputs = None,
                name = "relative_humidity_deep_learning_model_node",
            ),
        ]
    )
