"""
This is a boilerplate pipeline 'route_planning'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import data_collection, run_route_planner

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func = data_collection,
                inputs = ["dump", "bin", "starting"],
                outputs = "location_data",
                name = "data_collection_node",
            ),
            node(
                func = run_route_planner,
                inputs = ["dump", "starting", "location_data", "pairwise_distance", "params:model_parameters"],
                outputs = None, #"ortools_solution",
                name = "run_route_planner_node",
            ),
        ]
    )
