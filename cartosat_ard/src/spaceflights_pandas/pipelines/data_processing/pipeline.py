from kedro.pipeline import Pipeline, node, pipeline

from .nodes import  geometric_correction, detect_distortions , cloud_detection , pansharpening


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=geometric_correction,
                inputs="original_images",
                outputs="histogram_equalized_data",
                name="geometric_correction_node",
            ),
            node(
                func=detect_distortions,
                inputs="histogram_equalized_data",
                outputs="noise_masked_data",
                name="detect_distortions_node",
            ),
            node(
                func=cloud_detection,
                inputs="noise_masked_data",
                outputs="cloud_mask_data",
                name="cloud_detection_node",
            ),
            node(
                func=pansharpening,
                inputs="cloud_mask_data",
                outputs="pansharpened_data",
                name="pansharpening_node",
            )
        ]
    )
