from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_for_eda

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(preprocess_for_eda, "raw_data","processed_data"),
        ]
    )