from kedro.pipeline import Pipeline, node, pipeline

from .nodes import clean_data
from .nodes import scale_data
from .nodes import split_data

def create_pipeline(**kwargs):
    return Pipeline([
        node(clean_data, "processed_data", "clean_data"),
        node(scale_data, "clean_data", "scaled_data"),
        node(split_data, "scaled_data", ["train_data", "val_data", "test_data"]),
    ])
