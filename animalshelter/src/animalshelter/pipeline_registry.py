"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from .pipelines import eda as eda_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    
    return {
        "eda": eda_pipeline.create_pipeline(),
        "__default__": eda_pipeline.create_pipeline(),
    }
