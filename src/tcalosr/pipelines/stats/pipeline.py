"""
This is a boilerplate pipeline 'stats'
generated using Kedro 0.19.13
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import stats_1

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=stats_1,
            inputs="reservaciones_cleaned",
            outputs=["mensual_income", "nights_per_month"],
            name="stats_1_node",
        )
    ])
