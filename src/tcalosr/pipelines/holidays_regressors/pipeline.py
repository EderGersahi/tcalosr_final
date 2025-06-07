"""
This is a boilerplate pipeline 'holidays_regressors'
generated using Kedro 0.19.13
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import days_holidays, regressors_holidays

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=days_holidays,
                inputs="rename_to_prophet",
                outputs="holidays_days",
                name="holidays_days_node",
                ),
            node(
                func=regressors_holidays,
                inputs="rename_to_prophet",
                outputs="df_prophet_regressors",
                name="df_prophet_regressors_node",
                )
            
        ]
     )
