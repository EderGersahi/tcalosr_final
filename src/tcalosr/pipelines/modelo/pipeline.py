"""
This is a boilerplate pipeline 'modelo'
generated using Kedro 0.19.13
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import tune_hyperparams, create_best_model, evaluate_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
            node(
                func=tune_hyperparams,
                inputs=["df_prophet_regressors", "holidays_days","params:cross_parameters"],
                outputs="best_hyperparams",
                name="tune_hyperparams_node",
                ),
            node(
                func=create_best_model,
                inputs=[ "df_prophet_regressors","holidays_days","best_hyperparams"],
                outputs=["best_prophet_model","results"],
                name="best_prophet_model_node",
                ),
            node(
                func=evaluate_model,
                inputs=["best_prophet_model","params:cross_parameters"],
                outputs=["mape", "mdape", "rmse", "mdape1"],
                name="evaluate_model_node",
                )

               ]
    )

