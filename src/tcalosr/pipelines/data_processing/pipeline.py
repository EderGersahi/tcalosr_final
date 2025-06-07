"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.13
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import clean_reservaciones, imputar_tarifa, ingresos_por_fecha, suavizado, rename_columns

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=clean_reservaciones,
                inputs=["reservaciones", "params:fechas"],
                outputs="reservaciones_cleaned",
                name="clean_reservaciones_node",
            ),
            node(
                func=imputar_tarifa, 
                inputs="reservaciones_cleaned",
                outputs="imputar_data",
                name="imputar_tarifa_node",
            ),
            node(
                func=ingresos_por_fecha,
                inputs="imputar_data",
                outputs="ingresos_por_fecha",
                name="ingresos_por_fecha_node",
            ),
            node(
                func=suavizado,
                inputs="ingresos_por_fecha",
                outputs="suavizado_data",
                name="suavizado_node",
            ),
            node(
                func=rename_columns,
                inputs=["suavizado_data", "params:suavizado"],
                outputs="rename_to_prophet",
                name="rename_to_prophet_node",
            )

        ]
    )
