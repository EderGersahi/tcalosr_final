"""
This is a boilerplate pipeline 'holidays_regressors'
generated using Kedro 0.19.13
"""
import pandas as pd
import numpy as np
import holidays
from datetime import datetime , timedelta
from dateutil import easter

def _es_temporada_alta(fecha):
    a = fecha.year
    return (
        (datetime(a, 6, 22) <= fecha <= datetime(a, 8, 31)) or
        (datetime(a, 12, 20) <= fecha <= datetime(a, 12, 31)) or
        (datetime(a + 1, 1, 1) <= fecha <= datetime(a + 1, 1, 7))
    )


# Confinamiento COVID
# Si quitamos el año 2020 no aplica
def _es_confinamiento_covid(fecha):
    return (
        (datetime(2020, 3, 23) <= fecha <= datetime(2020, 6, 1)) or
        (datetime(2020, 12, 15) <= fecha <= datetime(2021, 2, 15)) 
    )
# Semana Santa
def _es_semana_santa(fecha):
    a = fecha.year
    pascua = easter.easter(a)
    inicio = pd.Timestamp(pascua - timedelta(days=6))
    fin = pd.Timestamp(pascua)
    return inicio <= fecha <= fin

def regressors_holidays(data: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega regresores de días festivos y temporadas altas al DataFrame de Prophet.
    
    Args:
        data (pd.DataFrame): DataFrame con una columna 'ds' de tipo datetime.
        
    Returns:
        pd.DataFrame: DataFrame con las columnas 'ds', 'es_temporada_alta', 
                      'es_confinamiento_covid', 'es_semana_santa'.
    """
    if 'ds' not in data.columns:
        raise ValueError("El DataFrame 'data' debe contener una columna 'ds' de tipo datetime.")
    
    data['ds'] = pd.to_datetime(data['ds'], errors='coerce')
    
    # Asegurarse de que la columna ds no tenga valores nulos
    data = data.dropna(subset=['ds'])
    
    data['es_temporada_alta'] = data['ds'].apply(_es_temporada_alta)
    #data['es_confinamiento_covid'] = data['ds'].apply(_es_confinamiento_covid)
    data['es_semana_santa'] = data['ds'].apply(_es_semana_santa)
    
    return data



def days_holidays(data: pd.DataFrame) -> pd.DataFrame:
    data['ds'] = pd.to_datetime(data['ds'], errors='coerce')  # Convierte a datetime
    years = data['ds'].dt.year.dropna().unique() 
    if len(years) == 0:
        raise ValueError("El DataFrame 'data' no contiene fechas válidas en la columna 'ds'.")
    feriados_mx = holidays.Mexico(years=years)

    # Convertir a df para Prophet
    df_feriados = pd.DataFrame([
        {'ds': pd.to_datetime(fecha), 'holiday': nombre}
        for fecha, nombre in feriados_mx.items()
    ])

    # Columans para extender el efecto del día festivo para prophet
    df_feriados['lower_window'] = 0
    df_feriados['upper_window'] = 1

    return df_feriados