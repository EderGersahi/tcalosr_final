"""
This is a boilerplate pipeline 'modelo'
generated using Kedro 0.19.13
"""

import itertools
import pandas as pd
import numpy as np
import mlflow
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from datetime import datetime , timedelta
from dateutil import easter
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error



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


def tune_hyperparams(data: pd.DataFrame,df_feriados: pd.DataFrame, parameters: dict)-> dict:
    
    rmses = []
    mape = []
    mdape = []
    param_grid = {
        'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
        'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0],
        'seasonality_mode': ['additive', 'multiplicative']
    }
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

    for params in all_params:
        model = Prophet(holidays = df_feriados, **params)
        model.add_regressor('es_temporada_alta')
        #model.add_regressor('es_confinamiento_covid')
        model.add_regressor('es_semana_santa')
        model.fit(data)
        parameters["initial"] 
        df_cv = cross_validation(model, initial=parameters["initial"] , period=parameters["period"], horizon=parameters["horizon"], parallel=parameters["parallel"])
        df_p = performance_metrics(df_cv, rolling_window=1)
        rmses.append(df_p['rmse'].values[0])
        mape.append(df_p['mape'].values[0])
        mdape.append(df_p['mdape'].values[0])

    tuning_results = pd.DataFrame(all_params)
    tuning_results['rmse'] = rmses
    tuning_results['mape'] = mape
    tuning_results['mdape'] = mdape

    best_params = tuning_results.sort_values(by=['mdape', 'mape','rmse'], ascending=[True, True, True]).iloc[0]
    for param, value in best_params.items():
        mlflow.log_param(param, value)
    best_params = best_params.drop(['rmse', 'mape', 'mdape'])
    best_params= best_params.to_dict()

    return best_params


def create_best_model(data: pd.DataFrame, df_feriados: pd.DataFrame, best_params: dict) -> Prophet:
    model = Prophet(holidays=df_feriados, **best_params)
    model.add_regressor('es_temporada_alta')
    #model.add_regressor('es_confinamiento_covid')
    model.add_regressor('es_semana_santa')
    model.fit(data)

    future = model.make_future_dataframe(periods=30)
    future['es_temporada_alta'] = future['ds'].apply(_es_temporada_alta)
    future['es_semana_santa'] = future['ds'].apply(_es_semana_santa)
    forecast = model.predict(future)

    forecast['ds']= pd.to_datetime(forecast['ds'])
    data['ds'] = pd.to_datetime(data['ds'])

    historico = forecast[forecast['ds'].isin(data['ds'])].copy()
    futuro = forecast[~forecast['ds'].isin(data['ds'])].copy()

    historico = historico.merge(data[['ds', 'y']], on='ds', how='left')
    historico['tipo'] = 'historico'

    futuro['y'] = None
    futuro['tipo'] = 'predicción'

    historico = historico[['ds', 'y', 'yhat', 'yhat_upper', 'yhat_lower','tipo']]
    futuro = futuro[['ds', 'y', 'yhat','yhat_upper', 'yhat_lower', 'tipo']]

    results = pd.concat([historico, futuro], ignore_index=True)

    results['residuals'] = results['y'] - results['yhat']

    return model, results

# def evaluate_model(model, data: pd.DataFrame):

#     forecast = model.predict(data)
#     y_true = data['y']
#     y_pred = forecast['yhat']
#     mape = mean_absolute_percentage_error(y_true, y_pred)
#     rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#     mdape = np.median(np.abs((y_true - y_pred) / y_true)) * 100

#     return  mape,  mdape, rmse

def evaluate_model(model: Prophet,parameters: dict):

    df_cv = cross_validation(model, initial=parameters["initial"] , period=parameters["period"], horizon=parameters["horizon"], parallel=parameters["parallel"])
    df_metrics = performance_metrics(df_cv, rolling_window=1)

    rmse = df_metrics['rmse'].iloc[0]
    mape = df_metrics['mape'].iloc[0]
    mdape = df_metrics['mdape'].iloc[0]
    mdape1 = pd.DataFrame({'MDAPE': [mdape]})

    return mape,  mdape, rmse, mdape1