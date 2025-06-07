"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.13
"""
import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
#Limpieza de datos
#Limpieza de datos
def clean_reservaciones(data:pd.DataFrame, fecha_limite: str) -> pd.DataFrame:

    #Eliminar columnas irrelevantes (se tomó la decisión con data wrangler)
    # Se eliminaron fechas que no se usan en el modelo
    # Se eliminaron columnas vacías y duplicadas 
    drop_cols = ['h_res_fec', 'h_res_fec_okt', 'h_fec_lld_okt', 'h_fec_lld', 'h_fec_reg', 
                 'h_fec_reg_okt', 'h_fec_sda', 'h_fec_sda_okt', 'h_nom', 'h_correo_e', 
                 'moneda_cve', 'h_ult_cam_fec', 'h_ult_cam_fec_okt', 'ID_Reserva', 
                 'Fecha_hoy', 'h_fec_reg_ok', 'h_res_fec_ok','h_ult_cam_fec_ok', 
                 'h_fec_sda_ok','Cliente_Disp', 'h_codigop', 'ID_empresa']
    data.drop(columns = [col for col in drop_cols if col in data.columns], inplace = True)

    #Cambiar las fechas de object a datetime
    data['h_fec_lld_ok'] = pd.to_datetime(data['h_fec_lld_ok'], errors='coerce')

    #Eliminar columnas con 'aa' (columnas del año anterior)
    data = data[data.columns.drop([col for col in data.columns if col.startswith('aa_')])]

    #Filtra por ID_programa (menos del 1% tiene ID_programa = 0)
    data = data[data['ID_Programa'] != 0]
    data.drop(columns=['ID_Programa'], inplace=True)

    # Filtrar filas con ID_Pais_Origen = 157
    data = data[data['ID_Pais_Origen'] == 157]
    data = data.drop(columns=['ID_Pais_Origen'])

    #Filtrar filas con número de noches, habitaciones, personas y la tarifa = 0
    data = data[(data['h_num_per'] > 0) & (data['h_num_noc'] > 0) & (data['h_tot_hab'] > 0) ]
    
    # Quitar outliers tarifa
    data = data[data['h_tfa_total'] >= 0]
    q1 = data['h_tfa_total'].quantile(0.25)
    q3 = data['h_tfa_total'].quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    data = data[(data['h_tfa_total'] >= low) & (data['h_tfa_total'] <= high)]

    # Quitar outliers noches
    data = data[data['h_num_noc'] >= 0]
    q1 = data['h_num_noc'].quantile(0.25)
    q3 = data['h_num_noc'].quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    data = data[(data['h_num_noc'] >= low) & (data['h_num_noc'] <= high)]

    # Reemplazar espacios por NaN y quitar duplicados en cod reserva (ignorando NaN)
    data['h_cod_reserva'] = data['h_cod_reserva'].replace(r'^\s*$', np.nan, regex = True)
    data = data[(~data['h_cod_reserva'].duplicated()) | data['h_cod_reserva'].isna()]
    # Elimar la columna al funcionar como un ID
    data.drop(columns=['h_cod_reserva'], inplace=True)

    # Incluir solo las reservas entre 2019 y 2021)
    data = data[(data['h_fec_lld_ok'] >= '2019-01-01') & (data['h_fec_lld_ok'] <= fecha_limite)]

    return data

def imputar_tarifa(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    # Guardar la columna de fecha como datetime 
    data['h_fec_lld_ok'] = pd.to_datetime(data['h_fec_lld_ok'])

    # Identificar columnas categóricas, excluyendo la fecha
    object_columns = data.select_dtypes(include=['object']).columns
    object_columns = [col for col in object_columns if col != 'h_fec_lld_ok']

    # Codificar columnas categóricas
    le = LabelEncoder()
    for col in object_columns:
        data[col] = le.fit_transform(data[col].astype(str))

    # Separar datos a imputar y válidos
    to_impute = data[data['h_tfa_total'] == 0].copy()
    data_valid = data[data['h_tfa_total'] != 0].copy()

    # Guardar fechas antes de eliminar
    fechas_to_impute = to_impute['h_fec_lld_ok']

    # Preparar X e y
    X = data_valid.drop(columns=['h_tfa_total', 'h_fec_lld_ok'])
    y = data_valid['h_tfa_total']

    # Entrenar modelo
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Imputar
    X_to_impute = to_impute.drop(columns=['h_tfa_total', 'h_fec_lld_ok'])
    to_impute['h_tfa_total'] = model.predict(X_to_impute)

    # Restaurar la columna de fecha original
    to_impute['h_fec_lld_ok'] = fechas_to_impute

    # Concatenar
    final_df = pd.concat([data_valid, to_impute], ignore_index=True)

    # Asegurar tipo datetime y ordenar
    final_df['h_fec_lld_ok'] = pd.to_datetime(final_df['h_fec_lld_ok'])
  
    return final_df


def ingresos_por_fecha(data: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa los ingresos por fecha de llegada y devuelve un DataFrame con la suma de ingresos por fecha.
    """
    ingresos = data.groupby('h_fec_lld_ok', as_index = False)['h_tfa_total'].sum().sort_values('h_fec_lld_ok')
    return ingresos

def suavizado(data: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """
    Aplica suavizado a la columna 'h_tfa_total' usando una media móvil y un suavizado exponencial.
    Args:
        data (pd.DataFrame): DataFrame de ingresos.
        window (int): Tamaño de la ventana para la media móvil.
    """
    data['smoothed'] = data['h_tfa_total'].rolling(window=window).mean() #media movil simple
    data['smoothed2'] = data['h_tfa_total'].ewm(span=7, adjust=False).mean() # Suavizado exponencial

    return data

def rename_columns(data: pd.DataFrame,suavizado: str ) -> pd.DataFrame:
    """
    Renombra las columnas del DataFrame para que sean aceptadas por el modelo Prophet.
    Args:
        data (pd.DataFrame): DataFrame suavizado con los ingresos por fecha.
    """
    if suavizado == 'smoothed':
        data=data.dropna(subset=[suavizado])
        data_prophet = data[['h_fec_lld_ok', suavizado]].copy().rename(columns = {'h_fec_lld_ok': 'ds', suavizado: 'y'})
    else:
        data_prophet = data[['h_fec_lld_ok', 'smoothed2']].copy().rename(columns = {'h_fec_lld_ok': 'ds', 'smoothed2': 'y'})
    return data_prophet