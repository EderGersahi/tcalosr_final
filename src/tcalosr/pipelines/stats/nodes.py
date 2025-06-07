"""
This is a boilerplate pipeline 'stats'
generated using Kedro 0.19.13
"""

import pandas as pd
import numpy as np

def stats_1(reservaciones_cleaned: pd.DataFrame) -> pd.DataFrame:

    reservations = reservaciones_cleaned.copy()
    reservations['h_fec_lld_ok'] = pd.to_datetime(reservations['h_fec_lld_ok'])
    reservations['mes'] = reservations['h_fec_lld_ok'].dt.to_period('M')
    mensual_income = reservations.groupby('mes').agg({
        'h_tfa_total': ['std', 'sum']
    }).reset_index()

    mensual_income.columns = ['Mes'] + [f'{col}_{func}' for col, func in mensual_income.columns[1:]]

    def exapand_reservation(row):
        fechas = pd.date_range(start=row['h_fec_lld_ok'], periods=row['h_num_noc'],freq='D')
        return pd.DataFrame({'fecha_noche': fechas})

    expanded_reservations = pd.concat([exapand_reservation(row) for _, row in reservations.iterrows()], ignore_index=True)
    expanded_reservations['año_mes']=expanded_reservations['fecha_noche'].dt.to_period('M')

    nights_per_month = expanded_reservations.groupby('año_mes').size().reset_index(name='nights_per_month')

    return mensual_income, nights_per_month
