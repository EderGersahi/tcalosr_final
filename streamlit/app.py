import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

#Titulo del dhashboard
st.set_page_config(
    page_title="Ingresos Hoteleros: Predicción y Análisis",
    page_icon=":hotel:",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("<h1 style='text-align: center; color: #442C52;'>Análisis y Predicción de Ingresos Hoteleros</h1>", unsafe_allow_html=True)

#Dataframes
model_results = pd.read_csv("results.csv")
mensual_data = pd.read_csv("mensual_income.csv")
nights_data = pd.read_csv("nights_per_month.csv")
df_mdape = pd.read_csv("mdape1.csv")

# -------------------- Grid --------------------

col1, col2, col3 = st.columns([2,1,1])
col4, col5 = st.columns([1.5,1])

# -------------------- Volatibilidad --------------------
volatility = mensual_data[['Mes','h_tfa_total_std']]

palette = px.colors.sequential.BuPu
colors = [palette[i % len(palette)] for i in range(len(volatility))]

bar_fig = go.Figure()

bar_fig.add_trace(go.Bar(
    y=volatility['Mes'],
    x=volatility['h_tfa_total_std'],
    orientation='h',
    marker=dict(color=colors, line=dict(color='DarkSlateGrey', width=0.5)),
    text=volatility['h_tfa_total_std'].apply(lambda x: f'${x:.2f}'),
    textposition='outside',
    textfont=dict(size=12, color='black'),
    name='Volatibilidad por mes'
))
bar_fig.update_layout(
    title=dict(
        text='Volatibilidad de Ingresos por Mes',
        x=0.5,
        xanchor='center',
        font=dict(size=16, color='#442C52')),
    xaxis_title='Volatibilidad',
    yaxis_title='Mes',
    template='plotly_white',
    height=300,
    margin=dict(l=20, r=20, t=50, b=20))

with col1:
    st.plotly_chart(bar_fig, use_container_width=True)

# -------------------- Ranking de Ingresos Mensuales -------------

ranking = mensual_data.copy()
ranking['Mes'] = pd.to_datetime(ranking['Mes'], format='%Y-%m')

ranking['mes'] = ranking['Mes'].dt.month_name()

ranking= ranking.groupby('mes')['h_tfa_total_sum'].mean().reset_index()
ranking['h_tfa_total_sum'] = ranking['h_tfa_total_sum'].round(2)

top3 = ranking[['mes', 'h_tfa_total_sum']].sort_values(by='h_tfa_total_sum', ascending=False).head(6)
bottom3 = ranking[['mes', 'h_tfa_total_sum']].sort_values(by='h_tfa_total_sum', ascending=True).head(6)


top3['Total Ingresos'] = top3['h_tfa_total_sum'].apply(lambda x: f'${x/1000000:.3f} M')
bottom3['Total Ingresos'] = bottom3['h_tfa_total_sum'].apply(lambda x: f'${x/1000000:.3f} M')


with col2:
    st.markdown("<div style='font-size:16px;font-weight:bold;text-align: center; color: #442C52;'>Meses con mayores ingresos</div>", unsafe_allow_html=True)
    st.table(top3[['mes', 'Total Ingresos']].reset_index(drop=True))

with col3:
    st.markdown("<div style='font-size:16px;font-weight:bold;text-align: center; color: #442C52;'>Meses con menores ingresos</div>", unsafe_allow_html=True)
    st.table(bottom3[['mes', 'Total Ingresos']].reset_index(drop=True))


# -------------------- Prediccion de Ingresos --------------------

error = df_mdape['MDAPE'].values[0]

if error < 0.15:
    mensaje = f"Precisión alta: {100-error*100:.2f}%. "
    color = "green"
elif error < 0.3:
    mensaje = f"Precisión moderada: {100-error*100:.2f}%. "
    color = "orange"
else:
    mensaje = f"Precisión baja: {100-error*100:.2f}%. "
    color = "red"

bias = model_results['residuals'].mean()
if bias > 0:
    mensaje += "<br>El modelo tiende a sobreestimar"
elif bias < 0:
    mensaje += "<br>El modelo tiende a subestimar"
else:
    mensaje += "<br>El modelo no presenta sesgo"


fig = go.Figure()
fig.add_trace(go.Scatter(
    x=model_results['ds'],
    y=model_results['y'],
    mode='lines+markers',
    name='Ingresos',
    line=dict(color='#7ABEFE', width=2),
    marker=dict(size=3, color="#7ABEFE", line=dict(width=1, color='DarkSlateGrey'))
))



fig.add_trace(go.Scatter(
    x = model_results[model_results['tipo']=='predicción']['ds'],
    y = model_results[model_results['tipo']=='predicción']['yhat_lower'],
    line=dict(width=0),
    showlegend=False,
    hoverinfo='skip')
)

fig.add_trace(go.Scatter(
    x = model_results[model_results['tipo']=='predicción']['ds'],
    y = model_results[model_results['tipo']=='predicción']['yhat_upper'],
    fill='tonexty',
    fillcolor="#B8BBF9",
    line=dict(width=0),
    showlegend=True,
    name='Intervalo de confianza')
)

fig.add_trace(go.Scatter(
    x=model_results['ds'],
    y=model_results['yhat'],
    mode='lines',
    name='Prediccion',
    line=dict(color="#965AD2", dash = 'dash', width=2)
))

fig.add_annotation(
    text=mensaje + f". Error: ${error*1000:.2f} por cada $1000",
    xref="paper", yref="paper",
    x=1, y=1,
    showarrow=False,
    font=dict(size=12, color='white'),
    bgcolor=color
)


fig.update_layout(
     title=dict(
        text='Predicción de Ingresos Hoteleros',
        x=0.5,
        xanchor='center',
        font=dict(size=16, color='#442C52')),
    template='plotly_white',
    xaxis_title='Fecha',
    yaxis_title='Ingresos',
    legend=dict(x=0, y=1.1, traceorder='normal', orientation='h'),
    hovermode='x unified',
    height=350,
    width=500,
    margin=dict(l=20, r=20, t=50, b=20))

with col4:
    st.plotly_chart(fig, use_container_width=True)

# -------------------- Ingresos y noches Mensuales --------------------

mensual_data['income_per_night'] = mensual_data['h_tfa_total_sum']/nights_data['nights_per_month']

fig_bar = go.Figure()

fig_bar.add_trace(go.Bar(
    x = mensual_data['Mes'],
    y = mensual_data['income_per_night'],
    marker=dict(color=colors, line=dict(color='DarkSlateGrey', width=0.5)),
    text=mensual_data['income_per_night'].apply(lambda x: f'${x:.2f}'),
    textposition='outside',
    textfont=dict(size=12, color='black'),
    name='Ingresos por noche reservada'
))
fig_bar.update_layout(
    title=dict(
        text='Ingresos por noche reservada',
        x=0.5,
        xanchor='center',
        font=dict(size=16, color='#442C52')),
    xaxis_title='Mes',
    yaxis_title='Ingresos por noche',
    template='plotly_white',
    height=350,
    margin=dict(l=20, r=20, t=50, b=20))

with col5:
    st.plotly_chart(fig_bar, use_container_width=True)