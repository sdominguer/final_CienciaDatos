import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configuración de la página
st.set_page_config(page_title="COVID-19 Global Dashboard", layout="wide")

# Carga de datos
@st.cache_data
def load_data():
    df = pd.read_csv('dataset_covid_completo.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df

try:
    df = load_data()
except:
    st.error("No se encontró el archivo 'dataset_covid_completo.csv'. Por favor, asegúrate de cargarlo.")
    st.stop()

# --- SIDEBAR (Filtros) ---
st.sidebar.header("Filtros de Análisis")
continente = st.sidebar.multiselect(
    "Selecciona Continente:",
    options=df["continent"].unique(),
    default=df["continent"].unique()
)

paises = st.sidebar.multiselect(
    "Selecciona Países:",
    options=df[df["continent"].isin(continente)]["country"].unique(),
    default=df[df["continent"].isin(continente)]["country"].unique()[:3]
)

df_selection = df[(df["continent"].isin(continente)) & (df["country"].isin(paises))]

# --- TÍTULO PRINCIPAL ---
st.title("📊 Dashboard COVID-19 & Factores Externos")
st.markdown("""
Esta aplicación analiza la relación entre los casos de COVID-19, 
la **temperatura media mensual** y la **capacidad hospitalaria**.
""")

# --- MÉTRICAS CLAVE ---
col1, col2, col3 = st.columns(3)
total_cases = df_selection[df_selection['indicator'] == 'cases']['weekly_count'].sum()
avg_temp = df_selection['avg_temp'].mean()
avg_beds = df_selection['hospital_beds'].mean()

col1.metric("Total Casos (Selección)", f"{total_cases:,.0f}")
col2.metric("Temp. Media", f"{avg_temp:.2f} °C")
col3.metric("Camas Hospital (Promedio)", f"{avg_beds:.2f}")

st.divider()

# --- GRÁFICOS INTERACTIVOS ---
c1, c2 = st.columns(2)

with c1:
    st.subheader("📈 Evolución Temporal: Casos vs Temp")
    # Gráfico de doble eje con Plotly
    fig_temp = px.line(df_selection, x="date", y="weekly_count", color="country",
                      title="Casos Semanales por País", labels={'weekly_count': 'Casos'})
    st.plotly_chart(fig_temp, use_container_width=True)

with c2:
    st.subheader("🌡️ Correlación: Temperatura vs Contagios")
    fig_scatter = px.scatter(df_selection, x="avg_temp", y="weekly_count", 
                             color="country", size="hospital_beds",
                             hover_name="month_name",
                             title="¿Influye el clima en los contagios?")
    st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()

# --- MAPA DE CALOR ---
st.subheader("🗺️ Mapa Global de Incidencia (14 días)")
fig_map = px.choropleth(df_selection, locations="ISO3", color="rate_14_day",
                        hover_name="country", animation_frame="month_name",
                        color_continuous_scale=px.colors.sequential.Reds)
st.plotly_chart(fig_map, use_container_width=True)

# --- TABLA DE DATOS ---
with st.expander("👀 Ver datos crudos"):
    st.dataframe(df_selection)

st.sidebar.markdown("---")
st.sidebar.write("Desarrollado con ❤️ usando Streamlit y Pandas")
