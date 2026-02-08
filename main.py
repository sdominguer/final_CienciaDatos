import streamlit as st
import pandas as pd
import plotly.express as px

# Configuración de la página
st.set_page_config(page_title="COVID-19 Global Analytics", layout="wide", page_icon="📈")

st.title("📊 Dashboard Integrado: COVID, Clima y Salud")
st.markdown("Sube tu archivo para limpiar las gráficas y explorar los datos.")

# 1. Carga de Archivo
uploaded_file = st.file_uploader("Carga tu CSV", type="csv")

if uploaded_file is not None:
    @st.cache_data
    def load_and_clean(file):
        df = pd.read_csv(file)
        # Convertir fecha
        df['date'] = pd.to_datetime(df['date'])
        # --- SOLUCIÓN AL GRÁFICO RARO ---
        # Ordenamos por fecha para que las líneas no se crucen
        df = df.sort_values(['country', 'date'])
        # Llenamos vacíos en camas y casos con 0 para evitar errores de Plotly
        df['hospital_beds'] = df['hospital_beds'].fillna(0)
        df['weekly_count'] = df['weekly_count'].fillna(0)
        return df

    df = load_and_clean(uploaded_file)

    # --- SIDEBAR ---
    st.sidebar.header("Filtros")
    
    # Filtro de Indicador (Fundamental para que no se mezclen Casos con Muertes en la misma línea)
    indicadores = df['indicator'].unique().tolist()
    sel_ind = st.sidebar.selectbox("Selecciona Indicador:", indicadores)

    continentes = sorted(df["continent"].unique().tolist())
    sel_cont = st.sidebar.multiselect("Continentes:", continentes, default=continentes)

    paises_disp = sorted(df[df["continent"].isin(sel_cont)]["country"].unique().tolist())
    sel_paises = st.sidebar.multiselect("Países:", paises_disp, default=paises_disp[:5])

    # Filtrado final
    mask = (df["continent"].isin(sel_cont)) & (df["country"].isin(sel_paises)) & (df["indicator"] == sel_ind)
    df_filtered = df[mask]

    # --- DASHBOARD ---
    m1, m2, m3 = st.columns(3)
    m1.metric("Total registros", len(df_filtered))
    m2.metric("Temp Promedio", f"{df_filtered['avg_temp'].mean():.2f} °C")
    m3.metric("Promedio Camas", f"{df_filtered['hospital_beds'].mean():.2f}")

    tab1, tab2 = st.tabs(["📈 Evolución Temporal", "🌡️ Clima vs Capacidad"])

    with tab1:
        st.subheader(f"Línea de Tiempo: {sel_ind}")
        # Usamos line_group para asegurar que cada país sea una línea independiente
        fig_line = px.line(
            df_filtered, 
            x="date", 
            y="weekly_count", 
            color="country",
            title=f"Curva de {sel_ind} (Ordenada cronológicamente)",
            template="plotly_dark",
            markers=True # Agregamos puntos para ver mejor los datos
        )
        st.plotly_chart(fig_line, use_container_width=True)

    with tab2:
        st.subheader("Análisis de Correlación")
        
        # Corrección del ValueError: aseguramos que el tamaño sea > 0
        df_plot = df_filtered.copy()
        df_plot['size_viz'] = df_plot['hospital_beds'].apply(lambda x: x if x > 0 else 0.1)
        
        fig_scat = px.scatter(
            df_plot,
            x="avg_temp",
            y="weekly_count",
            size="size_viz",
            color="country",
            hover_name="month_name",
            hover_data={"size_viz": False, "hospital_beds": True},
            title="Temperatura vs Contagios (Tamaño = Camas)",
            template="plotly_dark"
        )
        st.plotly_chart(fig_scat, use_container_width=True)

    with st.expander("Ver tabla de datos"):
        st.write(df_filtered)

else:
    st.info("Esperando archivo CSV...")
