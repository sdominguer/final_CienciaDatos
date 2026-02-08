import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from groq import Groq

# Configuración de página
st.set_page_config(page_title="Pandemic Decision Support System", layout="wide", page_icon="📈")

st.title("🚀 Sistema de Soporte a la Decisión: Análisis COVID-19")
st.markdown("---")

# --- SIDEBAR: MÓDULO ETL Y LIMPIEZA ---
st.sidebar.header("⚙️ Configuración del Sistema")
uploaded_file = st.sidebar.file_uploader("Cargar Base de Datos (CSV)", type="csv")

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    
    # --- LIMPIEZA INTERACTIVA ---
    st.sidebar.subheader("Módulo de Limpieza")
    
    # Manejo de ISO3 Nulos (Solución a tu pregunta)
    if st.sidebar.checkbox("Limpiar códigos de país (ISO3)"):
        nulos_iso = df_raw['ISO3'].isnull().sum()
        df_raw['ISO3'] = df_raw['ISO3'].fillna('N/A')
        st.sidebar.info(f"Se marcaron {nulos_iso} países como N/A")

    if st.sidebar.checkbox("Eliminar duplicados"):
        df_raw = df_raw.drop_duplicates()

    metodo_nulos = st.sidebar.selectbox(
        "Imputación de variables numéricas:",
        ["Mantener nulos", "Llenar con Media", "Llenar con Cero"]
    )
    
    df_clean = df_raw.copy()
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    if metodo_nulos == "Llenar con Media":
        df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].mean())
    elif metodo_nulos == "Llenar con Cero":
        df_clean[num_cols] = df_clean[num_cols].fillna(0)

    # --- FILTROS ---
    indicador = st.sidebar.selectbox("Indicador:", df_clean['indicator'].unique())
    continentes = st.sidebar.multiselect("Continentes:", df_clean['continent'].unique(), default=df_clean['continent'].unique())
    df_final = df_clean[(df_clean['indicator'] == indicador) & (df_clean['continent'].isin(continentes))]

    # --- TABS ---
    tab_desc, tab_cuant, tab_graf, tab_ia = st.tabs([
        "📖 Análisis Descriptivo", "🔢 Análisis Cuantitativo", "📊 Análisis Gráfico", "🤖 AI Insights"
    ])

    # TAB 1: DESCRIPTIVO (Glosario y Calidad de Datos)
    with tab_desc:
        st.subheader("Diccionario de Variables Generadas")
        with st.expander("🔍 Ver definiciones de indicadores senior"):
            st.markdown("""
            * **casos_100k (Incidencia):** Número de casos por cada 100,000 habitantes. Permite comparar la intensidad del brote entre países de distinto tamaño poblacional.
            * **camas_por_100k (Capacidad):** Cantidad de camas hospitalarias disponibles por cada 100,000 habitantes. Mide la robustez del sistema de salud.
            * **letalidad_pct (Tasa de Letalidad):** Porcentaje de muertes respecto al total de la población (ajustado semanalmente). Es un indicador crítico de riesgo de mortalidad.
            """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Calidad de Datos (Nulos):**")
            st.write(df_final.isnull().sum())
        with col2:
            st.write("**Estadísticas Descriptivas Rápidas:**")
            st.write(df_final.describe())

    # TAB 2: CUANTITATIVO
    with tab_cuant:
        st.subheader("Análisis de Correlación Multivariable")
        # Aseguramos solo numéricas para la matriz
        df_corr = df_final.select_dtypes(include=[np.number])
        fig_corr, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df_corr.corr(), annot=True, cmap="RdBu_r", center=0, ax=ax)
        st.pyplot(fig_corr)
        

    # TAB 3: GRÁFICO
    with tab_graf:
        st.subheader("Visualización Dinámica")
        # Mapa de incidencia
        fig_map = px.choropleth(df_final, locations="ISO3", color="casos_100k", 
                               hover_name="country", title="Mapa de Incidencia Global",
                               color_continuous_scale="Reds")
        st.plotly_chart(fig_map, use_container_width=True)
        
        # Dispersión
        fig_scat = px.scatter(df_final, x="avg_temp", y="letalidad_pct", size="camas_por_100k",
                             color="continent", hover_name="country", 
                             title="Relación Temperatura vs Letalidad (Tamaño = Capacidad Hospitalaria)")
        st.plotly_chart(fig_scat, use_container_width=True)

    # TAB 4: IA
    with tab_ia:
        st.subheader("Analista Virtual con Groq")
        api_key = st.text_input("Groq API Key:", type="password")
        if st.button("Generar Reporte Ejecutivo"):
            if api_key:
                client = Groq(api_key=api_key)
                resumen = df_final.describe().to_string()
                prompt = f"""Analiza estos indicadores de COVID: {resumen}. 
                Explica la relación entre 'camas_por_100k' y 'letalidad_pct'. 
                Sé técnico y actúa como un consultor de salud pública."""
                
                chat = client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="llama3-8b-8192")
                st.markdown(chat.choices[0].message.content)
            else:
                st.error("Falta la API Key.")

else:
    st.info("Por favor, sube el archivo generado en Colab.")
