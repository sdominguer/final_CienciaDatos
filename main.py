import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from groq import Groq # Debes instalar: pip install groq

# --- CONFIGURACIÓN Y ESTILO ---
st.set_page_config(page_title="DSS - Pandemic Insight", layout="wide")
st.title("🚀 Sistema de Soporte a la Decisión (DSS) - COVID-19")

# --- BARRA LATERAL: INGESTA Y ETL ---
st.sidebar.header("1. Módulo ETL (Ingesta y Limpieza)")
uploaded_file = st.sidebar.file_uploader("Cargar Dataset CSV", type="csv")

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    
    # --- LIMPIEZA INTERACTIVA ---
    st.sidebar.subheader("Limpieza de Datos")
    if st.sidebar.checkbox("Eliminar Duplicados"):
        df_raw = df_raw.drop_duplicates()
        st.sidebar.success("Duplicados eliminados")

    metodo_imputacion = st.sidebar.selectbox(
        "Método de Imputación (Nulos):",
        ["Ninguno", "Media", "Mediana", "Cero"]
    )
    
    # Aplicar imputación a variables numéricas
    num_cols = df_raw.select_dtypes(include=[np.number]).columns
    if metodo_imputacion == "Media":
        df_raw[num_cols] = df_raw[num_cols].fillna(df_raw[num_cols].mean())
    elif metodo_imputacion == "Mediana":
        df_raw[num_cols] = df_raw[num_cols].fillna(df_raw[num_cols].median())
    elif metodo_imputacion == "Cero":
        df_raw[num_cols] = df_raw[num_cols].fillna(0)

    # --- FEATURE ENGINEERING ---
    # Requisito 2.1: Crear nueva columna calculada
    if 'weekly_count' in df_raw.columns and 'population' in df_raw.columns:
        df_raw['casos_por_100k'] = (df_raw['weekly_count'] / df_raw['population']) * 100000

    # --- FILTROS GLOBALES ---
    st.sidebar.subheader("2. Filtros Globales")
    selected_cont = st.sidebar.multiselect("Continente", df_raw['continent'].unique(), default=df_raw['continent'].unique())
    slider_beds = st.sidebar.slider("Rango Camas Hospital", 0.0, float(df_raw['hospital_beds'].max()), (0.0, 10.0))

    df_filtered = df_raw[(df_raw['continent'].isin(selected_cont)) & 
                         (df_raw['hospital_beds'].between(slider_beds[0], slider_beds[1]))]

    # --- TABS: EDA DINÁMICO ---
    tab_uni, tab_bi, tab_ia = st.tabs(["📊 Análisis Univariado", "🔗 Correlaciones (Bivariado)", "🤖 AI Analyst (Groq)"])

    with tab_uni:
        st.subheader("Distribución de Variables")
        col_var = st.selectbox("Selecciona Variable para ver Distribución:", num_cols)
        # Requisito 2.2: Histograma/Boxplot
        fig_dist = px.histogram(df_filtered, x=col_var, marginal="box", title=f"Distribución de {col_var}")
        st.plotly_chart(fig_dist, use_container_width=True)

    with tab_bi:
        st.subheader("Matriz de Correlación Total")
        corr = df_filtered[num_cols].corr()
        fig_corr, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig_corr)

    with tab_ia:
        st.subheader("AI-Driven Insights (Groq)")
        api_key = st.text_input("Introduce tu Groq API Key:", type="password")
        
        if st.button("Generar Insights con IA"):
            if not api_key:
                st.error("Por favor, ingresa la API Key.")
            else:
                client = Groq(api_key=api_key)
                # Resumen estadístico para el Prompt
                stats_summary = df_filtered.describe().to_string()
                
                prompt = f"""
                Actúa como un Consultor Senior de Datos. Analiza el siguiente resumen estadístico de datos de COVID-19:
                {stats_summary}
                
                Responde a:
                1. ¿Qué tendencias principales detectas?
                2. ¿Qué riesgos sugieren los datos?
                3. ¿Qué oportunidad de política pública existe?
                """
                
                completion = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[{"role": "user", "content": prompt}]
                )
                st.markdown("### Análisis de la IA:")
                st.write(completion.choices[0].message.content)

else:
    st.info("Carga un CSV para activar el Sistema de Soporte a la Decisión.")
