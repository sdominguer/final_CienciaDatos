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
    # Lectura inicial
    df_raw = pd.read_csv(uploaded_file)
    
    # 1. Limpieza Interactiva (Requisito 2.1)
    st.sidebar.subheader("Módulo de Limpieza")
    
    if st.sidebar.checkbox("Eliminar registros duplicados"):
        antes = len(df_raw)
        df_raw = df_raw.drop_duplicates()
        st.sidebar.success(f"Eliminados {antes - len(df_raw)} duplicados")

    metodo_nulos = st.sidebar.selectbox(
        "¿Cómo quieres tratar los nulos restantes?",
        ["Mantener nulos", "Llenar con Media", "Llenar con Cero"]
    )
    
    # Aplicar limpieza según selección
    df_clean = df_raw.copy()
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    if metodo_nulos == "Llenar con Media":
        df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].mean())
        st.sidebar.info("Nulos imputados con la media")
    elif metodo_nulos == "Llenar con Cero":
        df_clean[num_cols] = df_clean[num_cols].fillna(0)
        st.sidebar.info("Nulos imputados con cero")

    # 2. Filtros de Navegación
    st.sidebar.subheader("Filtros de Análisis")
    indicador = st.sidebar.selectbox("Indicador:", df_clean['indicator'].unique())
    continentes = st.sidebar.multiselect("Continentes:", df_clean['continent'].unique(), default=df_raw['continent'].unique())
    paises = st.sidebar.multiselect("Países:", df_clean[df_clean['continent'].isin(continentes)]['country'].unique())

    # Dataframe filtrado final
    df_final = df_clean[(df_clean['indicator'] == indicador) & (df_clean['continent'].isin(continentes))]
    if paises:
        df_final = df_final[df_final['country'].isin(paises)]

    # --- TABS: ORGANIZACIÓN DEL PROYECTO ---
    tab_desc, tab_cuant, tab_graf, tab_ia = st.tabs([
        "📖 Análisis Descriptivo", 
        "🔢 Análisis Cuantitativo", 
        "📊 Análisis Gráfico", 
        "🤖 AI Insights (Groq)"
    ])

    # TAB 1: DESCRIPTIVO (Resumen del dataset y nulos)
    with tab_desc:
        st.subheader("Resumen Cualitativo y Estadístico")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Estructura del Dataset:**")
            st.write(f"Filas: {df_final.shape[0]} | Columnas: {df_final.shape[1]}")
            st.write(df_final.dtypes.value_counts())
        with col2:
            st.write("**Conteo de Nulos actual:**")
            st.write(df_final.isnull().sum())
        
        st.write("**Muestra de Datos Limpios:**")
        st.dataframe(df_final.head(10))

    # TAB 2: CUANTITATIVO (Métricas y Correlaciones)
    with tab_cuant:
        st.subheader("Análisis de Correlaciones y Métricas")
        # Métricas clave
        m1, m2, m3 = st.columns(3)
        m1.metric("Letalidad Promedio (%)", f"{df_final['letalidad_pct'].mean():.2f}%")
        m2.metric("Incidencia x 100k", f"{df_final['casos_100k'].mean():.2f}")
        m3.metric("Camas x 100k hab", f"{df_final['camas_por_100k'].mean():.2f}")
        
        # Heatmap
        st.write("**Matriz de Correlación de Pearson:**")
        fig_corr, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df_final[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig_corr)

    # TAB 3: GRÁFICO (Interactividad Plotly)
    with tab_graf:
        st.subheader("Visualización Dinámica")
        c1, c2 = st.columns(2)
        
        with c1:
            st.write("**Distribución de la Letalidad:**")
            fig_hist = px.histogram(df_final, x="letalidad_pct", nbins=30, color="continent", marginal="box")
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with c2:
            st.write("**Relación Temperatura vs Casos:**")
            fig_scat = px.scatter(df_final, x="avg_temp", y="casos_100k", size="camas_por_100k", 
                                 color="continent", hover_name="country", trendline="ols")
            st.plotly_chart(fig_scat, use_container_width=True)

    # TAB 4: INTELIGENCIA ARTIFICIAL (Groq)
    with tab_ia:
        st.subheader("Consultor de IA Virtual")
        st.info("Esta sección utiliza Llama-3 de Groq para interpretar los datos filtrados.")
        
        api_key = st.text_input("Ingresa tu Groq API Key:", type="password")
        
        if st.button("Generar Insights Estratégicos"):
            if not api_key:
                st.warning("Por favor, ingresa una API Key válida de Groq.")
            else:
                try:
                    client = Groq(api_key=api_key)
                    # Resumen para la IA
                    stats = df_final.describe().to_string()
                    
                    prompt = f"""
                    Actúa como un Consultor Senior de Datos. Basado en este resumen estadístico de COVID-19:
                    {stats}
                    
                    Proporciona:
                    1. Un análisis cualitativo de los datos.
                    2. Tres hallazgos cuantitativos clave.
                    3. Una recomendación estratégica para la toma de decisiones.
                    """
                    
                    response = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model="llama3-8b-8192",
                    )
                    st.success("Análisis generado con éxito:")
                    st.markdown(response.choices[0].message.content)
                except Exception as e:
                    st.error(f"Error al conectar con Groq: {e}")

else:
    st.info("Esperando carga de archivo CSV para iniciar el sistema...")
