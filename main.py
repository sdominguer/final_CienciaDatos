import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from groq import Groq

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="DSS - Pandemic Global Analysis", 
    layout="wide", 
    page_icon="🧬"
)

# --- ESTILOS PERSONALIZADOS ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("🛰️ Sistema de Soporte a la Decisión (DSS): COVID-19")
st.markdown("---")

# --- SIDEBAR: MÓDULO ETL (INGESTA Y LIMPIEZA) ---
st.sidebar.header("⚙️ Módulo 1: ETL e Ingesta")
uploaded_file = st.sidebar.file_uploader("Cargar Base de Datos Final (CSV)", type="csv")

if uploaded_file is not None:
    # Lectura de datos
    df_raw = pd.read_csv(uploaded_file)
    
    # 1. Limpieza Interactiva (Requisito 2.1)
    st.sidebar.subheader("Limpieza de Datos")
    
    if st.sidebar.checkbox("Eliminar Duplicados"):
        df_raw = df_raw.drop_duplicates()
        st.sidebar.success("Duplicados eliminados.")

# --- CAMBIO EN EL SIDEBAR ---
    if st.sidebar.checkbox("Tratar Nulos en ISO3 (Códigos País)"):
        # Contamos nulos solo en lo que el usuario está viendo actualmente
        nulos_iso = df_raw['ISO3'].isnull().sum() 
        df_raw['ISO3'] = df_raw['ISO3'].fillna('UNK')
        
        # Si quieres que coincida con la tabla de nulos de la pestaña, 
        # el mensaje debería decir que se limpió la base completa:
        st.sidebar.info(f"Se imputaron {nulos_iso} códigos en el dataset global.")

    metodo_nulos = st.sidebar.selectbox(
        "Método de Imputación (Variables Numéricas):",
        ["Ninguno", "Llenar con Media", "Llenar con Cero"]
    )
    
    df_clean = df_raw.copy()
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    if metodo_nulos == "Llenar con Media":
        df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].mean())
    elif metodo_nulos == "Llenar con Cero":
        df_clean[num_cols] = df_clean[num_cols].fillna(0)

    # 2. Filtros de Navegación
    st.sidebar.subheader("Navegación de Análisis")
    indicador = st.sidebar.selectbox("Seleccione Indicador:", df_clean['indicator'].unique())
    continentes = st.sidebar.multiselect("Filtrar Continentes:", df_clean['continent'].unique(), default=df_clean['continent'].unique())
    
    # Filtrado Final
    df_final = df_clean[(df_clean['indicator'] == indicador) & (df_clean['continent'].isin(continentes))]

    # --- ESTRUCTURA DE PESTAÑAS (Requisito 2.2) ---
    tab_desc, tab_cuant, tab_graf, tab_ia = st.tabs([
        "📖 Análisis Descriptivo", 
        "🔢 Análisis Cuantitativo", 
        "📊 Análisis Gráfico", 
        "🤖 AI Analyst (Groq)"
    ])

    # --- TAB 1: ANÁLISIS DESCRIPTIVO (Cualitativo) ---
    with tab_desc:
        st.subheader("📚 Glosario y Resumen de Datos")
        
        with st.expander("🔍 Interpretación de Variables (Feature Engineering)"):
            st.markdown("""
            * **casos_100k:** Tasa de incidencia. Casos por cada 100,000 habitantes.
            * **camas_por_100k:** Capacidad hospitalaria instalada por cada 100,000 habitantes.
            * **letalidad_pct:** Porcentaje de fallecimientos respecto a la población semanal.
            * **avg_temp:** Temperatura promedio mensual del país.
            """)
        
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Calidad del Dataset (Nulos):**")
            st.write(df_final.isnull().sum())
        with c2:
            st.write("**Estructura de Datos:**")
            st.write(df_final.dtypes.value_counts())
        
        st.write("**Vista Previa del Dataset Limpio:**")
        st.dataframe(df_final.head(15), use_container_width=True)

    # --- TAB 2: ANÁLISIS CUANTITATIVO ---
    with tab_cuant:
        st.subheader("🧬 Correlaciones Estadísticas")
        
        # Métricas de Resumen
        m1, m2, m3 = st.columns(3)
        m1.metric("Letalidad Media (%)", f"{df_final['letalidad_pct'].mean():.4f}%")
        m2.metric("Promedio Incidencia x 100k", f"{df_final['casos_100k'].mean():.2f}")
        m3.metric("Temp. Promedio Selección", f"{df_final['avg_temp'].mean():.1f} °C")
        
        st.write("**Matriz de Correlación de Pearson:**")
        df_corr = df_final[num_cols].corr()
        fig_corr, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df_corr, annot=True, cmap="RdBu_r", center=0, fmt=".2f", ax=ax)
        st.pyplot(fig_corr)
        

    # --- TAB 3: ANÁLISIS GRÁFICO (EDA Dinámico) ---
    with tab_graf:
        st.subheader("📊 Visualizaciones Interactivas")
        
        # Fix para el error de 'size' en Plotly (no permite ceros ni nulos)
        df_viz = df_final.copy()
        df_viz['size_ref'] = df_viz['camas_por_100k'].fillna(0).apply(lambda x: x if x > 0 else 0.1)

        col_a, col_b = st.columns(2)
        
        with col_a:
            st.write("**Distribución de la Tasa de Letalidad:**")
            fig_hist = px.histogram(df_viz, x="letalidad_pct", marginal="box", color_discrete_sequence=['#ff4b4b'])
            st.plotly_chart(fig_hist, use_container_width=True)
            
            
        with col_b:
            st.write("**Relación Temperatura vs Incidencia:**")
            fig_scat = px.scatter(
                df_viz, x="avg_temp", y="casos_100k", 
                size="size_ref", color="continent", 
                hover_name="country",
                hover_data={"size_ref": False, "camas_por_100k": True},
                trendline="ols"
            )
            st.plotly_chart(fig_scat, use_container_width=True)

        st.write("**Impacto Geográfico Global:**")
        fig_map = px.choropleth(
            df_viz, locations="ISO3", color="casos_100k", 
            hover_name="country", color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_map, use_container_width=True)

    # --- TAB 4: IA DRIVEN INSIGHTS (GROQ) ---
    with tab_ia:
        st.subheader("🤖 Analista Virtual (Llama-3 Groq)")
        st.markdown("Este módulo utiliza inteligencia artificial para interpretar los resultados estadísticos.")
        
        api_key = st.text_input("Ingrese su API Key de Groq:", type="password")
        
        if st.button("Generar Insights con IA"):
            if not api_key:
                st.error("Por favor, ingrese la API Key para continuar.")
            else:
                try:
                    client = Groq(api_key=api_key)
                    # Resumen estadístico para el LLM
                    resumen_desc = df_final.describe().to_string()
                    
                    prompt = f"""
                    Eres un Consultor Senior de Datos. Analiza el siguiente resumen estadístico de COVID-19:
                    {resumen_desc}
                    
                    Basado en estos datos, redacta un informe ejecutivo breve que incluya:
                    1. Interpretación de la relación entre infraestructura (camas_por_100k) y letalidad.
                    2. Análisis de cómo la temperatura (avg_temp) parece afectar los casos.
                    3. Dos recomendaciones estratégicas para políticas de salud pública.
                    """
                    
                    completion = client.chat.completions.create(
                        model="llama3-8b-8192",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.5
                    )
                    st.markdown("### 📝 Reporte de Consultoría:")
                    st.write(completion.choices[0].message.content)
                    
                except Exception as e:
                    st.error(f"Error al conectar con la API de Groq: {e}")

else:
    st.info("👋 Bienvenid@. Por favor, cargue el archivo CSV final para habilitar el dashboard.")

# --- PIE DE PÁGINA ---
st.markdown("---")
st.caption("Proyecto Final: Sistema de Soporte a la Decisión | Consultoría de Datos Senior")
