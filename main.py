import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="Pandemic Big Data Analysis", layout="wide", page_icon="📊")

st.title("🧬 Análisis de Correlación Multivariable")
st.markdown("Explora cómo interactúan **todas** las variables de tu dataset (Clima, Población, Salud y COVID).")

uploaded_file = st.file_uploader("Carga tu archivo CSV definitivo", type="csv")

if uploaded_file is not None:
    @st.cache_data
    def load_and_clean(file):
        df = pd.read_csv(file)
        # Convertir fechas
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['fecha_str'] = df['date'].dt.strftime('%Y-%m-%d')
        
        # Seleccionamos solo las columnas que realmente aportan al análisis numérico
        # Eliminamos IDs o columnas de texto que no tienen sentido en una correlación
        cols_drop = ['year', 'week', 'month'] # Evitamos variables temporales redundantes
        df_clean = df.drop(columns=[c for c in cols_drop if c in df.columns])
        
        return df_clean

    df = load_and_clean(uploaded_file)

    # --- SIDEBAR ---
    st.sidebar.header("Filtros Globales")
    indicador = st.sidebar.selectbox("Indicador COVID:", df['indicator'].unique())
    
    conts = sorted(df['continent'].unique().tolist())
    sel_conts = st.sidebar.multiselect("Continentes:", conts, default=conts)
    
    paises_disp = sorted(df[df["continent"].isin(sel_conts)]["country"].unique().tolist())
    sel_paises = st.sidebar.multiselect("Países en análisis:", paises_disp, default=paises_disp[:10])

    # Filtrado dinámico
    df_filtered = df[(df['indicator'] == indicador) & 
                     (df['continent'].isin(sel_conts)) & 
                     (df['country'].isin(sel_paises))]

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["🌍 Mapa del Tiempo", "📉 Correlación Total", "📋 Datos Crudos"])

    with tab1:
        st.subheader(f"Propagación Histórica: {indicador}")
        fig_map = px.choropleth(
            df[df['indicator'] == indicador], # Mapa global para contexto
            locations="ISO3",
            color="weekly_count",
            hover_name="country",
            animation_frame="fecha_str",
            color_continuous_scale="Viridis",
            template="plotly_dark",
            height=600
        )
        st.plotly_chart(fig_map, use_container_width=True)

    with tab2:
        st.subheader("🧪 Matriz de Correlación de Pearson (Todas las Variables)")
        st.write("Esta matriz analiza la fuerza de la relación entre cada par de variables numéricas disponibles.")
        
        # 1. Filtramos solo columnas numéricas
        df_numeric = df_filtered.select_dtypes(include=['float64', 'int64'])
        
        # 2. Calculamos la matriz
        corr_matrix = df_numeric.corr()

        # 3. Visualización con Seaborn
        fig_corr, ax = plt.subplots(figsize=(12, 8))
        # Ajustamos el estilo para que se vea profesional
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            fmt=".2f", 
            cmap='coolwarm', 
            center=0, 
            linewidths=0.5,
            ax=ax
        )
        plt.title(f"Correlación para {indicador} en Países Seleccionados")
        st.pyplot(fig_corr)

        # INSIGHTS AUTOMÁTICOS
        st.markdown("### 💡 ¿Qué estamos viendo?")
        st.info("""
        * **1.0 (Rojo intenso):** Relación positiva perfecta (si una sube, la otra también).
        * **-1.0 (Azul intenso):** Relación negativa perfecta (si una sube, la otra baja).
        * **0.0 (Blanco/Gris):** No hay relación estadística entre las variables.
        """)
        
        # Mostrar las correlaciones más fuertes con el weekly_count
        if 'weekly_count' in corr_matrix.columns:
            st.write(f"**Top correlaciones con {indicador}:**")
            top_corr = corr_matrix['weekly_count'].sort_values(ascending=False)
            st.dataframe(top_corr)

    with tab3:
        st.subheader("Vista Previa del Dataset Filtrado")
        st.dataframe(df_filtered)

else:
    st.info("👋 Sube tu archivo CSV procesado para generar el análisis de correlación.")
