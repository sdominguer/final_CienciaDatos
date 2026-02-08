import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="COVID-19 Scientific Insights", layout="wide", page_icon="🧬")

st.title("🔬 Análisis de Correlaciones Pandémicas")
st.markdown("Este dashboard busca relaciones estadísticas reales entre variables climáticas, demográficas y epidemiológicas.")

uploaded_file = st.file_uploader("Carga el archivo CSV", type="csv")

if uploaded_file is not None:
    @st.cache_data
    def load_data(file):
        df = pd.read_csv(file)
        df['date'] = pd.to_datetime(df['date'])
        # Limpieza estándar
        df = df.sort_values('date')
        # Formato para el mapa
        df['fecha_str'] = df['date'].dt.strftime('%Y-%m-%d')
        return df

    df = load_data(uploaded_file)

    # --- SIDEBAR ---
    st.sidebar.header("Configuración")
    indicador = st.sidebar.selectbox("Indicador Principal:", df['indicator'].unique())
    
    conts = sorted(df['continent'].unique().tolist())
    sel_conts = st.sidebar.multiselect("Continentes:", conts, default=conts)
    
    paises_disp = sorted(df[df["continent"].isin(sel_conts)]["country"].unique().tolist())
    sel_paises = st.sidebar.multiselect("Países para comparar:", paises_disp, default=paises_disp[:5])

    # Filtrado
    df_anim = df[(df['indicator'] == indicador) & (df['continent'].isin(sel_conts))].copy()
    df_filtered = df_anim[df_anim['country'].isin(sel_paises)]

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["🌍 Mapa de Difusión", "📉 Análisis de Correlación", "📊 Comparativa Temporal"])

    with tab1:
        st.subheader(f"Evolución Histórica de {indicador}")
        fig_map = px.choropleth(
            df_anim,
            locations="ISO3",
            color="weekly_count",
            hover_name="country",
            animation_frame="fecha_str",
            color_continuous_scale="Viridis",
            range_color=[0, df_anim['weekly_count'].quantile(0.95)],
            template="plotly_dark",
            height=600
        )
        st.plotly_chart(fig_map, use_container_width=True)

    with tab2:
        st.subheader("🧬 Matriz de Correlación (Pearson)")
        st.write("¿Qué variables están realmente relacionadas? (1.0 es relación perfecta, 0 es ninguna).")
        
        # Seleccionamos solo columnas numéricas relevantes para la matriz
        cols_corr = ['weekly_count', 'avg_temp', 'hospital_beds', 'population', 'rate_14_day']
        # Nos aseguramos de que existan en el df
        cols_existentes = [c for c in cols_corr if c in df_filtered.columns]
        
        corr_matrix = df_filtered[cols_existentes].corr()

        # Dibujar con Matplotlib/Seaborn
        fig_corr, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, ax=ax)
        plt.title("Interdependencia entre Variables")
        st.pyplot(fig_corr)
        
        st.info("""
        **Cómo leer esto:** * Si **avg_temp** y **weekly_count** tienen un número negativo alto, significa que cuando hace más frío, los casos aumentan.
        * Si **hospital_beds** tiene correlación baja con **weekly_count**, la infraestructura no varió según los contagios.
        """)

    with tab3:
        st.subheader("Tendencia por País")
        fig_line = px.line(
            df_filtered, 
            x="date", y="weekly_count", color="country",
            title=f"Línea de tiempo detallada: {indicador}",
            template="plotly_dark",
            markers=True
        )
        st.plotly_chart(fig_line, use_container_width=True)

else:
    st.info("Esperando el archivo CSV...")
