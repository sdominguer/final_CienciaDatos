import streamlit as st
import pandas as pd
import plotly.express as px

# Configuración de la página
st.set_page_config(page_title="Pandemic Time Machine", layout="wide", page_icon="🌍")

st.title("🛰️ Monitor Histórico e Interactivo de Pandemia")
st.markdown("Carga tu dataset para activar las visualizaciones animadas y comparativas.")

uploaded_file = st.file_uploader("Sube tu CSV (dataset_covid_completo.csv)", type="csv")

if uploaded_file is not None:
    @st.cache_data
    def load_full_data(file):
        df = pd.read_csv(file)
        df['date'] = pd.to_datetime(df['date'])
        # Limpieza de nulos para cálculos y visualización
        df[['weekly_count', 'hospital_beds', 'avg_temp']] = df[['weekly_count', 'hospital_beds', 'avg_temp']].fillna(0)
        # Formato de fecha para que la animación sea fluida y ordenada
        df['fecha_str'] = df['date'].dt.strftime('%Y-%m-%d')
        df = df.sort_values('date')
        return df

    df = load_full_data(uploaded_file)

    # --- SIDEBAR (CONTROLES Y FILTROS) ---
    st.sidebar.header("🎯 Filtros de Análisis")
    
    # 1. Seleccionar el indicador
    indicador = st.sidebar.selectbox("Tipo de Dato (Indicador):", df['indicator'].unique())
    
    # 2. Seleccionar continentes
    conts = sorted(df['continent'].unique().tolist())
    sel_conts = st.sidebar.multiselect("Filtrar por Continente:", conts, default=conts)

    # 3. Seleccionar países (¡Aquí regresamos el filtro!)
    paises_disp = sorted(df[df["continent"].isin(sel_conts)]["country"].unique().tolist())
    sel_paises = st.sidebar.multiselect("Seleccionar Países específicos:", paises_disp, default=paises_disp[:5])

    # --- FILTRADO DE DATOS ---
    # df_anim: Usado para el mapa y carrera de barras (incluye todo lo del continente)
    df_anim = df[(df['indicator'] == indicador) & (df['continent'].isin(sel_conts))].copy()
    
    # df_paises: Usado para la línea de tiempo y comparativas específicas
    df_paises = df_anim[df_anim['country'].isin(sel_paises)]

    # --- MÉTRICAS ---
    m1, m2, m3 = st.columns(3)
    m1.metric("Países en análisis", len(sel_paises))
    m2.metric(f"Total {indicador}", f"{df_paises['weekly_count'].sum():,.0f}")
    m3.metric("Temp. Promedio", f"{df_paises['avg_temp'].mean():.1f} °C")

    st.divider()

    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(["🌍 Mapa Animado", "📊 Carrera de Países", "📈 Tendencia Temporal", "🌡️ Clima vs Camas"])

    with tab1:
        st.subheader(f"Propagación Global de {indicador}")
        st.write("Animación histórica de principio a fin.")
        fig_map = px.choropleth(
            df_anim,
            locations="ISO3",
            color="weekly_count",
            hover_name="country",
            animation_frame="fecha_str",
            color_continuous_scale=px.colors.sequential.YlOrRd,
            range_color=[0, df_anim['weekly_count'].quantile(0.98)],
            template="plotly_dark",
            height=650
        )
        st.plotly_chart(fig_map, use_container_width=True)

    with tab2:
        st.subheader("Ranking Dinámico")
        st.write("Países con mayor incidencia semana a semana.")
        fig_race = px.bar(
            df_anim,
            x="weekly_count",
            y="country",
            color="continent",
            animation_frame="fecha_str",
            orientation='h',
            range_x=[0, df_anim['weekly_count'].max() * 1.1],
            template="plotly_dark",
            height=600
        )
        fig_race.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_race, use_container_width=True)

    with tab3:
        st.subheader("Comparativa entre países seleccionados")
        fig_line = px.line(
            df_paises, 
            x="date", y="weekly_count", color="country",
            title=f"Curva de {indicador} a través del tiempo",
            template="plotly_dark",
            markers=True
        )
        st.plotly_chart(fig_line, use_container_width=True)

    with tab4:
        st.subheader("Análisis Estadístico: Clima y Salud")
        
        # Corrección de tamaño para las burbujas
        df_paises['size_viz'] = df_paises['hospital_beds'].apply(lambda x: x if x > 0 else 0.1)
        
        fig_clima = px.scatter(
            df_paises,
            x="avg_temp",
            y="weekly_count",
            size="size_viz",
            color="country",
            hover_name="month_name",
            trendline="ols", # AHORA FUNCIONARÁ con statsmodels instalado
            title="Correlación: ¿Influye la temperatura en los casos?",
            template="plotly_dark"
        )
        st.plotly_chart(fig_clima, use_container_width=True)

else:
    st.info("👋 Sube tu archivo CSV para activar el dashboard.")
