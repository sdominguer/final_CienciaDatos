import streamlit as st
import pandas as pd
import plotly.express as px

# Configuración de la página
st.set_page_config(page_title="Pandemic Time Machine", layout="wide", page_icon="🌡️")

st.title("🛰️ Línea de Tiempo Global Pandémica")
st.markdown("Carga tu dataset para activar la máquina del tiempo interactiva.")

uploaded_file = st.file_uploader("Sube tu CSV (dataset_covid_completo.csv)", type="csv")

if uploaded_file is not None:
    @st.cache_data
    def load_full_data(file):
        df = pd.read_csv(file)
        df['date'] = pd.to_datetime(df['date'])
        # Llenado de nulos para evitar errores en gráficos de tamaño
        df[['weekly_count', 'hospital_beds', 'avg_temp']] = df[['weekly_count', 'hospital_beds', 'avg_temp']].fillna(0)
        
        # Crear una columna de texto para la animación (YYYY-MM-DD) 
        # Esto asegura que el mapa recorra toda la historia en orden
        df['fecha_str'] = df['date'].dt.strftime('%Y-%m-%d')
        df = df.sort_values('date')
        return df

    df = load_full_data(uploaded_file)

    # --- FILTROS SIDEBAR ---
    st.sidebar.header("🎯 Controles")
    indicador = st.sidebar.selectbox("Indicador:", df['indicator'].unique())
    
    # Filtro de continentes para el gráfico de barras
    conts = sorted(df['continent'].unique())
    sel_conts = st.sidebar.multiselect("Continentes:", conts, default=conts)

    # --- PROCESAMIENTO PARA ANIMACIÓN ---
    # Filtramos por indicador y continente para que la animación no pese demasiado
    df_anim = df[(df['indicator'] == indicador) & (df['continent'].isin(sel_conts))].copy()

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["🌍 Mapa del Tiempo", "📊 Carrera de Países", "📉 Análisis Climático"])

    with tab1:
        st.subheader(f"Evolución Global de {indicador}")
        st.write("Usa el botón 'Play' para ver la propagación desde el inicio hasta el final.")
        
        fig_map = px.choropleth(
            df_anim,
            locations="ISO3",
            color="weekly_count",
            hover_name="country",
            animation_frame="fecha_str", # Animación por fecha completa
            color_continuous_scale=px.colors.sequential.YlOrRd,
            range_color=[0, df_anim['weekly_count'].quantile(0.95)], # Escala balanceada
            template="plotly_dark",
            height=700
        )
        
        fig_map.update_geos(projection_type="natural earth")
        st.plotly_chart(fig_map, use_container_width=True)

    with tab2:
        st.subheader("Top Países con más impacto")
        st.write("Gráfico de barras animado semana a semana.")
        
        # Agrupamos por fecha y país para el ranking
        fig_race = px.bar(
            df_anim,
            x="weekly_count",
            y="country",
            color="continent",
            animation_frame="fecha_str",
            animation_group="country",
            orientation='h',
            range_x=[0, df_anim['weekly_count'].max()],
            title=f"Ranking Semanal de {indicador}",
            template="plotly_dark",
            height=600
        )
        
        # Ordenar las barras para que la más alta esté arriba
        fig_race.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_race, use_container_width=True)

    with tab3:
        st.subheader("Relación Clima vs Infraestructura")
        
        # Slider para filtrar rango de temperatura
        t_min, t_max = st.slider("Rango de Temperatura (°C):", 
                                 float(df['avg_temp'].min()), 
                                 float(df['avg_temp'].max()), 
                                 (0.0, 30.0))
        
        df_clima = df_anim[(df_anim['avg_temp'] >= t_min) & (df_anim['avg_temp'] <= t_max)]
        
        # Gráfico interactivo de burbujas (Camas como tamaño)
        df_clima['size_viz'] = df_clima['hospital_beds'].apply(lambda x: x if x > 0 else 0.1)
        
        fig_clima = px.scatter(
            df_clima,
            x="avg_temp",
            y="weekly_count",
            size="size_viz",
            color="country",
            hover_data=["date", "hospital_beds"],
            title="¿Cómo afecta la temperatura según la capacidad hospitalaria?",
            template="plotly_dark",
            trendline="ols" # Línea de tendencia estadística
        )
        st.plotly_chart(fig_clima, use_container_width=True)

else:
    st.info("🚀 ¡Hola! Sube tu CSV para empezar la animación histórica.")
