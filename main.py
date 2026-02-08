import streamlit as st
import pandas as pd
import plotly.express as px

# Configuración de la página
st.set_page_config(page_title="COVID-19 Global Monitor", layout="wide", page_icon="🌍")

# --- ESTILO Y TÍTULO ---
st.title("📊 Monitor Global de Pandemia")
st.markdown("Carga tu dataset para analizar la relación entre COVID, clima y salud.")

# --- CARGA DE DATOS ---
uploaded_file = st.file_uploader("Sube tu archivo CSV procesado", type="csv")

if uploaded_file is not None:
    @st.cache_data
    def load_and_prepare(file):
        df = pd.read_csv(file)
        
        # 1. Convertir fechas y limpiar
        df['date'] = pd.to_datetime(df['date'])
        df['hospital_beds'] = df['hospital_beds'].fillna(0)
        df['weekly_count'] = df['weekly_count'].fillna(0)
        
        # 2. Ordenar cronológicamente (Crucial para que las líneas no se crucen)
        df = df.sort_values(['country', 'date'])
        
        # 3. Ordenar meses para el mapa animado (Evita orden alfabético)
        orden_meses = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        df['month_name'] = pd.Categorical(df['month_name'], categories=orden_meses, ordered=True)
        
        return df

    df = load_and_prepare(uploaded_file)

    # --- SIDEBAR (Filtros Globales) ---
    st.sidebar.header("🕹️ Panel de Control")
    
    # Filtro de Indicador (Evita mezclar peras con manzanas)
    indicadores = df['indicator'].unique().tolist()
    sel_ind = st.sidebar.selectbox("Tipo de Dato:", indicadores)

    # Filtro de Geografía
    continentes = sorted(df["continent"].unique().tolist())
    sel_cont = st.sidebar.multiselect("Continentes:", continentes, default=continentes)

    paises_disp = sorted(df[df["continent"].isin(sel_cont)]["country"].unique().tolist())
    sel_paises = st.sidebar.multiselect("Países para comparar:", paises_disp, default=paises_disp[:3])

    # Aplicar filtros
    mask_completa = (df["indicator"] == sel_ind) & (df["continent"].isin(sel_cont))
    df_filtered = df[mask_completa & (df["country"].isin(sel_paises))]
    df_global = df[mask_completa] # Para el mapa usamos todos los países del continente

    # --- MÉTRICAS ---
    st.subheader("📌 Datos clave de la selección")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Registros", len(df_filtered))
    c2.metric("Casos/Muertes Totales", f"{df_filtered['weekly_count'].sum():,.0f}")
    c3.metric("Temp. Media", f"{df_filtered['avg_temp'].mean():.1f} °C")
    c4.metric("Camas (Promedio)", f"{df_filtered['hospital_beds'].mean():.1f}")

    st.divider()

    # --- TABS DE VISUALIZACIÓN ---
    tab1, tab2, tab3 = st.tabs(["📈 Línea de Tiempo", "🌍 Mapa Animado", "🌡️ Clima vs Capacidad"])

    with tab1:
        st.subheader(f"Evolución Semanal: {sel_ind}")
        # El parámetro 'markers' ayuda a identificar puntos sin datos
        fig_line = px.line(
            df_filtered, 
            x="date", 
            y="weekly_count", 
            color="country",
            line_group="country",
            title=f"Tendencia de {sel_ind} (Datos Ordenados)",
            template="plotly_dark",
            markers=True
        )
        st.plotly_chart(fig_line, use_container_width=True)

    with tab2:
        st.subheader("Difusión del Virus en el Tiempo")
        st.write("Presiona 'Play' para ver cómo cambia el mapa mes a mes.")
        
        # Mapa animado usando df_global para ver todo el mapa
        df_mapa = df_global.sort_values('month_name')
        
        fig_map = px.choropleth(
            df_mapa,
            locations="ISO3",
            color="weekly_count",
            hover_name="country",
            animation_frame="month_name",
            color_continuous_scale=px.colors.sequential.YlOrRd,
            title=f"Impacto Mensual: {sel_ind}",
            template="plotly_dark",
            height=600
        )
        # Mantener el rango de color constante para evitar parpadeos
        fig_map.update_layout(coloraxis_cmax=df_mapa['weekly_count'].max())
        st.plotly_chart(fig_map, use_container_width=True)

    with tab3:
        st.subheader("Factores Condicionantes")
        col_left, col_right = st.columns(2)
        
        with col_left:
            # Gráfico de burbujas (Camas de hospital)
            df_scat = df_filtered.copy()
            df_scat['size_ref'] = df_scat['hospital_beds'].apply(lambda x: x if x > 0 else 0.1)
            
            fig_scat = px.scatter(
                df_scat, x="avg_temp", y="weekly_count", 
                size="size_ref", color="country",
                hover_name="month_name",
                hover_data={"size_ref": False, "hospital_beds": True},
                title="Temperatura vs Contagios",
                template="plotly_dark"
            )
            st.plotly_chart(fig_scat, use_container_width=True)
            
        with col_right:
            # Comparativa de Camas
            beds_df = df_filtered.groupby('country')['hospital_beds'].mean().reset_index()
            fig_bar = px.bar(beds_df, x='country', y='hospital_beds', color='country',
                             title="Capacidad Hospitalaria Media")
            st.plotly_chart(fig_bar, use_container_width=True)

    # --- DATOS CRUDOS ---
    with st.expander("📄 Ver Base de Datos"):
        st.dataframe(df_filtered)

else:
    st.info("👋 Por favor, carga tu archivo CSV para generar el dashboard interactivo.")
