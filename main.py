import streamlit as st
import pandas as pd
import plotly.express as px

# Configuración de la página
st.set_page_config(page_title="COVID-19 Data Explorer", layout="wide", page_icon="🦠")

# --- TÍTULO Y CARGA DE ARCHIVO ---
st.title("🦠 Analizador Interactivo: COVID, Clima y Salud")
st.markdown("Sube tu archivo CSV procesado para comenzar el análisis.")

# Componente para subir el archivo
uploaded_file = st.file_uploader("Elige tu archivo CSV (ej. dataset_covid_completo.csv)", type="csv")

if uploaded_file is not None:
    # Cargar los datos
    @st.cache_data
    def load_data(file):
        df = pd.read_csv(file)
        # Intentar convertir fecha si existe
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df

    df = load_data(uploaded_file)
    
    # Validar que las columnas necesarias existan
    required_cols = ['country', 'continent', 'ISO3', 'avg_temp', 'hospital_beds']
    missing_cols = [c for c in required_cols if c not in df.columns]
    
    if missing_cols:
        st.warning(f"Faltan algunas columnas en el archivo: {', '.join(missing_cols)}. Algunos gráficos podrían no funcionar.")

    # --- SIDEBAR (Filtros) ---
    st.sidebar.header("⚙️ Configuración")
    
    continentes = sorted(df["continent"].unique().tolist())
    sel_continentes = st.sidebar.multiselect("Continentes", continentes, default=continentes)

    # Filtrar países según continente seleccionado
    paises_disp = sorted(df[df["continent"].isin(sel_continentes)]["country"].unique().tolist())
    sel_paises = st.sidebar.multiselect("Países", paises_disp, default=paises_disp[:2])

    # Aplicar filtros al DataFrame
    df_filtered = df[(df["continent"].isin(sel_continentes)) & (df["country"].isin(sel_paises))]

    # --- MÉTRICAS ---
    st.subheader("📌 Resumen de Selección")
    m1, m2, m3, m4 = st.columns(4)
    
    # Cálculo dinámico de métricas
    total_casos = df_filtered[df_filtered['indicator'] == 'cases']['weekly_count'].sum() if 'indicator' in df.columns else 0
    temp_min = df_filtered['avg_temp'].min()
    temp_max = df_filtered['avg_temp'].max()
    camas_promedio = df_filtered['hospital_beds'].mean()

    m1.metric("Total Casos", f"{total_casos:,.0f}")
    m2.metric("Temp Mínima", f"{temp_min:.1f} °C")
    m3.metric("Temp Máxima", f"{temp_max:.1f} °C")
    m4.metric("Camas (Promedio)", f"{camas_promedio:.2f}")

    st.divider()

    # --- VISUALIZACIONES ---
    tab1, tab2, tab3 = st.tabs(["📈 Tendencias", "🌡️ Clima vs Salud", "📊 Datos Crudos"])

    with tab1:
        st.subheader("Evolución de Casos Semanales")
        if 'date' in df_filtered.columns and 'weekly_count' in df_filtered.columns:
            fig_line = px.line(df_filtered, x='date', y='weekly_count', color='country',
                               title="Curva de contagios por país",
                               template="plotly_dark")
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("No hay datos de fecha o conteo semanal para este gráfico.")

    with tab2:
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("Relación Temperatura/Casos")
            fig_scat = px.scatter(df_filtered, x='avg_temp', y='weekly_count', 
                                  color='country', size='hospital_beds',
                                  hover_name='month_name',
                                  title="Temperatura vs Contagios (Tamaño = Camas)")
            st.plotly_chart(fig_scat, use_container_width=True)
            
        with col_b:
            st.subheader("Distribución de Camas por País")
            # Promedio de camas por país en la selección
            beds_df = df_filtered.groupby('country')['hospital_beds'].mean().reset_index()
            fig_bar = px.bar(beds_df, x='country', y='hospital_beds', 
                             color='country', title="Camas de hospital por país")
            st.plotly_chart(fig_bar, use_container_width=True)

    with tab3:
        st.subheader("Tabla de Datos Filtrada")
        st.dataframe(df_filtered, use_container_width=True)
        
        # Botón para descargar lo que se filtró
        csv_download = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button("Descargar selección como CSV", data=csv_download, file_name="filtro_custom.csv", mime="text/csv")

else:
    # Pantalla de bienvenida cuando no hay archivo
    st.info("👋 ¡Bienvenido! Por favor, carga el archivo CSV que generamos en el paso anterior para visualizar los datos.")
    
    # Imagen de ejemplo de cómo debe verse un dashboard interactivo
    st.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=200)
    
    st.markdown("""
    ### Instrucciones:
    1. Asegúrate de tener el archivo `dataset_covid_completo.csv`.
    2. Arrástralo al recuadro de arriba.
    3. Usa el menú de la izquierda para filtrar continentes y países.
    """)
