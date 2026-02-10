import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from groq import Groq
from datetime import datetime
import geopandas as gpd
import folium
from streamlit_folium import st_folium

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="DSS - Pandemic Global Analysis", 
    layout="wide", 
    page_icon="🧬"
)

# --- PALETA DE COLORES (DARK MODE) ---
# Background: #0B0F19 (deep navy-black)
# Surface:    #111827 (dark card)
# Elevated:   #1F2937 (raised surface)
# Border:     #1E293B (subtle border)
# Accent:     #22D3EE (vivid cyan)
# Accent 2:   #818CF8 (indigo)
# Text:       #F1F5F9 (off-white)
# Muted:      #64748B (slate)

# --- ESTILOS PERSONALIZADOS ---
st.markdown("""
<style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, .main, [data-testid="stAppViewContainer"],
    [data-testid="stApp"], .stApp {
        background-color: #0B0F19 !important;
        font-family: 'Inter', sans-serif;
        color: #E2E8F0 !important;
    }

    /* All text defaults */
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
    label, .stText, p {
        color: #CBD5E1 !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #F1F5F9 !important;
    }
    strong, b {
        color: #F1F5F9 !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D1117 0%, #111827 100%) !important;
        border-right: 1px solid #1E293B !important;
    }
    [data-testid="stSidebar"] * {
        color: #CBD5E1 !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2, [data-testid="stSidebar"] .stMarkdown h3 {
        color: #F1F5F9 !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] .stFileUploader label,
    [data-testid="stSidebar"] .stCheckbox label {
        color: #64748B !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        font-size: 11px !important;
        letter-spacing: 0.05em;
    }
    [data-testid="stSidebar"] .stSelectbox > div > div,
    [data-testid="stSidebar"] .stMultiSelect > div > div {
        background: #1F2937 !important;
        border-color: #334155 !important;
        color: #E2E8F0 !important;
    }
    [data-testid="stSidebar"] .stFileUploader > div {
        border: 2px dashed #334155 !important;
        border-radius: 12px !important;
        background: rgba(255,255,255,0.02) !important;
    }
    [data-testid="stSidebar"] .stFileUploader > div:hover {
        border-color: #22D3EE !important;
        background: rgba(34,211,238,0.04) !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #1E293B !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: #111827;
        border-radius: 14px;
        padding: 5px;
        border: 1px solid #1E293B;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        color: #64748B !important;
        background: transparent !important;
        border: none !important;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: #1F2937 !important;
        color: #CBD5E1 !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #22D3EE 0%, #818CF8 100%) !important;
        color: #0B0F19 !important;
        font-weight: 700 !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }

    /* ── Metric Cards ── */
    [data-testid="stMetric"] {
        background: #111827;
        padding: 20px 24px;
        border-radius: 16px;
        border: 1px solid #1E293B;
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: #22D3EE;
        box-shadow: 0 4px 20px rgba(34,211,238,0.1);
    }
    [data-testid="stMetric"] label {
        color: #64748B !important;
        font-weight: 600 !important;
        font-size: 12px !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #F1F5F9 !important;
        font-weight: 800 !important;
        font-size: 28px !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        border: 1px solid #334155 !important;
        background: #1F2937 !important;
        color: #CBD5E1 !important;
        padding: 8px 16px !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        border-color: #22D3EE !important;
        color: #22D3EE !important;
        background: rgba(34,211,238,0.08) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(34,211,238,0.15) !important;
    }

    /* ── Expanders ── */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        font-size: 14px !important;
        color: #E2E8F0 !important;
        background: #111827 !important;
        border-radius: 12px !important;
        border: 1px solid #1E293B !important;
    }
    [data-testid="stExpander"] {
        background: #111827 !important;
        border: 1px solid #1E293B !important;
        border-radius: 12px !important;
    }

    /* ── DataFrame ── */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #1E293B;
    }

    /* ── Chat ── */
    [data-testid="stChatMessage"] {
        background: #111827 !important;
        border-radius: 16px !important;
        padding: 16px 20px !important;
        margin-bottom: 12px !important;
        border: 1px solid #1E293B !important;
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] li {
        color: #CBD5E1 !important;
    }

    [data-testid="stChatInput"] > div {
        border-radius: 16px !important;
        border: 2px solid #1E293B !important;
        background: #111827 !important;
        transition: border-color 0.2s ease;
    }
    [data-testid="stChatInput"] > div:focus-within {
        border-color: #22D3EE !important;
        box-shadow: 0 0 0 3px rgba(34,211,238,0.1) !important;
    }
    [data-testid="stChatInput"] input,
    [data-testid="stChatInput"] textarea {
        color: #E2E8F0 !important;
    }

    /* ── Text Input (API key, etc.) ── */
    .stTextInput > div > div {
        background: #1F2937 !important;
        border-color: #334155 !important;
        border-radius: 12px !important;
        color: #E2E8F0 !important;
    }
    .stTextInput input {
        color: #E2E8F0 !important;
    }

    /* ── Selectbox / Multiselect ── */
    .stSelectbox > div > div, .stMultiSelect > div > div {
        border-radius: 12px !important;
        background: #1F2937 !important;
        border-color: #334155 !important;
    }

    /* ── Custom Components ── */
    .hero-header {
        background: linear-gradient(135deg, #0D1117 0%, #111827 40%, #0D1117 100%);
        padding: 40px 48px;
        border-radius: 24px;
        color: white;
        margin-bottom: 32px;
        position: relative;
        overflow: hidden;
        border: 1px solid #1E293B;
    }
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(34,211,238,0.12) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-header::after {
        content: '';
        position: absolute;
        bottom: -40%;
        left: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(129,140,248,0.08) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-header h1 {
        font-size: 32px;
        font-weight: 800;
        margin: 0 0 8px 0;
        position: relative;
        z-index: 1;
        letter-spacing: -0.5px;
        color: #F1F5F9;
    }
    .hero-header p {
        color: #64748B;
        font-size: 16px;
        margin: 0;
        position: relative;
        z-index: 1;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(34,211,238,0.1);
        color: #22D3EE;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.08em;
        margin-bottom: 16px;
        position: relative;
        z-index: 1;
        border: 1px solid rgba(34,211,238,0.2);
    }

    .chat-header-card {
        background: linear-gradient(135deg, #0D1117 0%, #111827 100%);
        padding: 32px;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
        border: 1px solid #1E293B;
    }
    .chat-header-card::before {
        content: '';
        position: absolute;
        top: -30%;
        right: -15%;
        width: 200px;
        height: 200px;
        background: radial-gradient(circle, rgba(34,211,238,0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    .chat-header-card h2 {
        font-size: 24px;
        font-weight: 700;
        margin: 0 0 8px 0;
        position: relative;
        z-index: 1;
        color: #F1F5F9;
    }
    .chat-header-card p {
        color: #64748B;
        font-size: 14px;
        margin: 0;
        position: relative;
        z-index: 1;
    }

    .section-card {
        background: #111827;
        padding: 28px;
        border-radius: 20px;
        border: 1px solid #1E293B;
        margin-bottom: 20px;
    }
    .section-title {
        font-size: 18px;
        font-weight: 700;
        color: #F1F5F9;
        margin: 0 0 4px 0;
    }
    .section-subtitle {
        font-size: 13px;
        color: #64748B;
        margin: 0 0 20px 0;
    }

    .welcome-container {
        text-align: center;
        padding: 80px 40px;
    }
    .welcome-icon {
        width: 80px;
        height: 80px;
        background: linear-gradient(135deg, #22D3EE 0%, #818CF8 100%);
        border-radius: 24px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 36px;
        margin-bottom: 28px;
        box-shadow: 0 8px 32px rgba(34,211,238,0.25);
    }
    .welcome-container h2 {
        font-size: 28px;
        font-weight: 800;
        color: #F1F5F9;
        margin: 0 0 12px 0;
        letter-spacing: -0.5px;
    }
    .welcome-container .subtitle {
        font-size: 16px;
        color: #64748B;
        max-width: 500px;
        margin: 0 auto 40px auto;
        line-height: 1.6;
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 16px;
        max-width: 560px;
        margin: 0 auto;
    }
    .feature-item {
        background: #111827;
        border: 1px solid #1E293B;
        border-radius: 16px;
        padding: 20px;
        text-align: left;
        transition: all 0.2s ease;
    }
    .feature-item:hover {
        border-color: #22D3EE;
        box-shadow: 0 4px 20px rgba(34,211,238,0.08);
    }
    .feature-item .icon {
        font-size: 24px;
        margin-bottom: 8px;
    }
    .feature-item .label {
        font-size: 14px;
        font-weight: 600;
        color: #E2E8F0;
        margin: 0 0 4px 0;
    }
    .feature-item .desc {
        font-size: 12px;
        color: #64748B;
        margin: 0;
    }

    .footer {
        text-align: center;
        padding: 24px 0 8px 0;
        color: #475569;
        font-size: 13px;
    }
    .footer a {
        color: #22D3EE;
        text-decoration: none;
        font-weight: 600;
    }

    /* Hide default Streamlit header gap */
    .block-container {
        padding-top: 2rem !important;
    }

    /* Alerts */
    .stAlert > div {
        border-radius: 12px !important;
        background: #1F2937 !important;
        border-color: #334155 !important;
    }

    /* Horizontal rule */
    hr {
        border-color: #1E293B !important;
    }

    /* Caption */
    .stCaption, [data-testid="stCaption"] {
        color: #475569 !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0B0F19; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #475569; }
</style>
""", unsafe_allow_html=True)

# --- HERO HEADER ---
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">COVID-19 ANALYTICS</div>
    <h1>Sistema de Soporte a la Decision (DSS simon)</h1>
    <p>Plataforma integral de analisis epidemiologico con inteligencia artificial</p>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR: MÓDULO ETL (INGESTA Y LIMPIEZA) ---
st.sidebar.markdown("### Modulo ETL e Ingesta")
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Cargar Base de Datos Final (CSV)", type="csv")

if uploaded_file is not None:
    # Lectura de datos
    df_raw = pd.read_csv(uploaded_file)
    
    # 1. Limpieza Interactiva (Requisito 2.1)
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Limpieza de Datos")
    
    if st.sidebar.checkbox("Eliminar Duplicados"):
        df_raw = df_raw.drop_duplicates()
        st.sidebar.success("Duplicados eliminados.")

    if st.sidebar.checkbox("Tratar Nulos en ISO3 (Codigos Pais)"):
        nulos_iso = df_raw['ISO3'].isnull().sum() 
        df_raw['ISO3'] = df_raw['ISO3'].fillna('UNK')
        st.sidebar.info(f"Se imputaron {nulos_iso} codigos en el dataset global.")

    metodo_nulos = st.sidebar.selectbox(
        "Metodo de Imputacion (Variables Numericas):",
        ["Ninguno", "Llenar con Media", "Llenar con Cero"]
    )
    
    df_clean = df_raw.copy()
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    if metodo_nulos == "Llenar con Media":
        df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].mean())
    elif metodo_nulos == "Llenar con Cero":
        df_clean[num_cols] = df_clean[num_cols].fillna(0)

    # 2. Filtros de Navegación
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Navegacion de Analisis")
    indicador = st.sidebar.selectbox("Seleccione Indicador:", df_clean['indicator'].unique())
    continentes = st.sidebar.multiselect("Filtrar Continentes:", df_clean['continent'].unique(), default=df_clean['continent'].unique())
    
    # Filtrado Final
    df_final = df_clean[(df_clean['indicator'] == indicador) & (df_clean['continent'].isin(continentes))]

    # --- ESTRUCTURA DE PESTAÑAS (Requisito 2.2) ---
    tab_desc, tab_cuant, tab_graf, tab_ia = st.tabs([
        "  📋 Analisis Descriptivo  ", 
        "  📊 Analisis Cuantitativo  ", 
        "  📈 Visualizaciones Dinamicas  ", 
        "  🤖 AI Analyst  "
    ])

    # --- TAB 1: ANÁLISIS DESCRIPTIVO (Cualitativo) ---
    with tab_desc:
        st.markdown("""
        <div class="section-card">
            <p class="section-title">Glosario y Resumen de Datos</p>
            <p class="section-subtitle">Descripcion general del dataset, calidad y estructura de variables</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("Interpretacion de Variables (Feature Engineering)", expanded=False):
            st.markdown("""
            | Variable | Descripcion |
            |---|---|
            | `casos_100k` | Tasa de incidencia. Casos por cada 100,000 habitantes. |
            | `camas_por_100k` | Capacidad hospitalaria instalada por cada 100,000 habitantes. |
            | `letalidad_pct` | Porcentaje de fallecimientos respecto a la poblacion semanal. |
            | `avg_temp` | Temperatura promedio mensual del pais. |
            """)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Calidad del Dataset (Nulos)**")
            st.dataframe(df_final.isnull().sum().rename("Nulos"), use_container_width=True)
        with c2:
            st.markdown("**Estructura de Datos**")
            st.dataframe(df_final.dtypes.value_counts().rename("Cantidad"), use_container_width=True)
        
        st.markdown("**Vista Previa del Dataset Limpio**")
        st.dataframe(df_final.head(15), use_container_width=True)

    # --- TAB 2: ANÁLISIS CUANTITATIVO ---
    with tab_cuant:
        st.markdown("""
        <div class="section-card">
            <p class="section-title">Correlaciones y Estadisticas</p>
            <p class="section-subtitle">Metricas de resumen, distribuciones y matriz de correlacion</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Métricas de Resumen
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Letalidad Media (%)", f"{df_final['letalidad_pct'].mean():.4f}%")
        m2.metric("Incidencia x 100k", f"{df_final['casos_100k'].mean():.2f}")
        m3.metric("Temp. Promedio", f"{df_final['avg_temp'].mean():.1f} °C")
        m4.metric("Paises Analizados", f"{df_final['country'].nunique()}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Sección de estadísticas descriptivas
        col_stat1, col_stat2 = st.columns(2)
        
        with col_stat1:
            st.markdown("**📊 Estadisticas Descriptivas**")
            stats_df = df_final[num_cols].describe().T
            stats_df = stats_df[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
            stats_df.columns = ['Media', 'Desv.Est', 'Min', 'Q1', 'Mediana', 'Q3', 'Max']
            st.dataframe(stats_df.round(2), use_container_width=True)
        
        with col_stat2:
            st.markdown("**📦 Box Plot - Distribucion por Continente**")
            var_boxplot = st.selectbox("Variable para Box Plot:", 
                                       ['letalidad_pct', 'casos_100k', 'camas_por_100k', 'avg_temp'],
                                       key='boxplot_var')
            
            fig_box = px.box(
                df_final, 
                x="continent", 
                y=var_boxplot,
                color="continent",
                color_discrete_sequence=['#22D3EE', '#818CF8', '#34D399', '#F472B6', '#FBBF24', '#FB923C'],
                template="plotly_dark"
            )
            fig_box.update_layout(
                plot_bgcolor='#111827',
                paper_bgcolor='#0B0F19',
                font=dict(family="Inter", size=11, color='#94A3B8'),
                margin=dict(t=10, b=10, l=10, r=10),
                xaxis=dict(gridcolor='#1E293B', title=""),
                yaxis=dict(gridcolor='#1E293B'),
                showlegend=False,
                height=300
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        st.markdown("---")
        st.markdown("**🔗 Matriz de Correlacion de Pearson**")
        df_corr = df_final[num_cols].corr()
        fig_corr, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            df_corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax,
            linewidths=0.5, linecolor='#1E293B',
            cbar_kws={"shrink": 0.8},
            annot_kws={"color": "#E2E8F0", "fontsize": 10}
        )
        ax.set_facecolor('#111827')
        fig_corr.patch.set_facecolor('#0B0F19')
        ax.tick_params(colors='#94A3B8', labelsize=10)
        for text in ax.get_xticklabels() + ax.get_yticklabels():
            text.set_color('#94A3B8')
        cbar = ax.collections[0].colorbar
        if cbar:
            cbar.ax.tick_params(colors='#94A3B8')
            cbar.outline.set_edgecolor('#1E293B')
        plt.tight_layout()
        st.pyplot(fig_corr)

    # --- TAB 3: VISUALIZACIONES DINÁMICAS (MEJORADO) ---
    with tab_graf:
        st.markdown("""
        <div class="section-card">
            <p class="section-title">Exploracion Visual Interactiva</p>
            <p class="section-subtitle">Graficos dinamicos con controles de filtrado y comparacion</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Fix para el error de 'size' en Plotly
        df_viz = df_final.copy()
        df_viz['size_ref'] = df_viz['camas_por_100k'].fillna(0).apply(lambda x: x if x > 0 else 0.1)
        
        # --- SECCIÓN 1: EVOLUCIÓN TEMPORAL ---
        if 'date' in df_viz.columns or 'year_week' in df_viz.columns:
            st.markdown("### ⏱️ Evolucion Temporal")
            
            # Determinar columna temporal
            time_col = 'date' if 'date' in df_viz.columns else 'year_week'
            
            # Convertir a datetime si es necesario
            if time_col == 'date':
                try:
                    df_viz[time_col] = pd.to_datetime(df_viz[time_col])
                except:
                    pass
            
            col_time1, col_time2 = st.columns([1, 3])
            
            with col_time1:
                paises_time = st.multiselect(
                    "Selecciona Paises:",
                    options=sorted(df_viz['country'].unique()),
                    default=sorted(df_viz['country'].unique())[:5] if len(df_viz['country'].unique()) >= 5 else sorted(df_viz['country'].unique()),
                    key='time_countries'
                )
                
                var_time = st.selectbox(
                    "Variable a graficar:",
                    ['casos_100k', 'letalidad_pct', 'camas_por_100k'],
                    key='time_var'
                )
            
            with col_time2:
                if paises_time:
                    df_time = df_viz[df_viz['country'].isin(paises_time)].sort_values(time_col)
                    
                    fig_time = px.line(
                        df_time,
                        x=time_col,
                        y=var_time,
                        color='country',
                        title=f"Evolucion de {var_time} en el tiempo",
                        color_discrete_sequence=['#22D3EE', '#818CF8', '#34D399', '#F472B6', '#FBBF24'],
                        template="plotly_dark"
                    )
                    fig_time.update_layout(
                        plot_bgcolor='#111827',
                        paper_bgcolor='#0B0F19',
                        font=dict(family="Inter", size=12, color='#94A3B8'),
                        margin=dict(t=40, b=20),
                        xaxis=dict(gridcolor='#1E293B'),
                        yaxis=dict(gridcolor='#1E293B'),
                        hovermode='x unified',
                        height=400
                    )
                    st.plotly_chart(fig_time, use_container_width=True)
                else:
                    st.info("Selecciona al menos un pais para visualizar")
            
            st.markdown("---")
        
        # --- SECCIÓN 2: COMPARACIÓN DE PAÍSES (TOP/BOTTOM) ---
        st.markdown("### 🏆 Ranking de Paises")
        
        col_rank1, col_rank2 = st.columns(2)
        
        with col_rank1:
            var_ranking = st.selectbox(
                "Variable para Ranking:",
                ['letalidad_pct', 'casos_100k', 'camas_por_100k'],
                key='rank_var'
            )
            
            n_paises = st.slider("Numero de paises:", 5, 20, 10, key='n_rank')
        
        with col_rank2:
            tipo_ranking = st.radio(
                "Tipo de Ranking:",
                ['Top (Mayores)', 'Bottom (Menores)'],
                horizontal=True,
                key='rank_type'
            )
        
        # Calcular ranking
        df_ranking = df_viz.groupby('country')[var_ranking].mean().reset_index()
        df_ranking = df_ranking.sort_values(var_ranking, ascending=(tipo_ranking == 'Bottom (Menores)'))
        df_ranking = df_ranking.head(n_paises)
        
        fig_rank = px.bar(
            df_ranking,
            y='country',
            x=var_ranking,
            orientation='h',
            title=f"{tipo_ranking} {n_paises} paises por {var_ranking}",
            color=var_ranking,
            color_continuous_scale=[[0, '#0D3B4F'], [0.5, '#22D3EE'], [1, '#818CF8']],
            template="plotly_dark"
        )
        fig_rank.update_layout(
            plot_bgcolor='#111827',
            paper_bgcolor='#0B0F19',
            font=dict(family="Inter", size=12, color='#94A3B8'),
            margin=dict(t=40, b=20),
            xaxis=dict(gridcolor='#1E293B'),
            yaxis=dict(gridcolor='#1E293B', title=""),
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_rank, use_container_width=True)
        
        st.markdown("---")
        
        # --- SECCIÓN 3: ANÁLISIS BIVARIADO AVANZADO ---
        st.markdown("### 🔍 Analisis de Relaciones")
        
        col_scatter1, col_scatter2 = st.columns([1, 2])
        
        with col_scatter1:
            var_x = st.selectbox("Variable X:", ['avg_temp', 'camas_por_100k', 'casos_100k'], key='scatter_x')
            var_y = st.selectbox("Variable Y:", ['casos_100k', 'letalidad_pct', 'camas_por_100k'], key='scatter_y')
            color_by = st.selectbox("Colorear por:", ['continent', 'country'], key='scatter_color')
            show_trendline = st.checkbox("Mostrar linea de tendencia", value=True, key='scatter_trend')
        
        with col_scatter2:
            fig_scatter = px.scatter(
                df_viz,
                x=var_x,
                y=var_y,
                size='size_ref',
                color=color_by,
                hover_name='country',
                hover_data={'size_ref': False, 'camas_por_100k': True, 'continent': True},
                trendline='ols' if show_trendline else None,
                title=f"Relacion entre {var_x} y {var_y}",
                color_discrete_sequence=['#22D3EE', '#818CF8', '#34D399', '#F472B6', '#FBBF24', '#FB923C'],
                template="plotly_dark"
            )
            fig_scatter.update_layout(
                plot_bgcolor='#111827',
                paper_bgcolor='#0B0F19',
                font=dict(family="Inter", size=12, color='#94A3B8'),
                margin=dict(t=40, b=20),
                xaxis=dict(gridcolor='#1E293B'),
                yaxis=dict(gridcolor='#1E293B'),
                height=450
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.markdown("---")
        
        # --- SECCIÓN 4: DISTRIBUCIONES Y MAPAS ---
        col_dist1, col_dist2 = st.columns(2)
        
        with col_dist1:
            st.markdown("**📊 Distribucion con Histograma**")
            var_hist = st.selectbox("Variable:", ['letalidad_pct', 'casos_100k', 'avg_temp'], key='hist_var')
            
            fig_hist = px.histogram(
                df_viz,
                x=var_hist,
                marginal='box',
                nbins=30,
                color_discrete_sequence=['#22D3EE'],
                template="plotly_dark"
            )
            fig_hist.update_layout(
                plot_bgcolor='#111827',
                paper_bgcolor='#0B0F19',
                font=dict(family="Inter", size=11, color='#94A3B8'),
                margin=dict(t=10, b=10),
                xaxis=dict(gridcolor='#1E293B'),
                yaxis=dict(gridcolor='#1E293B'),
                height=350,
                showlegend=False
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col_dist2:
            st.markdown("**🌡️ Heatmap de Correlaciones por Continente**")
            cont_heatmap = st.selectbox("Continente:", df_viz['continent'].unique(), key='heatmap_cont')
            
            df_cont = df_viz[df_viz['continent'] == cont_heatmap]
            corr_cont = df_cont[num_cols].corr()
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=corr_cont.values,
                x=corr_cont.columns,
                y=corr_cont.columns,
                colorscale=[[0, '#0D3B4F'], [0.5, '#22D3EE'], [1, '#818CF8']],
                text=corr_cont.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 10, "color": "#E2E8F0"}
            ))
            fig_heatmap.update_layout(
                plot_bgcolor='#111827',
                paper_bgcolor='#0B0F19',
                font=dict(family="Inter", size=10, color='#94A3B8'),
                margin=dict(t=10, b=10, l=10, r=10),
                height=350
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        st.markdown("---")
        
        # --- SECCIÓN 5: MAPA GEOGRÁFICO ---
        st.markdown("### 🌍 Impacto Geografico Global")
        
        col_map1, col_map2 = st.columns([1, 3])

        
        with col_map1:
            var_map = st.selectbox(
                "Variable para Mapa:",
                ['casos_100k', 'letalidad_pct', 'camas_por_100k'],
                key='map_var'
            )
            
            st.info(f"**Visualizando:** {var_map}\n\n"
                   f"**Max:** {df_viz[var_map].max():.2f}\n"
                   f"**Min:** {df_viz[var_map].min():.2f}\n"
                   f"**Media:** {df_viz[var_map].mean():.2f}")
        
        with col_map2:
            fig_map = px.choropleth(
                df_viz,
                locations='ISO3',
                color=var_map,
                hover_name='country',
                hover_data={
                    'ISO3': False,
                    var_map: ':.2f',
                    'continent': True
                },
                color_continuous_scale=[[0, '#0D3B4F'], [0.5, '#22D3EE'], [1, '#818CF8']],
                template="plotly_dark"
            )
            fig_map.update_layout(
                paper_bgcolor='#0B0F19',
                geo=dict(
                    bgcolor='#0B0F19',
                    showframe=False,
                    showcoastlines=True,
                    coastlinecolor='#334155',
                    landcolor='#1F2937',
                    lakecolor='#111827',
                    oceancolor='#0B0F19',
                    showocean=True,
                    projection_type='natural earth'
                ),
                font=dict(family="Inter", size=11, color='#94A3B8'),
                margin=dict(t=0, b=0, l=0, r=0),
                height=450,
                coloraxis_colorbar=dict(
                    tickfont=dict(color='#94A3B8'),
                    title=dict(font=dict(color='#94A3B8', size=10))
                )
            )
            st.plotly_chart(fig_map, use_container_width=True)
        
        st.markdown("---")

        st.markdown("### 🗺️ Mapa GIS Avanzado (GeoPandas + Folium)")

        # =============================
        # MAPA INTERACTIVO + CLICK + HOVER + TIME
        # =============================

        col_map1, col_map2 = st.columns([1, 3])
        with col_map1:
            var_map2 = st.selectbox(
                "Variable para Mapa:",
                ['casos_100k', 'letalidad_pct', 'camas_por_100k'],
                key='map_var'
            )
            st.info(f"**Visualizando:** {var_map2}\n\n"
                f"**Max:** {df_viz[var_map2].max():.2f}\n"
                f"**Min:** {df_viz[var_map2].min():.2f}\n"
                f"**Media:** {df_viz[var_map2].mean():.2f}")
        with col_map2:

                    # TIME SLIDER
            fechas = sorted(df_viz["date"].unique())
            fecha_sel = st.select_slider("Fecha del mapa", options=fechas)
            
            # DATA DE LA FECHA
            df_fecha = df_viz[df_viz["date"] == fecha_sel]
            
            # AGRUPACIÓN PARA MAPA
            df_map = df_fecha.groupby(["ISO3", "country"]).agg({
                var_map: "mean",
                "casos_100k": "sum",
                "letalidad_pct": "sum"
            }).reset_index()
            
            pais_click = None  # valor inicial
            
            try:
                world = gpd.read_file("countries.geojson")
            
                df_fecha["ISO3"] = df_fecha["ISO3"].str.upper().str.strip()
            
                gdf = world.merge(
                    df_map,
                    left_on="ISO3166-1-Alpha-3",
                    right_on="ISO3",
                    how="left"
                )
            
                m = folium.Map(
                    location=[20, 0],
                    zoom_start=2,
                    tiles="cartodb positron"
                )
            
                folium.Choropleth(
                    geo_data=gdf.to_json(),
                    data=gdf,
                    columns=["ISO3", var_map],
                    key_on="feature.properties.ISO3166-1-Alpha-3",
                    fill_color="YlGnBu",
                    fill_opacity=0.8,
                    line_opacity=0.3,
                    nan_fill_color="#d3d3d3",
                    legend_name=f"{var_map} - {fecha_sel}"
                ).add_to(m)
            
                tooltip = folium.GeoJsonTooltip(
                    fields=["name", var_map, "casos_100k", "letalidad_pct"],
                    aliases=["País:", "Indicador:", "Casos:", "Letalidad:"],
                    localize=True,
                    labels=True,
                )
            
                folium.GeoJson(
                    gdf,
                    tooltip=tooltip,
                    highlight_function=lambda x: {
                        "weight": 2,
                        "color": "black",
                        "fillOpacity": 0.9,
                    },
                ).add_to(m)
            
                # 👇 SIEMPRE renderizamos
                map_data = st_folium(m, use_container_width=True, height=520)
            
                # 👇 luego vemos si hubo click
                if map_data and map_data.get("last_active_drawing"):
                    props = map_data["last_active_drawing"]["properties"]
                    pais_click = props.get("ISO3166-1-Alpha-3")
            
            except Exception as e:
                st.error(f"Error en el mapa: {e}")
        
        
        # =============================
        # FILTRO POR PAÍS
        # =============================
        
        if pais_click:
            st.success(f"País seleccionado: {pais_click}")
            df_filtrado = df_viz[df_viz["ISO3"] == pais_click]
        else:
            df_filtrado = df_viz


        
        # --- SECCIÓN 6: COMPARACIÓN DIRECTA ENTRE PAÍSES ---
        st.markdown("### ⚖️ Comparacion Detallada entre Paises")
        
        paises_comparar = st.multiselect(
            "Selecciona 2-4 paises para comparar:",
            options=sorted(df_viz['country'].unique()),
            default=sorted(df_viz['country'].unique())[:3] if len(df_viz['country'].unique()) >= 3 else sorted(df_viz['country'].unique()),
            max_selections=4,
            key='compare_countries'
        )
        
        if len(paises_comparar) >= 2:
            df_compare = df_viz[df_viz['country'].isin(paises_comparar)]
            df_compare_agg = df_compare.groupby('country')[['casos_100k', 'letalidad_pct', 'camas_por_100k', 'avg_temp']].mean().reset_index()
            
            # Crear gráfico de barras agrupadas
            fig_compare = go.Figure()
            
            variables = ['casos_100k', 'letalidad_pct', 'camas_por_100k', 'avg_temp']
            colors = ['#22D3EE', '#818CF8', '#34D399', '#F472B6']
            
            for i, var in enumerate(variables):
                fig_compare.add_trace(go.Bar(
                    name=var,
                    x=df_compare_agg['country'],
                    y=df_compare_agg[var],
                    marker_color=colors[i]
                ))
            
            fig_compare.update_layout(
                barmode='group',
                title="Comparacion de Metricas entre Paises",
                plot_bgcolor='#111827',
                paper_bgcolor='#0B0F19',
                font=dict(family="Inter", size=12, color='#94A3B8'),
                margin=dict(t=40, b=20),
                xaxis=dict(gridcolor='#1E293B', title=""),
                yaxis=dict(gridcolor='#1E293B', title="Valor"),
                height=400,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                    font=dict(color='#94A3B8')
                )
            )
            st.plotly_chart(fig_compare, use_container_width=True)
            
            # Tabla comparativa
            st.markdown("**📋 Tabla Comparativa**")
            df_compare_display = df_compare_agg.set_index('country')
            df_compare_display.columns = ['Casos/100k', 'Letalidad %', 'Camas/100k', 'Temp °C']
            st.dataframe(df_compare_display.round(2).style.background_gradient(cmap='viridis'), use_container_width=True)
        else:
            st.info("Selecciona al menos 2 paises para ver la comparacion")

    # --- TAB 4: CHAT INTERACTIVO CON IA ---
    with tab_ia:
        # Header atractivo
        st.markdown("""
        <div class="chat-header-card">
            <h2>Chat Inteligente con Analista IA</h2>
            <p>Conversa con la IA sobre tu dataset de COVID-19 usando Groq</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Inicializar el historial de chat en session_state
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        if "groq_api_key" not in st.session_state:
            st.session_state.groq_api_key = ""
        
        # Sección de configuración
        with st.expander("⚙️ Configuracion de API", expanded=not st.session_state.groq_api_key):
            api_key_input = st.text_input(
                "API Key de Groq:", 
                type="password",
                value=st.session_state.groq_api_key,
                placeholder="Ingresa tu API key aqui...",
                help="Obten tu API key gratis en console.groq.com"
            )
            
            if api_key_input:
                st.session_state.groq_api_key = api_key_input
                st.success("✅ API Key configurada correctamente")
            
        # Controles del chat - CENTRADO
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            if st.button("🗑️ Limpiar Chat", use_container_width=True, type="secondary"):
                st.session_state.messages = []
                st.rerun()
        
        st.markdown("---")
        
        # Contenedor del chat
        chat_container = st.container()
        
        with chat_container:
            # Mostrar historial de mensajes
            for message in st.session_state.messages:
                with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🤖"):
                    st.markdown(message["content"])
            
            # Mensaje de bienvenida si no hay historial
            if len(st.session_state.messages) == 0:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown("""
                        Hola! Soy tu asistente de analisis de datos especializado en COVID-19.
                        
                        **Que te gustaria saber sobre tus datos?**
                    """)
        
        # Input del usuario
        if prompt := st.chat_input("Escribe tu pregunta aqui...", key="chat_input"):
            if not st.session_state.groq_api_key:
                st.error("⚠️ Por favor, configura tu API Key de Groq primero en la seccion de Configuracion.")
            else:
                # Agregar mensaje del usuario
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                with st.chat_message("user", avatar="🧑‍💻"):
                    st.markdown(prompt)
                
                # Generar respuesta de la IA
                with st.chat_message("assistant", avatar="🤖"):
                    message_placeholder = st.empty()
                    
                    try:
                        # Preparar contexto rico del dataset
                        resumen_estadistico = df_final.describe().to_string()
                        columnas_disponibles = ", ".join(df_final.columns.tolist())
                        total_registros = len(df_final)
                        continentes_unicos = ", ".join(df_final['continent'].unique().tolist())
                        paises_unicos = df_final['country'].nunique()
                        
                        # Calcular algunas métricas clave
                        letalidad_max = df_final['letalidad_pct'].max()
                        letalidad_min = df_final['letalidad_pct'].min()
                        casos_max = df_final['casos_100k'].max()
                        
                        # Contexto del sistema para la IA
                        contexto_sistema = f"""
Eres un analista de datos senior especializado en epidemiología y salud pública global.
Tienes acceso a un dataset completo de COVID-19 con las siguientes características:

📋 INFORMACIÓN GENERAL:
- Total de registros: {total_registros:,}
- Países únicos: {paises_unicos}
- Continentes: {continentes_unicos}
- Indicador actual filtrado: {indicador}

📊 VARIABLES DISPONIBLES:
{columnas_disponibles}

📈 ESTADÍSTICAS DESCRIPTIVAS COMPLETAS:
{resumen_estadistico}

🔑 DEFINICIONES DE VARIABLES CLAVE:
- casos_100k: Tasa de incidencia por cada 100,000 habitantes
- camas_por_100k: Capacidad hospitalaria instalada por cada 100,000 habitantes
- letalidad_pct: Porcentaje de fallecimientos respecto a la población
- avg_temp: Temperatura promedio mensual del país (°C)

📌 DATOS DESTACADOS:
- Letalidad máxima registrada: {letalidad_max:.4f}%
- Letalidad mínima registrada: {letalidad_min:.4f}%
- Tasa de casos más alta: {casos_max:.2f} por 100k habitantes

INSTRUCCIONES DE RESPUESTA:
1. Sé específico y usa datos reales del dataset cuando sea relevante
2. Si mencionas números, cítalos correctamente de las estadísticas
3. Mantén un tono profesional pero accesible
4. Si detectas patrones interesantes, mencionálos
5. Ofrece insights accionables cuando sea apropiado
6. Usa emojis ocasionalmente para hacer las respuestas más amigables
7. Si no tienes información suficiente, sé honesto al respecto
"""
                        
                        # Crear cliente de Groq
                        client = Groq(api_key=st.session_state.groq_api_key)
                        
                        # Construir historial de mensajes
                        messages_for_api = [
                            {"role": "system", "content": contexto_sistema}
                        ]
                        
                        # Agregar últimos 10 mensajes de contexto
                        for msg in st.session_state.messages[-10:]:
                            messages_for_api.append({
                                "role": msg["role"],
                                "content": msg["content"]
                            })
                        
                        # Llamada streaming a Groq con modelo actualizado
                        full_response = ""
                        
                        stream = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=messages_for_api,
                            temperature=0.7,
                            max_tokens=2048,
                            top_p=0.95,
                            stream=True
                        )
                        
                        # Mostrar respuesta en tiempo real con cursor
                        for chunk in stream:
                            if chunk.choices[0].delta.content:
                                full_response += chunk.choices[0].delta.content
                                message_placeholder.markdown(full_response + "▌")
                        
                        # Respuesta final sin cursor
                        message_placeholder.markdown(full_response)
                        
                        # Guardar en historial
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": full_response
                        })
                        
                    except Exception as e:
                        error_msg = f"**Error al conectar con Groq:**\n\n`{str(e)}`\n\nVerifica que tu API Key sea valida y tengas conexion a internet."
                        message_placeholder.error(error_msg)
        
        # Sección de preguntas sugeridas (minimalista)
        if len(st.session_state.messages) == 0:
            with st.expander("💡 Preguntas sugeridas", expanded=False):
                suggestions = [
                    ("📊", "Correlacion temperatura vs casos"),
                    ("🏥", "Impacto infraestructura hospitalaria"),
                    ("🌍", "Continente con mayor impacto"),
                    ("📈", "Top 5 paises por letalidad"),
                    ("🔍", "Patrones interesantes en datos"),
                    ("💊", "Recomendaciones del analisis")
                ]
                
                questions_full = [
                    "Cual es la correlacion entre temperatura e incidencia de casos?",
                    "Como afecta la infraestructura hospitalaria a la tasa de letalidad?",
                    "Que continente presenta el mayor impacto segun los datos?",
                    "Identifica los 5 paises con mayor tasa de letalidad",
                    "Hay algun patron interesante en los datos que deba considerar?",
                    "Dame 3 recomendaciones basadas en el analisis del dataset"
                ]
                
                for idx, (emoji, question) in enumerate(suggestions):
                    if st.button(f"{emoji} {question}", key=f"sugg_{idx}", use_container_width=True):
                        if st.session_state.groq_api_key:
                            st.session_state.messages.append({"role": "user", "content": questions_full[idx]})
                            st.rerun()
                        else:
                            st.error("⚠️ Configura tu API Key primero")
        else:
            # Estadísticas del chat (minimalista)
            user_msgs = len([m for m in st.session_state.messages if m["role"] == "user"])
            st.caption(f"💬 {user_msgs} preguntas realizadas")

else:
    # Pantalla de bienvenida mejorada
    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-icon">📊</div>
        <h2>Bienvenido al Sistema de Soporte a la Decision</h2>
        <p class="subtitle">Para comenzar, carga tu archivo CSV con datos de COVID-19 usando el panel lateral izquierdo.</p>
        <div class="feature-grid">
            <div class="feature-item">
                <div class="icon">🧹</div>
                <p class="label">Limpieza de Datos</p>
                <p class="desc">ETL interactivo con imputacion y filtros</p>
            </div>
            <div class="feature-item">
                <div class="icon">📈</div>
                <p class="label">Visualizaciones</p>
                <p class="desc">Graficos interactivos y mapas globales</p>
            </div>
            <div class="feature-item">
                <div class="icon">🧬</div>
                <p class="label">Correlaciones</p>
                <p class="desc">Analisis estadistico de Pearson</p>
            </div>
            <div class="feature-item">
                <div class="icon">🤖</div>
                <p class="label">Chat con IA</p>
                <p class="desc">Asistente inteligente con Groq AI</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- PIE DE PÁGINA ---
st.markdown("""
<div class="footer">
    <hr style="border-color: #1E293B; margin-bottom: 16px;">
    Proyecto Final: Sistema de Soporte a la Decision &nbsp;·&nbsp; Powered by <a href="https://groq.com" target="_blank">Groq AI</a> & <a href="https://streamlit.io" target="_blank">Streamlit</a>
</div>
""", unsafe_allow_html=True)
