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
    .chat-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .suggestion-button {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 8px;
        margin: 5px;
        cursor: pointer;
        transition: all 0.3s;
    }
    .suggestion-button:hover {
        background-color: #e0e2e6;
        transform: translateY(-2px);
    }
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

    # --- TAB 4: CHAT INTERACTIVO CON IA ---
    with tab_ia:
        # Header atractivo
        st.markdown("""
            <div class="chat-header">
                <h2>🤖 Chat Inteligente con Analista IA</h2>
                <p>Conversa con la IA sobre tu dataset de COVID-19 usando Groq</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Inicializar el historial de chat en session_state
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        if "groq_api_key" not in st.session_state:
            st.session_state.groq_api_key = ""
        
        # Sección de configuración
        with st.expander("⚙️ Configuración de API", expanded=not st.session_state.groq_api_key):
            api_key_input = st.text_input(
                "🔑 API Key de Groq:", 
                type="password",
                value=st.session_state.groq_api_key,
                placeholder="Ingresa tu API key aquí...",
                help="Obtén tu API key gratis en console.groq.com"
            )
            
            if api_key_input:
                st.session_state.groq_api_key = api_key_input
                st.success("✅ API Key configurada correctamente")
            
            st.markdown("""
                **¿Cómo obtener tu API Key?**
                1. Visita [console.groq.com](https://console.groq.com)
                2. Crea una cuenta gratuita
                3. Ve a "API Keys" en el menú
                4. Genera una nueva key
                5. Cópiala y pégala arriba 🔝
            """)
        
        # Controles del chat
        col_info, col_clear = st.columns([4, 1])
        with col_info:
            if st.session_state.groq_api_key:
                st.info("💬 La IA tiene contexto completo de tu dataset. ¡Hazle preguntas!")
            else:
                st.warning("⚠️ Configura tu API Key para comenzar a chatear")
        
        with col_clear:
            if st.button("🗑️ Limpiar Chat", use_container_width=True):
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
                        ¡Hola! 👋 Soy tu asistente de análisis de datos especializado en COVID-19.
                        
                        Puedo ayudarte a:
                        - 📊 Interpretar estadísticas y correlaciones
                        - 🌍 Analizar tendencias por continente o país
                        - 🏥 Evaluar el impacto de la infraestructura hospitalaria
                        - 🌡️ Estudiar la relación entre temperatura y casos
                        - 💡 Generar insights y recomendaciones
                        
                        **¿Qué te gustaría saber sobre tus datos?**
                    """)
        
        # Input del usuario
        if prompt := st.chat_input("💭 Escribe tu pregunta aquí...", key="chat_input"):
            if not st.session_state.groq_api_key:
                st.error("⚠️ Por favor, configura tu API Key de Groq primero en la sección de Configuración.")
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
                        
                        # Llamada streaming a Groq
                        full_response = ""
                        
                        stream = client.chat.completions.create(
                            model="llama3-8b-8192",
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
                        error_msg = f"❌ **Error al conectar con Groq:**\n\n`{str(e)}`\n\nVerifica que tu API Key sea válida y tengas conexión a internet."
                        message_placeholder.error(error_msg)
        
        # Sección de preguntas sugeridas
        st.markdown("---")
        st.markdown("### 💡 Preguntas Sugeridas")
        st.caption("Haz clic en cualquier pregunta para enviarla automáticamente")
        
        col1, col2, col3 = st.columns(3)
        
        suggestions = [
            ("📊", "¿Cuál es la correlación entre temperatura e incidencia de casos?"),
            ("🏥", "¿Cómo afecta la infraestructura hospitalaria a la tasa de letalidad?"),
            ("🌍", "¿Qué continente presenta el mayor impacto según los datos?"),
            ("📈", "Identifica los 5 países con mayor tasa de letalidad"),
            ("🔍", "¿Hay algún patrón interesante en los datos que deba considerar?"),
            ("💊", "Dame 3 recomendaciones basadas en el análisis del dataset")
        ]
        
        cols = [col1, col2, col3, col1, col2, col3]
        
        for idx, (emoji, question) in enumerate(suggestions):
            with cols[idx]:
                if st.button(f"{emoji} {question[:30]}...", key=f"sugg_{idx}", use_container_width=True):
                    # Simular envío de la pregunta
                    if st.session_state.groq_api_key:
                        st.session_state.messages.append({"role": "user", "content": question})
                        st.rerun()
                    else:
                        st.error("Configura tu API Key primero")
        
        # Estadísticas del chat
        if len(st.session_state.messages) > 0:
            st.markdown("---")
            user_msgs = len([m for m in st.session_state.messages if m["role"] == "user"])
            st.caption(f"📊 Conversación activa: {user_msgs} preguntas realizadas")

else:
    # Pantalla de bienvenida mejorada
    st.markdown("""
        <div style='text-align: center; padding: 50px;'>
            <h2>👋 Bienvenido al Sistema de Soporte a la Decisión</h2>
            <p style='font-size: 18px; color: #666;'>
                Para comenzar, carga tu archivo CSV con datos de COVID-19 usando el panel lateral
            </p>
            <p style='margin-top: 30px;'>
                <span style='font-size: 48px;'>📊</span><br>
                <strong>Funcionalidades principales:</strong><br>
                ✅ Limpieza y análisis de datos<br>
                ✅ Visualizaciones interactivas<br>
                ✅ Chat con IA especializada<br>
                ✅ Insights automáticos
            </p>
        </div>
    """, unsafe_allow_html=True)

# --- PIE DE PÁGINA ---
st.markdown("---")
st.caption("🚀 Proyecto Final: Sistema de Soporte a la Decisión | Powered by Groq AI & Streamlit")
