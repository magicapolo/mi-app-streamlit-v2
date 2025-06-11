import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go # Para gráficos interactivos

# --- Funciones de Lógica de Trading ---

def detectar_soporte_resistencia(data: pd.Series, window: int):
    """
    Calcula soporte y resistencia como el mínimo y máximo de una ventana móvil.
    Ajusta dinámicamente la ventana si hay menos datos de los solicitados.
    
    Args:
        data (pd.Series): Serie de datos de precios.
        window (int): Tamaño deseado de la ventana.
        
    Returns:
        tuple: (soporte, resistencia) como Series de Pandas.
    """
    actual_window = max(1, min(window, len(data)))
    if len(data) == 0:
        return pd.Series([]), pd.Series([]) 
    
    soporte = data.rolling(window=actual_window, min_periods=1).min()
    resistencia = data.rolling(window=actual_window, min_periods=1).max()
    
    return soporte, resistencia

def analizar_tendencia(data: pd.Series, short_sma_window: int = 5, long_sma_window: int = 20):
    """
    Analiza la tendencia de los datos usando Medias Móviles Simples (SMA).
    
    Args:
        data (pd.Series): Serie de datos de precios.
        short_sma_window (int): Ventana para la SMA corta.
        long_sma_window (int): Ventana para la SMA larga.
        
    Returns:
        str: "Alcista", "Bajista", "Estable" o "Insuficientes datos".
    """
    if len(data) < long_sma_window:
        return "Insuficientes datos para tendencia"

    sma_short = data.rolling(window=short_sma_window, min_periods=1).mean()
    sma_long = data.rolling(window=long_sma_window, min_periods=1).mean()

    if data.iloc[-1] > sma_short.iloc[-1] and sma_short.iloc[-1] > sma_long.iloc[-1]:
        return "Alcista"
    elif data.iloc[-1] < sma_short.iloc[-1] and sma_short.iloc[-1] < sma_long.iloc[-1]:
        return "Bajista"
    else:
        return "Estable"

# Esta función ahora no imprime directamente, sino que retorna los mensajes para control externo
def get_prediction_and_alerts(data: pd.Series, target_value: float, window: int, umbral: float,
                               short_sma_window: int, long_sma_window: int):
    """
    Calcula la predicción, análisis y alertas, retornándolos para su impresión ordenada.
    """
    results = {
        "prediction_sr_message": None,
        "alert_2_0_message": None,
        "alert_5_0_message": None,
        "alert_10_0_message": None,
        "tendency_message": None,
        "analysis_current": {}, # Para soporte, resistencia, actual, etc.
        "warnings": []
    }

    if len(data) == 0:
        results["warnings"].append("⚠️ **Error:** No hay datos para predecir.")
        return results
    
    # --- Manejar 'menos coeficientes' / ventana dinámica para S/R ---
    adjusted_window = max(2, min(window, len(data)))
    if len(data) == 1: 
        adjusted_window = 1
        results["warnings"].append("ℹ️ **Nota:** Con un solo punto de datos, el soporte y la resistencia son el mismo valor. La predicción será limitada.")
    elif adjusted_window < window:
        results["warnings"].append(f"ℹ️ **Nota:** La ventana deseada de `{window}` para S/R es mayor que los datos disponibles (`{len(data)}`). Se ajustó la ventana a `{adjusted_window}`.")

    soporte, resistencia = detectar_soporte_resistencia(data, adjusted_window)
    
    actual = data.iloc[-1]
    soporte_actual = soporte.iloc[-1]
    resistencia_actual = resistencia.iloc[-1]

    results["analysis_current"] = {
        "actual": actual,
        "soporte": soporte_actual,
        "resistencia": resistencia_actual,
        "target": target_value,
        "umbral": umbral
    }

    # --- Alertas para valores altos (10.00, 5.00, 2.00) ---
    if actual >= 10.00:
        results["alert_10_0_message"] = f"🔥🔥 **¡ALERTA MÁXIMA!** El valor actual ({actual:.4f}) es igual o mayor a **10.00**. ¡EXTREMA ATENCIÓN!"
    elif resistencia_actual >= 10.00:
        results["alert_10_0_message"] = f"🔔 **Advertencia Alta:** El nivel de resistencia detectado ({resistencia_actual:.4f}) es igual o mayor a **10.00**. ¡Potencialmente muy alto!"
    
    if actual >= 5.00 and actual < 10.00: 
        results["alert_5_0_message"] = f"🚨 **¡ALERTA!** El valor actual ({actual:.4f}) es igual o mayor a **5.00**."
    elif resistencia_actual >= 5.00 and resistencia_actual < 10.00: 
        results["alert_5_0_message"] = f"🔔 **Advertencia:** El nivel de resistencia detectado ({resistencia_actual:.4f}) es igual o mayor a **5.00**."

    if actual >= 2.00 and actual < 5.00: # Se ejecuta solo si no es ya >= 5.00 o >= 10.00
        results["alert_2_0_message"] = f"🟢 **¡Alerta Importante!** El valor actual ({actual:.4f}) es igual o mayor a **2.00**. ¡Buen punto!"
    elif resistencia_actual >= 2.00 and resistencia_actual < 5.00:
        results["alert_2_0_message"] = f"🔵 **Nota:** El nivel de resistencia detectado ({resistencia_actual:.4f}) es igual o mayor a **2.00**."

    # --- Análisis de Tendencia ---
    tendencia = analizar_tendencia(data, short_sma_window, long_sma_window)
    if tendencia == "Alcista":
        results["tendency_message"] = f"🚀 **TENDENCIA:** ¡Actualmente estamos en una tendencia **ALCISTA**!"
    elif tendencia == "Bajista":
        results["tendency_message"] = f"📉 **TENDENCIA:** Se ha detectado una tendencia **BAJISTA**."
    elif tendencia == "Estable":
        results["tendency_message"] = f"↔️ **TENDENCIA:** La tendencia actual parece **ESTABLE**."
    else:
        results["tendency_message"] = f"⚠️ **TENDENCIA:** {tendencia} (Necesitas más datos para un análisis de tendencia fiable)."

    # Validación final para la predicción S/R principal
    if np.isnan(soporte_actual) or np.isnan(resistencia_actual):
        results["warnings"].append("⚠️ **Advertencia:** Los niveles de soporte/resistencia no se pudieron calcular completamente con los datos/ventana actuales. La predicción basada en S/R puede ser menos precisa.")
        results["prediction_sr_message"] = "Insuficientes datos o S/R no calculable"
    else:
        if actual <= soporte_actual + umbral:
            results["prediction_sr_message"] = f"📈 **PREDICCIÓN:** Se espera un movimiento **MAYOR a {target_value:.4f}** (cerca del soporte)."
        elif actual >= resistencia_actual - umbral:
            results["prediction_sr_message"] = f"📉 **PREDICCIÓN:** Se espera un movimiento **MENOR a {target_value:.4f}** (cerca de la resistencia)."
        else:
            results["prediction_sr_message"] = "🤔 **PREDICCIÓN:** El valor actual está entre soporte y resistencia. La dirección es **INCIERTA**."
    
    return results

# --- Configuración de la Página Streamlit ---
st.set_page_config(
    page_title="Predicción de Trading Avanzada",
    page_icon="📈",
    layout="wide" # Diseño amplio para mejor visualización
)

st.title("🤖 Bot Predictor de Trading (Soporte, Resistencia y Tendencias)")

st.markdown("""
Esta aplicación te ayuda a predecir la dirección de un activo basándose en soporte, resistencia y tendencias.
También te alerta sobre coeficientes altos.
""")

# --- Entrada de Datos del Usuario ---
st.sidebar.header("⚙️ Configuración y Datos")

datos_input_str = st.sidebar.text_area(
    "**Ingresa tus valores numéricos históricos (separados por comas):**",
    value="1.45, 1.48, 1.52, 1.49, 1.51, 1.47, 1.53, 1.46, 1.50, 1.49, 1.51, 1.52, 1.50, 1.48, 1.47, 1.49, 1.51, 1.50, 2.10, 3.50, 5.20, 9.80, 10.10",
    height=180, 
    help="Introduce una serie de números que representen los precios del activo. El teclado numérico de tu teléfono se debería desplegar automáticamente."
)

# --- Botón de Predicción (MOVIMIENTO) ---
if st.sidebar.button("✨ Realizar Predicción Avanzada"):
    if not datos_input_str:
        st.error("Por favor, ingresa los valores numéricos para poder realizar la predicción.")
    else:
        try:
            datos_numericos = [float(x.strip()) for x in datos_input_str.split(',') if x.strip()]
            
            if not datos_numericos:
                st.error("No se detectaron números válidos en la entrada. Revisa el formato.")
            else:
                data_series = pd.Series(datos_numericos)
                
                # Obtener parámetros de los sliders
                target_value_sidebar = st.session_state.get('target_value', 1.50) # Usar session_state para mantener valores
                window_size_sidebar = st.session_state.get('window_size', 10)
                threshold_sidebar = st.session_state.get('threshold', 0.01)
                short_sma_sidebar = st.session_state.get('short_sma', 5)
                long_sma_sidebar = st.session_state.get('long_sma', 20)

                # Realizar la predicción y obtener todos los resultados
                results = get_prediction_and_alerts(data_series, target_value_sidebar, window_size_sidebar, 
                                                    threshold_sidebar, short_sma_sidebar, long_sma_sidebar)
                
                # --- ORDEN DE APARICIÓN EN EL CUERPO PRINCIPAL ---
                
                # 1. Predicción S/R Principal
                st.subheader("🎯 Predicción S/R Principal")
                if "PREDICCIÓN: Mayor" in results["prediction_sr_message"]:
                    st.success(results["prediction_sr_message"])
                elif "PREDICCIÓN: Menor" in results["prediction_sr_message"]:
                    st.error(results["prediction_sr_message"])
                else: # Incluye "INCIERTA" y "Insuficientes datos o S/R no calculable"
                    st.warning(results["prediction_sr_message"])

                # 2. Alerta 2.0 (si existe)
                if results["alert_2_0_message"]:
                    if "🟢" in results["alert_2_0_message"]:
                        st.success(results["alert_2_0_message"])
                    elif "🔵" in results["alert_2_0_message"]:
                        st.info(results["alert_2_0_message"])

                # 3. Análisis Actual (Detalles del Valor Actual, S/R)
                st.subheader("📊 Análisis Actual")
                st.write(f"**Valor actual:** `{results['analysis_current']['actual']:.4f}`")
                st.info(f"**Soporte detectado:** `{results['analysis_current']['soporte']:.4f}`")
                st.info(f"**Resistencia detectada:** `{results['analysis_current']['resistencia']:.4f}`")
                st.write(f"**Valor Objetivo de Comparación:** `{results['analysis_current']['target']:.4f}`")
                st.write(f"**Umbral de sensibilidad:** `{results['analysis_current']['umbral']:.4f}`")

                # 4. Alertas de 5.00 y 10.00
                if results["alert_10_0_message"]:
                    st.error(results["alert_10_0_message"])
                if results["alert_5_0_message"]:
                    st.error(results["alert_5_0_message"]) # Ya está ajustado en la función para no duplicar

                # 5. Análisis y Alertas de Tendencia
                st.subheader("📈 Análisis de Tendencia")
                if "ALCISTA" in results["tendency_message"]:
                    st.success(results["tendency_message"])
                elif "BAJISTA" in results["tendency_message"]:
                    st.error(results["tendency_message"])
                elif "ESTABLE" in results["tendency_message"]:
                    st.info(results["tendency_message"])
                else:
                    st.warning(results["tendency_message"])
                
                # 6. Advertencias generales que no encajan en las categorías anteriores
                for warning_msg in results["warnings"]:
                    st.warning(warning_msg)


                # --- Visualización de Datos y S/R ---
                st.subheader("📈 Gráfico de Precios con Soporte, Resistencia y Tendencias")
                
                plot_window_for_sr = max(1, min(window_size_sidebar, len(data_series)))
                soporte_line, resistencia_line = detectar_soporte_resistencia(data_series, plot_window_for_sr)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=list(range(len(data_series))), y=data_series, mode='lines+markers', name='Precios', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=list(range(len(soporte_line))), y=soporte_line, mode='lines', name='Soporte', line=dict(color='green', dash='dot')))
                fig.add_trace(go.Scatter(x=list(range(len(resistencia_line))), y=resistencia_line, mode='lines', name='Resistencia', line=dict(color='red', dash='dot')))
                
                # Añadir SMAs al gráfico
                if len(data_series) >= long_sma_sidebar: 
                    sma_short_line = data_series.rolling(window=short_sma_sidebar, min_periods=1).mean()
                    sma_long_line = data_series.rolling(window=long_sma_sidebar, min_periods=1).mean()
                    fig.add_trace(go.Scatter(x=list(range(len(sma_short_line))), y=sma_short_line, mode='lines', name=f'SMA {short_sma_sidebar}', line=dict(color='purple', dash='solid')))
                    fig.add_trace(go.Scatter(x=list(range(len(sma_long_line))), y=sma_long_line, mode='lines', name=f'SMA {long_sma_sidebar}', line=dict(color='brown', dash='dash')))

                # Añadir líneas de alerta
                fig.add_hline(y=target_value_sidebar, line_dash="dot", line_color="purple", annotation_text=f"Objetivo: {target_value_sidebar:.2f}", annotation_position="top right")
                fig.add_hline(y=2.00, line_dash="dot", line_color="blue", annotation_text="Alerta 2.00", annotation_position="top left") 
                fig.add_hline(y=5.00, line_dash="dash", line_color="orange", annotation_text="Alerta 5.00", annotation_position="bottom right")
                fig.add_hline(y=10.00, line_dash="dashdot", line_color="red", annotation_text="Alerta 10.00", annotation_position="top left")


                fig.update_layout(
                    title='Historial de Precios, Niveles S/R y Tendencias',
                    xaxis_title='Puntos de Datos',
                    yaxis_title='Valor',
                    height=500,
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)

        except ValueError:
            st.error("Error: Por favor, asegúrate de que los datos ingresados sean solo números válidos separados por comas.")

# --- Sliders para parámetros ---
st.sidebar.markdown("---") # Divisor para separar
st.sidebar.header("📊 Parámetros de Análisis") # Nuevo subtítulo

# Para mantener los valores de los sliders después de la predicción, usamos st.session_state
# Esto es importante porque el botón en la sidebar redibuja toda la app
st.sidebar.number_input(
    "**Valor Objetivo para la Predicción (ej. 1.50):**",
    min_value=0.01, max_value=100.0, value=1.50, step=0.01,
    key='target_value', # Usar key para session_state
    help="El valor contra el cual se compara la predicción (mayor o menor)."
)

st.sidebar.slider(
    "**Tamaño Ventana Móvil (S/R):**",
    min_value=1, max_value=50, value=10, step=1, 
    key='window_size', # Usar key para session_state
    help="Número de puntos para calcular S/R. Se ajusta dinámicamente si hay menos datos."
)

st.sidebar.slider(
    "**Umbral de Sensibilidad (Margen S/R):**",
    min_value=0.001, max_value=0.1, value=0.01, format="%.3f", step=0.001,
    key='threshold', # Usar key para session_state
    help="Define qué tan cerca debe estar el precio de S/R para activar una predicción."
)

st.sidebar.markdown("---")
st.sidebar.header("📈 Configuración de Tendencia")
st.sidebar.slider(
    "**Ventana SMA Corta (Tendencia):**",
    min_value=2, max_value=20, value=5, step=1,
    key='short_sma', # Usar key para session_state
    help="Número de puntos para la Media Móvil Simple (SMA) corta."
)
st.sidebar.slider(
    "**Ventana SMA Larga (Tendencia):**",
    min_value=5, max_value=50, value=20, step=1,
    key='long_sma', # Usar key para session_state
    help="Número de puntos para la Media Móvil Simple (SMA) larga (debe ser mayor que la corta)."
)

st.markdown("---")
st.caption("Desarrollado con Streamlit por tu AI asistente.")
