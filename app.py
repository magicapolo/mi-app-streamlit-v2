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

    # Si el precio actual está por encima de la SMA corta
    # y la SMA corta está por encima de la SMA larga
    if data.iloc[-1] > sma_short.iloc[-1] and sma_short.iloc[-1] > sma_long.iloc[-1]:
        return "Alcista"
    # Si el precio actual está por debajo de la SMA corta
    # y la SMA corta está por debajo de la SMA larga
    elif data.iloc[-1] < sma_short.iloc[-1] and sma_short.iloc[-1] < sma_long.iloc[-1]:
        return "Bajista"
    else:
        return "Estable"

def predecir_direccion(data: pd.Series, target_value: float = 1.50, window: int = 10, umbral: float = 0.01,
                       short_sma_window: int = 5, long_sma_window: int = 20):
    """
    Predice si el siguiente valor será mayor o menor al target_value (ej. 1.50)
    basándose en la posición actual respecto a soporte y resistencia.
    También incluye alertas para valores altos y análisis de tendencias.
    
    Args:
        data (pd.Series): Serie de datos de precios.
        target_value (float): El valor objetivo (ej. 1.50) para la predicción.
        window (int): Tamaño deseado de la ventana para calcular soporte/resistencia.
        umbral (float): Margen alrededor de soporte/resistencia para la predicción.
        short_sma_window (int): Ventana para la SMA corta para tendencia.
        long_sma_window (int): Ventana para la SMA larga para tendencia.
        
    Returns:
        str: Mensaje de predicción o estado.
    """
    if len(data) == 0:
        st.error("⚠️ **Error:** No hay datos para predecir.")
        return "No hay datos"
    
    # --- Manejar 'menos coeficientes' / ventana dinámica para S/R ---
    adjusted_window = max(2, min(window, len(data)))
    if len(data) == 1: 
        adjusted_window = 1
        st.warning("ℹ️ **Nota:** Con un solo punto de datos, el soporte y la resistencia son el mismo valor. La predicción será limitada.")
    elif adjusted_window < window:
        st.warning(f"ℹ️ **Nota:** La ventana deseada de `{window}` para S/R es mayor que los datos disponibles (`{len(data)}`). Se ajustó la ventana a `{adjusted_window}`.")

    soporte, resistencia = detectar_soporte_resistencia(data, adjusted_window)
    
    actual = data.iloc[-1]
    soporte_actual = soporte.iloc[-1]
    resistencia_actual = resistencia.iloc[-1]

    st.subheader("📊 Análisis Actual")
    st.write(f"**Valor actual:** `{actual:.4f}`")
    st.info(f"**Soporte detectado:** `{soporte_actual:.4f}`")
    st.info(f"**Resistencia detectada:** `{resistencia_actual:.4f}`")
    st.write(f"**Valor Objetivo de Comparación:** `{target_value:.4f}`")
    st.write(f"**Umbral de sensibilidad:** `{umbral:.4f}`")

    # --- Alerta para valores mayores a 10.00 ---
    if actual >= 10.00:
        st.error(f"🔥🔥 **¡ALERTA MÁXIMA!** El valor actual ({actual:.4f}) es igual o mayor a **10.00**. ¡EXTREMA ATENCIÓN!")
    elif resistencia_actual >= 10.00:
        st.warning(f"🔔 **Advertencia Alta:** El nivel de resistencia detectado ({resistencia_actual:.4f}) es igual o mayor a **10.00**. ¡Potencialmente muy alto!")
    
    # --- Alerta para valores mayores a 5.00 (general) ---
    if actual >= 5.00 and actual < 10.00: 
        st.error(f"🚨 **¡ALERTA!** El valor actual ({actual:.4f}) es igual o mayor a **5.00**.")
    elif resistencia_actual >= 5.00 and resistencia_actual < 10.00: 
        st.warning(f"🔔 **Advertencia:** El nivel de resistencia detectado ({resistencia_actual:.4f}) es igual o mayor a **5.00**.")

    # --- Alerta para valores mayores a 2.00 (NUEVA) ---
    if actual >= 2.00 and actual < 5.00: # Se ejecuta solo si no es ya >= 5.00 o >= 10.00
        st.success(f"🟢 **¡Alerta Importante!** El valor actual ({actual:.4f}) es igual o mayor a **2.00**. ¡Buen punto!")
    elif resistencia_actual >= 2.00 and resistencia_actual < 5.00:
        st.info(f"🔵 **Nota:** El nivel de resistencia detectado ({resistencia_actual:.4f}) es igual o mayor a **2.00**.")

    # --- Análisis y Alertas de Tendencia (NUEVO) ---
    st.subheader("📈 Análisis de Tendencia")
    tendencia = analizar_tendencia(data, short_sma_window, long_sma_window)
    if tendencia == "Alcista":
        st.success(f"🚀 **TENDENCIA:** ¡Actualmente estamos en una tendencia **ALCISTA**!")
    elif tendencia == "Bajista":
        st.error(f"📉 **TENDENCIA:** Se ha detectado una tendencia **BAJISTA**.")
    elif tendencia == "Estable":
        st.info(f"↔️ **TENDENCIA:** La tendencia actual parece **ESTABLE**.")
    else:
        st.warning(f"⚠️ **TENDENCIA:** {tendencia} (Necesitas más datos para un análisis de tendencia fiable).")

    # Validación final para la predicción S/R principal
    if np.isnan(soporte_actual) or np.isnan(resistencia_actual):
        st.warning("⚠️ **Advertencia:** Los niveles de soporte/resistencia no se pudieron calcular completamente con los datos/ventana actuales. La predicción basada en S/R puede ser menos precisa.")
        return "Insuficientes datos o S/R no calculable"

    # Lógica de predicción principal (mayor/menor a 1.50)
    st.subheader("🎯 Predicción S/R Principal")
    prediccion_mensaje = "No hay una señal clara."
    if actual <= soporte_actual + umbral:
        prediccion_mensaje = f"📈 **PREDICCIÓN:** Se espera un movimiento **MAYOR a {target_value:.4f}** (cerca del soporte)."
        st.success(prediccion_mensaje)
    elif actual >= resistencia_actual - umbral:
        prediccion_mensaje = f"📉 **PREDICCIÓN:** Se espera un movimiento **MENOR a {target_value:.4f}** (cerca de la resistencia)."
        st.error(prediccion_mensaje)
    else:
        prediccion_mensaje = "🤔 **PREDICCIÓN:** El valor actual está entre soporte y resistencia. La dirección es **INCIERTA**."
        st.warning(prediccion_mensaje)
    
    return prediccion_mensaje

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
    height=180, # Un poco más de altura para más datos
    help="Introduce una serie de números que representen los precios del activo. El teclado numérico de tu teléfono se debería desplegar automáticamente."
)

target_value = st.sidebar.number_input(
    "**Valor Objetivo para la Predicción (ej. 1.50):**",
    min_value=0.01, max_value=100.0, value=1.50, step=0.01,
    help="El valor contra el cual se compara la predicción (mayor o menor)."
)

window_size = st.sidebar.slider(
    "**Tamaño Ventana Móvil (S/R):**",
    min_value=1, max_value=50, value=10, step=1, 
    help="Número de puntos para calcular S/R. Se ajusta dinámicamente si hay menos datos."
)

threshold = st.sidebar.slider(
    "**Umbral de Sensibilidad (Margen S/R):**",
    min_value=0.001, max_value=0.1, value=0.01, format="%.3f", step=0.001,
    help="Define qué tan cerca debe estar el precio de S/R para activar una predicción."
)

st.sidebar.markdown("---")
st.sidebar.header("📊 Configuración de Tendencia")
short_sma = st.sidebar.slider(
    "**Ventana SMA Corta (Tendencia):**",
    min_value=2, max_value=20, value=5, step=1,
    help="Número de puntos para la Media Móvil Simple (SMA) corta."
)
long_sma = st.sidebar.slider(
    "**Ventana SMA Larga (Tendencia):**",
    min_value=5, max_value=50, value=20, step=1,
    help="Número de puntos para la Media Móvil Simple (SMA) larga (debe ser mayor que la corta)."
)

st.sidebar.markdown("---") # Divisor en la barra lateral

# --- Botón de Predicción ---
if st.sidebar.button("✨ Realizar Predicción Avanzada"):
    if not datos_input_str:
        st.error("Por favor, ingresa los valores numéricos para poder realizar la predicción.")
    else:
        try:
            datos_numericos = [float(x.strip()) for x in datos_input_str.split(',') if x.strip()]
            
            if not datos_numericos:
                st.error("No se detectaron números válidos en la entrada. Revisa el formato.")
            elif len(datos_numericos) < max(short_sma, long_sma):
                st.warning(f"⚠️ **Advertencia:** Necesitas al menos **{max(short_sma, long_sma)}** puntos de datos para un análisis de tendencia completo con las configuraciones actuales.")
                data_series = pd.Series(datos_numericos)
                # Ejecutar predicción incluso si no hay suficientes datos para tendencia completa
                prediccion_mensaje = predecir_direccion(data_series, target_value, window_size, threshold, short_sma, long_sma)
            else:
                data_series = pd.Series(datos_numericos)
                prediccion_mensaje = predecir_direccion(data_series, target_value, window_size, threshold, short_sma, long_sma)
                
                # --- Visualización de Datos y S/R ---
                st.subheader("📈 Gráfico de Precios con Soporte, Resistencia y Tendencias")
                
                plot_window_for_sr = max(1, min(window_size, len(data_series)))
                soporte_line, resistencia_line = detectar_soporte_resistencia(data_series, plot_window_for_sr)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=list(range(len(data_series))), y=data_series, mode='lines+markers', name='Precios', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=list(range(len(soporte_line))), y=soporte_line, mode='lines', name='Soporte', line=dict(color='green', dash='dot')))
                fig.add_trace(go.Scatter(x=list(range(len(resistencia_line))), y=resistencia_line, mode='lines', name='Resistencia', line=dict(color='red', dash='dot')))
                
                # Añadir SMAs al gráfico
                if len(data_series) >= long_sma: # Solo si hay suficientes datos para las SMAs
                    sma_short_line = data_series.rolling(window=short_sma, min_periods=1).mean()
                    sma_long_line = data_series.rolling(window=long_sma, min_periods=1).mean()
                    fig.add_trace(go.Scatter(x=list(range(len(sma_short_line))), y=sma_short_line, mode='lines', name=f'SMA {short_sma}', line=dict(color='purple', dash='solid')))
                    fig.add_trace(go.Scatter(x=list(range(len(sma_long_line))), y=sma_long_line, mode='lines', name=f'SMA {long_sma}', line=dict(color='brown', dash='dash')))

                # Añadir líneas de alerta
                fig.add_hline(y=target_value, line_dash="dot", line_color="purple", annotation_text=f"Objetivo: {target_value:.2f}", annotation_position="top right")
                fig.add_hline(y=2.00, line_dash="dot", line_color="blue", annotation_text="Alerta 2.00", annotation_position="top left") # NUEVA LÍNEA 2.0
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

st.markdown("---")
st.caption("Desarrollado con Streamlit por tu AI asistente.")
