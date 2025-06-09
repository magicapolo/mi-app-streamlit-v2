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
    # Aseguramos que la ventana sea al menos 1 y no mayor que los datos disponibles.
    # min_periods=1 permite calcular el rolling min/max incluso con un solo dato en la ventana inicial.
    actual_window = max(1, min(window, len(data)))
    
    if len(data) == 0:
        return pd.Series([]), pd.Series([]) # Retorna series vacías si no hay datos
    
    soporte = data.rolling(window=actual_window, min_periods=1).min()
    resistencia = data.rolling(window=actual_window, min_periods=1).max()
    
    return soporte, resistencia

def predecir_direccion(data: pd.Series, target_value: float = 1.50, window: int = 10, umbral: float = 0.01):
    """
    Predice si el siguiente valor será mayor o menor al target_value (ej. 1.50)
    basándose en la posición actual respecto a soporte y resistencia.
    También incluye alertas para valores altos.
    
    Args:
        data (pd.Series): Serie de datos de precios.
        target_value (float): El valor objetivo (ej. 1.50) para la predicción.
        window (int): Tamaño deseado de la ventana para calcular soporte/resistencia.
        umbral (float): Margen alrededor de soporte/resistencia para la predicción.
        
    Returns:
        str: Mensaje de predicción o estado.
    """
    if len(data) == 0:
        st.error("⚠️ **Error:** No hay datos para predecir.")
        return "No hay datos"
    
    # --- Manejar 'menos coeficientes' / ventana dinámica ---
    # Si la ventana seleccionada es mayor que los datos disponibles, ajustamos la ventana.
    # Un mínimo de 2 suele ser más significativo para S/R, pero permitimos 1 para casos extremos.
    adjusted_window = max(2, min(window, len(data)))
    if len(data) == 1: # Caso especial para un solo punto de datos
        adjusted_window = 1
        st.warning("ℹ️ **Nota:** Con un solo punto de datos, el soporte y la resistencia son el mismo valor. La predicción será limitada.")
    elif adjusted_window < window:
        st.warning(f"ℹ️ **Nota:** La ventana deseada de `{window}` es mayor que los datos disponibles (`{len(data)}`). Se ajustó la ventana a `{adjusted_window}`.")

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

    # --- Alerta para valores mayores a 5.00 ---
    if actual >= 5.00:
        st.error(f"🚨 **¡ALERTA!** El valor actual ({actual:.4f}) es igual o mayor a **5.00**. ¡Presta atención!")
    elif resistencia_actual >= 5.00: # También alertamos si la resistencia ya está muy alta
        st.warning(f"🔔 **Advertencia:** El nivel de resistencia detectado ({resistencia_actual:.4f}) es igual o mayor a **5.00**. Podría indicar un movimiento fuerte.")
    
    if np.isnan(soporte_actual) or np.isnan(resistencia_actual):
        st.warning("⚠️ **Advertencia:** Los niveles de soporte/resistencia no se pudieron calcular completamente con los datos/ventana actuales. La predicción puede ser menos precisa.")
        return "Insuficientes datos o S/R no calculable"

    # Lógica de predicción principal
    prediccion_mensaje = "No hay una señal clara."
    if actual <= soporte_actual + umbral:
        # Si el precio está cerca o por debajo del soporte, es probable que rebote al alza
        predicacion_mensaje = f"📈 **PREDICCIÓN:** Se espera un movimiento **MAYOR a {target_value:.4f}** (cerca del soporte)."
        st.success(prediccion_mensaje)
    elif actual >= resistencia_actual - umbral:
        # Si el precio está cerca o por encima de la resistencia, es probable que rebote a la baja
        prediccion_mensaje = f"📉 **PREDICCIÓN:** Se espera un movimiento **MENOR a {target_value:.4f}** (cerca de la resistencia)."
        st.error(prediccion_mensaje)
    else:
        # Si está en medio, es incierto
        prediccion_mensaje = "🤔 **PREDICCIÓN:** El valor actual está entre soporte y resistencia. La dirección es **INCIERTA**."
        st.warning(prediccion_mensaje)
    
    return prediccion_mensaje

# --- Configuración de la Página Streamlit ---
st.set_page_config(
    page_title="Predicción de Trading (Soporte/Resistencia)",
    page_icon="📈",
    layout="wide" # Diseño amplio para mejor visualización
)

st.title("🤖 Bot Predictor de Trading (Soporte y Resistencia)")

st.markdown("""
Esta aplicación te ayuda a predecir si el siguiente valor de un activo será **MAYOR** o **MENOR** a un valor objetivo
basándose en los conceptos de soporte y resistencia de una ventana móvil. **Ahora con ajustes para datos limitados y alertas de valores altos.**
""")

# --- Entrada de Datos del Usuario ---
st.sidebar.header("⚙️ Configuración y Datos")

datos_input_str = st.sidebar.text_area(
    "**Ingresa tus valores numéricos históricos (separados por comas):**",
    value="1.45, 1.48, 1.52, 1.49, 1.51, 1.47, 1.53, 1.46, 1.50, 1.49, 1.51, 1.52, 1.50, 1.48, 1.47, 1.49, 1.51, 1.50",
    height=150,
    help="Introduce una serie de números que representen los precios del activo."
)

target_value = st.sidebar.number_input(
    "**Valor Objetivo para la Predicción (ej. 1.50):**",
    min_value=0.01, max_value=100.0, value=1.50, step=0.01,
    help="El valor contra el cual se compara la predicción (mayor o menor)."
)

window_size = st.sidebar.slider(
    "**Tamaño de la Ventana Móvil (para S/R):**",
    min_value=1, max_value=50, value=10, step=1, # Se puede ajustar hasta 1
    help="Número de puntos de datos para calcular el soporte y la resistencia. Se ajustará dinámicamente si hay menos datos."
)

threshold = st.sidebar.slider(
    "**Umbral de Sensibilidad (Margen S/R):**",
    min_value=0.001, max_value=0.1, value=0.01, format="%.3f", step=0.001,
    help="Define qué tan cerca debe estar el precio de S/R para activar una predicción."
)

st.sidebar.markdown("---") # Divisor en la barra lateral

# --- Botón de Predicción ---
if st.sidebar.button("✨ Realizar Predicción"):
    if not datos_input_str:
        st.error("Por favor, ingresa los valores numéricos para poder realizar la predicción.")
    else:
        try:
            # Convierte el string de entrada a una lista de floats
            datos_numericos = [float(x.strip()) for x in datos_input_str.split(',') if x.strip()]
            
            if not datos_numericos:
                st.error("No se detectaron números válidos en la entrada. Revisa el formato.")
            else:
                data_series = pd.Series(datos_numericos)
                
                # Genera la predicción
                prediccion_mensaje = predecir_direccion(data_series, target_value, window_size, threshold)
                
                # --- Visualización de Datos y S/R ---
                st.subheader("📈 Gráfico de Precios con Soporte y Resistencia")
                
                # Para la gráfica, usamos la ventana que se usó realmente para calcular S/R
                # y la pasamos para dibujar las líneas
                plot_window_for_sr = max(1, min(window_size, len(data_series)))
                soporte_line, resistencia_line = detectar_soporte_resistencia(data_series, plot_window_for_sr)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=list(range(len(data_series))), y=data_series, mode='lines+markers', name='Precios', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=list(range(len(soporte_line))), y=soporte_line, mode='lines', name='Soporte', line=dict(color='green', dash='dot')))
                fig.add_trace(go.Scatter(x=list(range(len(resistencia_line))), y=resistencia_line, mode='lines', name='Resistencia', line=dict(color='red', dash='dot')))
                
                # Añadir el valor objetivo
                fig.add_hline(y=target_value, line_dash="dot", line_color="purple", annotation_text=f"Objetivo: {target_value:.2f}", annotation_position="top right")

                # Añadir línea de alerta de 5.00
                fig.add_hline(y=5.00, line_dash="dash", line_color="orange", annotation_text="Alerta 5.00", annotation_position="top left")


                fig.update_layout(
                    title='Historial de Precios y Niveles S/R',
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
