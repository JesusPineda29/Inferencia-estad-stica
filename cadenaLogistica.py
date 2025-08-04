import numpy as np
import scipy.stats as stats

# SIMULACIÓN Y ANÁLISIS
lambda_param = 14  # Nueva tasa esperada (cambio por el sistema automatizado)
tamaño_muestra = 60
np.random.seed(42)  # Para reproducibilidad

# Simulación de datos
muestra_logistica = np.random.poisson(lambda_param, tamaño_muestra)
print("Muestra de pedidos por hora:", muestra_logistica[:10])  # Primeros 10 valores

# PARTE 1: INTERVALOS DE CONFIANZA
def intervalo_confianza_poisson(muestra, confianza=0.95):
    media_muestral = np.mean(muestra)
    n = len(muestra)
    error_estandar = np.sqrt(media_muestral / n)
    z = stats.norm.ppf((1 + confianza) / 2)
    margen_error = z * error_estandar
    return media_muestral - margen_error, media_muestral + margen_error

ic_logistica = intervalo_confianza_poisson(muestra_logistica)
print(f"Intervalo de confianza: [{ic_logistica[0]:.2f}, {ic_logistica[1]:.2f}]")

# PARTE 2: CONTRASTE DE HIPÓTESIS
def prueba_hipotesis_poisson(muestra, lambda_0, alpha=0.05):
    media_muestral = np.mean(muestra)
    n = len(muestra)
    z = (media_muestral - lambda_0) / np.sqrt(lambda_0 / n)
    p_valor = 2 * (1 - stats.norm.cdf(abs(z)))
    return p_valor < alpha, p_valor

# Hipótesis
# H0: λ = 12 (la tasa no ha cambiado)
# H1: λ ≠ 12 (la tasa ha cambiado significativamente)

lambda_0 = 12
rechazo, p_valor = prueba_hipotesis_poisson(muestra_logistica, lambda_0)

print(f"Valor p: {p_valor:.6f}")
print(f"¿Se rechaza H0? {rechazo}")

# DECISIÓN EMPRESARIAL
if rechazo:
    print("DECISIÓN: El nuevo sistema SÍ ha afectado significativamente la tasa de pedidos.")
    print("LogiMéxico debe reajustar su plantilla y capacidad operativa.")
else:
    print("DECISIÓN: No hay evidencia suficiente de cambio.")
    print("Pueden mantener la operación actual.")