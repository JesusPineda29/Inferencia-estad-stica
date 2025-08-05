import numpy as np
import scipy.stats as stats

# SIMULACIÓN Y ANÁLISIS
n_piezas = 500  # Piezas por turno
p_defecto = 0.025  # Nueva tasa de defectos (mejora esperada)
turnos = 40
np.random.seed(123)

# Simulación
muestra_produccion = np.random.binomial(n_piezas, p_defecto, turnos)
print("Defectos por turno (primeros 10):", muestra_produccion[:10])

# PARTE 1: INTERVALOS DE CONFIANZA
def intervalo_confianza_binomial(muestra, n, confianza=0.95):
    p_estimado = np.mean(muestra) / n
    z = stats.norm.ppf((1 + confianza) / 2)
    margen_error = z * np.sqrt((p_estimado * (1 - p_estimado)) / len(muestra))
    return (p_estimado - margen_error, p_estimado + margen_error)

ic_produccion = intervalo_confianza_binomial(muestra_produccion, n_piezas)
print(f"Intervalo de confianza: [{ic_produccion[0]*100:.2f}%, {ic_produccion[1]*100:.2f}%]")


# PARTE 2: CONTRASTE DE HIPÓTESIS
def prueba_hipotesis_binomial(muestra, n, p_0, alpha=0.05):
    p_estimado = np.mean(muestra) / n
    se = np.sqrt(p_0 * (1 - p_0) / len(muestra))
    z = (p_estimado - p_0) / se
    p_valor = 2 * (1 - stats.norm.cdf(abs(z)))
    return p_valor < alpha, p_valor

# Hipótesis
# H0: p = 0.03 (la calidad no ha cambiado)
# H1: p ≠ 0.03 (la calidad ha cambiado)

p_0 = 0.03
rechazo, p_valor = prueba_hipotesis_binomial(muestra_produccion, n_piezas, p_0)

print(f"Valor p: {p_valor:.6f}")
print(f"¿Se rechaza H0? {rechazo}")
