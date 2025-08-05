import numpy as np
import scipy.stats as stats

# SIMULACIÓN Y ANÁLISIS
n_ensayo = 1  # Cada cliente: satisfecho (1) o no satisfecho (0)
p_satisfaccion = 0.88  # Mejora esperada con productos smart
n_clientes = 120
np.random.seed(789)

# Simulación
muestra_satisfaccion = np.random.binomial(n_ensayo, p_satisfaccion, n_clientes)
print("Satisfacción clientes (1=satisfecho, 0=no):", muestra_satisfaccion[:20])

# PARTE 1: INTERVALOS DE CONFIANZA
def intervalo_confianza_satisfaccion(muestra, confianza=0.95):
    p_estimado = np.mean(muestra)
    n = len(muestra)
    z = stats.norm.ppf((1 + confianza) / 2)
    margen_error = z * np.sqrt((p_estimado * (1 - p_estimado)) / n)
    return (p_estimado - margen_error, p_estimado + margen_error)

ic_satisfaccion = intervalo_confianza_satisfaccion(muestra_satisfaccion)
print(f"Intervalo de confianza: [{ic_satisfaccion[0]*100:.2f}%, {ic_satisfaccion[1]*100:.2f}%]")


# PARTE 2: CONTRASTE DE HIPÓTESIS
def prueba_hipotesis_satisfaccion(muestra, p_0, alpha=0.05):
    p_estimado = np.mean(muestra)
    n = len(muestra)
    se = np.sqrt(p_0 * (1 - p_0) / n)
    z = (p_estimado - p_0) / se
    p_valor = 2 * (1 - stats.norm.cdf(abs(z)))
    return p_valor < alpha, p_valor

# Hipótesis
# H0: p = 0.85 (mantenemos el 85% de satisfacción)
# H1: p ≠ 0.85 (la satisfacción ha cambiado)

p_0 = 0.85
rechazo, p_valor = prueba_hipotesis_satisfaccion(muestra_satisfaccion, p_0)

print(f"Valor p: {p_valor:.6f}")
print(f"¿Se rechaza H0? {rechazo}")

# DECISIÓN EMPRESARIAL
if rechazo:
    satisfaccion_actual = np.mean(muestra_satisfaccion)
    if satisfaccion_actual > p_0:
        print("DECISIÓN: La satisfacción SÍ ha MEJORADO significativamente.")
        print(f"Ahora tienen {satisfaccion_actual*100:.1f}% de satisfacción. ¡Pueden promocionarlo!")
    else:
        print("DECISIÓN: La satisfacción ha EMPEORADO significativamente.")
        print("Deben tomar acciones correctivas urgentemente.")
else:
    print("DECISIÓN: La satisfacción se mantiene estable en el 85%.")
    print("Pueden continuar con la estrategia actual.")