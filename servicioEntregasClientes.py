import numpy as np
import scipy.stats as stats

# SIMULACIÓN Y ANÁLISIS
media_entrega = 4.2  # Ligeramente por encima de la promesa
desv_std = 0.8  # Menor variabilidad por mejores procesos
n_entregas = 80
np.random.seed(456)

# Simulación
muestra_entregas = np.random.normal(media_entrega, desv_std, n_entregas)
print("Tiempos de entrega (primeros 10):", muestra_entregas[:10])

# PARTE 1: INTERVALOS DE CONFIANZA
def intervalo_confianza_normal(muestra, confianza=0.95):
    media_muestra = np.mean(muestra)
    error_estandar = stats.sem(muestra)
    return stats.t.interval(confianza, len(muestra)-1, 
                           loc=media_muestra, scale=error_estandar)

ic_entregas = intervalo_confianza_normal(muestra_entregas)
print(f"Intervalo de confianza: [{ic_entregas[0]:.2f}, {ic_entregas[1]:.2f}] días")


# PARTE 2: CONTRASTE DE HIPÓTESIS
def prueba_hipotesis_normal(muestra, mu_0, alpha=0.05):
    t_stat, p_valor = stats.ttest_1samp(muestra, mu_0)
    return p_valor < alpha, p_valor

# Hipótesis
# H0: μ = 4.0 (cumplimos la promesa de 4 días)
# H1: μ ≠ 4.0 (no cumplimos la promesa)

mu_0 = 4.0
rechazo, p_valor = prueba_hipotesis_normal(muestra_entregas, mu_0)

print(f"Valor p: {p_valor:.6f}")
print(f"¿Se rechaza H0? {rechazo}")

# DECISIÓN EMPRESARIAL
if rechazo:
    tiempo_promedio = np.mean(muestra_entregas)
    if tiempo_promedio > mu_0:
        print("DECISIÓN: NO están cumpliendo la promesa de 4 días.")
        print("Se están tardando MÁS de lo prometido. Deben mejorar logística o cambiar la promesa comercial.")
    else:
        print("DECISIÓN: Están entregando MÁS RÁPIDO de lo prometido.")
        print("¡Excelente! Pueden usar esto como ventaja competitiva.")
else:
    print("DECISIÓN: SÍ están cumpliendo la promesa de 4 días.")
    print("Pueden mantener o incluso mejorar la oferta comercial.")