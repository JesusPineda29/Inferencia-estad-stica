# JUPYTER NOTEBOOK - ACTIVIDAD II: INFERENCIA ESTADÍSTICA
# Análisis de 4 Sistemas Empresariales

# =============================================================================
# CELDA 1: IMPORTACIÓN DE LIBRERÍAS
# =============================================================================

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

print("Librerías importadas correctamente")
print("Listo para análisis de inferencia estadística")

# =============================================================================
# SISTEMA 1: CONTROL DE CADENA LOGÍSTICA DE DISTRIBUCIÓN
# Distribución: POISSON
# =============================================================================

"""
## 🏢 SISTEMA 1: CADENA LOGÍSTICA - LogiMéxico S.A.

**CONTEXTO EMPRESARIAL:**
LogiMéxico S.A., distribuidora farmacéutica líder en CDMX, implementó un nuevo 
sistema automatizado de gestión de pedidos. Históricamente manejaban 12 pedidos/hora.
Necesitan verificar si el nuevo sistema ha afectado esta tasa.

**PARÁMETROS DEL ANÁLISIS:**
- Tasa histórica: 12 pedidos/hora
- Muestra: 60 horas de operación
- Nivel de confianza: 95%
- Nueva tasa esperada: 14 pedidos/hora (mejora por automatización)
"""

# SIMULACIÓN DE DATOS - SISTEMA LOGÍSTICO
lambda_param = 14  # PARÁMETRO MODIFICADO: nueva tasa esperada
tamaño_muestra_logistica = 60  # PARÁMETRO MODIFICADO: más horas de análisis
np.random.seed(42)  # Para reproducibilidad

muestra_logistica = np.random.poisson(lambda_param, tamaño_muestra_logistica)

print("SISTEMA 1: CADENA LOGÍSTICA")
print("="*50)
print(f" Muestra de pedidos por hora (primeros 15): {muestra_logistica[:15]}")
print(f" Media muestral: {np.mean(muestra_logistica):.2f} pedidos/hora")
print(f" Desviación estándar: {np.std(muestra_logistica):.2f}")

# PARTE 1: INTERVALO DE CONFIANZA (usando código original)
def intervalo_confianza_poisson(muestra, confianza=0.95):
    media_muestral = np.mean(muestra)
    n = len(muestra)
    error_estandar = np.sqrt(media_muestral / n)
    z = stats.norm.ppf((1 + confianza) / 2)
    margen_error = z * error_estandar
    return media_muestral - margen_error, media_muestral + margen_error

ic_logistica = intervalo_confianza_poisson(muestra_logistica)
print(f"\n INTERVALO DE CONFIANZA (95%):")
print(f"   Rango: {ic_logistica[0]:.2f} - {ic_logistica[1]:.2f} pedidos/hora")

# PARTE 2: CONTRASTE DE HIPÓTESIS (usando código original)
def prueba_hipotesis_poisson(muestra, lambda_0, alpha=0.05):
    media_muestral = np.mean(muestra)
    n = len(muestra)
    z = (media_muestral - lambda_0) / np.sqrt(lambda_0 / n)
    p_valor = 2 * (1 - stats.norm.cdf(abs(z)))
    return p_valor < alpha, p_valor

# H0: λ = 12 (la tasa no ha cambiado)
# H1: λ ≠ 12 (la tasa ha cambiado significativamente)
lambda_0_logistica = 12  # Tasa histórica
rechazo_logistica, p_valor_logistica = prueba_hipotesis_poisson(muestra_logistica, lambda_0_logistica)

print(f"\n CONTRASTE DE HIPÓTESIS:")
print(f"   H0: λ = {lambda_0_logistica} pedidos/hora (sin cambio)")
print(f"   H1: λ ≠ {lambda_0_logistica} pedidos/hora (hay cambio)")
print(f"   P-valor: {p_valor_logistica:.4f}")
print(f"   ¿Se rechaza H0?: {'SÍ' if rechazo_logistica else 'NO'}")

# INTERPRETACIÓN EMPRESARIAL
print(f"\n DECISIÓN EMPRESARIAL:")
if rechazo_logistica:
    print("    El nuevo sistema SÍ cambió significativamente la tasa de pedidos")
    print("    LogiMéxico debe reajustar plantilla y capacidad operativa")
else:
    print("    No hay evidencia suficiente de cambio en la tasa")
    print("    Pueden mantener la operación actual")

# =============================================================================
# SISTEMA 2: CONTROL DE CENTRO DE PRODUCCIÓN
# Distribución: BINOMIAL
# =============================================================================

"""
##  SISTEMA 2: CENTRO DE PRODUCCIÓN - Manufacturas Precisión S.A.

**CONTEXTO EMPRESARIAL:**
Manufacturas Precisión S.A. produce componentes de frenos automotrices en Aguascalientes.
Instalaron nueva maquinaria y necesitan verificar si la calidad cambió.
Tasa histórica de defectos: 3%

**PARÁMETROS DEL ANÁLISIS:**
- Tasa histórica de defectos: 3%
- Piezas por turno: 500
- Turnos analizados: 40
- Nueva tasa esperada: 2.5% (mejora esperada)
"""

# SIMULACIÓN DE DATOS - SISTEMA PRODUCCIÓN
n_piezas = 500  # Piezas por turno
p_defecto = 0.025  # PARÁMETRO MODIFICADO: mejora esperada (2.5% vs 3% histórico)
turnos = 40
np.random.seed(123)

muestra_produccion = np.random.binomial(n_piezas, p_defecto, turnos)

print("\n\n SISTEMA 2: CENTRO DE PRODUCCIÓN")
print("="*50)
print(f"🔧 Defectos por turno (primeros 15): {muestra_produccion[:15]}")
print(f" Promedio de defectos por turno: {np.mean(muestra_produccion):.1f}")
print(f" Proporción estimada de defectos: {np.mean(muestra_produccion)/n_piezas:.4f} ({np.mean(muestra_produccion)/n_piezas*100:.2f}%)")

# PARTE 1: INTERVALO DE CONFIANZA (usando código original)
def intervalo_confianza_binomial(muestra, n, confianza=0.95):
    p_estimado = np.mean(muestra) / n
    z = stats.norm.ppf((1 + confianza) / 2)
    margen_error = z * np.sqrt((p_estimado * (1 - p_estimado)) / len(muestra))
    return (p_estimado - margen_error, p_estimado + margen_error)

ic_produccion = intervalo_confianza_binomial(muestra_produccion, n_piezas)
print(f"\n INTERVALO DE CONFIANZA (95%):")
print(f"   Proporción de defectos: {ic_produccion[0]:.4f} - {ic_produccion[1]:.4f}")
print(f"   En porcentaje: {ic_produccion[0]*100:.2f}% - {ic_produccion[1]*100:.2f}%")

# PARTE 2: CONTRASTE DE HIPÓTESIS (usando código original)
def prueba_hipotesis_binomial(muestra, n, p_0, alpha=0.05):
    p_estimado = np.mean(muestra) / n
    se = np.sqrt(p_0 * (1 - p_0) / len(muestra))
    z = (p_estimado - p_0) / se
    p_valor = 2 * (1 - stats.norm.cdf(abs(z)))
    return p_valor < alpha, p_valor

# H0: p = 0.03 (la calidad no ha cambiado)
# H1: p ≠ 0.03 (la calidad ha cambiado)
p_0_produccion = 0.03  # Tasa histórica de defectos
rechazo_produccion, p_valor_produccion = prueba_hipotesis_binomial(muestra_produccion, n_piezas, p_0_produccion)

print(f"\n CONTRASTE DE HIPÓTESIS:")
print(f"   H0: p = {p_0_produccion} (3% defectos - sin cambio)")
print(f"   H1: p ≠ {p_0_produccion} (hay cambio en calidad)")
print(f"   P-valor: {p_valor_produccion:.4f}")
print(f"   ¿Se rechaza H0?: {'SÍ' if rechazo_produccion else 'NO'}")

# INTERPRETACIÓN EMPRESARIAL
print(f"\n DECISIÓN EMPRESARIAL:")
if rechazo_produccion:
    if np.mean(muestra_produccion)/n_piezas < p_0_produccion:
        print("    ¡EXCELENTE! La nueva maquinaria MEJORÓ la calidad")
        print("    Pueden promocionar la mejora en calidad")
    else:
        print("    La nueva maquinaria EMPEORÓ la calidad")
        print("    Requieren ajustes urgentes en el proceso")
else:
    print("    La calidad se mantiene estable (3% defectos)")
    print("    Pueden continuar con la operación normal")

# =============================================================================
# SISTEMA 3: NIVEL DE SERVICIO DE ENTREGAS
# Distribución: NORMAL
# =============================================================================

"""
##  SISTEMA 3: SERVICIO DE ENTREGAS - E-Commerce Plus

**CONTEXTO EMPRESARIAL:**
E-Commerce Plus compite con Amazon México prometiendo entregas en 4 días o menos.
Con nuevos centros de distribución, necesitan verificar si cumplen esta promesa.
CEO requiere datos estadísticos para decisiones comerciales.

**PARÁMETROS DEL ANÁLISIS:**
- Promesa comercial: 4.0 días máximo
- Muestra: 80 entregas
- Tiempo real esperado: 4.2 días (ligero retraso)
- Desviación estándar: 0.8 días (menor variabilidad)
"""

# SIMULACIÓN DE DATOS - SISTEMA ENTREGAS
media_entrega = 4.2  # PARÁMETRO MODIFICADO: ligeramente por encima de promesa
desv_std = 0.8  # PARÁMETRO MODIFICADO: menor variabilidad
n_entregas = 80  # PARÁMETRO MODIFICADO: más entregas analizadas
np.random.seed(456)

muestra_entregas = np.random.normal(media_entrega, desv_std, n_entregas)

print("\n\n SISTEMA 3: SERVICIO DE ENTREGAS")
print("="*50)
print(f" Tiempos de entrega en días (primeros 15): {muestra_entregas[:15].round(2)}")
print(f" Tiempo promedio de entrega: {np.mean(muestra_entregas):.2f} días")
print(f" Desviación estándar: {np.std(muestra_entregas):.2f} días")

# PARTE 1: INTERVALO DE CONFIANZA (usando código original)
def intervalo_confianza_normal(muestra, confianza=0.95):
    media_muestra = np.mean(muestra)
    error_estandar = stats.sem(muestra)
    return stats.t.interval(confianza, len(muestra)-1, loc=media_muestra, scale=error_estandar)

ic_entregas = intervalo_confianza_normal(muestra_entregas)
print(f"\n INTERVALO DE CONFIANZA (95%):")
print(f"   Tiempo promedio real: {ic_entregas[0]:.2f} - {ic_entregas[1]:.2f} días")

# PARTE 2: CONTRASTE DE HIPÓTESIS (usando código original)
def prueba_hipotesis_normal(muestra, mu_0, alpha=0.05):
    t_stat, p_valor = stats.ttest_1samp(muestra, mu_0)
    return p_valor < alpha, p_valor

# H0: μ = 4.0 (cumplimos la promesa de 4 días)
# H1: μ ≠ 4.0 (no cumplimos la promesa)
mu_0_entregas = 4.0  # Promesa comercial
rechazo_entregas, p_valor_entregas = prueba_hipotesis_normal(muestra_entregas, mu_0_entregas)

print(f"\n CONTRASTE DE HIPÓTESIS:")
print(f"   H0: μ = {mu_0_entregas} días (cumplimos promesa)")
print(f"   H1: μ ≠ {mu_0_entregas} días (no cumplimos promesa)")
print(f"   P-valor: {p_valor_entregas:.4f}")
print(f"   ¿Se rechaza H0?: {'SÍ' if rechazo_entregas else 'NO'}")

# INTERPRETACIÓN EMPRESARIAL
print(f"\n DECISIÓN EMPRESARIAL:")
if rechazo_entregas:
    if np.mean(muestra_entregas) > mu_0_entregas:
        print("    NO estamos cumpliendo la promesa de 4 días")
        print("    Opciones: Mejorar logística O cambiar promesa comercial")
    else:
        print("    ¡Entregamos MÁS RÁPIDO que la promesa!")
        print("    Oportunidad de mejorar la oferta comercial")
else:
    print("    SÍ cumplimos la promesa de 4 días")
    print("    Pueden mantener o mejorar la oferta comercial")

# =============================================================================
# SISTEMA 4: SATISFACCIÓN DE CLIENTES
# Distribución: BINOMIAL
# =============================================================================

"""
##  SISTEMA 4: SATISFACCIÓN DE CLIENTES - TecnoHogar México

**CONTEXTO EMPRESARIAL:**
TecnoHogar México fabrica electrodomésticos de línea blanca.
Lanzaron productos "smart" y buscan mantener 85% de satisfacción.
Necesitan medir el impacto de la nueva tecnología en la satisfacción.

**PARÁMETROS DEL ANÁLISIS:**
- Objetivo de satisfacción: 85%
- Clientes encuestados: 120
- Satisfacción esperada: 88% (mejora con productos smart)
"""

# SIMULACIÓN DE DATOS - SISTEMA SATISFACCIÓN
n_ensayo = 1  # Cada cliente: satisfecho (1) o no satisfecho (0)
p_satisfaccion = 0.88  # PARÁMETRO MODIFICADO: mejora esperada con productos smart
n_clientes = 120  # PARÁMETRO MODIFICADO: más clientes encuestados
np.random.seed(789)

muestra_satisfaccion = np.random.binomial(n_ensayo, p_satisfaccion, n_clientes)

print("\n\n SISTEMA 4: SATISFACCIÓN DE CLIENTES")
print("="*50)
print(f" Respuestas (1=satisfecho, 0=no) primeras 30: {muestra_satisfaccion[:30]}")
print(f" Clientes satisfechos: {np.sum(muestra_satisfaccion)} de {n_clientes}")
print(f" Proporción de satisfacción: {np.mean(muestra_satisfaccion):.3f} ({np.mean(muestra_satisfaccion)*100:.1f}%)")

# PARTE 1: INTERVALO DE CONFIANZA (usando código original)
def intervalo_confianza_satisfaccion(muestra, confianza=0.95):
    p_estimado = np.mean(muestra)
    n = len(muestra)
    z = stats.norm.ppf((1 + confianza) / 2)
    margen_error = z * np.sqrt((p_estimado * (1 - p_estimado)) / n)
    return (p_estimado - margen_error, p_estimado + margen_error)

ic_satisfaccion = intervalo_confianza_satisfaccion(muestra_satisfaccion)
print(f"\n INTERVALO DE CONFIANZA (95%):")
print(f"   Proporción de satisfacción: {ic_satisfaccion[0]:.3f} - {ic_satisfaccion[1]:.3f}")
print(f"   En porcentaje: {ic_satisfaccion[0]*100:.1f}% - {ic_satisfaccion[1]*100:.1f}%")

# PARTE 2: CONTRASTE DE HIPÓTESIS (usando código original)
def prueba_hipotesis_satisfaccion(muestra, p_0, alpha=0.05):
    p_estimado = np.mean(muestra)
    n = len(muestra)
    se = np.sqrt(p_0 * (1 - p_0) / n)
    z = (p_estimado - p_0) / se
    p_valor = 2 * (1 - stats.norm.cdf(abs(z)))
    return p_valor < alpha, p_valor

# H0: p = 0.85 (mantenemos el 85% de satisfacción)
# H1: p ≠ 0.85 (la satisfacción ha cambiado)
p_0_satisfaccion = 0.85  # Objetivo de satisfacción
rechazo_satisfaccion, p_valor_satisfaccion = prueba_hipotesis_satisfaccion(muestra_satisfaccion, p_0_satisfaccion)

print(f"\n CONTRASTE DE HIPÓTESIS:")
print(f"   H0: p = {p_0_satisfaccion} (85% satisfacción - objetivo)")
print(f"   H1: p ≠ {p_0_satisfaccion} (satisfacción cambió)")
print(f"   P-valor: {p_valor_satisfaccion:.4f}")
print(f"   ¿Se rechaza H0?: {'SÍ' if rechazo_satisfaccion else 'NO'}")

# INTERPRETACIÓN EMPRESARIAL
print(f"\n DECISIÓN EMPRESARIAL:")
if rechazo_satisfaccion:
    if np.mean(muestra_satisfaccion) > p_0_satisfaccion:
        print("    ¡EXCELENTE! Los productos smart MEJORARON la satisfacción")
        print("   📢 Pueden promocionar este logro en marketing")
    else:
        print("    Los productos smart REDUJERON la satisfacción")
        print("    Necesitan revisar la experiencia del usuario")
else:
    print("    La satisfacción se mantiene estable (85%)")
    print("    Pueden continuar con la estrategia actual")

# =============================================================================
# GRÁFICA DE DISTRIBUCIONES (OPCIONAL)
# =============================================================================

"""
##  VISUALIZACIÓN DE LAS DISTRIBUCIONES
"""

plt.figure(figsize=(15, 10))

# Gráfica 1: Distribución Poisson (Logística)
plt.subplot(2, 2, 1)
plt.hist(muestra_logistica, bins=15, alpha=0.7, color='blue', edgecolor='black')
plt.axvline(np.mean(muestra_logistica), color='red', linestyle='--', 
           label=f'Media: {np.mean(muestra_logistica):.1f}')
plt.axvline(lambda_0_logistica, color='green', linestyle='--', 
           label=f'Histórico: {lambda_0_logistica}')
plt.title('Sistema 1: Pedidos por Hora (Poisson)')
plt.xlabel('Pedidos por hora')
plt.ylabel('Frecuencia')
plt.legend()

# Gráfica 2: Distribución Binomial (Producción)
plt.subplot(2, 2, 2)
plt.hist(muestra_produccion, bins=15, alpha=0.7, color='orange', edgecolor='black')
plt.axvline(np.mean(muestra_produccion), color='red', linestyle='--', 
           label=f'Media: {np.mean(muestra_produccion):.1f}')
plt.axvline(p_0_produccion * n_piezas, color='green', linestyle='--', 
           label=f'Histórico: {p_0_produccion * n_piezas:.1f}')
plt.title('Sistema 2: Defectos por Turno (Binomial)')
plt.xlabel('Número de defectos')
plt.ylabel('Frecuencia')
plt.legend()

# Gráfica 3: Distribución Normal (Entregas)
plt.subplot(2, 2, 3)
plt.hist(muestra_entregas, bins=15, alpha=0.7, color='green', edgecolor='black')
plt.axvline(np.mean(muestra_entregas), color='red', linestyle='--', 
           label=f'Media: {np.mean(muestra_entregas):.1f}')
plt.axvline(mu_0_entregas, color='blue', linestyle='--', 
           label=f'Promesa: {mu_0_entregas}')
plt.title('Sistema 3: Tiempo de Entrega (Normal)')
plt.xlabel('Días')
plt.ylabel('Frecuencia')
plt.legend()

# Gráfica 4: Distribución Binomial (Satisfacción)
plt.subplot(2, 2, 4)
valores_satisfaccion = [0, 1]
conteos = [np.sum(muestra_satisfaccion == 0), np.sum(muestra_satisfaccion == 1)]
plt.bar(valores_satisfaccion, conteos, alpha=0.7, color='purple', edgecolor='black')
plt.title('Sistema 4: Satisfacción de Clientes')
plt.xlabel('Satisfacción (0=No, 1=Sí)')
plt.ylabel('Número de clientes')
plt.xticks([0, 1], ['No Satisfecho', 'Satisfecho'])

plt.tight_layout()
plt.show()

# =============================================================================
# RESUMEN EJECUTIVO
# =============================================================================

print("\n\n" + "="*70)
print("📋 RESUMEN EJECUTIVO - INFERENCIA ESTADÍSTICA")
print("="*70)

sistemas = [
    (" LogiMéxico (Logística)", rechazo_logistica, p_valor_logistica),
    (" Manufacturas Precisión (Producción)", rechazo_produccion, p_valor_produccion),
    (" E-Commerce Plus (Entregas)", rechazo_entregas, p_valor_entregas),
    (" TecnoHogar (Satisfacción)", rechazo_satisfaccion, p_valor_satisfaccion)
]

for i, (nombre, rechazo, p_valor) in enumerate(sistemas, 1):
    print(f"\n{i}. {nombre}")
    print(f"   P-valor: {p_valor:.4f}")
    print(f"   Decisión: {'RECHAZAR H0' if rechazo else 'NO RECHAZAR H0'}")
    print(f"   Estado: {'CAMBIO SIGNIFICATIVO' if rechazo else 'SIN CAMBIO SIGNIFICATIVO'}")

print(f"\n NIVEL DE CONFIANZA UTILIZADO: 95% (α = 0.05)")
print(f" TODOS LOS ANÁLISIS COMPLETADOS EXITOSAMENTE")
print("="*70)