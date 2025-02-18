'''
Análisis de Desempeño Entrega Rápida S.A.
Universidad Fidelitas
Grupo 2
Estudio de caso 1
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, binom, poisson

# Cargar datos actividad 1
conductor_A = [15, 18, 22, 20, 17]
conductor_B = [10, 12, 15, 20, 18]
conductor_C = [25, 30, 28, 27, 29]

# Entregas realizadas
entregas = np.add(conductor_A, conductor_B)
entregas = np.add(entregas, conductor_C)
print(f"Entregas realizadas: {entregas}")

#productoescalar
#distancia recorrida
conductor_A_dis = [5,6,7,5,6]
conductor_B_dis = [3,4,5,3,4]
conductor_C_dis = [10,12,11,10,11]
#tiempos de entrega
conductor_A_tiempo = [30,35,32,30,31]
conductor_B_tiempo = [25,27,26,24,25]
conductor_C_tiempo = [50,52,51,49,50]
#producto escalar
producto_escalar_A = np.dot(conductor_A_dis,conductor_A_tiempo)
print(f"Producto escalar: {producto_escalar_A}")
producto_escalar_B = np.dot(conductor_B_dis,conductor_B_tiempo)
print(f"Producto escalar: {producto_escalar_B}")
producto_escalar_C = np.dot(conductor_C_dis,conductor_C_tiempo)
print(f"Producto escalar: {producto_escalar_C}")

#calculo matrices
Matriz_A = np.array([[15,18,22,20,17],
                    [10,12,15,20,18],
                    [25,30,28,27,29]
])
Matriz_B = np.array([[30,35,32,30,31],
                    [25,27,26,24,25],
                    [50,52,51,49,50]
])
Matriz_c = np.array([[5,6,7,5,6],
                    [3,4,5,3,4],
                    [10,12,11,10,11]
])
#suma matriz a y b
sum = np.add(Matriz_A,Matriz_B)
print(f"suma es igual{sum}")
#resta matriz a y b
resta = np.subtract(Matriz_A,Matriz_B)
print(f"resta es igual{resta}")
#multiplicacion matriz a y c
multy = (Matriz_A*Matriz_c)
print(f"multiplicacion es igual: {multy}")
#estadistica
data_tiempos_entrega = [30,32,35,30,28,40,45,32,36,30,50,42,38,33,31,29,37,36,35,31,44,48,36,34,30,40,41,39,38,33,34]
#media
media = np.mean(data_tiempos_entrega)
print(f"Media: {media}")
#mediana
mediana = np.median(data_tiempos_entrega)
print(f"Mediana: {mediana}")
#moda
moda = stats.mode(data_tiempos_entrega)
print(f"Moda: {moda}")
#rango
rango = np.ptp(data_tiempos_entrega)
print(f"Rango: {rango}")
#varianza
varianza = np.var(data_tiempos_entrega)
#desviacion estandar
desviacion_estandar = np.std(data_tiempos_entrega)
print(f"Desviacion estandar: {desviacion_estandar}")

#desviacion normal
# Parámetros de la distribución normal
mu = 36  # Media
sigma = 5  # Desviación estándar

# Intervalo de tiempo de entrega
# x = np.linspace(30,40,100)
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
y = stats.norm.pdf(x, mu, sigma)

# Probabilidad de que el tiempo de entrega esté entre 21 y 51 minutos
probabilidad = stats.norm.cdf(51, mu, sigma) - stats.norm.cdf(21, mu, sigma)
print(f"Probabilidad de que el tiempo de entrega esté entre 21 y 51 minutos: {probabilidad:.4f}")

# Graficar
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Distribución Normal', color='blue')
plt.fill_between(x, y, where=(x >= 21) & (x <= 51), color='skyblue', alpha=0.5) # Mostrar espacio entre numeros de probabilidad calculados
plt.axvline(mu, color='red', linestyle='dashed', label='Media (36 min)')
plt.axvline(21, color='red', linestyle='dashed', label='Límite Inferior (21 min)')
plt.axvline(51, color='green', linestyle='dashed', label='Límite Superior (51 min)')
plt.title('Distribución Normal de Tiempos de Entrega')
plt.xlabel('Tiempo de Entrega (minutos)')
plt.ylabel('Densidad de Probabilidad')
plt.legend()
plt.grid()
plt.show()

#distribucion binomial
# Parámetros de la distribución binomial
n = 20
p = 0.7
k = 15
#calcular la probabilidad de que exactamente 15 entregas se realicen a tiempo
probabilidad_k = binom.pmf(k, n, p)
#calcular la probabilidad de que exactamente 10 entregas se realicen a tiempo
probabilidad_k = binom.pmf(10, n, p)
#calcular la probabilidad de que exactamente 5 entregas se realicen a tiempo
probabilidad_k = binom.pmf(5, n, p)

#mostrar el resultado
print(f'La probabilidad de que se realicen exactamente 15 entregas a tiempo es: {probabilidad_k:.4f}')
print(f'La probabilidad de que se realicen exactamente 10 entregas a tiempo es: {probabilidad_k:.4f}')
print(f'La probabilidad de que se realicen exactamente 5 entregas a tiempo es: {probabilidad_k:.4f}')

#Valores posibles de k (número de éxitos)
k2 = np.arange(0,n+1)
# Calcular la probabilidad de cada valor de k
probabilidades = binom.pmf(k2, n, p)
#calcular la funcion de probabilidad (PMF)
pmf = binom.pmf(k2, n, p)
#Graficar
plt.figure(figsize=(10,6))
plt.bar(k2, pmf, label = "Distribución Binomial", color = "orange")

plt.title("Distribución Binomial")
plt.xlabel("Número de Éxitos")
plt.ylabel("Probabilidad")
plt.xticks(k2)
plt.axvline(x=15, color='r', linestyle='--', label=f"15 Entregas deseadas")
plt.axvline(x=10, color='b', linestyle='--', label=f"10 Entregas deseadas")
plt.axvline(x=5, color='g', linestyle='--', label=f"5 Entregas deseadas")
plt.grid(axis="y")
plt.legend()
plt.show()

#distribucion poisson
# Parámetro de la distribución de Poisson
lambda_ = 3 #taza promedio
k = 2
# Calcular la probabilidad de recibir 2 quejas
probabilidad_k = poisson.pmf(k, lambda_)
# Calcular la probabilidad de recibir 4 quejas
probabilidad_k = poisson.pmf(4, lambda_)
# Calcular la probabilidad de recibir 6 quejas
probabilidad_k = poisson.pmf(6, lambda_)
#mostrar el resultado
print(f'La probabilidad de recibir exactamente 2 quejas es: {probabilidad_k:.4f}')
print(f'La probabilidad de recibir exactamente 4 quejas es: {probabilidad_k:.4f}')
print(f'La probabilidad de recibir exactamente 6 quejas es: {probabilidad_k:.4f}')

#Valores posibles de k (número de quejas)
k2 = np.arange(0,11) # 0 a 10 quejas
#calcular la funcion de probabilidad (PMF)
pmf = poisson.pmf(k2, lambda_)
# Graficar
plt.figure(figsize=(10, 6))
plt.bar(k2, pmf, label='Distribución de Poisson', color='green')
plt.title('Distribución de Poisson')
plt.xlabel('Número de quejas')
plt.ylabel('Probabilidad')
plt.xticks(k2)
plt.grid(axis='y')
plt.legend()
plt.show()
