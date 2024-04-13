# Importar librerías necesarias
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Ruta al archivo CSV
ruta_csv = "C:\\Users\\wilme\\Pictures\\siza\\transporte_masivo.csv"

# Cargar datos desde un archivo CSV
datos = pd.read_csv(ruta_csv)

# Preprocesamiento de datos
X = datos[['distancia', 'tiempo', 'costo']]  # Características

# Crear un modelo de agrupamiento (K-Means)
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(X)

# Asignar las etiquetas de los clusters a los datos
datos['cluster'] = kmeans.labels_


# Calcular el coeficiente de silueta
score_silueta = silhouette_score(X, datos['cluster'])
print(f"Coeficiente de Silueta: {score_silueta}")

# Calcular el tiempo promedio de viaje para cada cluster
tiempo_promedio_por_cluster = datos.groupby('cluster')['tiempo'].mean()

# Mostrar el tiempo promedio de viaje para cada cluster
print("\nTiempo promedio de viaje por cluster:")
print(tiempo_promedio_por_cluster)

# Identificar el cluster más rápido
cluster_mas_rapido = tiempo_promedio_por_cluster.idxmin()
tiempo_cluster_mas_rapido = tiempo_promedio_por_cluster[cluster_mas_rapido]

print(f"\nEl cluster más rápido es el número {cluster_mas_rapido}, con un tiempo promedio de viaje de {tiempo_cluster_mas_rapido:.2f} minutos.")