# Importar librerías necesarias
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Ruta al archivo CSV
ruta_csv = "C:\\Users\\wilme\\Pictures\\siza\\transporte_masivo.csv"

# Cargar datos desde un archivo CSV
datos = pd.read_csv(ruta_csv)