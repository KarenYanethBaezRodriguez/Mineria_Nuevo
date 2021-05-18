import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

def print_tabulate(df: pd.DataFrame):
    print(tabulate(df, headers=df.columns, tablefmt='orgtbl'))

covid = pd.read_csv('Covid.csv', names=['Economia','code','Disminución_de_crecimiento_en_consumo_CP','Disminución_del_crecimiento_del_consumo_largo_plazo','Disminucion_de_crecimiento_de_la_inversion_CP','Disminucion_de_crecimiento_de_la_inversion_LP','Disminucion_de_ingresos_por_turismo_CP','Disminucion_de_ingresos_por_turismo_LP','Indice_de_rigurosidad_promedio','Movilidad_promedio'])
print_tabulate(covid)
# create a figure and axis
fig, ax = plt.subplots()

# scatter the sepal_length against the sepal_width
ax.scatter(covid['Disminución_del_crecimiento_del_consumo_largo_plazo'], covid['code'])
# set a title and labels
ax.set_title('Covid')
ax.set_xlabel('porcentaje')
ax.set_ylabel('Paises')
plt.show()
