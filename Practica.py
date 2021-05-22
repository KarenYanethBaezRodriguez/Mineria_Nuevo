from os import linesep
import matplotlib.pyplot as plt
import requests
import io
from bs4 import BeautifulSoup
import pandas as pd
from tabulate import tabulate
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np
from sklearn import linear_model
import numbers
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from pandas import DataFrame



#from sklearn.metrics import mean_squared_error, r2_score

#KAREN YANETH BAEZ RODRIGUEZ / MINERIA DE DATOS

#La base de Datos trata sobre Los contagios y falleciomiento en México por COVID 

#DATA ADQUISITION
df = pd.read_csv("Covid.csv") #get_csv_from_url()


def print_tabulate(df: pd.DataFrame):
    print(tabulate(df, headers=df.columns, tablefmt='orgtbl'))

#DATA CLAENING
df= df.drop(['Casos por 100000 habitantes'], axis=1)

#'Disminucion_de_crecimiento_en_consumo_CP'
#DATA PARSING

df.columns=['Estados','Contagios','Fallecidos','Recuperados']
print_tabulate(df)

#La base de Datos trata sobre las perdidas econimicas por el covid en ciertos paises 


#DATA VISUALIZATION
valores = df[["Estados","Contagios"]]

ax= valores.plot.bar(x="Estados",y="Contagios", rot = 0)

ax.set_title('Covid en Mexico')
ax.set_xlabel('estados')
ax.set_ylabel('contagios')
#plt.show()
plt.xticks(rotation=90)
plt.savefig(f"img/covid.png")

fallecidos = df[["Estados","Fallecidos"]]

ax= fallecidos.plot.bar(x="Estados",y="Fallecidos", rot = 0)

ax.set_title('Covid en Mexico')
ax.set_xlabel('estados')
ax.set_ylabel('muertes')
#plt.show()
plt.xticks(rotation=90)
plt.savefig(f"img/fallecidos.png")

recuperados = df[["Estados","Recuperados"]]

ax= recuperados.plot.bar(x="Estados",y="Recuperados", rot = 0)

ax.set_title('Covid en Mexico')
ax.set_xlabel('estados')
ax.set_ylabel('recuperados')
#plt.show()
plt.xticks(rotation=90)
plt.savefig(f"img/recuperados.png")

#STATIC TEST

modl = ols("Recuperados ~ Contagios", data=df).fit()
anova_df = sm.stats.anova_lm(modl, typ=2)
if anova_df["PR(>F)"][0] < 0.005:
    print("hay diferencias")
    print(anova_df)
    # Prueba tukey
    # imprimir los resultados
else:
    print("No hay diferencias")

from bioinfokit.analys import stat
res = stat()
res.anova_stat(df=df, res_var="Recuperados", anova_model="Recuperados ~ Contagios")
res.anova_summary


sm.qqplot(res.anova_std_residuals, line='45')
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Standardized Residuals")
plt.show()

# histogram
plt.hist(res.anova_model_out.resid, bins='auto', histtype='bar', ec='k')
plt.xlabel("Residuals")
plt.ylabel('Frequency')
plt.show()

#Como los residuos estandarizados se encuentran alrededor de la línea de 45 grados, sugiere que los residuos están 
# distribuidos aproximadamente normalmente.

#En el histograma, la distribución parece aproximadamente normal y sugiere que los residuos tienen una distribución
#  aproximadamente normal.

#LINEAR MODELS

#Solo queremos 10 datos para hacer nuestra regresion lineal así que eliminaremos  22 hileras

a = {'x':df["Recuperados"],'y':df["Contagios"]}
b = pd.DataFrame(a)
from sklearn.linear_model import LinearRegression

Linea_Regresion = LinearRegression()
Eje_X = b[['x']] #ESTA ES LA NOTACION PARA LAS COLUMNAS
Eje_Y = b[['y']]
Linea_Regresion.fit(Eje_X,Eje_Y)

#El valor de la pendiente (β1) es; 
Linea_Regresion.coef_
print(Linea_Regresion.coef_)

#El valor de la interseccion (β0) es; 
Linea_Regresion.intercept_
print(Linea_Regresion.intercept_)

print("Podemos concluir que la línea de regresión ajustada a los datos es  y", Linea_Regresion.intercept_,"+",Linea_Regresion.coef_,"xi")

#Podemos concluir que la linea de regresión ajustada a los datos es y=2486.7496+1.1123xi


plt.plot(df["Recuperados"],df["Contagios"],'o')
plt.plot([min(df["Recuperados"]),max(df["Recuperados"])],[2486.7496+1.1123 * min(df["Recuperados"]),-3.15446137+1.01078752*max(df["Recuperados"])])
plt.show()

#FORECASTING
k = {'Recuperados(x)':df["Recuperados"],'Contagios(y)':df["Contagios"]}
g = pd.DataFrame(k)
print(g)

Linea_Regresion = LinearRegression()

Linea_Regresion.fit(g[['Recuperados(x)']],g[['Contagios(y)']])

print('La pendiente es : ',Linea_Regresion.coef_,'\nLa interseccion es : ',Linea_Regresion.intercept_)

import seaborn as sns

sns.regplot( x = 'Recuperados(x)', y = 'Contagios(y)', data = g,ci = 95)
plt.ylim(0,)
plt.savefig(f"img/analisvisual.png")
plt.show()

#Para el analisis de person y p-valor
from scipy import stats

Pearson = stats.pearsonr(g.loc[:,'Recuperados(x)'], g.loc[:,'Contagios(y)'])
print('El coeficiente de Pearson es ',Pearson[0],'\nEl P-Valor es ',Pearson[1])


#El coeficiente de Pearson es  0.9990843015371831 
#El P-Valor es  1.255705711994477e-42


if 0.01 < Pearson[1] and Pearson[1] < 0.05 :
  print('Fuerte relacion lineal')
elif 0.05 < Pearson[1] and Pearson[1] < 0.1 :
  print('Moderada  relacion lineal')
elif Pearson[1] < 0.1:
  print('Poca relacion lineal')
elif Pearson[1] > 0.1:
  print('No existe relacion lineal')

  #Poca relación lineal
#Ahora para el coeficiente de determinación

ModLin = LinearRegression()


ModLin.fit(g[['Recuperados(x)']], g[['Contagios(y)']])


print('El coeficiente de determinacion es ',ModLin.score(g[['Recuperados(x)']], g[['Contagios(y)']]))

#El coeficiente de determinacion es  0.998169441578041

#Nos ayuda a reflejar la bondad del ajuste entre 0 y 1, de un modelo a la variable que se pretende explicar

#DATA CLASSIFICATION


k = {'x':df["Recuperados"],'y':df["Contagios"]}
k_df= DataFrame(k,columns=['x','y'])
print(k_df)

#Disperción
plt.scatter(k_df['x'],k_df['y'])
plt.savefig(f"img/dispercion.png")
plt.show()

from sklearn.cluster import KMeans
KMeans = KMeans(n_clusters=3).fit(k_df)
centroids = KMeans.cluster_centers_
print(centroids)

#[[ 33629.62962963  39857.14814815]
 #[531532.         592456.        ]
 #[166591.75       188359.25      ]]

plt.scatter(k_df['x'],k_df['y'],c=KMeans.labels_.astype(float),s=30,alpha=0.5)
plt.scatter(centroids[:,0],centroids[:,1],c="red",s=50)
plt.savefig(f"img/centroide.png")
plt.show()
