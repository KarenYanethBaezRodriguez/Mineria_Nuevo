from os import linesep
import matplotlib.pyplot as plt
from numpy import rot90
import requests
import io
from bs4 import BeautifulSoup
import pandas as pd
from tabulate import tabulate
import statsmodels.api as sm
from statsmodels.formula.api import ols
 

def get_soup(url: str) -> BeautifulSoup:
    response = requests.get(url)
    return BeautifulSoup(response.content, 'html.parser')

 

def wiki() -> pd.DataFrame:
    soup = get_soup("https://en.wikipedia.org/wiki/List_of_states_of_Mexico")
    list_of_lists = [] # :List
    rows = soup.table.find_all('tr')
    for row in rows[1:]:
        columns = row.find_all('td')
        #  listado_de_valores_en_columnas = []
        #  for column in columns:
        #    listado_de_valores_en_columnas.append(coulmn.text.strip())
        listado_de_valores_en_columnas = [column.text.strip() for column in columns]
        list_of_lists.append(listado_de_valores_en_columnas)

 

    return pd.DataFrame(list_of_lists, columns=[header.text.strip() for header in  rows[0].find_all('th')])

 

def get_info_transparencia_uanl(page:int = 1) -> pd.DataFrame:
    soup = get_soup(f"http://transparencia.uanl.mx/remuneraciones_mensuales/bxd.php?pag_act={page}&id_area_form=2305&mya_det=082020")
    table = soup.find_all("table")
    table_row = table[2].find_all('tr')
    list_of_lists = [[row_column.text.strip() for row_column in row.find_all('td')] for row in table_row]
    df = pd.DataFrame(list_of_lists[1:], columns=list_of_lists[0])
    return df

 

#def get_csv_from_url() -> pd.DataFrame:
    #url="https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv"
    #s=requests.get(url).content
    #return pd.read_csv(io.StringIO(s.decode('utf-8')))

 

def print_tabulate(df: pd.DataFrame):
    print(tabulate(df, headers=df.columns, tablefmt='orgtbl'))

 


# df1 = get_info_transparencia_uanl(1)

 

# dfs = [get_info_transparencia_uanl(page) for page in range(1,10)]

 

#dfs = [get_info_transparencia_uanl(page) for page in range(1,10)]
#df = pd.concat(dfs)

df = pd.read_csv("Covid.csv") #get_csv_from_url()
#print(df.columns) 
df= df.drop(['code'], axis=1)
df.columns=['Economia','Disminución_de_crecimiento_en_consumo_CP','Disminución del crecimiento del consumo largo plazo','Disminucion_de_crecimiento_de_la_inversion_CP','Disminucion_de_crecimiento_de_la_inversion_LP','Disminucion_de_ingresos_por_turismo_CP','Disminucion_de_ingresos_por_turismo_LP','Indice_de_rigurosidad_promedio','Movilidad_promedio']

valores = df[["Economia","Disminución del crecimiento del consumo largo plazo"]]

ax= valores.plot.bar(x="Economia",y="Disminución del crecimiento del consumo largo plazo", rot = 0)

ax.set_title('Covid')
ax.set_xlabel('Paises')
ax.set_ylabel('Porcentaje')
#plt.show()
plt.xticks(rotation=90)
plt.savefig(f"img/covid.png")


#modl=ols("Disminución del crecimiento del consumo largo plazo ~ Economia", data=df.colums).fit()
#anova_df=sm.stats.anova_lm(modl,typ=2)
#if anova_df["PR(>F)"][0] < 0.5:
   # print("no hay diferencia")
    #else:
       # print("No hay diferncia")


#df = get_info_transparencia_uanl(1)
# print(df["Largest city"])
# df = get_csv_from_url()
# df = wiki()
print_tabulate(df)
#df.to_csv("Limpio/Covid_sincode.csv", index=False) 
