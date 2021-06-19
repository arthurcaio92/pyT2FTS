# -*- coding: utf-8 -*-
from pyT2FTS.sliding_window import janela_deslizante
from pyT2FTS.datasets import get_TAIEX,get_NASDAQ,get_Brent_Oil,get_SP500
import numpy as np
import pandas as pd



'------------------------------------------------ Data set import -------------------------------------------------'
"""
taiex_df = get_TAIEX()
taiex = taiex_df.avg               
taiex = taiex.to_numpy()  

nasdaq_df = get_NASDAQ()
nasdaq = nasdaq_df.avg               
nasdaq = nasdaq.to_numpy()    

sp500_df = get_SP500()
sp500 = sp500_df.Avg               
sp500 = sp500[11500:16000]
sp500 = sp500.to_numpy()     

df_brent_oil = get_Brent_Oil()
brent_oil = df_brent_oil.Price  
brent_oil = brent_oil.to_numpy()

crude_oil_df = pd.read_excel('Crude-oil-prices.xls', sheet_name='Data 1')
crude_oil_df = crude_oil_df['RWTC']
crude_oil = crude_oil_df[1:].to_numpy()
crude_oil = crude_oil[5524:]

'n sei pq mas precisa fazer isso'
a=[]
for x in crude_oil:
    a.append(x)
    
crude_oil  = np.array(a)

    
"""

taiex_df = get_TAIEX()
taiex = taiex_df.avg               
taiex = taiex.to_numpy() 


'------------------------------------------------ Gridsearch Parameters -------------------------------------------------'


datasets = [taiex]
dataset_names = ['NASDAQ']
diff = 1                                   #Se diff = 1, diferencia os dados. Se diff = 0, não diferencia
particoes = np.arange(3,4)                 #particoes deve ser uma lista
ordens = [1]
partitioners = ['SODA']            #partitioners: 'chen' 'SODA' 'ADP' 'DBSCAN' 'CMEANS' 'entropy' 'FCM'  
mfs = ['triangular']         #mfs: 'triangular' ou 'trapezoidal' ou 'gaussian'

'------------------------------------------------ Running the model -------------------------------------------------'


'Builds and runs the model'
janela_deslizante(datasets,dataset_names,diff,particoes,ordens,partitioners,mfs)



"""
"############ RUN WITHOUT SLIDING WINDOW ############"

from pyT2FTS.processo_completo import T2FTS


data = taiex
dataset_names = ['NASDAQ']
diff = 1                                   #Se diff = 1, diferencia os dados. Se diff = 0, não diferencia
numero_de_sets = 10          
order = 1
metodo_part = 'SODA'          
mf_type = 'triangular'         #mfs: 'triangular' ou 'trapezoidal' ou 'gaussian'


lista_erros,n_sets,FLR,FLRG = T2FTS(data,metodo_part,mf_type,partition_parameters=numero_de_sets,order=order,diff=diff)

mf_type = 'trapezoidal'         #mfs: 'triangular' ou 'trapezoidal' ou 'gaussian'


lista_erros,n_sets,FLR,FLRG = T2FTS(data,metodo_part,mf_type,partition_parameters=numero_de_sets,order=order,diff=diff)

mf_type = 'gaussian'         #mfs: 'triangular' ou 'trapezoidal' ou 'gaussian'


lista_erros,n_sets,FLR,FLRG = T2FTS(data,metodo_part,mf_type,partition_parameters=numero_de_sets,order=order,diff=diff)
"""
