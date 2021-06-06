# -*- coding: utf-8 -*-
from pyT2FTS.sliding_window import janela_deslizante
from pyT2FTS.datasets import get_TAIEX,get_NASDAQ,get_Brent_Oil,get_SP500
import numpy as np



'------------------------------------------------ Data set import -------------------------------------------------'

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

'------------------------------------------------ Gridsearch Parameters -------------------------------------------------'


datasets = [taiex,nasdaq]
dataset_names = ['TAIEX','nasdaq']
diff = 1                                   #Se diff = 1, diferencia os dados. Se diff = 0, não diferencia
particoes = np.arange(1,11)                 #particoes deve ser uma lista
ordens = [1]
partitioners = ['SODA']            #partitioners: 'chen' 'SODA' 'ADP' 'DBSCAN' 'CMEANS' 'entropy' 'FCM'  
mfs = ['triangular']         #mfs: 'triangular' ou 'trapezoidal' ou 'gaussian'

'------------------------------------------------ Running the model -------------------------------------------------'


'Builds and runs the model'
janela_deslizante(datasets,dataset_names,diff,particoes,ordens,partitioners,mfs)



"""
'Toca um som quando acabar'
import winsound
duration = 1000  # milliseconds
freq = 440  # Hz
winsound.Beep(freq, duration)
"""





