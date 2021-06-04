<<<<<<< HEAD
# -*- coding: utf-8 -*-
=======
>>>>>>> f983be4b3a672a40fc0708c2377770347e7eee84
from pyT2FTS.sliding_window import janela_deslizante
from pyT2FTS.datasets import get_TAIEX,get_NASDAQ,get_Brent_Oil
import numpy as np



'------------------------------------------------ Data set import -------------------------------------------------'

taiex = get_TAIEX()
taiex = taiex.avg               
taiex = taiex.to_numpy()      


nasdaq = get_NASDAQ()
nasdaq = nasdaq.avg               
nasdaq = nasdaq.to_numpy()      

<<<<<<< HEAD
"""
df_brent_oil = get_Brent_Oil()
brent_oil = df_brent_oil.Price  
brent_oil = brent_oil.to_numpy()
"""
=======

df_brent_oil = get_Brent_Oil()
brent_oil = df_brent_oil.Price  
brent_oil = brent_oil.to_numpy()

>>>>>>> f983be4b3a672a40fc0708c2377770347e7eee84

'------------------------------------------------ Gridsearch Parameters -------------------------------------------------'


datasets = [taiex,nasdaq]
dataset_names = ['TAIEX','NASDAQ']
diff = 1                                   #Se diff = 1, diferencia os dados. Se diff = 0, não diferencia
particoes = np.arange(3,15)                 #particoes deve ser uma lista
ordens = [1,2]
partitioners = ['SODA','ADP']            #partitioners: 'chen' 'SODA' 'ADP' 'DBSCAN' 'CMEANS' 'entropy' 'FCM'  
<<<<<<< HEAD
mfs = ['triangular','triangular']         #mfs: 'triangular' ou 'trapezoidal' ou 'gaussian'
=======
mfs = ['trapezoidal','triangular']         #mfs: 'triangular' ou 'trapezoidal' ou 'gaussian'
>>>>>>> f983be4b3a672a40fc0708c2377770347e7eee84

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





