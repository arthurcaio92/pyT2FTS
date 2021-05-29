from pyT2FTS.sliding_window import janela_deslizante
from pyT2FTS.datasets import get_TAIEX,get_NASDAQ
import numpy as np



'------------------------------------------------ Data set import -------------------------------------------------'

taiex = get_TAIEX()
taiex = taiex.avg               # Pega somente a última coluna de dados: a média (avg)
taiex = taiex.to_numpy()      # Covnerte de panda dataframe para array numpy


nasdaq = get_NASDAQ()
nasdaq = nasdaq.avg               # Pega somente a última coluna de dados: a média (avg)
nasdaq = nasdaq.to_numpy()      # Covnerte de panda dataframe para array numpy


'------------------------------------------------ Gridsearch Parameters -------------------------------------------------'


datasets = [taiex,nasdaq]
dataset_names = ['TAIEX','NASDAQ']
diff = 1                                   #Se diff = 1, diferencia os dados. Se diff = 0, não diferencia
particoes = np.arange(3,15)                 #particoes deve ser uma lista
ordens = [1,2]
partitioners = ['SODA','ADP']            #partitioners: 'chen' 'SODA' 'ADP' 'DBSCAN' 'CMEANS' 'entropy' 'FCM'  
mfs = ['trapezoidal','triangular']         #mfs: 'triangular' ou 'trapezoidal' ou 'gaussian'

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





