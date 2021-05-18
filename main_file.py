from pyT2FTS.sliding_window import janela_deslizante
from pyT2FTS.datasets import get_TAIEX
import numpy as np



'------------------------------------------------ Data set import -------------------------------------------------'

taiex = get_TAIEX()
data = taiex.avg               # Pega somente a última coluna de dados: a média (avg)
data = data.to_numpy()      # Covnerte de panda dataframe para array numpy

'------------------------------------------------ Gridsearch Parameters -------------------------------------------------'


ordens = [1]

'particoes must be a list'
#particoes = [10]
particoes = np.arange(19,20)

diff = 1
        
metodo_part = 'DBSCAN'    

'------------------------------------------------ Running the model -------------------------------------------------'


'Builds and runs the model'
df_geral,df_especifico = janela_deslizante(data,diff,particoes,ordens,metodo_part)



"""
'Toca um som quando acabar'
import winsound
duration = 1000  # milliseconds
freq = 440  # Hz
winsound.Beep(freq, duration)
"""