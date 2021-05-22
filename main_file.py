from pyT2FTS.sliding_window import janela_deslizante
from pyT2FTS.datasets import get_TAIEX
import numpy as np



'------------------------------------------------ Data set import -------------------------------------------------'

taiex = get_TAIEX()
data = taiex.avg               # Pega somente a última coluna de dados: a média (avg)
data = data.to_numpy()      # Covnerte de panda dataframe para array numpy

'------------------------------------------------ Gridsearch Parameters -------------------------------------------------'


ordens = [1,2,3]

'particoes must be a list'
#particoes = [10]
particoes = np.arange(1,11)

diff = 1
        
metodo_part = 'soda'    

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





"""

from pyT2FTS.processo_completo import T2FTS
from pyT2FTS.datasets import get_TAIEX,get_NASDAQ
import matplotlib.pyplot as plt

'------------------------------------------------ Data set import -------------------------------------------------'

taiex = get_TAIEX()
data = taiex.avg               # Pega somente a última coluna de dados: a média (avg)
data = data.to_numpy()      # Covnerte de panda dataframe para array numpy

'------------------------------------------------ Initial parameters -------------------------------------------------'

'Define a ordem do sistema'
order = 1

'Define o número de conjuntos'
numero_de_sets = 12

'Se diff = 1, diferencia os dados. Se diff = 0, não diferencia'
diff = 1

metodo_part = 'FCM' 

'Builds and runs the model'
T2FTS(data,metodo_part,partition_parameters=numero_de_sets,order=order,diff=diff)
"""



