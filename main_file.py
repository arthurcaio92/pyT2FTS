from pyT2FTS.sliding_window import janela_deslizante
from pyT2FTS.datasets import get_TAIEX




'------------------------------------------------ Data set import -------------------------------------------------'

taiex = get_TAIEX()
data = taiex.avg               # Pega somente a última coluna de dados: a média (avg)
data = data.to_numpy()      # Covnerte de panda dataframe para array numpy

'------------------------------------------------ Gridsearch Parameters -------------------------------------------------'


ordens = [1,2]

'particoes must be a list'
particoes = [1,2]
#particoes = np.arange(1,6)

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