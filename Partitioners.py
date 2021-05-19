from pandas import DataFrame



    
    

def conjuntos_soda(data,gridsize):
    
    """Retorna apenas o numero de conjuntos encontrado pelo SODA"""
    
    print("Gridsize:",gridsize )
    
    from pyT2FTS.SODA import SODA_function 
    
    soda_idx = SODA_function(data,gridsize)
                
    dados_idx = DataFrame(soda_idx,columns=['idx'])
            
    maximo = dados_idx['idx'].max()
    
    'Define o numero de sets'
    numero_de_sets = maximo
                         
    return numero_de_sets
    
 
def conjuntos_adp(data,gridsize, distancetype='chebyshev'):
    'A distancia pode ser chebyshev, euclidean, cityblock, sqeuclidean ou cosine'
	
    """Retorna apenas o numero de conjuntos encontrado pelo ADP"""
    
    print("Gridsize: {}".format(gridsize))
    
    from pyT2FTS.OfflineADP import ADP 
    #from OfflineADP import ADP 
    
	#Make it a two-column dataframe       
    dados = DataFrame(data, columns = ['avg'])
    dados.insert(0, '#', range(1,len(dados)+1))
    
    centre, idx = ADP(dados,gridsize)           
    
    'Define o numero de sets'
    numero_de_sets = len(centre)
                         
    return numero_de_sets


def conjuntos_dbscan(data, eps):
    from sklearn.cluster import DBSCAN
    
    #Transforma a base de treinos em um array 2D
    dados = DataFrame(data, columns = ['avg'])
    dados.insert(0, '#', range(1,len(dados)+1))
    dados = dados.to_numpy()

    #Executa o DBSCAN
    db = DBSCAN(eps = eps).fit(dados)
    
    #Salva as labels do modelo em uma variável
    labels = db.labels_

    #Conta o número de labels ignorando o -1, se presente, e salva numa variável
    numero_de_sets = len(set(labels)) - (1 if -1 in labels else 0)
    
    'Identifica a quantidade de outliers'
    n_noise = list(labels).count(-1)
    'A pocentagem de noise/total'
    r = n_noise/len(dados)
    
    
    print(f'EPS: {eps}')
    print(f'Noise points: {n_noise} ({r*100}%)')
    
    
    #Caso o algoritmo não crie clusters, retornar 1 para evitar problemas futuros
    if numero_de_sets == 0:
        return 1
    
    return numero_de_sets


def conjuntos_cmeans(data, k):
    from pyFTS.partitioners import CMeans
    from pyFTS.common import Membership as mf
    
    #Executa o particionamento e salva num objeto
    obj = CMeans.CMeansPartitioner(data = data, npart = k, func = mf.trimf)
    
    #Lista para dict keys
    keys = []
    for k in range(1,k+1):
        keys.append('A'+str(k))
    
    ##Lista para guardar os parâmetros (a, b, c) da função triangular de cada set
    cmeans_params = []
    for k in keys:
        cmeans_params.append(obj.sets[k].parameters)
    
    return cmeans_params

def conjuntos_entropy(data):
    from pyFTS.partitioners import Entropy
    
    #Executa o particionamento e salva num objeto
    obj = Entropy.EntropyPartitioner(data=data)
    
    #Lista para dict keys
    keys = []
    for i in range(0,len(obj.sets)):
        keys.append('A'+str(i))
    
    ##Lista para guardar os parâmetros (a, b, c) da função triangular de cada set
    entropy_params = []
    for i in keys:
        entropy_params.append(obj.sets[i].parameters)
    
    return entropy_params

def conjuntos_fcm(data):
    from pyFTS.partitioners import FCM
    
    #Executa o particionamento e salva num objeto
    obj = FCM.FCMPartitioner(data=data)
    
    #Lista para dict keys
    keys = []
    for i in range(1,len(obj.sets)+1):
        keys.append('A'+str(i))
    
    ##Lista para guardar os parâmetros (a, b, c) da função triangular de cada set
    fcm_params = []
    for i in keys:
        fcm_params.append(obj.sets[i].parameters)
    
    return fcm_params

def conjuntos_huarng(data):
    from pyFTS.partitioners import Huarng
    
    #Executa o particionamento e salva num objeto
    obj = Huarng.HuarngPartitioner(data=data)
    
    #Lista para dict keys
    keys = []
    for i in range(1,len(obj.sets)+1):
        keys.append('A'+str(i))
    
    ##Lista para guardar os parâmetros (a, b, c) da função triangular de cada set
    huarng_params = []
    for i in keys:
        huarng_params.append(obj.sets[i].parameters)
    
    return huarng_params