# -*- coding: utf-8 -*-
from pyT2FTS.T2FTS import Type2Model,IT2FS_plot
from pyT2FTS.ferramentas import metricas_erro,plotar_previsao 
from pyT2FTS.Partitioners import conjuntos_soda,conjuntos_adp,conjuntos_dbscan,conjuntos_cmeans,conjuntos_entropy,conjuntos_fcm,conjuntos_huarng
from pyT2FTS.Transformations import Differential
import numpy as np


"""
    Função que realiza o treinamento e teste de um detemrinado vetor 'data'

    :params:
    :data: dados que serao treinados e testados
    :diff: comando para diferenciar ou nao os valores de 'data'
    :order: ordem do sistema ( n. de amostras na previsao)
    :numero_de_sets: numero de conjuntos fuzzy criados    
    
"""
    
def T2FTS(data,metodo_part,mf_type,partition_parameters,order,diff):
        
    '------------------------------------------------Definição dos intervalos ------------------------------------------'
    
    'Treinamento correponde a 80% do total de dados '
    intervalo_treino = int(0.9 * len(data))         
     
    treino = data[:intervalo_treino]
  
    'Teste correponde a 20% do total de dados '
    teste = data[intervalo_treino:]
          
    'Verifica se é para diferenciar os dados'
    if diff == True:
        treino_orig = treino
        teste_orig = teste
    
        tdiff = Differential(1) 
        treino = tdiff.apply(treino_orig)
        teste = tdiff.apply(teste_orig)
    
    'Objeto da classe tipo2'
    modelo = Type2Model(treino,order) 
    

        
    '------------------------------------------------ Geração de sets  -------------------------------------------------'

    if metodo_part == 'chen':
        numero_de_sets = partition_parameters
        modelo.chen_model_sobreposto(partition_parameters, mf_type)
        
    elif metodo_part == 'SODA':
        gridsize = partition_parameters
        numero_de_sets = conjuntos_soda(treino,gridsize)
        modelo.chen_model_sobreposto(numero_de_sets, mf_type)   
        
    elif metodo_part == 'ADP':
        gridsize = partition_parameters
        numero_de_sets = conjuntos_adp(treino, gridsize)
        modelo.chen_model_sobreposto(numero_de_sets, mf_type)
        
    elif metodo_part == 'DBSCAN':
        eps = partition_parameters
        numero_de_sets = conjuntos_dbscan(treino, eps)
        modelo.chen_model_sobreposto(numero_de_sets, mf_type)
        
    elif metodo_part == 'CMEANS': 
        k = partition_parameters
        cmeans_params = conjuntos_cmeans(treino, k, mf_type)
        numero_de_sets = len(cmeans_params)
        modelo.generate_uneven_length_mfs(numero_de_sets, mf_type, cmeans_params)
    
    elif metodo_part == 'entropy':
        k = partition_parameters
        entropy_params = conjuntos_entropy(treino,k, mf_type)
        numero_de_sets = len(entropy_params)
        modelo.generate_uneven_length_mfs(numero_de_sets, mf_type, entropy_params)
        
    elif metodo_part == 'FCM':
        k = partition_parameters
        fcm_params = conjuntos_fcm(treino,k, mf_type)
        numero_de_sets = len(fcm_params)
        modelo.generate_uneven_length_mfs(numero_de_sets, mf_type, fcm_params)
    
    elif metodo_part == 'huarng':
        huarng_params = conjuntos_huarng(treino)
        numero_de_sets = len(huarng_params)
        modelo.generate_uneven_length_mfs(numero_de_sets,huarng_params)
        
    else:
        raise Exception("Method %s not implemented" % metodo_part)
        
        
    #Plot partition graphs
    #plot_title = str(numero_de_sets) + ' partitions'
    #IT2FS_plot(*modelo.dict_sets.values(),title= plot_title)
    
    '------------------------------------------------ Treinamento  ------------------------------------------'
        
    'Treina o modelo'
    FLR,FLRG = modelo.treinamento()
    
    '------------------------------------------------  Teste  ------------------------------------------'
    'Clipa os dados para estarem dentro do universo de discurso'
    teste = np.clip(teste, modelo.dominio_inf+1, modelo.dominio_sup-1)

    
    print("Partitioner:",metodo_part,"| N. of sets:", numero_de_sets, "| Order:", order)
    print("")
    forecast_result = modelo.predict(teste)   #A lista com as previsoes da janela do momento sao retornadas para esta variavel

    
    'Retorna os valores para a escala original (ou seja, desfaz a diferenciacao)'
    if diff == True:
        forecast_result = forecast_result[1:] #faz isso por causa da diferenciação
        forecast_result = tdiff.inverse(forecast_result,teste_orig)
        teste = teste_orig[order:]  # Para plotar e metricas de erro deve usar a serie original
        
    else:       
        teste = teste[order:]  # O primeiro item nao tem correspondente na previsao
        forecast_result = forecast_result[:-1]

    '------------------------------------------------  Métricas de erro  ------------------------------------------'
    lista_erros = metricas_erro(teste,forecast_result)
        
    'Plotar o grafico teste x previsao'      
    #plotar_previsao(teste,forecast_result)
    
    
    return lista_erros,numero_de_sets,FLR,FLRG
    
    
