import pandas as pd
import numpy as np
from pyT2FTS.processo_completo import T2FTS
import pickle #To save the data as the process goes
import time

    
def janela_deslizante(data,diff,particoes,ordens,metodo_part):
    
    """Realiza a janela deslizante sobre um conjunto de dados.
    Tamanho da janela padrão é de 1000 amostras.
    
    params:
    :data: dados que serao treinados e testados
    :diff: comando para diferenciar ou nao os valores de 'data'
    :particoes: Lista contendo numeros de particoes que serão testados pela janela. Se método for SODA, então refere-se ao gridsize
    :ordens: Lista contendo numeros de ordens que serão testados pela janela
    
    Return
    
    :df_geral: dataframe contendo erros gerais, referentes a cada janela
    :df_especifico: dataframe contendo erros especificos, referentes a cada janela
    Salva arquivo em Excel contendo duas planilhas:
        Erros gerais: Mostra erros por janela
        Errros especificos: Mostra media e desvio de padrao de RMSE de todas as janelas 
        por gridsize/numero de particoes

    """
    
    'Let''s measure the total elapsed time for the whole process to be completed'
    start_time = time.time()
    
    
    
    'Indicates the percentage of the windowsize to move the window'
    increment = 0.2  

    'list to sabe the errors'
    lista_rmse = []
    lista_particoes = []
    lista_regras = []
    lista_flrg = []
    
    erros_geral = {'Gridsize':[],
                   'Particoes':[],
                   'Ordem':[],
                   'Janela':[],
                   'UDETHEIL':[],
                   'MAPE':[],
                   'MSE':[],
                   'RMSE':[],
                   'MAE':[],
                   'NDEI':[],
                   'Media_RMSE':[],
                   'Desv_pad_RMSE':[],
                   'FLR':[],
                   'FLRG':[]
                  
                   }
    
    erros_especifico = {'Gridsize':[],
                        'Ordem':[],
                        'Particoes': [],
                        'mean_RMSE': [],
                        'std_RMSE': [],
                        'FLR':[],
                        'FLRG':[],
                        'Time(s)': [],
                        'Total Time(s)': []
                        
                         }  
    
    
    '-----Begins the Gridsearch------'
    
    for numero in particoes:
        
        gridsize = numero
        
        for lag in ordens:
        
            '------------------------------------------------ Janela deslizante -------------------------------------------------'
               
            #tamanho_janela = int(0.2*len(data))      #Tamanho da janela deslizante
            tamanho_janela = 1000
            
            janela_inf = 0
            janela_sup = tamanho_janela
            
            'Let''s measure the METHOD elapsed time '
            method_start_time = time.time()
            
            
            while (janela_sup <= len(data)):
                
                dados = data[janela_inf:janela_sup]
                print("Dados: [", janela_inf, ":",janela_sup,"]")
            
                '------------------------------------------------ Confirgurações da janela -------------------------------------------------'
    
                'Define a ordem do sistema'
                order = lag
                
                'Define o numero de conjuntos'
                numero_de_sets = numero
                                
                lista_erros,n_sets,FLR,FLRG = T2FTS(dados,metodo_part,partition_parameters=numero_de_sets,order=order,diff=diff)
               
                print("---------------------------------")
                   
                '------------------------------------------------  Metricas de erro  ------------------------------------------'
                'acrescenta na lista de erros o rmse'
                lista_rmse.append(lista_erros[3])
                
                'acrescenta na lista do numero de regras'
                lista_regras.append(FLR)
                lista_flrg.append(FLRG)
                lista_particoes.append(n_sets)
                
                'Constroi o dicionario de erros gerais ( que mostras as janelas)'
                erros_geral['Gridsize'].append(gridsize)                
                erros_geral['Particoes'].append(n_sets)
                erros_geral['Ordem'].append(order)
                erros_geral['Janela'].append("{}:{}".format(janela_inf,janela_sup))

                erros_geral['UDETHEIL'].append(lista_erros[0])
                erros_geral['MAPE'].append(lista_erros[1])
                erros_geral['MSE'].append(lista_erros[2])
                erros_geral['RMSE'].append(lista_erros[3])
                erros_geral['MAE'].append(lista_erros[4])                
                erros_geral['NDEI'].append(lista_erros[5]) 
                erros_geral['Media_RMSE'].append(None)   
                erros_geral['Desv_pad_RMSE'].append(None)   
                erros_geral['FLR'].append(FLR)
                erros_geral['FLRG'].append(FLRG)
               
                'Desliza a janela'
                janela_inf = janela_inf+200
                janela_sup = janela_sup+200
                
                #janela_inf = int(janela_inf + tamanho_janela * increment)
                #janela_sup = int(janela_sup + tamanho_janela * increment)
                
            
            'Ends time measurement'
            method_end_time = time.time()
            
            method_elapsed_time = method_end_time - method_start_time
            
            'Calcula a media e desvio padrao de RMSE de todas as instancias da janela para a ordem e partição do momento'
           
            avg_rmse = np.mean(lista_rmse)
            std_rmse = np.std(lista_rmse)
            avg_particoes = np.mean(lista_particoes)
            avg_regras = np.mean(lista_regras)
            avg_flrg = np.mean(lista_flrg)

            'Preenche as linhas que informam os valores médios das métricas. Todas as outras colunas sao zeradas'
            'Algumas sao None porque nao deve aparecer nada na tabela do Excel'
            
            '### Primeira linha: vazia ###'
            erros_geral['Gridsize'].append(None)
            erros_geral['Particoes'].append(None)
            erros_geral['Ordem'].append(None)
            erros_geral['Janela'].append(None)
       
            erros_geral['UDETHEIL'].append(None)
            erros_geral['MAPE'].append(None)
            erros_geral['MSE'].append(None)
            erros_geral['RMSE'].append(None)
            erros_geral['MAE'].append(None)                
            erros_geral['NDEI'].append(None)   
            erros_geral['Media_RMSE'].append(None) 
            erros_geral['Desv_pad_RMSE'].append(None)  
            erros_geral['FLR'].append(None)
            erros_geral['FLRG'].append(None)
            
            '### Segunda linha: Médias das métricas ###'
            erros_geral['Gridsize'].append('Médias:')
            erros_geral['Particoes'].append(avg_particoes)
            erros_geral['Ordem'].append(None)
            erros_geral['Janela'].append(None)
       
            erros_geral['UDETHEIL'].append(None)
            erros_geral['MAPE'].append(None)
            erros_geral['MSE'].append(None)
            erros_geral['RMSE'].append(None)
            erros_geral['MAE'].append(None)                
            erros_geral['NDEI'].append(None)   
            erros_geral['Media_RMSE'].append(avg_rmse) 
            erros_geral['Desv_pad_RMSE'].append(std_rmse)  
            erros_geral['FLR'].append(avg_regras)
            erros_geral['FLRG'].append(avg_flrg)

            '### Terceira linha: vazia ###'
            erros_geral['Gridsize'].append(None)
            erros_geral['Particoes'].append(None)
            erros_geral['Ordem'].append(None)
            erros_geral['Janela'].append(None)
       
            erros_geral['UDETHEIL'].append(None)
            erros_geral['MAPE'].append(None)
            erros_geral['MSE'].append(None)
            erros_geral['RMSE'].append(None)
            erros_geral['MAE'].append(None)                
            erros_geral['NDEI'].append(None)   
            erros_geral['Media_RMSE'].append(None) 
            erros_geral['Desv_pad_RMSE'].append(None)  
            erros_geral['FLR'].append(None)
            erros_geral['FLRG'].append(None)
            
            'Constroi dicionario de erros especificos'   
            
            
            erros_especifico['Gridsize'].append(gridsize)   
            erros_especifico['Ordem'].append(order)         
            erros_especifico['Particoes'].append(avg_particoes) 
            erros_especifico['FLR'].append(avg_regras)  
            erros_especifico['FLRG'].append(avg_flrg)  
            erros_especifico['mean_RMSE'].append(avg_rmse)
            erros_especifico['std_RMSE'].append(std_rmse)
            erros_especifico['Time(s)'].append(method_elapsed_time)
            erros_especifico['Total Time(s)'].append(None)


            'Printa na tela o resultado das janelas'
            if metodo_part == 'chen':
                r = "RMSE Médio - part: " + str(numero) + ", ordem: " + str(order)
                print("[",r,"]:",avg_rmse)
                print("---------------------------------")

            elif metodo_part == 'soda' or metodo_part == 'ADP': 
                r = "RMSE Médio - Gridsize: " + str(gridsize) + ", ordem: " + str(order)
                print("[",r,"]:",avg_rmse)
                print("---------------------------------")
                
                            
            'usa Pickle para salvar os dicionarios a cada janela'
            pickle_out = open("gerais.pickle","wb")
            pickle.dump(erros_geral, pickle_out)
            pickle_out = open("especifico.pickle","wb")          
            pickle.dump(erros_especifico, pickle_out)         
            pickle_out.close()
        
            'Resets the lists'
            lista_rmse = []  
            lista_regras = []
            lista_flrg = []
            lista_particoes = []
            
    
    'Ends time measurement'
    end_time = time.time()      
    total_elapsed_time = end_time - start_time
    
    'Adds the final line with the total elapsed time'
    erros_especifico['Gridsize'].append(None)   
    erros_especifico['Ordem'].append(None)         
    erros_especifico['Particoes'].append(None) 
    erros_especifico['FLR'].append(None)  
    erros_especifico['FLRG'].append(None)  
    erros_especifico['mean_RMSE'].append(None)
    erros_especifico['std_RMSE'].append(None)
    erros_especifico['Time(s)'].append('Total Elapsed Time:')
    erros_especifico['Total Time(s)'].append(total_elapsed_time)
               
    
    '------------------------------------------------  Salva metricas em Excel  ------------------------------------------'

    'Define o nome do arquivo final com os erros'
    if diff == 0:  
        nome_arquivo = "erros" + "_semdiff_" + metodo_part + "_" + str(particoes[0]) + "a" + str(particoes[-1]) + ".xlsx"
        
    elif diff == 1:   
        nome_arquivo = "erros" + "_diff_" + metodo_part + "_" + str(particoes[0]) + "a" + str(particoes[-1]) + ".xlsx"      
        
    
    print("Arquivo salvo:",nome_arquivo)
    writer = pd.ExcelWriter(nome_arquivo, engine='xlsxwriter')
            
    df_geral = pd.DataFrame(data=erros_geral)
    df_especifico = pd.DataFrame(data=erros_especifico)
    #df_especifico.columns = ['Gridsize','Particoes','RMSE medio 1_Ordem', 'Desvio padrao RMSE 1_Ordem','RMSE medio 2_Ordem', 'Desvio padrao RMSE 2_Ordem','RMSE medio 3_Ordem', 'Desvio padrao RMSE 3_Ordem','FLR','FLRG']    
           
    df_geral.to_excel(writer, sheet_name='Geral',index = False)
    df_especifico.to_excel(writer, sheet_name='Específico',index = False)
   
    writer.save()
    
    #FAZ o download do arquivo de resultados para o computador
    #from google.colab import files
    #files.download(nome_arquivo)
    
    
 
    return df_geral,df_especifico
        
        













def comparison_sliding_window(datasets,dataset_names,diff,particoes,ordens,metodo_part):
    
    """
    Function to evaluate the performance of the model using different datasets
    
    Realiza a janela deslizante sobre um conjunto de dados.
    Tamanho da janela padrão é de 1000 amostras.
    
    params:
    :data: dados que serao treinados e testados
    :diff: comando para diferenciar ou nao os valores de 'data'
    :particoes: Lista contendo numeros de particoes que serão testados pela janela
    :ordens: Lista contendo numeros de ordens que serão testados pela janela
    
    Return
    
    :df_geral: dataframe contendo erros gerais, referentes a cada janela
    :df_especifico: dataframe contendo erros especificos, referentes a cada janela
    Salva arquivo em Excel contendo duas planilhas:
        Erros gerais: Mostra erros por janela
        Errros especificos: Mostra media e desvio de padrao de RMSE de todas as janelas 
        por gridsize/numero de particoes

    """
    
    comparacao = {}
    
    indice = 0
    
    'Indicates the percentage of the windowsize to move the window'
    increment = 0.2  

    
    '-----Começa o Gridsearch------'
    
    for data in datasets: 
        
        'lista para salvar os erros'
        lista_rmse = []
        
        erros_geral = {'Gridsize':[],
                       'Particoes':[],
                       'Ordem':[],
                       'Janela':[],
                       'UDETHEIL':[],
                       'MAPE':[],
                       'MSE':[],
                       'RMSE':[],
                       'MAE':[],
                       'NDEI':[],
                       'Media_RMSE':[],
                       'Desv_pad_RMSE':[]
                       }
        
        erros_especifico = {'Gridsize':[],
                            'media_1_ordem':[],
                            'desvio_1_ordem':[],
                            'media_2_ordem':[],
                            'desvio_2_ordem':[],
                            'media_3_ordem':[],
                            'desvio_3_ordem':[],
                             }  
        
        
        
        data_name = dataset_names[indice]
    
        for numero in particoes:
            
            gridsize = numero
            erros_especifico['Gridsize'].append(gridsize)                
            
            for lag in ordens:
            
                '------------------------------------------------ Janela deslizante -------------------------------------------------'
                   
                tamanho_janela = int(0.2*len(data))      #Tamanho da janela deslizante
                
                janela_inf = 0
                janela_sup = tamanho_janela
                
                while (janela_sup <= len(data)):
                    
                    dados = data[janela_inf:janela_sup]
                    print("Dataset: ", data_name)
                    print("Dados: [", janela_inf, ":",janela_sup,"]")

                
                    '------------------------------------------------ Confirgurações da janela -------------------------------------------------'
        
                    'Define a ordem do sistema'
                    order = lag
                    
                    'Define o numero de conjuntos'
                    numero_de_sets = numero
                                    
                    lista_erros,n_sets = T2FTS(dados,metodo_part,partition_parameters=numero_de_sets,order=order,diff=diff)
                   
                    print("---------------------------------")
                       
                    '------------------------------------------------  Metricas de erro  ------------------------------------------'
                    'acrescenta na lista de erros o rmse'
                    lista_rmse.append(lista_erros[3])
                    
                    'Constroi o dicionario de erros gerais ( que mostras as janelas)'
                    erros_geral['Gridsize'].append(gridsize)                
                    erros_geral['Particoes'].append(n_sets)
                    erros_geral['Ordem'].append(order)
                    erros_geral['Janela'].append("{}:{}".format(janela_inf,janela_sup))
    
                    erros_geral['UDETHEIL'].append(lista_erros[0])
                    erros_geral['MAPE'].append(lista_erros[1])
                    erros_geral['MSE'].append(lista_erros[2])
                    erros_geral['RMSE'].append(lista_erros[3])
                    erros_geral['MAE'].append(lista_erros[4])                
                    erros_geral['NDEI'].append(lista_erros[5]) 
                    erros_geral['Media_RMSE'].append(None)   
                    erros_geral['Desv_pad_RMSE'].append(None)   
                   
                    'Desliza a janela'
                    janela_inf = int(janela_inf + tamanho_janela * increment)
                    janela_sup = int(janela_sup + tamanho_janela * increment)
                    

                    
                    #janela_inf = janela_inf + 40
                    #janela_sup = janela_sup + 40
                    
                
                'Calcula a media e desvio padrao de RMSE de todas as instancias da janela para a ordem e partição do momento'
               
                avg_rmse = np.mean(lista_rmse)
                std_rmse = np.std(lista_rmse)
    
                'Preenche a linha que informa RMSE medio e desv.pad. Todas as outras colunas sao zeradas'
                'Algumas sao None porque nao deve aparecer nada na tabela do Excel'
                erros_geral['Gridsize'].append(gridsize)
                erros_geral['Particoes'].append(None)
                erros_geral['Ordem'].append(order)
                erros_geral['Janela'].append(None)
           
                erros_geral['UDETHEIL'].append(None)
                erros_geral['MAPE'].append(None)
                erros_geral['MSE'].append(None)
                erros_geral['RMSE'].append(None)
                erros_geral['MAE'].append(None)                
                erros_geral['NDEI'].append(None)   
                erros_geral['Media_RMSE'].append(avg_rmse) 
                erros_geral['Desv_pad_RMSE'].append(std_rmse)   
                
                'Constroi dicionario de erros especificos'    
                if order ==1:
                    erros_especifico['media_1_ordem'].append(avg_rmse)
                    erros_especifico['desvio_1_ordem'].append(std_rmse)
    
                if order ==2:
                    erros_especifico['media_2_ordem'].append(avg_rmse)
                    erros_especifico['desvio_2_ordem'].append(std_rmse)
                  
                if order ==3:
                    erros_especifico['media_3_ordem'].append(avg_rmse)
                    erros_especifico['desvio_3_ordem'].append(std_rmse)
    
    
                'Printa na tela o resultado das janelas'
                if metodo_part == 'chen':
                    r = "RMSE Médio - part: " + str(numero) + ", ordem: " + str(order)
                    print("[",r,"]:",avg_rmse)
                    print("---------------------------------")
    
                elif metodo_part == 'soda': 
                    r = "RMSE Médio - Gridsize: " + str(gridsize) + ", ordem: " + str(order)
                    print("[",r,"]:",avg_rmse)
                    print("---------------------------------")
                    
                    
                'usa Pickle para salvar os dicionarios a cada janela'
                pickle_out = open("gerais.pickle","wb")
                pickle.dump(erros_geral, pickle_out)
                pickle_out = open("especifico.pickle","wb")          
                pickle.dump(erros_especifico, pickle_out)         
                pickle_out.close()
            
                lista_rmse = []  #resets the list
                
        comparacao[data_name] = erros_especifico
        
        'Faz com que todos as colunas tenham o mesmo tamanho ( para satisfazer pandas)'
        if not erros_especifico['media_2_ordem']:
            erros_especifico['media_2_ordem'] =np.zeros(len(erros_especifico['media_1_ordem']))
            erros_especifico['desvio_2_ordem'] =np.zeros(len(erros_especifico['media_1_ordem']))
    
        if not erros_especifico['media_3_ordem']:
            erros_especifico['media_3_ordem'] =np.zeros(len(erros_especifico['media_1_ordem']))
            erros_especifico['desvio_3_ordem'] =np.zeros(len(erros_especifico['media_1_ordem']))
                   
        
        '>>>>>>>>> If you want individual metrics for each dataset uncomment this part'
        """
        '------------------------------------------------  Salva metricas em Excel  ------------------------------------------'
    
        'Define o nome do arquivo final com os erros'
        if diff == 0:  
            nome_arquivo = data_name + "_erros" + "_semdiff_" + metodo_part + "_" + str(particoes[0]) + "a" + str(particoes[-1]) + ".xlsx"
            
        elif diff == 1:   
            nome_arquivo = data_name + "_erros" + "_diff_" + metodo_part + "_" + str(particoes[0]) + "a" + str(particoes[-1]) + ".xlsx"      
            

        
        print("Arquivo salvo:",nome_arquivo)
        writer = pd.ExcelWriter(nome_arquivo, engine='xlsxwriter')
            
        df_geral = pd.DataFrame(data=erros_geral)
        df_especifico = pd.DataFrame(data=erros_especifico)
        df_especifico.columns = ['Gridsize','RMSE medio 1_Ordem', 'Desvio padrao RMSE 1_Ordem','RMSE medio 2_Ordem', 'Desvio padrao RMSE 2_Ordem','RMSE medio 3_Ordem', 'Desvio padrao RMSE 3_Ordem']    
               
        df_geral.to_excel(writer, sheet_name='Geral',index = False)
        df_especifico.to_excel(writer, sheet_name='Específico',index = False)
       
        writer.save()
        
        """
        indice = indice + 1
     
    return comparacao
        
        
        
        
 