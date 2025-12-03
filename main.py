from clases.clase_supermercado import Supermercado
import random
import numpy as np


np.random.seed(41)
random.seed(41)

super = Supermercado(lista_de_productos=[1, 2, 3, 4], 
                     lista_de_nivel_de_stock=[180, 200, 190, 130],
                     horarios_apertura_cierre=(8, 20), 
                     cantidad_de_horarios_de_entregas=3,
                     lambdas_online_por_hora= [1]*5+[5]*5+[10]*5+[12]*5+[10]*4,
                     lambdas_mixto_por_hora=[0]*8+[20]*4+[30]*4+[50]*4+[0]*4,
                     lambdas_presencial_por_hora=[0]*8+[10]*4+[15]*4+[20]*4+[0]*4,
                     gamma_mixtos=0.5)

# MOSTRAR 1
# print("MODELO 1: ", super.modelo1.probability_distribution_over([0,4]))
# print("MODELO 2: ", super.modelo2.probability_distribution_over([0,1,2]))

# super.graficar_lambdas()
# super.grafico_de_intensidad_por_hora()

# MOSTRAR 2
# tiempos, is_online, modelos, tiempos_de_entrega = super.obtener_instancia()

# super.simular_dia_online(tiempos, is_online, modelos, tiempos_de_entrega)
# print(super.compra_online_por_baches_dia_anterior)
# super.simular_con_graficos_un_dia()

# MOSTRAR 3


# instancias = super.generar_n_instancias(10)
# super.simular_n_dias(instancias, dia_online_ON=True)
# print(super.probabilidad_out_of_stock())

# super.simular_con_graficos_un_dia()