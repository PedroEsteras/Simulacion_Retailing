from archivos_viejos.clase_supermercado import Supermercado
import random
import numpy as np


np.random.seed(39)
random.seed(39)

super = Supermercado(lista_de_productos=[1, 2, 3, 4], 
                     lista_de_nivel_de_stock=[160, 90, 110, 50],
                     horarios_apertura_cierre=(8, 20), 
                     cantidad_de_horarios_de_entregas=3,
                     lambdas_online_por_hora= [1]*5+[3]*5+[2]*5+[2]*5+[2]*4,  
                     lambdas_mixto_por_hora=[0]*8+[20]*4+[30]*4+[50]*4+[0]*4,   
                     lambdas_presencial_por_hora=[0]*8+[18]*4+[21]*4+[24]*4+[0]*4,
                     gamma_mixtos=0.2)

# CANTIDADES
print("ON ", sum(super.lambdas_online_por_hora))
print("MIX ", sum(super.lambdas_mixto_por_hora))
print("PRE ", sum(super.lambdas_presencial_por_hora))

# MOSTRAR 1   
# print("MODELO 1: ", super.modelo1.probability_distribution_over([0,4]))
# print("MODELO 2: ", super.modelo2.probability_distribution_over([0,1,2]))

# MOSTRAR 2
# super.graficar_lambdas()
# super.grafico_de_intensidad_por_hora()
# super.grafico_histogramas_por_tipo()
#super.comparar_totales_con_n_simulaciones()

# print("ONLINE DS TEORICO: ", np.sqrt(sum([x**2 for x in super.lambdas_online_por_hora])))
# print("MIXTO ONLINE DS TEORICO: ", np.sqrt(sum([(x*super.gamma_mixtos)**2 for x in super.lambdas_mixto_por_hora])))
# # # # MOSTRAR 2
# tiempos, is_online, modelos, tiempos_de_entrega = super.obtener_instancia()

# super.simular_dia_online(tiempos, is_online, modelos, tiempos_de_entrega)
# print(super.compra_online_por_baches_dia_anterior)
# super.simular_con_graficos_un_dia()

# # MOSTRAR 3

print(super.horarios_de_entrega)

instancias = super.generar_n_instancias(3)
super.simular_n_dias(instancias, dia_online_ON=True)