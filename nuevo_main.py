from clases.clase_escenarios import Escenario
from clases.clase_llegadas import Llegada
from clases.clase_nuevo_supermercado import Nuevo_Supermercado
from src.models.models.multinomial_logit import MultinomialLogitModel


llegadas = Llegada(dias=3, 
                   horarios_apertura_cierre=(8, 20),
                   lambdas_online_por_hora=[2]*8+[3]*8+[1]*8,
                   lambdas_presencial_por_hora=[0]*7+[10]*6+[15]*7+[0]*4,
                   lambdas_mixto_por_hora=[0]*7+[15]*6+[19]*7+[0]*4,
                   gamma_mixtos_online=0.2,
                   seed=42)

lista_de_productos=[1,2,3,4]

modelo1 = MultinomialLogitModel.simple_random([0] + lista_de_productos)
modelo2 = MultinomialLogitModel.simple_random([0] + lista_de_productos)
modelo3 = MultinomialLogitModel.simple_random([0] + lista_de_productos)


escenario = Escenario(lista_de_productos=lista_de_productos,
                      lista_de_nivel_de_stock=[150,110,80,90],
                      cantidad_de_horarios_de_entregas=3,
                      modelos_de_eleccion=[modelo1, modelo2, modelo3],
                      llegada=llegadas)


super = Nuevo_Supermercado(escenario)

instancia_de_llegadas = llegadas.obtener_instancia_de_llegadas()
isntancia_con_asignaciones = escenario.asignar_modelo_y_horario(instancia_de_llegadas, seed=42)

super.simular_escenario(isntancia_con_asignaciones)


