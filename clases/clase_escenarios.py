import random
import math
import numpy as np
import matplotlib.pyplot as plt
from src.models.models.multinomial_logit import MultinomialLogitModel
from src.models.transactions.base import TransactionGenerator
import csv
import os
from clases.clase_llegadas import Llegada


class Escenario():
    def __init__(self, 
                 lista_de_productos: list[int],  
                 lista_de_nivel_de_stock: list[int],
                 cantidad_de_horarios_de_entregas: int,
                 modelos_de_eleccion: list[MultinomialLogitModel],
                 llegada: Llegada):


        self.dias = llegada.dias
        self.modelos_de_eleccion = modelos_de_eleccion

        self.horarios_apertura_cierre: tuple[int, int] = llegada.horarios_apertura_cierre
        self.lista_de_productos: list[int] = lista_de_productos
        self.lista_de_nivel_de_stock: list[int] = lista_de_nivel_de_stock

        self.horarios_de_entrega = calcular_horarios_entrega_str(self.horarios_apertura_cierre, cantidad_de_horarios_de_entregas)
        self.horarios_de_entrega_min = [self._horario_str_a_minutos(h) for h in self.horarios_de_entrega]

        self.llegada: Llegada = llegada
        
    def asignar_modelo_y_horario(self, llegadas: list[tuple[int, bool]], seed: int):
        random.seed(seed)

        dias_con_asignaciones = []

        for eventos_de_un_dia in llegadas:

            eventos_con_asignaciones = []

            for evento in eventos_de_un_dia:
                tiempo_de_llegada = evento[0]
                tipo_de_cliente =  evento[1] 
                modelo_de_eleccion = random.choice(self.modelos_de_eleccion)
                horario_de_entrega = random.choice(self.horarios_de_entrega) if tipo_de_cliente == 1 else None

                eventos_con_asignaciones.append((tiempo_de_llegada, tipo_de_cliente, modelo_de_eleccion, horario_de_entrega))

            dias_con_asignaciones.append(eventos_con_asignaciones)

        return dias_con_asignaciones

    def _horario_str_a_minutos(self, horario_str: str) -> int:
        hh, mm = map(int, horario_str.split(":"))
        return hh * 60 + mm









# Calculador horarios de entrega
def calcular_horarios_entrega_str(horarios_apertura_cierre: tuple[int, int],
                                cantidad_de_bloques_de_entregas: int) -> list[str]:

    apertura, cierre = horarios_apertura_cierre
    bloques = cantidad_de_bloques_de_entregas

    if bloques < 2:
        raise ValueError("Debe haber al menos 2 bloques (apertura y cierre).")

    # Pasamos a minutos
    apertura_min = apertura * 60
    cierre_min = cierre * 60

    # Intervalo en minutos
    intervalo = (cierre_min - apertura_min) / (bloques - 1)

    horarios = []
    for i in range(bloques):
        minutos = apertura_min + i * intervalo
        minutos = int(round(minutos))  # redondeo limpio

        hh = minutos // 60
        mm = minutos % 60

        horarios.append(f"{hh:02d}:{mm:02d}")

    return horarios