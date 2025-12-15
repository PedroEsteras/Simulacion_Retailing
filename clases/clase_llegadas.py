import random
import numpy as np

class Llegada():
    def __init__(self,
                 dias: int,
                 horarios_apertura_cierre: tuple[int, int],
                 lambdas_online_por_hora: list[float],
                 lambdas_presencial_por_hora: list[float],
                 lambdas_mixto_por_hora: list[float],
                 gamma_mixtos_online: float,
                 seed: int):
    
        random.seed(seed)

        self.seed = seed
        self.dias = dias

        self.lambdas_online_por_hora = lambdas_online_por_hora
        self.lambdas_presencial_por_hora = lambdas_presencial_por_hora
        self.lambdas_mixto_por_hora = lambdas_mixto_por_hora

        self.horarios_apertura_cierre: tuple[int, int] = horarios_apertura_cierre
    
        self.minutos_abierto: int = (self.horarios_apertura_cierre[1] - self.horarios_apertura_cierre[0])*60
        self.minutos_todo_el_dia: int = 24 * 60

        self.gamma_mixtos: float = gamma_mixtos_online

    def _horario_str_a_minutos(self, horario_str: str) -> int:
        hh, mm = map(int, horario_str.split(":"))
        return hh * 60 + mm

    def lambda_online_por_minuto(self, minuto_del_dia):
        hora = int(minuto_del_dia // 60)
        return self.lambdas_online_por_hora[hora] / 60

    def lambda_presencial_por_minuto(self, minuto_del_dia):
        hora = int(minuto_del_dia // 60)
        return self.lambdas_presencial_por_hora[hora] / 60

    def lambda_mixto_por_minuto(self, minuto_del_dia):
        hora = int(minuto_del_dia // 60)
        return self.lambdas_mixto_por_hora[hora] / 60

    def minutos_hasta_proximo_cliente_online(self, minuto_actual):
        lam = self.lambda_online_por_minuto(minuto_actual)
        if lam == 0:
            return  None # no llega nadie en esta hora
        return random.expovariate(lam)

    def minutos_hasta_proximo_cliente_presencial(self, minuto_actual):
        lam = self.lambda_presencial_por_minuto(minuto_actual)
        if lam == 0:
            return None   # nadie llega en esta hora
        return random.expovariate(lam)

    def minutos_hasta_proximo_cliente_mixto(self, minuto_actual):
        lam = self.lambda_mixto_por_minuto(minuto_actual)
        if lam == 0:
            return None
        return random.expovariate(lam)   

    def generar_lista_de_llegadas_online(self):
        llegadas = []
        for hora in range(24):
            inicio = hora * 60
            fin = (hora + 1) * 60
            minuto_actual = inicio

            while True:
                dt = self.minutos_hasta_proximo_cliente_online(minuto_actual)
                if dt is None:
                    break

                minuto_actual += dt
                if minuto_actual >= fin:
                    break

                llegadas.append(minuto_actual)

        return llegadas

    def generar_lista_de_llegadas_presencial(self):
        llegadas = []

        h0, h1 = self.horarios_apertura_cierre

        for hora in range(h0, h1):
            inicio = hora * 60
            fin = (hora + 1) * 60
            minuto_actual = inicio

            while True:
                dt = self.minutos_hasta_proximo_cliente_presencial(minuto_actual)

                if dt is None:         # lam == 0 → no hay llegadas
                    break

                minuto_actual += dt

                if minuto_actual >= fin:
                    # Nos pasamos de la hora → no contamos el cliente
                    break

                # Llegada válida
                llegadas.append(minuto_actual)

        return llegadas

    def generar_lista_de_llegadas_mixto(self):
        h0, h1 = self.horarios_apertura_cierre

        llegadas_online = []
        llegadas_presencial = []

        for hora in range(h0, h1):
            inicio = hora * 60
            fin = (hora + 1) * 60
            minuto_actual = inicio

            while True:
                dt = self.minutos_hasta_proximo_cliente_mixto(minuto_actual)
                if dt is None:
                    break

                minuto_actual += dt

                if minuto_actual >= fin:
                    break

                # Decisión online vs presencial
                if random.random() < self.gamma_mixtos:
                    llegadas_online.append(minuto_actual)
                else:
                    llegadas_presencial.append(minuto_actual)

        return llegadas_online, llegadas_presencial
    
    def obtener_instancia_de_llegadas(self):
        
        lista_de_llegadas_por_dia = []

        for _ in range(self.dias):
            at_online = self.generar_lista_de_llegadas_online()
            at_presencial = self.generar_lista_de_llegadas_presencial()
            at_mixto_online, at_mixto_presencial = self.generar_lista_de_llegadas_mixto()
        
            eventos = []
            eventos += [(t, True) for t in at_online]
            eventos += [(t, False) for t in at_presencial]
            eventos += [(t, True) for t in at_mixto_online]
            eventos += [(t, False) for t in at_mixto_presencial]

            eventos.sort(key=lambda x: x[0])

            lista_de_llegadas_por_dia.append(eventos)

        return lista_de_llegadas_por_dia




