import random
import math
import numpy as np
import matplotlib.pyplot as plt
from src.models.models.multinomial_logit import MultinomialLogitModel
from src.models.transactions.base import TransactionGenerator



class Supermercado():
    def __init__(self, lista_de_productos: list[int],  
                 lista_de_nivel_de_stock: list[int],
                 horarios_apertura_cierre: tuple[int, int],
                 cantidad_de_horarios_de_entregas: int,
                 lambdas_online_por_hora: list[float],
                 lambdas_presencial_por_hora: list[float],
                 lambdas_mixto_por_hora: list[float],
                 gamma_mixtos: float):

        self.dia = 0
        self.producto_sin_stock_al_final_del_dia = [0 for i in range(len(lista_de_productos))]
        self.cantidad_de_dias_sin_stock = [0 for i in range(len(lista_de_productos))]

        self.lambdas_online_por_hora = lambdas_online_por_hora
        self.lambdas_presencial_por_hora = lambdas_presencial_por_hora
        self.lambdas_mixto_por_hora = lambdas_mixto_por_hora

        self.horarios_apertura_cierre: tuple[int, int] = horarios_apertura_cierre
        self.lista_de_productos: list[int] = lista_de_productos
        self.lista_de_nivel_de_stock: list[int] = lista_de_nivel_de_stock

        self.cantidades: dict[str, int] = {self.lista_de_productos[i]: self.lista_de_nivel_de_stock[i] for i in range(len(self.lista_de_productos))}
    
        self.minutos_abierto: int = (self.horarios_apertura_cierre[1] - self.horarios_apertura_cierre[0])*60
        self.minutos_todo_el_dia: int = 24 * 60

        self.gamma_mixtos: float = gamma_mixtos

        self.horarios_de_entrega = calcular_horarios_entrega_str(self.horarios_apertura_cierre, cantidad_de_horarios_de_entregas)
        self.horarios_de_entrega_min = [self._horario_str_a_minutos(h) for h in self.horarios_de_entrega]


        self.compra_online_por_baches_dia_anterior: dict[str, dict[int, int]] = {horario: {} for horario in self.horarios_de_entrega}
        self.compra_online_por_baches_actual: dict[str, dict[int, int]] = {horario: {} for horario in self.horarios_de_entrega}

        self.modelo1 = MultinomialLogitModel.simple_random([0] + self.lista_de_productos)
        self.modelo2 = MultinomialLogitModel.simple_random([0] + self.lista_de_productos)

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
            return 1e9  # no llega nadie en esta hora
        return random.expovariate(lam)

    def minutos_hasta_proximo_cliente_presencial(self, minuto_actual):
        lam = self.lambda_presencial_por_minuto(minuto_actual)
        if lam == 0:
            return 1e9   # nadie llega en esta hora
        return random.expovariate(lam)

    def minutos_hasta_proximo_cliente_mixto(self, minuto_actual):
        lam = self.lambda_mixto_por_minuto(minuto_actual)
        if lam == 0:
            return 1e9
        return random.expovariate(lam)

    def reiniciar_compra_online_actual(self):
        self.compra_online_por_baches_actual: dict[str, dict[int, int]] = {horario: {} for horario in self.horarios_de_entrega}

    def obtener_productos_disponibles(self):
        productos_disponibles = []
        for prod in self.cantidades:
            if self.cantidades[prod] > 0:
                productos_disponibles.append(prod)
        return [0] + productos_disponibles 
    
    def reponer_stock(self):
        
        for i in range(len(self.lista_de_productos)):
            if self.cantidades[self.lista_de_productos[i]] == 0:
                self.cantidad_de_dias_sin_stock[i] += 1

        self.cantidades: dict[str, int] = {self.lista_de_productos[i]: self.lista_de_nivel_de_stock[i] for i in range(len(self.lista_de_productos))}

    def probabilidad_out_of_stock(self):
        probabilidades = []
        
        for i in range(len(self.lista_de_productos)):
            probabilidades.append(self.cantidad_de_dias_sin_stock[i] / self.dia)
        
        return probabilidades

    def consumir_stock(self, producto_comprado: int):
        if producto_comprado != 0:
            if self.cantidades[producto_comprado] == 0:
                #Se agoto el stock
                pass
            else:
                self.cantidades[producto_comprado] -= 1

    def generar_lista_de_llegadas_online(self): 
        horario_actual = 0
        horario_final = 24 * 60
        lista_de_llegadas = []

        while horario_actual < horario_final:
            dt = self.minutos_hasta_proximo_cliente_online(horario_actual)
            horario_actual += dt
            if horario_actual >= horario_final:
                break
            lista_de_llegadas.append(horario_actual)

        return lista_de_llegadas

    def generar_lista_de_llegadas_presencial(self):

        horario_inicial = self.horarios_apertura_cierre[0] * 60
        horario_final   = self.horarios_apertura_cierre[1] * 60
        horario_actual  = horario_inicial

        lista_de_llegadas = []

        while horario_actual < horario_final:

            dt = self.minutos_hasta_proximo_cliente_presencial(horario_actual)
            horario_actual += dt

            if horario_actual >= horario_final:
                break

            lista_de_llegadas.append(horario_actual)

        return lista_de_llegadas

    def generar_lista_de_llegadas_mixto(self):

        horario_inicial = self.horarios_apertura_cierre[0] * 60
        horario_final   = self.horarios_apertura_cierre[1] * 60
        horario_actual  = horario_inicial

        lista_de_llegadas_online = []
        lista_de_llegadas_presencial = []

        while horario_actual < horario_final:

            dt = self.minutos_hasta_proximo_cliente_mixto(horario_actual)
            horario_actual += dt

            if horario_actual >= horario_final:
                break

            # Decide si el mixto es online o presencial
            if random.random() < self.gamma_mixtos:
                lista_de_llegadas_online.append(horario_actual)
            else:
                lista_de_llegadas_presencial.append(horario_actual)

        return lista_de_llegadas_online, lista_de_llegadas_presencial

    def agregar_producto_a_la_compra_online_actual(self, horario, producto: int):
        if producto not in self.compra_online_por_baches_actual[horario]:
            self.compra_online_por_baches_actual[horario][producto] = 0
        self.compra_online_por_baches_actual[horario][producto] += 1

    def consumir_stock_de_un_bache_online(self, horario: str):
        productos_y_cantidades_del_horario = self.compra_online_por_baches_dia_anterior[horario]
        for producto, cantidad in productos_y_cantidades_del_horario.items():
            for _ in range(cantidad):
                self.consumir_stock(producto)

    def _horario_str_a_minutos(self, horario_str: str) -> int:
        hh, mm = map(int, horario_str.split(":"))
        return hh * 60 + mm

    def obtener_instancia(self):
        
        at_online = self.generar_lista_de_llegadas_online()
        at_presencial = self.generar_lista_de_llegadas_presencial()
        at_mixto_online, at_mixto_presencial = self.generar_lista_de_llegadas_mixto()
    
        eventos = []
        eventos += [(t, True) for t in at_online]
        eventos += [(t, False) for t in at_presencial]
        eventos += [(t, True) for t in at_mixto_online]
        eventos += [(t, False) for t in at_mixto_presencial]

        eventos.sort(key=lambda x: x[0])
        tiempos = [e[0] for e in eventos]
        is_online = [e[1] for e in eventos]
        modelos = [random.choice([self.modelo1, self.modelo2]) for _ in range(len(eventos))]
        tiempos_de_entrega = [random.choice(self.horarios_de_entrega) if is_online[i] else None for i in range(len(eventos))]

        return tiempos, is_online, modelos, tiempos_de_entrega

    def simular_dia(self, tiempos: list[int], is_online: list[bool], modelos: list[MultinomialLogitModel], tiempos_de_entrega: list[str]):
        self.dia += 1

        for i in range(len(tiempos)):

            horario_actual = tiempos[i]
            horario_siguiente = tiempos[i+1] if i+1 < len(tiempos) else 24*60 + 1

            # --- Ver si pasamos por un horario de entrega ---
            for h_str, h_min in zip(self.horarios_de_entrega, self.horarios_de_entrega_min):
                if horario_actual < h_min <= horario_siguiente:
                    self.consumir_stock_de_un_bache_online(h_str)

                   

            tipo_de_cliente = "online" if is_online[i] else "presencial"
            modelo_de_eleccion = modelos[i]
            horario_de_entrega = tiempos_de_entrega[i] if is_online[i] else None
            
            tg = TransactionGenerator(modelo_de_eleccion)

            if tipo_de_cliente == "presencial":
                ofrecidos = self.obtener_productos_disponibles()
                producto_comprado = tg.generate_transaction_for(ofrecidos).product
                self.consumir_stock(producto_comprado)
                

            elif tipo_de_cliente == "online":
                ofrecidos = self.lista_de_productos
                producto_comprado = tg.generate_transaction_for(ofrecidos).product
                self.agregar_producto_a_la_compra_online_actual(horario_de_entrega, producto_comprado)
        

        self.compra_online_por_baches_dia_anterior = self.compra_online_por_baches_actual
        self.reiniciar_compra_online_actual()
        print("Cantidades Dia ", self.dia, ": ", self.cantidades)
        self.reponer_stock()

    def simular_dia_online(self, tiempos: list[int], is_online: list[bool], modelos: list[MultinomialLogitModel], tiempos_de_entrega: list[str]):
        for i in range(len(tiempos)):

            horario_actual = tiempos[i]
            horario_siguiente = tiempos[i+1] if i+1 < len(tiempos) else 24*60 + 1

            # --- Ver si pasamos por un horario de entrega ---
            for h_str, h_min in zip(self.horarios_de_entrega, self.horarios_de_entrega_min):
                if horario_actual < h_min <= horario_siguiente:
                    self.consumir_stock_de_un_bache_online(h_str)

                   

            tipo_de_cliente = "online" if is_online[i] else "presencial"
            modelo_de_eleccion = modelos[i]
            horario_de_entrega = tiempos_de_entrega[i] if is_online[i] else None
            
            tg = TransactionGenerator(modelo_de_eleccion)
            if tipo_de_cliente == "online":
                ofrecidos = self.lista_de_productos
                producto_comprado = tg.generate_transaction_for(ofrecidos).product
                self.agregar_producto_a_la_compra_online_actual(horario_de_entrega, producto_comprado)
        

        self.compra_online_por_baches_dia_anterior = self.compra_online_por_baches_actual
        self.reiniciar_compra_online_actual()
       
    def generar_n_instancias(self, n):
        instancias = []
        for _ in range(n):
            instancias.append(self.obtener_instancia())
        return instancias

    def simular_n_dias(self, n_instancias, dia_online_ON):

        for i in range(len(n_instancias)):

            tiempos, is_online, modelos, tiempos_de_entrega = n_instancias[i]

            if dia_online_ON and i == 0:
                self.simular_dia_online(tiempos, is_online, modelos, tiempos_de_entrega)
            else:
                self.simular_dia(tiempos, is_online, modelos, tiempos_de_entrega)
        
    # GRAFICOS SIN EFECTO EN LA SIMULACION

    def simular_con_graficos_un_dia(self):
        self.dia += 1
        tiempos, is_online, modelos, tiempos_de_entrega = self.obtener_instancia()

        # Eje X y series de stock solo cuando hay cambios
        historial_stock = {pid: [] for pid in self.cantidades}
        historial_tiempo = []

        # Guardar instante actual del stock
        def registrar(t):
            historial_tiempo.append(t)
            for pid in historial_stock:
                historial_stock[pid].append(self.cantidades[pid])

        momentos_baches = []

        for i in range(len(tiempos)):
            horario_actual = tiempos[i]
            horario_siguiente = tiempos[i+1] if i+1 < len(tiempos) else 24*60+1

            # ----------- BACHES ONLINE (si ocurren en este intervalo) -----------
            for h_str, h_min in zip(self.horarios_de_entrega, self.horarios_de_entrega_min):
                if horario_actual < h_min <= horario_siguiente:
                    # registrar estado ANTES de consumir
                    registrar(h_min)

                    momentos_baches.append(h_min)
                    self.consumir_stock_de_un_bache_online(h_str)

                    # registrar estado DESPUÉS
                    registrar(h_min)

            # ----------- CLIENTE PRESENCIAL -----------
            if not is_online[i]:
                modelo = modelos[i]
                tg = TransactionGenerator(modelo)

                antes = list(self.cantidades.values())

                ofrecidos = self.obtener_productos_disponibles()
                producto = tg.generate_transaction_for(ofrecidos).product
                self.consumir_stock(producto)

                # registrar solo si cambió el stock
                if list(self.cantidades.values()) != antes:
                    registrar(horario_actual)

        # Final del día
        # REPONER STOCK SIN REGISTRAR
        self.cantidades: dict[str, int] = {self.lista_de_productos[i]: self.lista_de_nivel_de_stock[i] for i in range(len(self.lista_de_productos))}

        # ------------------ GRAFICAR ------------------
        plt.figure(figsize=(12, 7))

        for pid, valores in historial_stock.items():
            plt.step(historial_tiempo, valores, where="post", label=f"Producto {pid}")

        for t in momentos_baches:
            plt.axvline(t, color="black", linewidth=2, linestyle="--")

        plt.title("Evolución del stock durante el día (solo eventos reales)")
        plt.xlabel("Minutos desde apertura")
        plt.ylabel("Stock")
        plt.legend()
        plt.grid(True)
        plt.show()

    def graficar_lambdas(self):
        horas = list(range(24))

        # Ajusto cualquier lista que no tenga exactamente 24 valores
        def ajustar(l):
            if len(l) == 24:
                return l
            elif len(l) < 24:
                return l + [0]*(24 - len(l))
            else:
                return l[:24]

        lambdas_online  = ajustar(self.lambdas_online_por_hora)
        lambdas_mixto   = ajustar(self.lambdas_mixto_por_hora)
        lambdas_pres    = ajustar(self.lambdas_presencial_por_hora)

        plt.figure(figsize=(12,6))
        plt.plot(horas, lambdas_online,  label="Online")
        plt.plot(horas, lambdas_mixto,   label="Mixto")
        plt.plot(horas, lambdas_pres,    label="Presencial")

        plt.xlabel("Hora del día")
        plt.ylabel("Lambda (clientes por hora)")
        plt.title("Tasa de Llegadas por Hora (Online, Mixto, Presencial)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def grafico_de_intensidad_por_hora(self):
        import matplotlib.pyplot as plt
        import numpy as np

        # --- Generar llegadas ---
        online = self.generar_lista_de_llegadas_online()
        pres = self.generar_lista_de_llegadas_presencial()
        mix_on, mix_pre = self.generar_lista_de_llegadas_mixto()

        # Convertir minutos -> hora (int)
        def hora(t): return int(t // 60)

        horas = np.arange(24)

        conteo_online  = np.zeros(24, int)
        conteo_pres    = np.zeros(24, int)
        conteo_mixon   = np.zeros(24, int)
        conteo_mixpre  = np.zeros(24, int)

        for t in online:      conteo_online[hora(t)] += 1
        for t in pres:        conteo_pres[hora(t)] += 1
        for t in mix_on:      conteo_mixon[hora(t)] += 1
        for t in mix_pre:     conteo_mixpre[hora(t)] += 1

        # --- Graficar ---
        plt.figure(figsize=(14,6))

        plt.plot(horas, conteo_online, label="Online", linewidth=2)
        plt.plot(horas, conteo_mixon, label="Mixto Online", linewidth=2)
        plt.plot(horas, conteo_mixpre, label="Mixto Presencial", linewidth=2)
        plt.plot(horas, conteo_pres, label="Presencial", linewidth=2)

        plt.title("Intensidad de Llegadas por Hora")
        plt.xlabel("Hora del día")
        plt.ylabel("Cantidad de clientes")
        plt.xticks(horas)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        print("\nTotales:")
        print("Online:             ", sum(conteo_online))
        print("Mixto Online:       ", sum(conteo_mixon))
        print("Mixto Presencial:   ", sum(conteo_mixpre))
        print("Presencial:         ", sum(conteo_pres))










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

