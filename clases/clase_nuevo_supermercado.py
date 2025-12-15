from src.models.models.multinomial_logit import MultinomialLogitModel
from src.models.transactions.base import TransactionGenerator
from src.models.models import Model
import csv
import os
from clases.clase_escenarios import Escenario
from datetime import datetime

class Nuevo_Supermercado():
    def __init__(self, escenario: Escenario):
                

        self.dia = 0
       
        self.lista_de_productos: list[int] = escenario.lista_de_productos
        self.lista_de_nivel_de_stock: list[int] = escenario.lista_de_nivel_de_stock

        self.cantidades: dict[str, int] = {self.lista_de_productos[i]: self.lista_de_nivel_de_stock[i] for i in range(len(self.lista_de_productos))}

        self.compra_online_por_baches_dia_anterior: dict[str, list[int]] = {horario: [] for horario in escenario.horarios_de_entrega}
        self.compra_online_por_baches_actual: dict[str, list[int]] = {horario: [] for horario in escenario.horarios_de_entrega}

        self.escenario: Escenario = escenario

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.output_dir = os.path.join(
            f"informe_simulacion_{timestamp}"
        )

        os.makedirs(self.output_dir, exist_ok=True)

    def reiniciar_compra_online_actual(self):
        self.compra_online_por_baches_actual: dict[str, list[int]] = {horario: [] for horario in self.escenario.horarios_de_entrega}

    def obtener_productos_disponibles(self):
        productos_disponibles = []
        for prod in self.cantidades:
            if self.cantidades[prod] > 0:
                productos_disponibles.append(prod)
        return [0] + productos_disponibles 
    
    def reponer_stock(self):
        self.cantidades: dict[str, int] = {self.lista_de_productos[i]: self.lista_de_nivel_de_stock[i] for i in range(len(self.lista_de_productos))}

    def consumir_stock(self, producto_comprado: int):
        # DEVOLVER PRODUCTO CONSUMIDO!!!!!!
        if producto_comprado != 0:
            if self.cantidades[producto_comprado] == 0:
                #Se agoto el stock
                return None
            else:
                self.cantidades[producto_comprado] -= 1
                return producto_comprado
        else:
            return None

    def agregar_producto_a_la_compra_online_actual(self, horario, producto: int):
        self.compra_online_por_baches_actual[horario].append(producto)
            
    def consumir_stock_de_un_bache_online(self, horario: str):
        productos_del_horario = self.compra_online_por_baches_dia_anterior[horario]

        for producto in productos_del_horario:
            producto_comprado = self.consumir_stock(producto)

            
            self.guardar_transaccion(
                canal="ONLINE",
                tiempo_o_bache=horario,
                productos_disponibles=self.lista_de_productos,
                producto_comprado=producto_comprado if producto_comprado is not None else 0
            )

    def simular_dia(self, tiempos: list[int], is_online: list[bool], modelos: list[MultinomialLogitModel], tiempos_de_entrega: list[str]):
        self.dia += 1
        self.registrar_dia(self.dia)

        for i in range(len(tiempos)):

            horario_actual = tiempos[i]
            horario_siguiente = tiempos[i+1] if i+1 < len(tiempos) else 24*60 + 1

            # --- Ver si pasamos por un horario de entrega ---
            for h_str, h_min in zip(self.escenario.horarios_de_entrega, self.escenario.horarios_de_entrega_min):
                if horario_actual < h_min <= horario_siguiente:
                    self.consumir_stock_de_un_bache_online(h_str)
                   

            tipo_de_cliente = "online" if is_online[i] else "presencial"
            modelo_de_eleccion = modelos[i]
            horario_de_entrega = tiempos_de_entrega[i] if is_online[i] else None
            
            tg = TransactionGenerator(modelo_de_eleccion)

            if tipo_de_cliente == "presencial":
                ofrecidos = self.obtener_productos_disponibles()
                producto_elegido = tg.generate_transaction_for(ofrecidos).product
                prodcuto_comprado = self.consumir_stock(producto_elegido)
                if prodcuto_comprado is not None:
                    self.guardar_transaccion(
                        canal="PRESENCIAL",
                        tiempo_o_bache=horario_actual,
                        productos_disponibles=ofrecidos,
                        producto_comprado=prodcuto_comprado
                    )



            elif tipo_de_cliente == "online":
                ofrecidos = self.lista_de_productos   
                producto_elegido = tg.generate_transaction_for(ofrecidos).product
                self.agregar_producto_a_la_compra_online_actual(horario_de_entrega, producto_elegido)
        

        self.compra_online_por_baches_dia_anterior = self.compra_online_por_baches_actual
        self.reiniciar_compra_online_actual()
        print("Cantidades Dia ", self.dia, ": ", self.cantidades)
        self.reponer_stock()

    def simular_dia_online(self, tiempos: list[int], is_online: list[bool], modelos: list[MultinomialLogitModel], tiempos_de_entrega: list[str]):
        for i in range(len(tiempos)):

            horario_actual = tiempos[i]
            horario_siguiente = tiempos[i+1] if i+1 < len(tiempos) else 24*60 + 1

            # --- Ver si pasamos por un horario de entrega ---
            for h_str, h_min in zip(self.escenario.horarios_de_entrega, self.escenario.horarios_de_entrega_min):
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
       
    def simular_escenario(self, eventos_por_dia: list[list[tuple[int, bool, MultinomialLogitModel, str]]]):

        self.guardar_inventario_inicial()
        self.guardar_parametros_escenario()
        self.guardar_modelos_de_eleccion()
        self.guardar_porcentaje_modelos_por_dia(eventos_por_dia)


        dia_0 = eventos_por_dia[0]
        tiempos, is_online, modelos, tiempos_de_entrega = zip(*dia_0)

        self.simular_dia_online(tiempos, is_online, modelos, tiempos_de_entrega)



        for i in range(1, self.escenario.dias):

            dia_i = eventos_por_dia[i]
            tiempos, is_online, modelos, tiempos_de_entrega = zip(*dia_i)
            self.simular_dia(tiempos, is_online, modelos, tiempos_de_entrega)
            
    def guardar_transaccion(
        self,
        canal: str,
        tiempo_o_bache,
        productos_disponibles: list[int],
        producto_comprado: int
    ):
        archivo = os.path.join(self.output_dir, "transacciones.csv")
        escribir_header = not os.path.exists(archivo)

        productos_str = ";".join(str(p) for p in productos_disponibles)

        with open(archivo, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            if escribir_header:
                writer.writerow([
                    "canal",
                    "tiempo_o_bache",
                    "productos_disponibles",
                    "producto_comprado"
                ])

            writer.writerow([
                canal.upper(),
                tiempo_o_bache,
                productos_str,
                producto_comprado
            ])

    def registrar_dia(self, dia):
        archivo = os.path.join(self.output_dir, "transacciones.csv")

        escribir_header = not os.path.exists(archivo)

        with open(archivo, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            if escribir_header:
                writer.writerow([
                    "canal",
                    "tiempo_o_bache",
                    "productos_disponibles",
                    "producto_comprado"
                ])

            writer.writerow([f"DIA {dia}", "", "", ""])

    def guardar_inventario_inicial(self):
        archivo = os.path.join(self.output_dir, "inventario_inicial.csv")

        with open(archivo, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["producto", "stock_inicial"])

            for prod, stock in zip(
                self.escenario.lista_de_productos,
                self.escenario.lista_de_nivel_de_stock
            ):
                writer.writerow([prod, stock])

    def guardar_parametros_escenario(self):
        archivo = os.path.join(self.output_dir, "parametros_escenario.csv")

        llegada = self.escenario.llegada

        with open(archivo, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # ============================
            # Parámetros globales
            # ============================
            writer.writerow(["# PARAMETROS GLOBALES"])
            writer.writerow(["parametro", "valor"])

            writer.writerow(["dias", llegada.dias])
            writer.writerow(["hora_apertura", llegada.horarios_apertura_cierre[0]])
            writer.writerow(["hora_cierre", llegada.horarios_apertura_cierre[1]])
            writer.writerow([
                "cantidad_horarios_entrega",
                len(self.escenario.horarios_de_entrega)
            ])
            writer.writerow(["gamma_mixtos_online", llegada.gamma_mixtos])
            writer.writerow(["seed", "definido_en_llegada"])

            # Línea en blanco
            writer.writerow([])

            # ============================
            # Lambdas por hora
            # ============================
            writer.writerow(["# LAMBDAS POR HORA"])
            writer.writerow(["hora", "mixto", "online", "presencial"])

            for hora in range(24):
                writer.writerow([
                    hora,
                    llegada.lambdas_mixto_por_hora[hora],
                    llegada.lambdas_online_por_hora[hora],
                    llegada.lambdas_presencial_por_hora[hora]
                ])

    def guardar_modelos_de_eleccion(self):
        modelos_dir = os.path.join(self.output_dir, "modelos")
        os.makedirs(modelos_dir, exist_ok=True)

        for i, modelo in enumerate(self.escenario.modelos_de_eleccion):
            path = os.path.join(modelos_dir, f"modelo_{i}.json")
            modelo.save(path)

    def guardar_porcentaje_modelos_por_dia(self, eventos_por_dia):
        archivo = os.path.join(self.output_dir, "porcentaje_modelos_por_dia.csv")


        # Mapear modelo -> id
        modelo_a_id = {
            modelo: i for i, modelo in enumerate(self.escenario.modelos_de_eleccion)
        }

        with open(archivo, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["dia", "modelo_id", "modelo_clase", "porcentaje"])

            for dia, eventos in enumerate(eventos_por_dia, start=1):
                total_clientes = len(eventos)
                conteo = {}

                for _, _, modelo, _ in eventos:
                    modelo_id = modelo_a_id[modelo]
                    conteo[modelo_id] = conteo.get(modelo_id, 0) + 1

                for modelo_id, cantidad in conteo.items():
                    porcentaje = cantidad / total_clientes
                    modelo = self.escenario.modelos_de_eleccion[modelo_id]

                    writer.writerow([
                        dia,
                        modelo_id,
                        modelo.__class__.__name__,
                        porcentaje
                    ])



