from item import Pelicula, Llibre
from abc import ABC, abstractmethod
import numpy as np


class Dataset(ABC):
    def __init__(self, ruta_items, ruta_valoracions):
        self.ruta_items = ruta_items
        self.ruta_valoracions = ruta_valoracions
        #self.arxiu_csv = arxiu_csv 
        self.items = []       # Llista d'objectes (pel·lícules o llibres)
        self.items_ids = []   # IDs dels ítems
        self.valoracions_raw = []  # Llista de tuples (usuari, item, valoració)
        self.valoracions = np.empty((1, 1), dtype=object)  # Inicialització buida

        self.carrega_items()
        self.carrega_valoracions()

    @abstractmethod
    def carrega_items(self):
        pass

    @abstractmethod
    def carrega_valoracions():
        pass

    def carrega_matriu_valoracions(self):
        llista_usuaris = list({v[0] for v in self.valoracions_raw})
        n_rows = len(llista_usuaris) + 1
        n_cols = len(self.items_ids) + 1
        matriu = np.zeros((n_rows, n_cols), dtype=object)
        matriu[0][0] = "id"

        for j, item_id in enumerate(self.items_ids, start=1):
            matriu[0][j] = item_id
        for i, usuari_id in enumerate(llista_usuaris, start=1):
            matriu[i][0] = usuari_id
        for usuari_id, item_id, valoracio in self.valoracions_raw:
            i = llista_usuaris.index(usuari_id) + 1
            j = self.items_ids.index(item_id) + 1
            matriu[i][j] = valoracio

        self.valoracions = matriu

    def mostra_matriu(self):
        for fila in self.valoracions:
            print("\t".join(str(x) for x in fila))

    def get_valoracions(self):
        return self.valoracions


class DatasetPelicules(Dataset):
    def carrega_items(self):
        self.items = []
        self.items_ids = []
        with open(self.ruta_items, "r", encoding="utf8") as arxiu:
            next(arxiu)#ens saltem la primera línia
            for linia in arxiu:
                parts = linia.strip().split(",")

                movie_id = int(parts[0])
                titol = parts[1]
                generes = parts[2].split('|') if parts[2] else []

                self.items_ids.append(movie_id)
                self.items.append(Pelicula(movie_id, titol, generes))
        self.items_ids.sort()

        print("OPERACIO REALITZADA AMB EXIT")
        
        # self.items, self.items_ids = importar_pelicules(self.ruta_items)

    def carrega_valoracions(self):
        #mirar de canviar lo de la ruta
        with open(self.ruta_valoracions, "r", encoding="utf8") as arxiu:
            next(arxiu)  # Saltar la capçalera
            for linia in arxiu:
                parts = linia.strip().split(",")
                user = parts[0]
                item = int(parts[1])
                valoracio = float(parts[2])
                self.valoracions_raw.append((user, item, valoracio))


class DatasetLlibres(Dataset):
    def carrega_items(self):
        self.items = []
        self.items_ids = []
        """ara estan gaurdats dins de la classe"""
        with open(self.ruta_items, "r", encoding="utf8") as arxiu:
            next(arxiu)  # salta capçalera

            for linia in arxiu:
                parts = linia.strip().split(",")

                isbn = parts[0]
                titol = parts[1]
                autor = parts[2]
                any_publi = parts[3]
                editorial = parts[4]

                self.items_ids.append(isbn)
                self.items.append(Llibre(isbn, titol, autor, any_publi, editorial))

        self.items_ids.sort()
        print("OPERACIO REALITZADA AMB EXIT")
    
    """def carrega_items(self):
        self.items, self.items_ids = importar_llibres(self.ruta_items)"""

    def carrega_valoracions(self):
        with open(self.ruta_valoracions, "r") as arxiu:
            next(arxiu)
            for linia in arxiu:
                parts = linia.strip().split(",")
                user = parts[0]
                item = parts[1]
                raw_valoracio = float(parts[2])
                # Si és 0, considerem que no hi ha puntuació
                if raw_valoracio == 0:
                    valoracio = 0
                else:
                    valoracio = raw_valoracio / 2  # escalar de 0–10 a 0–5
                self.valoracions_raw.append((user, item, valoracio))


"""print("Pelicules")
dataset_pelicula = DatasetPelicules("dataset/MovieLens100k/prova_pelicules.csv", "dataset/MovieLens100k/prova_valoracions.csv")
print("Objecte Dataset Creat")
dataset_pelicula.carrega_items()
print("Importat items Creat")
dataset_pelicula.carrega_valoracions()
print("Valoracions Creades")
dataset_pelicula.carrega_matriu_valoracions()
print("Matriu Creada")
dataset_pelicula.mostra_matriu()
print("")

print("Llibres")
dataset_llibre = DatasetLlibres("dataset/Books/prova_llibres.csv", "dataset/Books/prova_valoracions.csv")
print("Objecte Dataset Creat")
dataset_llibre.carrega_items()
print("Importat items Creat")
dataset_llibre.carrega_valoracions()
print("Valoracions Creades")
dataset_llibre.carrega_matriu_valoracions()
print("Matriu Creada")
dataset_llibre.mostra_matriu()
print("")"""
