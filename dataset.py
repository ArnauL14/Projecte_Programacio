from importar import importar_pelicules, importar_llibres, importar_ids
from abc import ABCMeta, abstractmethod
import numpy as np


class Dataset(metaclass=ABCMeta):
    def __init__(self, ruta_items, ruta_valoracions):
        self.ruta_items = ruta_items
        self.ruta_valoracions = ruta_valoracions
        self.items = []
        self.items_ids = []
        self.items_per_id = {}
        self.raitings = np.empty((0, 0), dtype=object)

    def get_item(self, id_item):
        return self.items_per_id.get(id_item, None)

    def llista_items(self):
        return self.items

    @abstractmethod
    def carrega(self):
        pass

    def carrega_valoracions(self):
        llista_ids_usuaris = importar_ids(self.ruta_valoracions)
        llista_ids_items = self.items_ids

        n_files = len(llista_ids_usuaris) + 1
        n_columnes = len(llista_ids_items) + 1

        matriu = np.zeros((n_files, n_columnes), dtype=object)
        matriu[0][0] = "origen"

        for j, id_item in enumerate(llista_ids_items, start=1):
            matriu[0][j] = id_item

        for i, id_usuari in enumerate(llista_ids_usuaris, start=1):
            matriu[i][0] = id_usuari

        with open(self.ruta_valoracions, "r") as f:
            next(f)
            for linia in f:
                parts = linia.strip().split(",")

                id_usuari = parts[0]
                id_item = parts[1]
                valoracio = float(parts[2])

                fila = llista_ids_usuaris.index(id_usuari) + 1
                columna = llista_ids_items.index(id_item) + 1

                matriu[fila][columna] = valoracio
        
        self.raitings = matriu


class DatasetPelicules(Dataset):
    def __init__(self, ruta_items, ruta_valoracions):
        super().__init__(ruta_items, ruta_valoracions)
        self.carrega()
        self.carrega_valoracions()

    def carrega(self):
        self.items, self.items_ids = importar_pelicules(self.ruta_items)


class DatasetLlibres(Dataset):
    def __init__(self, ruta_items, ruta_valoracions):
        super().__init__(ruta_items, ruta_valoracions)
        self.carrega()
        self.carrega_valoracions()

    def carrega(self):
        self.items, self.items_ids = importar_llibres(self.ruta_items)