from item import Pelicula, Llibre
from abc import ABC, abstractmethod
import numpy as np
import csv


class Dataset(ABC):
    def __init__(self, ruta_items, ruta_valoracions):
        self.ruta_items = ruta_items
        self.ruta_valoracions = ruta_valoracions
        self.items = []       # Llista d'objectes (pel·lícules o llibres)
        self.items_ids = []   # IDs dels ítems
        self.valoracions_raw = []  # Llista de tuples (usuari, item, valoració)
        self.valoracions = np.empty((1, 1), dtype=object)  # Inicialització buida

        self.carrega_items()
        self.carrega_valoracions()
        self.carrega_matriu_valoracions()

    @abstractmethod
    def carrega_items(self):
        """
        Carrega els ítems del dataset. 
        Aquesta funció ha de ser implementada per les subclasses.
        """
        pass

    @abstractmethod
    def carrega_valoracions():
        """
        Carrega les valoracions del dataset.
        Aquesta funció ha de ser implementada per les subclasses.
        """
        pass

    def carrega_matriu_valoracions(self):
        """
        Crea una matriu de valoracions on les files són usuaris i les columnes són ítems.
        La matriu té la forma:
        | Usuari/Item | Item1 | Item2 | ... |
        | Usuari1     | Val1  | Val2  | ... |
        | Usuari2     | Val3  | Val4  | ... |
        Cada cel·la conté la valoració donada per l'usuari a l'ítem, o 0 si no hi ha valoració.
        Si un usuari no ha valorat un ítem, la cel·la corresponent serà 0.
        Args:
            None
        Returns:
            None    
        Aquesta funció construeix la matriu de valoracions a partir de les dades crues.
        La matriu té la primera fila i la primera columna com a identificadors d'usuari i ítem respectivament.
        La matriu es construeix de manera que cada cel·la conté la valoració donada per l'usuari a l'ítem.
        Si un usuari no ha valorat un ítem, la cel·la corresponent serà 0.
        """
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
        """
        Mostra la matriu de valoracions de manera llegible. Utilitzada per fer proves i debugging.
        """
        for fila in self.valoracions:
            print("\t".join(str(x) for x in fila))

    def get_valoracions(self):
        """
        Retorna la matriu de valoracions.
        Returns:
            np.ndarray: Matriu de valoracions on les files són usuaris i les columnes són ítems.
        """
        return self.valoracions
    
    def get_item(self, id_item):
        """
        Retorna l'objecte item corresponent a l'ID donat.
        Args:
            id_item (int | str): ID de l'item a cercar.
        
        Returns:
            Pelicula | Llibre | None: L'objecte item si es troba, o None si no existeix.
        """
        if id_item in self.items_ids:
            index = self.items_ids.index(id_item)
            return self.items[index]
        return None


class DatasetPelicules(Dataset):
    def carrega_items(self):
        self.items = []
        self.items_ids = []
        with open(self.ruta_items, "r", encoding="utf8") as arxiu:
            lector = csv.reader(arxiu, delimiter=",", quotechar='"')
            next(lector)  # salta la capçalera

            for parts in lector:
                try:
                    movie_id = int(parts[0])
                    titol = parts[1]
                    generes = parts[2].split('|') if parts[2] else []

                    self.items.append(Pelicula(movie_id, titol, generes))
                    self.items_ids.append(movie_id)
                except IndexError:
                    print(f"Línia malformada: {parts}")

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
    
    def get_descriptors(self):
        """
        Retorna una llista de descriptors dels ítems.
        Cada descriptor és una cadena de text que conté els gèneres d'una pel·lícula.
        Returns:
            list: Llista de descriptors on cada element és una cadena de text amb els gèneres d'una pel·lícula.
        """
        return [' '.join(p.genere) for p in self.items]


class DatasetLlibres(Dataset):
    def carrega_items(self):
        self.items = []
        self.items_ids = []
        MAX_LINIES = 10000  # o qualsevol valor que vulguis

        with open(self.ruta_items, "r", encoding="utf8") as arxiu:
            next(arxiu)  # salta la capçalera

            for i, linia in enumerate(arxiu):
                if i >= MAX_LINIES:
                    break  # parem quan arribem al límit

                parts = linia.strip().split(",")

                isbn = parts[0]
                titol = parts[1]
                autor = parts[2]
                any_publi = parts[3]
                editorial = parts[4]

                self.items_ids.append(isbn)
                self.items.append(Llibre(isbn, titol, autor, any_publi, editorial))

    def carrega_valoracions(self):
        with open(self.ruta_valoracions, "r") as arxiu:
            next(arxiu)
            for linia in arxiu:
                parts = linia.strip().split(",")
                user = parts[0]
                item = parts[1]
                if item not in self.items_ids:
                    print("Valoracio no afegida: item no trobat")
                    continue  # salta aquesta valoració
                raw_valoracio = float(parts[2])
                # Si és 0, considerem que no hi ha puntuació
                if raw_valoracio == 0:
                    valoracio = 0
                else:
                    valoracio = raw_valoracio / 2  # escalar de 0–10 a 0–5
                self.valoracions_raw.append((user, item, valoracio))
                print("Valoracio afegida")



"""dataset_pelicula = DatasetPelicules("dataset/MovieLens100k/prova_pelicules.csv", "dataset/MovieLens100k/prova_valoracions.csv")
print("Objecte Dataset Creat")
dataset_pelicula.carrega_items()
print("Importat items Creat")
dataset_pelicula.carrega_valoracions()
print("Valoracions Creades")
dataset_pelicula.carrega_matriu_valoracions()
print("Matriu Creada")
pelicula = dataset_pelicula.get_item(4)
print(pelicula)"""
