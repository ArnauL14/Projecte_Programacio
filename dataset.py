from item import Pelicula, Llibre
from abc import ABC, abstractmethod
import numpy as np
import csv


class Dataset(ABC):
    """
    Classe base per a la gestió de datasets de recomanació.
    Aquesta classe defineix la interfície per carregar ítems i valoracions,
    i proporciona funcionalitats per crear una matriu de valoracions.
    Attributes:
        ruta_items (str): Ruta al fitxer que conté els ítems.
        ruta_valoracions (str): Ruta al fitxer que conté les valoracions.
        items (list): Llista d'ítems carregats.
        items_ids (list): Llista d'IDs dels ítems.
        valoracions_raw (list): Llista de tuples amb les valoracions crues (usuari, ítem, valoració).
        valoracions (np.ndarray): Matriu de valoracions (usuaris x ítems).
    
    Métodes:
        carrega_items(): Carrega els ítems del dataset.
        carrega_valoracions(): Carrega les valoracions del dataset.
        carrega_matriu_valoracions(): Crea una matriu de valoracions a partir de les valoracions del dataset.
        
    """
    def __init__(self, ruta_items, ruta_valoracions):
        self.ruta_items = ruta_items
        self.ruta_valoracions = ruta_valoracions
        self.items = []
        self.items_ids = []
        self.valoracions_raw = []
        self.valoracions = []

    @abstractmethod
    def carrega_items(self):
        """Carrega els ítems del dataset."""
        pass

    @abstractmethod
    def carrega_valoracions(self):
        """Carrega les valoracions del dataset."""
        pass

    def carrega_matriu_valoracions(self):
        """Crea una matriu amb valoracions (files: usuaris, columnes: ítems)."""
        usuaris = sorted(list({v[0] for v in self.valoracions_raw}))
        id2user = {usuari: idx for idx, usuari in enumerate(usuaris)}
        id2item = {item: idx for idx, item in enumerate(self.items_ids)}

        matriu = np.zeros((len(usuaris), len(self.items_ids)))

        for usuari, item, valoracio in self.valoracions_raw:
            if item in id2item:
                matriu[id2user[usuari], id2item[item]] = valoracio

        self.valoracions = matriu

    def mostra_matriu(self):
        """Mostra la matriu de valoracions."""
        for fila in self.valoracions:
            print("\t".join(str(x) for x in fila))

    def get_valoracions(self):
        """Retorna la matriu de valoracions."""
        return self.valoracions

    def get_item(self, id_item):
        """Retorna l'ítem donat el seu ID."""
        if id_item in self.items_ids:
            index = self.items_ids.index(id_item)
            return self.items[index]
        return None


class DatasetPelicules(Dataset):
    """Classe per a gestionar un dataset de pel·lícules.
    Aquesta classe hereta de Dataset i implementa els mètodes per carregar ítems 
    i valoracions específics per a pel·lícules.
    
    Attributes:
        ruta_items (str): Ruta al fitxer CSV que conté les pel·lícules.
        ruta_valoracions (str): Ruta al fitxer CSV que conté les valoracions de les pel·lícules.
    
    Métodes:
        carrega_items(): Carrega les pel·lícules del dataset.
        carrega_valoracions(): Carrega les valoracions de les pel·lícules.
        get_descriptors(): Retorna una llista de descriptors (gèneres) de les pel·lícules.
    """
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
                except (IndexError, ValueError):
                    print(f"Línia malformada o invàlida: {parts}")
        self.items_ids.sort()
        print("Pelicules carregades correctament.")

    def carrega_valoracions(self):
        with open(self.ruta_valoracions, "r", encoding="utf8") as arxiu:
            next(arxiu)
            for linia in arxiu:
                parts = linia.strip().split(",")
                try:
                    user = parts[0]
                    item = int(parts[1])
                    valoracio = float(parts[2])
                    self.valoracions_raw.append((user, item, valoracio))
                except (IndexError, ValueError):
                    print(f"Valoració malformada: {parts}")

    def get_descriptors(self):
        """Retorna descriptors (gèneres) de cada pel·lícula."""
        return [' '.join(p.genere) for p in self.items]


class DatasetLlibres(Dataset):
    """Classe per a gestionar un dataset de llibres.
    Aquesta classe hereta de Dataset i implementa els mètodes per carregar ítems
    i valoracions específics per a llibres.
    Attributes:
        ruta_items (str): Ruta al fitxer CSV que conté els llibres.
        ruta_valoracions (str): Ruta al fitxer CSV que conté les valoracions dels llibres.
    Métodes:
            carrega_items(): Carrega els llibres del dataset.
            carrega_valoracions(): Carrega les valoracions dels llibres.
    """
    def carrega_items(self):
        """
        Aquesta funció llegeix un fitxer CSV i crea una llista d'objectes Llibre.
        Args:
            self.ruta_items (str): Ruta al fitxer CSV que conté els llibres.
        Returns:
            None
        Raises:
            FileNotFoundError: Si el fitxer no es troba.
            ValueError: Si hi ha un error en el format de les dades.
        Example:
            >>> dataset = DatasetLlibres("books.csv", "ratings.csv")
            >>> dataset.carrega_items()
            Llibres carregats correctament.
        """
        self.items = []
        self.items_ids = []
        MAX_LINIES = 10000

        with open(self.ruta_items, "r", encoding="utf8") as arxiu:
            next(arxiu)
            for i, linia in enumerate(arxiu):
                if i >= MAX_LINIES:
                    break
                parts = linia.strip().split(",")
                try:
                    isbn = parts[0]
                    titol = parts[1]
                    autor = parts[2]
                    any_publi = int(parts[3])
                    editorial = parts[4]
                    self.items.append(Llibre(isbn, titol, autor, any_publi, editorial))
                    self.items_ids.append(isbn)
                except (IndexError, ValueError):
                    print(f"Línia malformada: {parts}")

        self.items_ids.sort()
        print("Llibres carregats correctament.")

    def carrega_valoracions(self):
        """
        Aquesta funció llegeix un fitxer CSV i crea una llista de tuples amb les valoracions.
        Args:
            self.ruta_valoracions (str): Ruta al fitxer CSV que conté les valoracions.
        Returns:
            None
        Raises:
            FileNotFoundError: Si el fitxer no es troba.
            ValueError: Si hi ha un error en el format de les dades.
        Example:
            >>> dataset = DatasetLlibres("books.csv", "ratings.csv")
            >>> dataset.carrega_valoracions()
            Valoracions carregades correctament.
        """
        with open(self.ruta_valoracions, "r", encoding="utf8") as arxiu:
            next(arxiu)
            for linia in arxiu:
                parts = linia.strip().split(",")
                try:
                    user = parts[0]
                    item = parts[1]
                    if item not in self.items_ids:
                        print("Valoració no afegida: ítem no trobat")
                        continue
                    raw_valoracio = float(parts[2])
                    valoracio = 0 if raw_valoracio == 0 else raw_valoracio / 2
                    self.valoracions_raw.append((user, item, valoracio))
                except (IndexError, ValueError):
                    print(f"Valoració malformada: {parts}")
