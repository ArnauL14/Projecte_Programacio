from item import Pelicula, Llibre
from abc import ABC, abstractmethod
import numpy as np
import csv
import logging

class Dataset(ABC):
    """
    Classe abstracta base per a la gestió de datasets de recomanació.
    Aquesta classe defineix la interfície per carregar ítems i valoracions,
    i proporciona funcionalitats per crear una matriu de valoracions amb capçaleres.
    
    Attributes:
        _ruta_items (str): Ruta al fitxer que conté els ítems.
        _ruta_valoracions (str): Ruta al fitxer que conté les valoracions.
        _items (list): Llista d'objectes (pel·lícules o llibres).
        _items_ids (list): Llista d'IDs dels ítems.
        _valoracions_raw (list): Llista de tuples (usuari, item, valoració).
        _valoracions (np.ndarray): Matriu de valoracions amb capçaleres (usuaris x ítems).
    """
    
    def __init__(self, ruta_items, ruta_valoracions):
        """
        Inicialitza el dataset amb les rutes als fitxers d'ítems i valoracions.
        
        Args:
            ruta_items (str): Ruta al fitxer d'ítems.
            ruta_valoracions (str): Ruta al fitxer de valoracions.
        """
        self._ruta_items = ruta_items
        self._ruta_valoracions = ruta_valoracions
        self._items = []       # Llista d'objectes (pel·lícules o llibres)
        self._items_ids = []   # IDs dels ítems
        self._valoracions_raw = []  # Llista de tuples (usuari, item, valoració)
        self._valoracions = np.empty((1, 1), dtype=object)  # Inicialització buida

        logging.info("Inicialitzant dataset...")
        self.carrega_items()
        self.carrega_valoracions()
        self.carrega_matriu_valoracions()
        logging.info("Dataset inicialitzat correctament")

    def get_valoracions_numeriques(self):
        """
        Retorna la matriu de valoracions numèriques sense capçalera.
        
        Returns:
            np.ndarray: Submatriu amb les valoracions numèriques (sense capçaleres).
        """
        logging.info("Obtenint valoracions numèriques (sense capçaleres)")
        return self._valoracions[1:, 1:].astype(float)

    @abstractmethod
    def carrega_items(self):
        """
        Carrega els ítems del dataset. 
        Aquesta funció ha de ser implementada per les subclasses.
        """
        pass

    @abstractmethod
    def carrega_valoracions(self):
        """
        Carrega les valoracions del dataset.
        Aquesta funció ha de ser implementada per les subclasses.
        """
        pass

    def carrega_matriu_valoracions(self):
        """
        Crea una matriu de valoracions amb capçaleres on:
        - Les files són usuaris
        - Les columnes són ítems
        - La primera fila conté els IDs dels ítems
        - La primera columna conté els IDs dels usuaris
        - Les cel·les contenen la valoració o 0 si no n'hi ha
        
        La matriu té la forma:
        | Usuari/Item | Item1 | Item2 | ... |
        | Usuari1     | Val1  | Val2  | ... |
        | Usuari2     | Val3  | Val4  | ... |
        """
        logging.info("Iniciant càrrega de la matriu de valoracions...")
        
        llista_usuaris = list({v[0] for v in self._valoracions_raw})
        n_rows = len(llista_usuaris) + 1
        n_cols = len(self._items_ids) + 1
        
        logging.info(f"Creant matriu de {n_rows} files (usuaris) i {n_cols} columnes (ítems)")
        matriu = np.zeros((n_rows, n_cols), dtype=object)
        matriu[0][0] = "id"

        # Omplim capçalera d'ítems
        for j, item_id in enumerate(self._items_ids, start=1):
            matriu[0][j] = item_id
        
        # Omplim columna d'usuaris
        for i, usuari_id in enumerate(llista_usuaris, start=1):
            matriu[i][0] = usuari_id
        
        # Omplim valoracions
        for usuari_id, item_id, valoracio in self._valoracions_raw:
            i = llista_usuaris.index(usuari_id) + 1
            j = self._items_ids.index(item_id) + 1
            matriu[i][j] = valoracio

        self._valoracions = matriu
        logging.info("Matriu de valoracions creada correctament")

    def mostra_matriu(self):
        """
        Mostra la matriu de valoracions de manera llegible. 
        Utilitzada per fer proves i debugging.
        """
        logging.info("Mostrant matriu de valoracions...")
        for fila in self._valoracions:
            print("\t".join(str(x) for x in fila))

    def get_valoracions(self):
        """
        Retorna la matriu de valoracions completa amb capçaleres.
        
        Returns:
            np.ndarray: Matriu de valoracions amb capçaleres.
        """
        logging.info("Obtenint matriu de valoracions completa")
        return self._valoracions
    
    def get_item(self, id_item):
        """
        Retorna l'objecte item corresponent a l'ID donat.
        
        Args:
            id_item (int | str): ID de l'item a cercar.
        
        Returns:
            Pelicula | Llibre | None: L'objecte item si es troba, None si no existeix.
        """
        if id_item in self._items_ids:
            index = self._items_ids.index(id_item)
            return self._items[index]
        logging.warning(f"Ítem amb ID {id_item} no trobat")
        return None


class DatasetPelicules(Dataset):
    """
    Classe per a gestionar un dataset de pel·lícules.
    Implementa els mètodes per carregar ítems i valoracions específics per a pel·lícules.
    """
    
    def carrega_items(self):
        """
        Carrega les pel·lícules del fitxer CSV i les emmagatzema a self._items.
        El format esperat és: movieId,títol,gèneres (separats per |)
        """
        logging.info(f"Carregant pel·lícules des de {self._ruta_items}")
        self._items = []
        self._items_ids = []
        
        with open(self._ruta_items, "r", encoding="utf8") as arxiu:
            lector = csv.reader(arxiu, delimiter=",", quotechar='"')
            next(lector)  # salta la capçalera

            for parts in lector:
                try:
                    movie_id = int(parts[0])
                    titol = parts[1]
                    generes = parts[2].split('|') if parts[2] else []

                    self._items.append(Pelicula(movie_id, titol, generes))
                    self._items_ids.append(movie_id)
                except IndexError:
                    logging.error(f"Línia malformada: {parts}")

        logging.info(f"Carregades {len(self._items)} pel·lícules")

    def carrega_valoracions(self):
        """
        Carrega les valoracions de pel·lícules del fitxer CSV.
        El format esperat és: usuari,movieId,valoració
        """
        logging.info(f"Carregant valoracions des de {self._ruta_valoracions}")
        with open(self._ruta_valoracions, "r", encoding="utf8") as arxiu:
            next(arxiu)  # Saltar la capçalera
            for linia in arxiu:
                parts = linia.strip().split(",")
                try:
                    user = parts[0]
                    item = int(parts[1])
                    valoracio = float(parts[2])
                    self._valoracions_raw.append((user, item, valoracio))
                except (IndexError, ValueError):
                    logging.error(f"Valoració malformada: {parts}")
        
        logging.info(f"Carregades {len(self._valoracions_raw)} valoracions")

    def get_descriptors(self):
        """
        Retorna una llista de descriptors dels ítems.
        Cada descriptor és una cadena de text que conté els gèneres d'una pel·lícula.
        
        Returns:
            list: Llista de descriptors (gèneres concatenats).
        """
        logging.info("Generant descriptors de pel·lícules")
        return [' '.join(p.genere) for p in self._items]


class DatasetLlibres(Dataset):
    """
    Classe per a gestionar un dataset de llibres.
    Implementa els mètodes per carregar ítems i valoracions específics per a llibres.
    """
    
    def carrega_items(self):
        """
        Carrega els llibres del fitxer CSV i els emmagatzema a self._items.
        El format esperat és: isbn,títol,autor,any,editorial
        """
        logging.info(f"Carregant llibres des de {self._ruta_items}")
        self._items = []
        self._items_ids = []
        MAX_LINIES = 10000  # límit per a no sobrecarregar la memòria

        with open(self._ruta_items, "r", encoding="utf8") as arxiu:
            next(arxiu)  # salta la capçalera

            for i, linia in enumerate(arxiu):
                if i >= MAX_LINIES:
                    logging.warning(f"S'ha arribat al límit de {MAX_LINIES} llibres")
                    break

                parts = linia.strip().split(",")

                try:
                    isbn = parts[0]
                    titol = parts[1]
                    autor = parts[2]
                    any_publi = parts[3]
                    editorial = parts[4]

                    self._items_ids.append(isbn)
                    self._items.append(Llibre(isbn, titol, autor, any_publi, editorial))
                except IndexError:
                    logging.error(f"Línia malformada: {parts}")

        logging.info(f"Carregats {len(self._items)} llibres")

    def carrega_valoracions(self):
        """
        Carrega les valoracions de llibres del fitxer CSV.
        El format esperat és: usuari,isbn,valoració
        Les valoracions s'escalen de 0-10 a 0-5.
        """
        logging.info(f"Carregant valoracions des de {self._ruta_valoracions}")
        with open(self._ruta_valoracions, "r") as arxiu:
            next(arxiu)
            for linia in arxiu:
                parts = linia.strip().split(",")
                try:
                    user = parts[0]
                    item = parts[1]
                    if item not in self._items_ids:
                        logging.warning(f"Valoració no afegida: ítem {item} no trobat")
                        continue
                    raw_valoracio = float(parts[2])
                    # Si és 0, considerem que no hi ha puntuació
                    if raw_valoracio == 0:
                        valoracio = 0
                    else:
                        valoracio = raw_valoracio / 2  # escalar de 0–10 a 0–5
                    self._valoracions_raw.append((user, item, valoracio))
                except (IndexError, ValueError):
                    logging.error(f"Valoració malformada: {parts}")
        
        logging.info(f"Carregades {len(self._valoracions_raw)} valoracions (escalades 0-5)")


"""dataset_pelicula = DatasetPelicules("dataset/MovieLens100k/prova_pelicules.csv", "dataset/MovieLens100k/prova_valoracions.csv")
print("Objecte Dataset Creat")
pelicula = dataset_pelicula.get_item(4)
print(pelicula)"""
