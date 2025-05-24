from dataset import DatasetLlibres, DatasetPelicules, Dataset
from abc import ABC, abstractmethod
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.metrics.pairwise import cosine_similarity


class Recomanador(ABC):
    """
    Classe abstracta per a sistemes de recomanació.
    Aquesta classe defineix la interfície per a recomanadors que poden ser
    basats en contingut, col·laboratius o simples.
    Attributes:
        dataset (Dataset): Instància del conjunt de dades utilitzat per a les recomanacions.
        matriu (np.ndarray): Matriu de valoracions dels usuaris.
        Métodes:
            recomana(usuari_id, n): Recomana ítems per a un usuari donat.
        Funcions auxiliars:
            trobar_index_usuari(usuari_id): Troba l'índex d'un usuari a la matriu de valoracions.
            calcula_mitjana_global(): Calcula la mitjana global de les valoracions.
            get_valoracions_numeriques(): Retorna les valoracions com a matriu numèrica.
        """
    def __init__(self, tipus_dataset, ruta_items, ruta_valoracions):
        if tipus_dataset == "pelicules":
            self.dataset = DatasetPelicules(ruta_items, ruta_valoracions)
        elif tipus_dataset == "llibres":
            self.dataset = DatasetLlibres(ruta_items, ruta_valoracions)
        else:
            raise ValueError("Tipus de dataset no reconegut")
        self.matriu = self.dataset.get_valoracions().astype(object)

    @abstractmethod
    def recomana(self, usuari_id, n):
        """
        Mètode abstracte per recomanar ítems a un usuari donat.
        Args:
            usuari_id (str): Identificador de l'usuari per al qual es volen recomanar ítems.
            n (int): Nombre mínim de vots requerits per considerar un ítem.
        Returns:
            list: Llista de tuples amb (item_id, score) ordenats per score descendent.
        Exemple:
            recomana("123", 5)
        
        """
        pass

    def trobar_index_usuari(self, usuari_id):
        idx_usuari = None
        Trobat = False
        i = 1
        while i < self.matriu.shape[0] and not Trobat:
            if self.matriu[i][0] == usuari_id:
                idx_usuari = i
                Trobat = True
            else:
                i += 1

        return idx_usuari
    
    def calcula_mitjana_global(self):
        """
        Calcula la mitjana global de les valoracions de la matriu.
        Retorna:
            float: La mitjana global de les valoracions, o 0 si no hi ha valoracions.
        Exemple:
            3.5
        """
        valors = [
            float(self.matriu[i][j])
            for i in range(1, self.matriu.shape[0])
            for j in range(1, self.matriu.shape[1])
            if self.matriu[i][j] != 0
        ]
        return sum(valors) / len(valors) if valors else 0

    def get_valoracions_numeriques(self):
        """
        Retorna la matriu de valoracions com a matriu numèrica (float).
        Aquesta matriu no conté la capçalera (IDs d'usuaris i ítems).
        Retorna:
            np.ndarray: Matriu de valoracions numèriques sense capçalera.
        Exemple:
            array([[0.0, 3.5, 0.0],
                   [4.0, 0.0, 2.5],
                   [0.0, 1.5, 0.0]])
        """
        return self.matriu[1:, 1:].astype(float)


class RecomanadorSimple(Recomanador):
    """
    Sistema de recomanació simple basat en la mitjana de les valoracions.
    Hereta de la classe abstracta Recomanador i recomana ítems no valorats per l'usuari
    segons la mitjana global de les valoracions i el nombre mínim de vots requerits.
    Attributes:
        dataset (Dataset): Instància del conjunt de dades utilitzat per a les recomanacions.
        matriu (np.ndarray): Matriu de valoracions dels usuaris.
        
    Métodes:
        recomana(usuari_id, n): Recomana ítems per a un usuari donat. 
    """
    def recomana(self, usuari_id, n):  # n: mínim de vots requerits
        """
        Recomana ítems no valorats per l'usuari donat, basant-se en la mitjana global
        de les valoracions i el nombre mínim de vots requerits.
        Args:
            usuari_id (str): Identificador de l'usuari per al qual es volen recomanar ítems.
            n (int): Nombre mínim de vots requerits per considerar un ítem.
        Returns:
            list: Llista de tuples amb (item_id, score) ordenats per score descendent.
        Exemple:
            recomana("123", 5)
        """
        llista_scores = []
        matriu = self.matriu  # ja carregada i amb capçalera

        idx_usuari = self.trobar_index_usuari(usuari_id)
        if idx_usuari is None:
            return []

        avg_global = self.calcula_mitjana_global()

        for j in range(1, matriu.shape[1]):
            item_id = matriu[0][j]
            valoracio_usuari = matriu[idx_usuari][j]

            if valoracio_usuari != 0:
                continue  # Ja l'ha valorat

            # Recollir les valoracions no zero d'aquest ítem
            valors_item = [
                float(matriu[i][j])
                for i in range(1, matriu.shape[0])
                if matriu[i][j] != 0
            ]
            v = len(valors_item)

            if v >= n:
                avg_item = sum(valors_item) / v
                score = (v / (v + n)) * avg_item + (n / (v + n)) * avg_global
                llista_scores.append((item_id, score))

        llista_scores.sort(key=lambda x: x[1], reverse=True)
        return llista_scores[:5]
    

class RecomanadorCollaboratiu(Recomanador):
    """
    Sistema de recomanació col·laboratiu basat en la similitud entre usuaris.
    Hereta de la classe abstracta Recomanador i recomana ítems no valorats per l'usuari
    segons la similitud amb altres usuaris.
    Attributes:
        dataset (Dataset): Instància del conjunt de dades utilitzat per a les recomanacions.
        matriu (np.ndarray): Matriu de valoracions dels usuaris.
    Métodes:
        recomana(usuari_id, k): Recomana ítems per a un usuari donat basant-se en els k usuaris més similars.
    Args:
        usuari_id (str): Identificador de l'usuari per al qual es volen recomanar ítems.
        k (int): Nombre d'usuaris similars a considerar per a les recomanacions.
    Returns:
        list: Llista de tuples amb (item_id, score) ordenats per score descendent.
    Example:
        recomana("123", 5)
    """
    def recomana(self, usuari_id, k): # k: és els k usuaris mes similars que volem tenir en compte
        idx_usuari = self.trobar_index_usuari(usuari_id)
        if idx_usuari is None:
            return []
        
        matriu_num = self.get_valoracions_numeriques()  # np.array de valors float, no conté capçalera
        idx_usuari -= 1  # perquè get_valoracions_numeriques() no té capçalera

        vector_usuari = matriu_num[idx_usuari]
        similituds = []

        # PAS 2: Calcular similituds amb tots els altres usuaris
        for i in range(matriu_num.shape[0]):
            if i == idx_usuari:
                continue

            # Identificar les posicions on ambdós usuaris han valorat algun ítem
            # vector_usuari i vector_altre són dues files de la matriu (dos usuaris)
            vector_altre = matriu_num[i]
            mask = (vector_usuari != 0) & (vector_altre != 0)

            #Si no tenen cap valoració en comú, similitud = 0
            if not np.any(mask):
                sim = 0
            else:
                #Seleccionar només les valoracions comunes
                u = vector_usuari[mask]
                v = vector_altre[mask]

                #Calcular el producte escalar i Calcular els mòduls (normes) dels vectors
                numerador = np.dot(u, v)
                norm_u = np.linalg.norm(u)
                norm_v = np.linalg.norm(v)

                if norm_u == 0 or norm_v == 0: # Evitem divisió per 0
                    sim = 0
                else:
                    sim = numerador / (norm_u * norm_v)
                similituds.append((1, sim))

        similituds.sort(key=lambda x: x[1], reverse=True)
        top_k = similituds[:k]
        prediccions = []
        for j in range(matriu_num.shape[1]):
            if vector_usuari[j] != 0:
                continue

            numerador = 0
            denominador = 0
            for i, sim in top_k:
                val = matriu_num[i][j]
                if val != 0:
                    numerador += sim * val
                    denominador += abs(sim)

            if denominador > 0:
                prediccio = numerador / denominador
                item_id = self.matriu[0][j + 1]
                prediccions.append((item_id, float(prediccio)))

        prediccions.sort(key=lambda x: x[1], reverse=True)
        return prediccions[:5]
    

class RecomanadorContingut(Recomanador):
    """
    Sistema de recomanació basat en contingut.
    Hereta d'una classe abstracta Recomanador i utilitza TF-IDF per generar recomanacions
    segons el contingut (gèneres, descripcions, etc.) dels ítems.

    Attributes:
        dataset (Dataset): Instància del conjunt de dades (llibres o pel·lícules).
        matriu (np.ndarray): Matriu de valoracions dels usuaris.
        tfidf_matrix (np.ndarray): Matriu TF-IDF dels ítems.
        vectorizer (TfidfVectorizer): Vectoritzador TF-IDF de sklearn.
    
    """
def __init__(self, dataset):
    self.dataset = dataset
    self.matriu = self.dataset.get_valoracions().astype(object)
    self.vectorizer = TfidfVectorizer(stop_words="english")
    self.tfidf_matrix = None

def fit(self):
    """
    Genera la matriu TF-IDF dels ítems utilitzant les seves característiques textuals.
    """
    item_texts = self.dataset.get_descriptors()  # una llista de cadenes
    self.tfidf_matrix = self.vectorizer.fit_transform(item_texts).toarray()

def cosine_similarity(vector, matriu):
        """
        Calcula la similitud cosinus entre un vector i cada fila d'una matriu.
        Args:
            vector (np.ndarray): Vector de referència per comparar.
            matriu (np.ndarray): Matriu on cada fila és un vector a comparar amb el vector.
        Returns:
            np.ndarray: Un array de similituds cosinus entre el vector i cada fila de la matriu.
        Example:
            cosine_similarity(np.array([1, 2, 3]), np.array([[1, 0, 0], [0, 2, 3], [1, 1, 1]]))

        """
        norm_vector = np.linalg.norm(vector)
        norm_matriu = np.linalg.norm(matriu, axis=1)

        # Evitem divisió per 0
        with np.errstate(divide='ignore', invalid='ignore'):
            producte_escalar = matriu @ vector
            similituds = producte_escalar / (norm_matriu * norm_vector)
            similituds = np.nan_to_num(similituds)  # converteix NaN i inf a 0

        return similituds

def build_user_profile(self, user_id):
    """
    Construeix el perfil d'un usuari mitjançant una mitjana ponderada dels vectors TF-IDF
    dels ítems que ha valorat.

    Args:
        user_id (str): Identificador de l'usuari.

    Returns:
        np.ndarray: Vector del perfil de l'usuari.
        Si l'usuari no existeix o no té valoracions, retorna None.
    Example:   
        build_user_profile("123")
    """
    idx = self.trobar_index_usuari(user_id)
    if idx is None:
        return None

    valoracions = self.matriu[idx, 1:].astype(float)
    profile = np.zeros(self.tfidf_matrix.shape[1])
    for i, score in enumerate(valoracions):
        if score > 0:
            profile += score * self.tfidf_matrix[i]
    divisor = np.count_nonzero(valoracions)
    return profile / divisor if divisor > 0 else profile

def recomana(self, user_id, n=5):
    """
    Recomana ítems no puntuats per l'usuari segons la similitud entre el seu perfil i els ítems.

    Args:
        user_id (str): Identificador de l'usuari.
        n (int): Nombre d'ítems a recomanar.

    Returns:
        list[tuple]: Llista de tuples amb (item_id, score) ordenats per score descendent.
    """
    profile = self.build_user_profile(user_id)
    if profile is None:
        return []

    sim_scores = cosine_similarity([profile], self.tfidf_matrix)[0]
    idx_usuari = self.trobar_index_usuari(user_id)
    valoracions = self.matriu[idx_usuari, 1:].astype(float)
    recomanacions = [
        (self.matriu[0][i + 1], score) for i, score in enumerate(sim_scores) if valoracions[i] == 0
    ]
    recomanacions.sort(key=lambda x: x[1], reverse=True)
    return recomanacions[:n]

def trobar_index_usuari(self, usuari_id):
    """
    Troba l'índex del vector corresponent a l'usuari dins la matriu.

    Args:
        usuari_id (str): Identificador de l'usuari.

    Returns:
        int | None: Índex de l'usuari o None si no es troba.
    """
    for i in range(1, self.matriu.shape[0]):
        if self.matriu[i][0] == usuari_id:
            return i
    return None



"""print("Simple")
reco = RecomanadorSimple("pelicules","dataset/MovieLens100k/movies.csv", "dataset/MovieLens100k/ratings.csv")
scores = reco.recomana("10", 10)
print(scores)
print("")"""

print("Colaboratiu")
reco2 = RecomanadorCollaboratiu("pelicules","dataset/MovieLens100k/movies.csv", "dataset/MovieLens100k/ratings.csv")
scores2 = reco2.recomana("1", 5)
print(scores2)
print("")

"""
#dataset_pelicula = DatasetPelicules("dataset/MovieLens100k/prova_pelicules.csv", "dataset/MovieLens100k/prova_valoracions.csv")
dataset_pelicula = DatasetPelicules("dataset/MovieLens100k/movies.csv", "dataset/MovieLens100k/ratings.csv")
"""
