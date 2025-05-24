from dataset import DatasetLlibres, DatasetPelicules, Dataset
from abc import ABC, abstractmethod
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class Recomanador(ABC):
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
        Mètode abstracte per recomanar ítems a un usuari.
        Args:
            usuari_id (str): Identificador de l'usuari.
            n (int): Paràmetre per a la recomanació, pot ser el mínim de vots requerits o el nombre d'usuaris similars.
        Returns:
            None
        Aquesta funció ha de ser implementada per les subclasses.
        """
        pass

    def trobar_index_usuari(self, usuari_id):
        """
        Troba l'índex del vector corresponent a l'usuari dins la matriu.

        Args:
            usuari_id (str): Identificador de l'usuari.

        Returns:
            int | None: Índex de l'usuari o None si no es troba.
        """
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
        Returns:
            float: Mitjana global de les valoracions sense contar 0s.
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
        Retorna la matriu de valoracions numèriques sense capçalera.
        """
        return self.matriu[1:, 1:].astype(float)
    
    def mostrar_recomancions(self, recomanacions):
        """
        Mostra les recomanacions d'una manera llegible.
        
        Args:
            recomanacions (list): Llista de tuples (item_id, score).
        """
        for item_id, score in recomanacions:
            item = self.dataset.get_item(item_id)
            print(f"{item} --> Score: {score:.2f}")


class RecomanadorSimple(Recomanador):
    def recomana(self, usuari_id, n):  # n: mínim de vots requerits
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

        self.mostrar_recomancions(llista_scores[:5])
    

class RecomanadorCollaboratiu(Recomanador):
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
        
        self.mostrar_recomancions(prediccions[:5])
    

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
    def __init__(self, tipus_dataset, ruta_items, ruta_valoracions):
        super().__init__(tipus_dataset, ruta_items, ruta_valoracions)
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = None
        self.fit()

    def cosine_similarity(self, matriu, vector):
        """
        Calcula la similitud cosinus entre un vector i cada fila d'una matriu.
        :param matriu: np.array de shape (n, d)
        :param vector: np.array de shape (d,)
        :return: np.array de shape (n,) amb la similitud cosinus per fila
        """
        norm_vector = np.linalg.norm(vector)
        norm_matriu = np.linalg.norm(matriu, axis=1)

        with np.errstate(divide='ignore', invalid='ignore'):
            producte_escalar = matriu @ vector
            similituds = producte_escalar / (norm_matriu * norm_vector)
            similituds = np.nan_to_num(similituds)

        return similituds

    def fit(self):
        item_texts = self.dataset.get_descriptors()
        self.tfidf_matrix = self.vectorizer.fit_transform(item_texts).toarray()

    def build_user_profile(self, user_id):
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

    def recomana(self, user_id):
        profile = self.build_user_profile(user_id)
        if profile is None:
            return []

        sim_scores = self.cosine_similarity(self.tfidf_matrix, profile)

        idx_usuari = self.trobar_index_usuari(user_id)
        valoracions = self.matriu[idx_usuari, 1:].astype(float)
        recomanacions = [
            (self.matriu[0][i + 1], score)
            for i, score in enumerate(sim_scores)
            if valoracions[i] == 0
        ]
        recomanacions.sort(key=lambda x: x[1], reverse=True)

        self.mostrar_recomancions(recomanacions[:5])


"""print("Simple")
reco = RecomanadorSimple("pelicules","dataset/MovieLens100k/movies.csv", "dataset/MovieLens100k/ratings.csv")
scores = reco.recomana("10", 10)
print("")"""

"""print("Colaboratiu")
reco2 = RecomanadorCollaboratiu("pelicules","dataset/MovieLens100k/movies.csv", "dataset/MovieLens100k/ratings.csv")
scores2 = reco2.recomana("1", 5)
print("")"""

"""print("Contingut")
reco3 = RecomanadorContingut("pelicules","dataset/MovieLens100k/movies.csv", "dataset/MovieLens100k/ratings.csv")
reco3.recomana("1") #@Iker, els scores de aquesta funció sembla que van de 0 a 1, no sé si és correcte o no
print("") """

"""print("Simple Llibres")
reco = RecomanadorSimple("llibres", "dataset/Books/prova_llibres.csv", "dataset/Books/prova_valoracions.csv")
scores = reco.recomana("1", 0)
print(scores)"""

"""
dataset_pelicula = DatasetPelicules("dataset/MovieLens100k/prova_pelicules.csv", "dataset/MovieLens100k/prova_valoracions.csv")
dataset_pelicula = DatasetPelicules("dataset/MovieLens100k/movies.csv", "dataset/MovieLens100k/ratings.csv")
"""
