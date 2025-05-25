from dataset import DatasetLlibres, DatasetPelicules, Dataset
from abc import ABC, abstractmethod
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import logging


class Recomanador(ABC):
    """
    Classe abstracta base per a sistemes de recomanació.
    Defineix la interfície comuna per a tots els tipus de recomanadors i
    proporciona funcionalitats bàsiques com la gestió del dataset i càlculs de valoracions.
    
    Attributes:
        _dataset (Dataset): Instància del dataset utilitzat pel recomanador.
        _matriu (np.ndarray): Matriu de valoracions amb capçaleres.
    """
    
    def __init__(self, tipus_dataset, ruta_items, ruta_valoracions):
        """
        Inicialitza el recomanador amb el tipus de dataset i les rutes als fitxers.
        
        Args:
            tipus_dataset (str): Tipus de dataset ('pelicules' o 'llibres').
            ruta_items (str): Ruta al fitxer d'ítems.
            ruta_valoracions (str): Ruta al fitxer de valoracions.
            
        Raises:
            ValueError: Si el tipus de dataset no és reconegut.
        """
        if tipus_dataset == "pelicules":
            self._dataset = DatasetPelicules(ruta_items, ruta_valoracions)
        elif tipus_dataset == "llibres":
            self._dataset = DatasetLlibres(ruta_items, ruta_valoracions)
        else:
            raise ValueError("Tipus de dataset no reconegut")
        self._matriu = self._dataset.get_valoracions().astype(object)

    def get_dataset(self): 
        """
        Retorna el dataset associat al recomanador.
        
        Returns:
            Dataset: El dataset utilitzat pel recomanador.
        """
        return self._dataset

    @abstractmethod
    def recomana(self, usuari_id, n):
        """
        Mètode abstracte per recomanar ítems a un usuari.
        
        Args:
            usuari_id (str): Identificador de l'usuari.
            n/k (int): Paràmetre específic de cada algorisme:
                      - Per a recomanació simple: mínim de vots requerits
                      - Per a recomanació col·laborativa: nombre d'usuaris similars
        Returns:
            None
            
        Notes:
            Aquest mètode ha de ser implementat per les subclasses.
        """
        pass

    def trobar_index_usuari(self, usuari_id):
        """
        Troba l'índex del vector corresponent a l'usuari dins la matriu.
        
        Args:
            usuari_id (str): Identificador de l'usuari.
            
        Returns:
            int | None: Índex de l'usuari a la matriu o None si no es troba.
        """
        idx_usuari = None
        Trobat = False
        i = 1
        while i < self._matriu.shape[0] and not Trobat:
            if self._matriu[i][0] == usuari_id:
                idx_usuari = i
                Trobat = True
            else:
                i += 1

        return idx_usuari
    
    def calcula_mitjana_global(self):
        """
        Calcula la mitjana global de les valoracions de la matriu.
        
        Returns:
            float: Mitjana global de les valoracions (ignorant els zeros).
        """
        valors = [
            float(self._matriu[i][j])
            for i in range(1, self._matriu.shape[0])
            for j in range(1, self._matriu.shape[1])
            if self._matriu[i][j] != 0
        ]
        logging.info("Mitjana global calculada.")
        return sum(valors) / len(valors) if valors else 0

    def get_valoracions_numeriques(self):
        """
        Retorna la matriu de valoracions numèriques sense capçalera.
        
        Returns:
            np.ndarray: Matriu de valoracions numèriques (sense capçaleres).
        """
        return self._matriu[1:, 1:].astype(float)
    
    def mostrar_recomancions(self, recomanacions):
        """
        Mostra les recomanacions d'una manera llegible.
        
        Args:
            recomanacions (list): Llista de tuples (item_id, score).
        """
        print("")
        for item_id, score in recomanacions:
            item = self._dataset.get_item(item_id)
            print(f"{item} --> Score: {score:.2f}")
        print("")


class RecomanadorSimple(Recomanador):
    """
    Sistema de recomanació simple basat en popularitat i valoracions mitjanes.
    Utilitza una fórmula que combina la mitjana de l'ítem amb la mitjana global,
    ponderant per la quantitat de valoracions rebudes.
    """
    
    def recomana(self, usuari_id, n):  # n: mínim de vots requerits
        """
        Genera recomanacions basades en valoracions mitjanes i nombre de vots.
        
        Args:
            usuari_id (str): ID de l'usuari per al qual es generen recomanacions.
            n (int): Mínim de vots requerits per considerar un ítem.
            
        Returns:
            list: Llista de tuples (item_id, score) ordenada per score descendent.
        """
        logging.info("Generant recomanacions simples...")
        llista_scores = []
        matriu = self._matriu

        idx_usuari = self.trobar_index_usuari(usuari_id)
        if idx_usuari is None:
            return []

        avg_global = self.calcula_mitjana_global()

        # Per a cada ítem no valorat per l'usuari
        for j in range(1, matriu.shape[1]):
            item_id = matriu[0][j]
            valoracio_usuari = matriu[idx_usuari][j]

            if valoracio_usuari != 0:
                continue  # L'usuari ja ha valorat aquest ítem

            # Recollir les valoracions no zero d'aquest ítem
            valors_item = [
                float(matriu[i][j])
                for i in range(1, matriu.shape[0])
                if matriu[i][j] != 0
            ]
            v = len(valors_item)

            # Aplicar fórmula de ponderació si té suficients valoracions
            if v >= n:
                try:
                    avg_item = sum(valors_item) / v
                    score = (v / (v + n)) * avg_item + (n / (v + n)) * avg_global
                    llista_scores.append((item_id, score))
                except ZeroDivisionError:
                    logging.error(f"Divisió per zero en calcular la puntuació per l'ítem {item_id}.")
                    return []

        # Ordenar per score descendent
        llista_scores.sort(key=lambda x: x[1], reverse=True)
        logging.info(f"Recomanacions generades per a l'usuari {usuari_id}.")
        self.mostrar_recomancions(llista_scores[:5])
        return llista_scores
    

class RecomanadorCollaboratiu(Recomanador):
    """
    Sistema de recomanació col·laboratiu basat en filtrat col·laboratiu.
    Utilitza similitud cosinus entre usuaris per predir valoracions.
    """
    
    def recomana(self, usuari_id, k):
        """
        Genera recomanacions basades en usuaris similars.
        
        Args:
            usuari_id (str): ID de l'usuari per al qual es generen recomanacions.
            k (int): Nombre d'usuaris més similars a considerar.
            
        Returns:
            list: Llista de tuples (item_id, score) ordenada per score descendent.
        """
        logging.info("Generant recomanacions col·laboratives...")
        idx_usuari = self.trobar_index_usuari(usuari_id)
        if idx_usuari is None:
            return []
        
        matriu_num = self.get_valoracions_numeriques()
        idx_usuari -= 1  # Ajustar índex per matriu sense capçalera

        vector_usuari = matriu_num[idx_usuari]
        similituds = []

        # Calcular similituds amb tots els altres usuaris
        for i in range(matriu_num.shape[0]):
            if i == idx_usuari:
                continue

            vector_altre = matriu_num[i]
            mask = (vector_usuari != 0) & (vector_altre != 0)

            # Si no tenen valoracions en comú, similitud = 0
            if not np.any(mask):
                sim = 0
            else:
                # Calcular similitud cosinus només amb valoracions comunes
                u = vector_usuari[mask]
                v = vector_altre[mask]
                numerador = np.dot(u, v)
                norm_u = np.linalg.norm(u)
                norm_v = np.linalg.norm(v)

                sim = numerador / (norm_u * norm_v) if (norm_u * norm_v) != 0 else 0
            similituds.append((i, sim))

        # Seleccionar els k usuaris més similars
        similituds.sort(key=lambda x: x[1], reverse=True)
        top_k = similituds[:k]
        
        # Generar prediccions per a ítems no valorats
        prediccions = []
        for j in range(matriu_num.shape[1]):
            if vector_usuari[j] != 0:
                continue  # L'usuari ja ha valorat aquest ítem

            # Calcular predicció ponderada per similitud
            numerador = 0
            denominador = 0
            for i, sim in top_k:
                val = matriu_num[i][j]
                if val != 0:
                    numerador += sim * val
                    denominador += abs(sim)

            if denominador > 0:
                prediccio = numerador / denominador
                item_id = self._matriu[0][j + 1]
                prediccions.append((item_id, float(prediccio)))

        prediccions.sort(key=lambda x: x[1], reverse=True)
        logging.info(f"Recomanacions col·laboratives generades per a l'usuari {usuari_id}.")
        self.mostrar_recomancions(prediccions[:5])
        return prediccions
    

class RecomanadorContingut(Recomanador):
    """
    Sistema de recomanació basat en contingut.
    Utilitza TF-IDF per analitzar el contingut dels ítems i recomanar
    ítems similars als que l'usuari ha valorat positivament.
    
    Attributes:
        _vectorizer (TfidfVectorizer): Vectoritzador per a transformar text a vectors TF-IDF.
        _tfidf_matrix (np.ndarray): Matriu TF-IDF de tots els ítems.
    """
    
    def __init__(self, tipus_dataset, ruta_items, ruta_valoracions):
        """
        Inicialitza el recomanador i el vectoritzador TF-IDF.
        """
        super().__init__(tipus_dataset, ruta_items, ruta_valoracions)
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._tfidf_matrix = None
        self.fit()

    def cosine_similarity(self, matriu, vector):
        """
        Calcula la similitud cosinus entre un vector i cada fila d'una matriu.
        
        Args:
            matriu (np.ndarray): Matriu de vectors (shape n x d).
            vector (np.ndarray): Vector de consulta (shape d,).
            
        Returns:
            np.ndarray: Vector de similituds (shape n,).
        """
        norm_vector = np.linalg.norm(vector)
        norm_matriu = np.linalg.norm(matriu, axis=1)

        with np.errstate(divide='ignore', invalid='ignore'):
            producte_escalar = matriu @ vector
            similituds = producte_escalar / (norm_matriu * norm_vector)
            similituds = np.nan_to_num(similituds)

        logging.info("Similituds cosinus calculades.")
        return similituds

    def fit(self):
        """
        Entrena el vectoritzador TF-IDF amb les descripcions dels ítems.
        """
        item_texts = self._dataset.get_descriptors()
        self._tfidf_matrix = self._vectorizer.fit_transform(item_texts).toarray()

    def build_user_profile(self, user_id):
        """
        Construeix un perfil d'usuari com a combinació dels ítems que ha valorat.
        
        Args:
            user_id (str): ID de l'usuari.
            
        Returns:
            np.ndarray | None: Vector de perfil d'usuari o None si no es troba.
        """
        idx = self.trobar_index_usuari(user_id)
        if idx is None:
            return None

        valoracions = self._matriu[idx, 1:].astype(float)
        profile = np.zeros(self._tfidf_matrix.shape[1])
        for i, score in enumerate(valoracions):
            if score > 0:
                profile += score * self._tfidf_matrix[i]
        divisor = np.count_nonzero(valoracions)
        logging.info(f"Perfil d'usuari construït per a l'usuari {user_id}.")
        return profile / divisor if divisor > 0 else profile

    def recomana(self, user_id):
        """
        Genera recomanacions basades en similitud de contingut.
        
        Args:
            user_id (str): ID de l'usuari per al qual es generen recomanacions.
            
        Returns:
            list: Llista de tuples (item_id, score) ordenada per score descendent.
        """
        logging.info("Generant recomanacions basades en contingut...")
        profile = self.build_user_profile(user_id)
        if profile is None:
            return []

        # Calcular similituds amb tots els ítems
        sim_scores = self.cosine_similarity(self._tfidf_matrix, profile)

        # Filtrar ítems ja valorats
        idx_usuari = self.trobar_index_usuari(user_id)
        valoracions = self._matriu[idx_usuari, 1:].astype(float)
        recomanacions = [
            (self._matriu[0][i + 1], score)
            for i, score in enumerate(sim_scores)
            if valoracions[i] == 0
        ]
        recomanacions.sort(key=lambda x: x[1], reverse=True)

        logging.info(f"Recomanacions basades en contingut generades per a l'usuari {user_id}.")
        self.mostrar_recomancions(recomanacions[:5])
        return recomanacions


"""print("Simple")
reco = RecomanadorSimple("pelicules","dataset/MovieLens100k/movies.csv", "dataset/MovieLens100k/ratings.csv")
scores = reco.recomana("10", 0)
print("")

print("Colaboratiu")
reco2 = RecomanadorCollaboratiu("pelicules","dataset/MovieLens100k/movies.csv", "dataset/MovieLens100k/ratings.csv")
scores2 = reco2.recomana("1", 5)
print("")

print("Contingut")
reco3 = RecomanadorContingut("pelicules","dataset/MovieLens100k/movies.csv", "dataset/MovieLens100k/ratings.csv")
reco3.recomana("1") #@Iker, els scores de aquesta funció sembla que van de 0 a 1, no sé si és correcte o no
print("")"""

"""print("Simple Llibres")
reco = RecomanadorSimple("llibres", "dataset/Books/prova_llibres.csv", "dataset/Books/prova_valoracions.csv")
scores = reco.recomana("1", 0)
print(scores)"""

"""
dataset_pelicula = DatasetPelicules("dataset/MovieLens100k/prova_pelicules.csv", "dataset/MovieLens100k/prova_valoracions.csv")
dataset_pelicula = DatasetPelicules("dataset/MovieLens100k/movies.csv", "dataset/MovieLens100k/ratings.csv")
"""
