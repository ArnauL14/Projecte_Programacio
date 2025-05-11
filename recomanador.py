from dataset import DatasetLlibres, DatasetPelicules
from abc import ABC, abstractmethod

class Recomanador(ABC):
    def __init__(self, dataset):
        self.dataset = dataset

    @abstractmethod
    def recomana(self, usuari_id, n):
        pass

class RecomanadorSimple(Recomanador):
    def recomana(self, usuari_id, n):
        llista_scores = []
        matriu = self.dataset.get_valoracions().astype(object)

        # Trobar la fila de l'usuari
        idx_usuari = None
        for i in range(1, matriu.shape[0]):
            if matriu[i][0] == usuari_id:
                idx_usuari = i
                break
        if idx_usuari is None:
            return []

         # Calcular mitjana global (Avg_global)
        valors = [
            matriu[i][j]
            for i in range(1, matriu.shape[0])
            for j in range(1, matriu.shape[1])
            if matriu[i][j] != 0
        ]
        avg_global = sum(valors) / len(valors) if valors else 0

        #Recorre totes les columnes
        for j in range(1, matriu.shape[1]):
            item_id = matriu[0][j]
            valoracio_usuari = matriu[idx_usuari][j]

            if valoracio_usuari != 0:
                continue  # Ja valorat per aquest usuari

            # Recollir les valoracions d'aquest ítem
            valors_item = [
                matriu[i][j]
                for i in range(1, matriu.shape[0])
                if matriu[i][j] != 0
            ]
            v = len(valors_item)  # número de vots

            #On posem minim_vots? Quants vots minims posem?
            minim_vots = 10
            if v >= minim_vots:
                avg_item = sum(valors_item) / v
                m = minim_vots
                score = (v / (v + m)) * avg_item + (m / (v + m)) * avg_global
                llista_scores.append((item_id, score))

        llista_scores.sort(key=lambda x: x[1], reverse=True)
        return llista_scores[:n]
    

class RecomanadorCollaboratiu(Recomanador):
    def recomana(self, usuari_id, n=5):
        # Algorisme de similitud + predicció
        pass


print("Pelicules")
#dataset_pelicula = DatasetPelicules("dataset/MovieLens100k/prova_pelicules.csv", "dataset/MovieLens100k/prova_valoracions.csv")
dataset_pelicula = DatasetPelicules("dataset/MovieLens100k/movies.csv", "dataset/MovieLens100k/ratings.csv")
print("Objecte Dataset Creat")
dataset_pelicula.carrega_items()
print("Importat items Creat")
dataset_pelicula.carrega_valoracions()
print("Valoracions Creades")
dataset_pelicula.carrega_matriu_valoracions()
print("Matriu Creada")
#dataset_pelicula.mostra_matriu()
print("")

recomanador = RecomanadorSimple(dataset_pelicula)
print("Recomanador Creat")

#Quants items mostrem?
nombe_items_mostrats = 5
id_usuari = "2"
scores = recomanador.recomana(id_usuari, nombe_items_mostrats)
print(scores)
