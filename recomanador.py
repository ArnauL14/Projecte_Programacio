from dataset import DatasetLlibres, DatasetPelicules, Dataset
from abc import ABC, abstractmethod

class Recomanador(ABC):
    def __init__(self, tipus_dataset, ruta_items, ruta_valoracions):
        if tipus_dataset == "pelicules":
            self.dataset = DatasetPelicules(ruta_items, ruta_valoracions)
        elif tipus_dataset == "llibres":
            self.dataset = DatasetLlibres(ruta_items, ruta_valoracions)
        else:
            raise ValueError("Tipus de dataset no reconegut")

    @abstractmethod
    def recomana(self, usuari_id, n):
        pass

class RecomanadorSimple(Recomanador):
    def recomana(self, usuari_id, n):
        llista_scores = []
        matriu = self.dataset.get_valoracions().astype(object)

        # Trobar la fila de l'usuari (iker i while Arnau ;))
        idx_usuari = None
        Trobat = False
        i = 1
        while i < matriu.shape[0] and not Trobat:
            if matriu[i][0] == usuari_id:
                idx_usuari = i
                Trobat = True
            else:
                i += 1

        if not Trobat:
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

            """
            els demano al main i li passo a la classe, així els resultats 
            estan sota el control de l'usuari. 
            """
            if v >= n:
                avg_item = sum(valors_item) / v
                score = (v / (v + n)) * avg_item + (n / (v + n)) * avg_global
                llista_scores.append((item_id, score))

        llista_scores.sort(key=lambda x: x[1], reverse=True)
        return llista_scores[:5]
    

class RecomanadorCollaboratiu(Recomanador):
    def recomana(self, usuari_id, n=5):
        # Algorisme de similitud + predicció
        pass


print("Pelicules")
reco = RecomanadorSimple("pelicules","dataset/MovieLens100k/movies.csv", "dataset/MovieLens100k/ratings.csv")
scores = reco.recomana("1", 10)
print(scores)

"""
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

vots_minims = 10
recomanador = RecomanadorSimple(dataset_pelicula, vots_minims)
print("Recomanador Creat")

#Quants items mostrem? 5 està al document
nombe_items_mostrats = 5
id_usuari = "2"
scores = recomanador.recomana(id_usuari, nombe_items_mostrats)
print(scores)"""
