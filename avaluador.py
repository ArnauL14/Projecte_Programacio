import numpy as np
from dataset import Dataset


class Avaluador:
    """
    Classe per avaluar la qualitat de les recomanacions d'un sistema de recomanació.
    Aquesta classe calcula l'error absolut mitjà (MAE) entre les valoracions reals i les recomanades.
    Attributes:
        valoracins_reals (np.ndarray): Matriu de valoracions reals.
        dataset (Dataset): Instància del conjunt de dades utilitzat per a les recomanacions.
        valoracions (np.ndarray): Matriu de valoracions del dataset.
    """
    def __init__(self, valoracins_reals, dataset): 
        self._valoracins_reals = valoracins_reals # Matriu de valoracions reals
        self._dataset = dataset
        self._valoracions = dataset.get_valoracions_numeriques()

    def calcula_mae(self):
        """
        Aquesta funció calcula i retorna el valor (float) del
        MAE (Mean Absolute Error), una mètrica per valorar la precisió de les prediccions.
        """
        return np.mean(np.abs(self._valoracins_reals - self._valoracions))
    
    def calcula_rmse(self):
        """
        Aquesta funció calcula i retorna el valorn (float) del
        RMSE , una mètrica per valorar la precisió de les prediccions
        """
        mse = np.mean((self._valoracins_reals - self._valoracions) ** 2)
        return np.sqrt(mse)

"""dataset = DatasetPelicules("dataset/MovieLens100k/movies.csv", "dataset/MovieLens100k/ratings.csv")
matriu_real = dataset.get_valoracions_numeriques() + 2
avaluador = Avaluador(matriu_real, dataset)
print("MAE:", avaluador.calcula_mae())
print("RMSE:", avaluador.calcula_rmse())"""
