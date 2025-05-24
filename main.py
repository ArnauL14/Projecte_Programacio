import logging
from recomanador import RecomanadorSimple, RecomanadorCollaboratiu, RecomanadorContingut
#from avaluador import Avaluador  # assumeix que ja el tens
import sys
import pickle

# Configuració del logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("log.txt", mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)

def seleccionar_dataset():
    """
    Permet a l'usuari seleccionar el tipus de dataset amb el qual vol treballar.
    Retorna el tipus de dataset seleccionat.
    """
    while True:
        tipus = input("Vols treballar amb 'pelicules' o 'llibres'? ").strip().lower()
        if tipus in ["pelicules", "llibres"]:
            return tipus
        else:
            print("Opció no vàlida.")

def seleccionar_rutes(tipus):
    """
    Retorna les rutes dels fitxers de dades segons el tipus de dataset seleccionat.
    Args:
        tipus (str): El tipus de dataset ('pelicules' o 'llibres').
    Returns:
        tuple: Rutes dels fitxers de dades (ruta_items, ruta_valoracions).
    """
    if tipus == "pelicules":
        return "dataset/MovieLens100k/movies.csv", "dataset/MovieLens100k/ratings.csv"
    else:
        return "dataset/Books/books.csv", "dataset/Books/ratings.csv"
    
def seleccionar_recomanador():
    """
    Permet a l'usuari seleccionar el tipus de recomanador que vol utilitzar.
    Retorna el tipus de recomanador seleccionat.
    """
    while True:
        op = input("Tipus de recomanador (simple, colaboratiu, contingut): ").strip().lower()
        if op == "simple":
            return op
        elif op == "colaboratiu":
            return op
        elif op == "contingut":
            return op
        else:
            print("Opció no vàlida.")

def main():
    logging.info("Inici del sistema de recomanació")
    usuari = input("Introdueix l'ID de l'usuari: ").strip()

    while True:
        print("\nOpcions:")
        print("1. Recomanar ítems")
        print("2. Avaluar mètode")
        print("3. Sortir")
        opcio = input("Escull una opció (1-3): ").strip()

        if opcio == "1":
            logging.info(f"Usuari {usuari} ha seleccionat recomanar ítems")
            tipus = seleccionar_dataset()
            ruta_items, ruta_valoracions = seleccionar_rutes(tipus)
            tipus_recomanador = seleccionar_recomanador()
            if tipus_recomanador == "simple":
                recomanador = RecomanadorSimple(tipus, ruta_items, ruta_valoracions)
                num_minim_vots = int(input("Introdueix el mínim de vots requerits: "))
                recomanador.recomana(usuari, num_minim_vots)
            elif tipus_recomanador == "colaboratiu":
                recomanador = RecomanadorCollaboratiu(tipus, ruta_items, ruta_valoracions)
                num_usuaris_similars = int(input("Introdueix el nombre d'usuaris similars: "))
                recomanador.recomana(usuari, num_usuaris_similars)
            elif tipus_recomanador == "contingut":
                recomanador = RecomanadorContingut(tipus, ruta_items, ruta_valoracions)
                recomanador.recomana(usuari)
            logging.info(f"Recomanacions per a l'usuari {usuari} amb el recomanador {tipus_recomanador} fetes amb èxit")

        if opcio == "2":
            """Aquesta part del codi hauria de ser implementada per a l'avaluació del mètode."""
            
        elif opcio == "3":
            logging.info("Sortint del sistema de recomanació")
            print("Sortint...")
            break #@Iker, es pot canviar si vols per un while opcio =! 3

        else:
            print("Opció no vàlida. Torna-ho a intentar.")
            continue

        logging.info("Programa finalitzat amb èxit")

if __name__ == "__main__":
    main()
