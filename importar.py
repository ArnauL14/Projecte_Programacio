from item import Pelicula, Llibre

def importar_pelicules(arxiu_csv):
    pelicules = []
    ids = []
    with open(arxiu_csv, "r", encoding="utf-8") as arxiu:
        next(arxiu)

        for linia in arxiu:
            parts = linia.strip().split(",")

            movie_id = int(parts[0])
            titol = parts[1]
            generes = parts[2].split('|') if parts[2] else []

            ids.append(movie_id)
            pelicules.append(Pelicula(movie_id, titol, generes))

    ids.sort()
    return pelicules, ids


def importar_llibres(arxiu_csv):
    llibres = []
    isbns = []
    with open(arxiu_csv, "r", encoding="utf8") as arxiu:
        next(arxiu)  # salta capçalera

        for linia in arxiu:
            parts = linia.strip().split(",")

            isbn = parts[0]
            titol = parts[1]
            autor = parts[2]
            any_publi = parts[3]
            editorial = parts[4]

            isbns.append(isbn)
            llibres.append(Llibre(isbn, titol, autor, any_publi, editorial))

    isbns.sort()
    return llibres, isbns

def importar_ids(arxiu_csv):
    ids = []
    with open(arxiu_csv, "r", encoding="utf8") as arxiu:
        next(arxiu)

        for linia in arxiu:
            parts = linia.strip().split(",")
            id = parts[0]

            if id not in ids:
                ids.append(id)
    
    ids.sort()
    return ids
