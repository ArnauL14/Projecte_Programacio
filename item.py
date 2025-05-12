class Item:
    def __init__(self, id_item, titol):
        self.id = id_item
        self.titol = titol

    def __str__(self):
        return f"{self.titol}"


class Pelicula(Item):
    def __init__(self, id_item, titol, generes):
        super().__init__(id_item, titol)
        self.genere = generes  # llista de string

    def __str__(self):
        return f"{self.titol} - Gèneres: {','.join(self.genere)}"


class Llibre(Item):
    def __init__(self, isbn, titol, autor, any_publicacio, editorial):
        super().__init__(isbn, titol)
        self.autor = autor
        self.any = int(any_publicacio)
        self.editorial = editorial

    def __str__(self):
        return f"{self.titol} ({self.any}) - {self.autor}, {self.editorial}"
