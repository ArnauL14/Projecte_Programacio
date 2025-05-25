class Item:
    def __init__(self, id_item, titol):
        self._id = id_item
        self._titol = titol

    def __str__(self):
        return f"{self._titol}"


class Pelicula(Item):
    def __init__(self, id_item, titol, generes):
        super().__init__(id_item, titol)
        self._genere = generes  # llista de strings

    def __str__(self):
        return f"{self._titol} - {', '.join(self._genere)}"


class Llibre(Item):
    def __init__(self, isbn, titol, autor, any_publicacio, editorial):
        super().__init__(isbn, titol)
        self._autor = autor
        self._any = any_publicacio
        self._editorial = editorial

    def __str__(self):
        return f"{self._titol} ({self._any}) - {self._autor}, {self._editorial}"
