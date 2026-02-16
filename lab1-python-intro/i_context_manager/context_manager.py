from typing import TextIO, Type


class Supresser:
    def __init__(self, *types_: Type[BaseException]):
        self.types = types_

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class Retyper:
    def __init__(self, type_from: Type[BaseException], type_to: Type[BaseException]):
        self.type_from = type_from
        self.type_to = type_to

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class Dumper:
    def __init__(self, stream: TextIO | None = None):
        self.stream = stream

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def supresser(*types_: Type[BaseException]) -> Supresser:
    return Supresser(*types_)


def retyper(type_from: Type[BaseException], type_to: Type[BaseException]) -> Retyper:
    return Retyper(type_from, type_to)


def dumper(stream: TextIO | None = None) -> Dumper:
    return Dumper(stream)
