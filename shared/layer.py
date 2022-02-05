from abc import *

class Layer(ABC):
    def __init__(self, name) -> None:
        self.name: str = name

    @abstractmethod
    def activate(self, x, y):
        pass