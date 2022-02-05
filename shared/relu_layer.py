from shared.layer import Layer

class ReluLayer(Layer):
    def __init__(self, name) -> None:
        super().__init__(name)

    def activate(self, x, y):
        return [x if x > 0 else 0, y if y > 0 else 0]
