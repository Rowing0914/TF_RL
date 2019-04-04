"""

Boilerplate file

- Agent
    - take_action: action selection procedure
    - learn: learning procedure
    - save_weights: save the weights in a model
    - load_weights: load the weights in a model

"""

class Agent:
    def __init__(self, processor=None):
        self.processor = processor

    def fit(self):
        # TODO: think if this is neede
        raise NotImplementedError()

    def compile(self):
        # TODO: think if this is neede
        raise NotImplementedError()

    def load_weights(self):
        raise NotImplementedError()

    def save_weights(self):
        raise NotImplementedError()

    def take_action(self):
        raise NotImplementedError()

    def learn(self):
        raise NotImplementedError()

