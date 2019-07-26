import tensorflow as tf

class ActorWrapper:
    def __init__(self, model, graph_operator):
        self.model = model
        self.graph_operator = graph_operator

    @tf.contrib.eager.defun(autograph=False)
    def __call__(self, state):
        input_graph = self.graph_operator.obs_to_graph(state)
        output_graph = self.model(input_graph)
        action = self.graph_operator.readout_action(output_graph)
        return action
