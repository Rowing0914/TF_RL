import tensorflow as tf

class GCNBlock(tf.keras.Model):
    def __init__(self, adjacency_matrix, output_dim):
        super(GCNBlock, self).__init__()
        self.adjacency_matrix = adjacency_matrix
        self.dense = tf.keras.layers.Dense(output_dim, use_bias=True)
        self.lstm_cell = tf.compat.v1.keras.layers.CuDNNLSTM(output_dim, stateful=True)

    def call(self, features):
        output = self.dense(features)
        output = tf.linalg.matmul(self.adjacency_matrix, output)
        output = tf.keras.activations.relu(output)
        return output

class GCN(tf.keras.Model):
    def __init__(self, adjacency_matrix, num_node_feature):
        super(GCN, self).__init__()
        self.gcn1 = GCNBlock(adjacency_matrix, num_node_feature)
        self.gcn2 = GCNBlock(adjacency_matrix, num_node_feature)
        self.readout = tf.keras.layers.Dense(1, activation="tanh")

    def call(self, features):
        output = self.gcn1(features)
        output = self.gcn2(output)
        action = self.readout(output)
        return action