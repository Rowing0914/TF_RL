import tensorflow as tf
import tensorflow_probability as tfp

XAVIER_INIT = tf.contrib.layers.xavier_initializer()


class GRU_cell(tf.keras.Model):
    def __init__(self, hidden_unit, output_nodes):
        super(GRU_cell, self).__init__()
        self.i_to_r = tf.keras.layers.Dense(hidden_unit, activation="sigmoid", kernel_initializer=XAVIER_INIT)
        self.i_to_z = tf.keras.layers.Dense(hidden_unit, activation="sigmoid", kernel_initializer=XAVIER_INIT)
        self.i_to_h = tf.keras.layers.Dense(hidden_unit, activation="tanh", kernel_initializer=XAVIER_INIT)
        self.h_to_h = tf.keras.layers.Dense(hidden_unit, activation="linear", kernel_initializer=XAVIER_INIT)
        self.h_to_o = tf.keras.layers.Dense(output_nodes, activation="tanh", kernel_initializer=XAVIER_INIT)

    def call(self, x, previous_hidden_state):
        z = self.i_to_z(x)
        r = self.i_to_r(x)
        h = self.i_to_h(x) + self.h_to_h(previous_hidden_state) * r
        current_hidden_state = tf.multiply((1 - z), h) + tf.multiply(previous_hidden_state, z)
        output = self.h_to_o(current_hidden_state)
        return output, current_hidden_state


class GGNN(tf.keras.Model):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Implementation based on https://arxiv.org/abs/1511.05493
    """

    def __init__(self, state_dim, node_info, rec_hidden_unit=30, rec_output_unit=30, recurrent_step=3, batch_size=32):
        super(GGNN, self).__init__()
        self.batch_size = batch_size
        self.recurrent_step = recurrent_step
        self.state_dim = state_dim
        self.node_info = node_info

        # Each type of nodes has its own mapping func
        # num_node x obs(for each node)-> num_node x node_state_dim
        self.obs_to_node_embed = {
            node_id: tf.keras.layers.Dense(state_dim, activation="tanh", kernel_initializer=XAVIER_INIT)
            for node_id in self.node_info["input_dict"]
        }

        # Each type of nodes has its own propagation func
        # num_node x node_params_dim -> num_node x node_state_dim
        self.processed_node_params = {
            node_type: tf.keras.layers.Dense(state_dim, activation="tanh", kernel_initializer=XAVIER_INIT)(
                tf.cast(self.node_info["node_parameters"][node_type], dtype=tf.float32)
            )
            for node_type in self.node_info["node_type_dict"]
        }
        self.processed_node_params = self._dict_to_matrix(self.node_info["node_type_dict"], self.processed_node_params)

        # Update the internal memory
        # num_node x node_state_dim -> num_node x node_state_dim
        # self.recurrent_unit = tf.keras.layers.CuDNNGRU(state_dim, return_state=True, stateful=True)
        self.recurrent_unit = GRU_cell(hidden_unit=rec_hidden_unit, output_nodes=rec_output_unit)

        # Output Model
        # num_node x hidden_dim -> num_node x 1(used to construct the mu for each action)
        self.hidden_to_output = tf.keras.layers.Dense(1, activation="tanh", kernel_initializer=XAVIER_INIT)

        # STD for Action Distribution(Gaussian)
        self.sigmas = tf.Variable([0.3] * len(self.node_info["output_list"]))
        # self.sigmas = tf.keras.layers.Dense(len(self.node_info["output_list"]), activation='linear', kernel_initializer=XAVIER_INIT)

    def call(self, obs):
        # step 1: Mapping Observation to Node Embedding
        node_embed = {
            node_id: self.obs_to_node_embed[node_id](self._gather_obs_at_node(obs, node_id))
            for node_id in self.node_info["input_dict"]
        }
        node_embed = self._dict_to_matrix(self.node_info["input_dict"], node_embed)

        # Concat information
        if len(node_embed.shape) == 3:
            # (batch_size x num_node x node_state_dim), (batch_size x num_node x node_state_dim)
            # => batch_size x num_node x node_state_dim*2
            hidden_state = tf.concat([[self.processed_node_params]*self.batch_size, node_embed], axis=-1)
        else:
            # (num_node x node_state_dim), (num_node x node_state_dim) => num_node x node_state_dim*2
            hidden_state = tf.concat([self.processed_node_params, node_embed], axis=-1)

        prop_state = list()
        for t in range(self.recurrent_step):
            # step 2: Message Exchanging for each edge type
            # (num_node x num_node) x (num_node x node_state_dim*2) -> num_node x node_state_dim*2
            for edge in self.node_info["edge_type_list"]:
                prop_state.append(tf.compat.v1.linalg.matmul(self.node_info["adjacency_matrix"][edge], hidden_state))

            # step 4: Aggregate Exchanged Messages
            # num_edge_type x (num_node x node_state_dim) -> num_node x node_state_dim*2
            msgs = tf.math.reduce_sum(prop_state, axis=0)

            # step 5: Recurrent Update
            # num_node x node_state_dim*2 -> num_node x node_state_dim*2
            output, hidden_state = self.recurrent_unit(msgs, hidden_state)

        # step 6: Readout function
        output = self.hidden_to_output(hidden_state)

        # step 7: Construct the Gaussian distribution over the action
        means = tf.gather(tf.reshape(output, output.shape[:-1]), self.node_info["output_list"], axis=-1)
        actions = tfp.distributions.Normal(loc=means, scale=self.sigmas).sample(1)
        return actions

    def _gather_obs_at_node(self, state, node_id):
        return tf.gather(state, self.node_info["input_dict"][node_id], axis=-1)

    def _dict_to_matrix(self, loop_list, item):
        for i, data in enumerate(loop_list):
            if i == 0:
                temp = item[data]
            else:
                if len(item[data].shape) == 3: # in the case of batch learning
                    temp = tf.concat([temp, item[data]], axis=1)
                else:
                    temp = tf.concat([temp, item[data]], axis=0)
        return tf.cast(temp, dtype=tf.float32)
