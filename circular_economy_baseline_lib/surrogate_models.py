import tensorflow as tf
import numpy as np

# Load the saved model

apap_path = "best_model_apap.h5"
pap_path = "best_model_pap.h5"
hyd_path = "best_model_hyd.h5"

class SurrogateModel():
    def __init__(self):
        #import pdb; pdb.set_trace()
        self.apap_model = tf.keras.models.load_model(apap_path, compile=False)
        self.pap_model = tf.keras.models.load_model(pap_path, compile=False)
        self.hyd_model = tf.keras.models.load_model(hyd_path, compile=False)

        print("---APAP model loaded---")
        self.apap_model.summary()
        print("---PAP model loaded---")
        self.pap_model.summary()
        print("---hyd model loaded---")
        self.hyd_model.summary()

    def get_apap_model_outputs(self, apap_input):
        return self.apap_model(apap_input)

    def get_pap_model_outputs(self, pap_input):
        return self.pap_model(pap_input)

    def get_hyd_model_outputs(self, hyd_input):
        return self.hyd_model(hyd_input)

    def get_agent_model(self, agent_id):
        if agent_id == 0:
            return self.apap_model
        elif agent_id == 1:
            return self.pap_model
        else:
            return self.hyd_model

    def get_agent_model_params(self, agent_id):
        if agent_id == 0:
            curr_model = self.apap_model
        elif agent_id == 1:
            curr_model = self.pap_model
        else:
            curr_model = self.hyd_model

        weights = []
        biases = []
        bn_params = []  # For batch normalization: [gamma, beta, mean, variance]
        layer_type_list = []

        for layer in curr_model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                w, b = layer.get_weights()
                weights.append(w)
                biases.append(b)
                layer_type_list.append(0)
            elif isinstance(layer, tf.keras.layers.BatchNormalization):
                gamma, beta, mean, variance = layer.get_weights()
                bn_params.append((gamma, beta, mean, variance))
                layer_type_list.append(1)

        return weights, biases, bn_params, layer_type_list









