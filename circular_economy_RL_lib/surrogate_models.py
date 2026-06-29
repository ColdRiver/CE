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




