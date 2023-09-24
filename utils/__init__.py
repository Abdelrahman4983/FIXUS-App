from tensorflow.keras.models import load_model


af_classifier = load_model("Models/foot-ankle-classifier.h5")

ankle_ap_view = load_model("Models/Ankle_ap_veiw.hdf5")
ankle_lateral_view = load_model("Models/Ankle_lateral_veiw.hdf5")
ankle_oblique_view = load_model("Models/Ankle_oblique_veiw.hdf5")

foot_ap_view = load_model("Models/Foot_ap_veiw.hdf5")
foot_lateral_view = load_model("Models/Foot_lateral_veiw.hdf5")
foot_oblique_view = load_model("Models/Foot_oblique_veiw.hdf5")