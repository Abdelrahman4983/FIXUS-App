import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pipline import handle_image_size


classifier = tf.keras.models.load_model('/home/abdelrahman/Desktop/Dr. Sohiel/FIXUS-App/Models/foot-leg-classifier.h5')

img = Image.open('/home/abdelrahman/Desktop/Dr. Sohiel/Images/Foot/Normal/2/1.jpeg')
img = handle_image_size(img)

pred_to_label = {0:"Ankle",1:"Foot"}
prediction = classifier.predict(img)
print(pred_to_label[round(prediction[0][0])])

