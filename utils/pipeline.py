from PIL import Image
import numpy as np
from tensorflow.keras import backend as K
import cv2

views_to_label = {0:"Abnormal", 1:"Normal"}

def make_prediction(model, image):
    pred = model.predict(image)
    i = np.argmax(pred[0])
    label = views_to_label[i]

    heatmap = CAM_multiclass(model, image)
    return label, heatmap


def load_image(img_path, shape, gray=False):
    img = Image.open(img_path)
    img = img.resize(shape)
    if gray:
        img = img.convert('L')
        img = np.asarray(img)/255.
        img = img.reshape(1, img.shape[0], img.shape[1])
    else:
        img = np.asarray(img)/255.
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
    return img


def CAM_multiclass(model,img_set,which_label=None):
    class_weights = model.layers[-1].get_weights()[0]
    final_conv = model.layers[-3]
    get_output = K.function([model.layers[0].input],[final_conv.output,model.layers[-1].output])

    cam_output = np.zeros(shape = [img_set.shape[0],img_set.shape[1],img_set.shape[2],3])

    for img_index in range(img_set.shape[0]):
        input_img = img_set[img_index:img_index+1,...]
        #input_img_rgb = np.concatenate([input_img, input_img, input_img], axis = 3)
        [conv_output,predictions] = get_output([input_img])
        conv_output = conv_output[0,...]
        pred_label = np.argmax(predictions)

        cam = np.zeros(shape = conv_output.shape[:-1])
        if which_label==None:
            for i, w in enumerate(class_weights[:, pred_label]):
                cam += w * conv_output[...,i]
        if not (which_label==None):
            for i, w in enumerate(class_weights[:,  which_label[img_index] ]):
                cam += w * conv_output[...,i]
        cam /= np.max(cam)

        cam_resize = cv2.resize(cam[:,:], (img_set.shape[2],img_set.shape[1]))
        heatmap = cv2.applyColorMap(np.uint8(255*cam_resize), cv2.COLORMAP_JET)
        heatmap = heatmap[...,(2,1,0)]
        heatmap[np.where(cam_resize < 0.1)] = 0
        cam_output[img_index,:,:,:] = heatmap*0.3 + input_img[0,:,:,:]*255
        #cam_output[img_index,:,:,:] = heatmap*0.5 + input_img_rgb[0,:,:,:]

    return cam_output