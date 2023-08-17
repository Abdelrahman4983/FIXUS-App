import numpy as np

def handle_image_size(img):
    '''
        img: Pillow Image
    '''
    img = img.convert('L')
    img = np.asarray(img.resize((256,256)))/255.
    img = img.reshape(1,256,256)

    return img

