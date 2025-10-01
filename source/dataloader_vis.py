import os
import numpy as np

import cv2
import nibabel as nib
from skimage import transform

def read_data(image_size=(50, 50, 1), address='./images', flip=False, debug=False):
    folder_list = [item[0] for item in os.walk(address)]
    folder_list = folder_list[1:]
    
    if address[-4:] == '.npz':
        data = np.load(address)
        images = np.vstack((data['train_images'], data['val_images'], data['test_images']))
        labels = np.vstack((data['train_labels'], data['val_labels'], data['test_labels']))
        octs = []
        oct_label = []
        for i, d in enumerate(images):
            d_ = d.astype('float')
            d = d.astype('float')
            if len(d_.shape) == 2:
                d_ = np.tile(np.expand_dims(d_, 2), 3)
                d = np.tile(np.expand_dims(d, 2), 3)
            d_[:, :, 2] = d[:, :, 0]
            d_[:, :, 0] = d[:, :, 2]
            d_ /= np.max(d_)
            oct_image_standard_size = transform.resize(d_, image_size, order=1, preserve_range=True)
            octs.append(oct_image_standard_size)
            if labels[i] == 0:
                oct_label.append(0)
            else:
                oct_label.append(1)
        print("Fraction of AD images:", sum(oct_label)/len(oct_label))
    
        return np.asarray(octs), np.asarray(oct_label), address
    
    octs = []
    oct_label = []
    addresses = []
    label_counter = 0
    for _folder in folder_list:
        file_list = os.listdir(os.path.join(os.getcwd(), _folder))
        for _file in file_list:
            #if (_file[-4:] == '.tif') or (_file[-4:] == '.JPG'):
            if (_file[-4:] == '.png') or (_file[-4:] == '.JPG') or (_file[-5:] == '.jpeg'):
                oct_image = cv2.imread(os.path.join(os.getcwd(), _folder, _file)).astype('float64')
                addresses.append(os.path.join(os.getcwd(), _folder, _file))
                oct_image /= np.max(oct_image)
                
                oct_image_standard_size = np.zeros(image_size)
                larger_side = np.argmax(oct_image.shape)
                smaller_side = 1-larger_side
                scaling_factor = image_size[0] / oct_image.shape[larger_side]
                if larger_side == 0:
                    image_ = transform.resize(oct_image, (image_size[0], int(scaling_factor*oct_image.shape[smaller_side]), image_size[2]), order=1, preserve_range=True)
                else:
                    image_ = transform.resize(oct_image, (int(scaling_factor*oct_image.shape[smaller_side]), image_size[1], image_size[2]), order=1, preserve_range=True)
                for i in range(image_.shape[0]):
                    for j in range(image_.shape[1]):
                        oct_image_standard_size[i, j, :] = image_[i, j, :]
                if address.find('OCTID') != -1:
                    oct_image_standard_size = transform.rotate(oct_image_standard_size, 270)
                
                octs.append(oct_image_standard_size)
                if flip:
                    oct_image_flipped = cv2.flip(oct_image_standard_size, 1).reshape(image_size)
                    octs.append(oct_image_flipped)
                
                if _folder.find('hc') != -1:
                    oct_label.append(0)
                    if flip:
                        oct_label.append(0)
                else:
                    oct_label.append(1)
                    if flip:
                        oct_label.append(1)
                
                if debug:
                    print(_file)
                
                
    label_counter = 2
    
    print("Fraction of AD images:", sum(oct_label)/len(oct_label))
    
    return np.asarray(octs), np.asarray(oct_label), addresses
    
if __name__ == '__main__':
    x, y = read_data(image_size=(224, 224, 3), address='./[[[OCT RAW FOR DRYAD]]][2020-01JAN-16]', debug=False)
    for i in range(x.shape[0]):
        if y[i] == 0:
            cv2.imwrite('./oct_data/hc/data_'+str(i)+'.png', 255*x[i])
        else:
            cv2.imwrite('./oct_data/ad/data_'+str(i)+'.png', 255*x[i])
