
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img

# Based on https://keras.io/examples/vision/oxford_pets_image_segmentation/
# Ideas and code taken from 
# 1. https://github.com/Tramac/awesome-semantic-segmentation-pytorch/blob/master/core/data/dataloader/cityscapes.py
# 2. https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/loader/cityscapes_loader.py 
# 3. https://github.com/divamgupta/image-segmentation-keras/blob/master/keras_segmentation/data_utils/data_loader.py
# 4. https://towardsdatascience.com/a-keras-pipeline-for-image-segmentation-part-1-6515a421157d
# And many others 

#CHANGE THE PATHS!!!
input_dir = "../data/semantic-segmentation/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train" #CHANGE THE PATHS!!!
target_dir = "../data/semantic-segmentation/cityscapes/gtFine_trainvaltest/gtFine/train" #CHANGE THE PATHS!!!
img_size = (160, 160)
num_classes = 20
batch_size = 32

input_img_paths = sorted(
    [
        os.path.join(input_dir, city, fname) 
        for city in os.listdir(input_dir) 
        for fname in os.listdir(os.path.join(input_dir, city))
            if fname.endswith(".png")
    ]
)
target_img_paths = sorted(
    [
        os.path.join(target_dir, city, fname) 
        for city in os.listdir(target_dir) 
        for fname in os.listdir(os.path.join(target_dir, city))
            if fname.endswith("gtFine_labelIds.png") and not fname.startswith(".")
    ]
)


class Cityscapes(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1] #not to train
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ['road', 'sidewalk', 'building', 'wall', 'fence', \
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']

        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(num_classes)))

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y_temp = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        y = np.zeros((self.batch_size,) + self.img_size + (20,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y_temp[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, ..., 30. Subtract one to make them 0, 1, ..., 29:
            y_temp[j] -= 1
            y_temp[j] = self.fix_indxs(y_temp[j])
            y[j] = self.one_hot_encode(y_temp[j])
        return x, y
    
    def fix_indxs(self,mask):
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        mask[mask == 255] = len(self.valid_classes)
        return mask
    
    def one_hot_encode(self,lbl):
        new_lbl = np.array(self.get_one_hot(lbl.reshape(-1),num_classes))
        new_lbl = new_lbl.reshape(160,160,num_classes)
        return new_lbl
        
    def get_one_hot(self,targets, nb_classes):
        res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
        return res.reshape(list(targets.shape)+[nb_classes])
