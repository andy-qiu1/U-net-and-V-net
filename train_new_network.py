from __future__ import division, print_function
import numpy as np
np.random.seed(98764)
from tf_unet import image_gen
from tf_unet import unet
from tf_unet import util
import os
from PIL import Image
image_dir = './images'
ground_truth_dir = './gts'
image_file_list = []
for subdir, dirs, files in os.walk(image_dir):
    for file in files:
        if(file.split('.')[1] == 'tif'):
            image_file_list.append(os.path.join(subdir, file))
print("size of dataset:"+str(len(image_file_list)))
test_set = image_file_list[:5000]
verification_set = image_file_list[5000:6000]
print("size of test_set:")
print(len(test_set))
print("size of verification_set:")
print(len(verification_set))
test_set_gt = map((lambda x : x.replace('images','processed_gts')),test_set)
verification_set_gt = map((lambda x : x.replace('images','processed_gts')),verification_set)
nodule_image_count= 0
for file in test_set_gt:
    im = Image.open(file)
    imarray = np.array(im)
    if np.max(imarray)==1:
        nodule_image_count=nodule_image_count+1
print("size of test_set that has nodule:")
print(nodule_image_count)

nodule_image_count= 0
for file in verification_set_gt:
    im = Image.open(file)
    imarray = np.array(im)
    if np.max(imarray)==1:
        nodule_image_count=nodule_image_count+1
print("size of verification_set that has nodule:")
print(nodule_image_count)

shapes=[]
for file in test_set_gt:
    im = Image.open(file)
    imarray = np.array(im)
    shape = imarray.shape
    if shape not in shapes:
        shapes.append(shape)
print("different shapes of image:")
print(shapes)

from tf_unet.image_util import BaseDataProvider


class My_Own_Data_Provider(BaseDataProvider):
    channels = 1
    n_class = 2


    def __init__(self, test_set_paths, a_min=None, a_max=None,shuffle_data=True):
        self.a_min = a_min if a_min is not None else -np.inf
        self.a_max = a_max if a_min is not None else np.inf
        self.file_idx = 0
        self.data_files = test_set_paths
        self.shuffle_data = shuffle_data
        np.random.shuffle(self.data_files)

    def _next_data(self):
        self._cylce_file()
        image_name = self.data_files[self.file_idx]
        label_name = image_name.replace('images','processed_gts')

        img = self._load_file(image_name)
        label = self._load_file(label_name)

        return img,label

    def _load_file(self, path, dtype=np.float32):
        return np.array(Image.open(path), dtype)

    def _cylce_file(self):
        self.file_idx += 1
        if self.file_idx >= len(self.data_files):
            self.file_idx = 0
            if self.shuffle_data:
                np.random.shuffle(self.data_files)

    def _load_data_and_label(self):
        data, label = self._next_data()

        train_data = self._process_data(data)
        labels = self._process_labels(label)

        train_data, labels = self._post_process(train_data, labels)

        nx = train_data.shape[1]
        ny = train_data.shape[0]

        return train_data.reshape(1, ny, nx, self.channels), labels.reshape(1, ny, nx, self.n_class),

    def _process_labels(self, label):
        if self.n_class == 2:
            nx = label.shape[1]
            ny = label.shape[0]
            labels = np.zeros((ny, nx, self.n_class), dtype=np.float32)
            labels[..., 1] = label
            labels[..., 0] = np.ones(label.shape,dtype=np.float32)- label
            return labels

        return label

    def _process_data(self, data):
        # normalization
        #data = np.clip(np.fabs(data), self.a_min, self.a_max)
        data -= np.amin(data)
        data /= np.amax(data)
        return data

    def _post_process(self, data, labels):
        """
        Post processing hook that can be used for data augmentation

        :param data: the data array
        :param labels: the label array
        """
        data= np.pad(data, ((30, 30), (30, 30)), 'constant', constant_values=((0, 0), (0, 0)))
        temp_1 =np.pad(labels[..., 1], ((30, 30), (30, 30)), 'constant', constant_values=((0, 0), (0, 0)))
        temp_2 = np.pad(labels[..., 0], ((30, 30), (30, 30)), 'constant', constant_values=((1, 1), (1, 1)))
        labels = np.zeros((572,572,2))
        labels[..., 1] = temp_1
        labels[..., 0] = temp_2
        return data, labels

    def __call__(self, n):
        train_data, labels = self._load_data_and_label()
        nx = train_data.shape[1]
        ny = train_data.shape[2]

        X = np.zeros((n, nx, ny, self.channels))
        Y = np.zeros((n, nx, ny, self.n_class))

        X[0] = train_data
        Y[0] = labels
        for i in range(1, n):
            train_data, labels = self._load_data_and_label()
            X[i] = train_data
            Y[i] = labels

        return X, Y
my_generator =My_Own_Data_Provider(test_set)

net = unet.Unet(channels=my_generator.channels, n_class=my_generator.n_class, layers=5, features_root=16,cost ='cross_entropy',cost_kwargs=dict(class_weights=[1,10]))
trainer = unet.Trainer(net, optimizer="adam", opt_kwargs=dict())
path = trainer.train(my_generator, "./unet_trained", training_iters=32, epochs=100, display_step=2,restore=True)
