import os
import tensorflow as tf
import numpy as np
import cv2
import shuffle

class PF_Pascal():
    def __init__(self, path_to_root):
        '''
        Args:
            path_to_root (string): the root path to PF-Pascal dataset extracted from zip file.
        Returns:
            None
        '''
        self.image_dir = os.join(path_to_root, "PF-dataset-PASCAL", "JPEGImages")
        self.annotation_dir = os.join(path_to_root, "PF-dataset-PASCAL", "Annotations")
        self.category_list = os.listdir(self.annotation_dir)
        self.category_list = [cat for cat in self.category_list if cat != ".DS_Store"]
        #filter out .DS_Store file, Its not category.
        self.num_category = len(self.category_list)
        self.category_images = []
        for cat in self.category_list:
            tmp_dir = os.listdir(os.path.join(self.PATH_TO_ANNOTATIONS, cat))
            image_lists_of_the_category = [ os.path.splitext(m_file)[0] + ".jpg" for m_file in tmp_dir]
            #convert extensions(.m -> .jpg) each file names
            self.category_images.append(image_lists_of_the_category)

    def synthesized_pair(self, input_shape, num_examples, shuffle=True):
        '''
        return a iterative object to generate batches of image pairs which are
        original image and synthesized image by geometric transformation from orignal image.
        Args:
            path_to_root (string): the root path to PF-Pascal dataset extracted from zip file.
        Returns:
            None
        '''
        image_list = os.listdir(self.image_dir)
        if shuffle:
            shuffle(image_list)
        for file in image_list:
            imageA = cv2.imread(os.path.join(self.image_dir, file))[:,:,::-1]
            imageB, parameter = synthesize_with_transform(imageA)
            yield image, parameter

    def categorical_pair(self, input_shape, num_examples, shuffle=True):
        pass

    def episode(self):
        pass

    def load_generator(self, method, input_shape, num_examples, shuffle_buffer):
        '''
        load the tensorflow generator using tf.data API.
        you can choose the three way of data loading for proper tasks.
        Args:
            input_shape (int): the shape of input. (H,W,C)
            num_examples (int) : the limitation of data. -1 represents full dataset.
            shuffle_buffer (int) : buffer size for tf.data.shuffle. -1 represents no usage of shuffle.
        Returns:
            None
        '''
        if method == "synthesized_pair":
            gen = self.synthesized_pair
        elif method == "categorical_pair":
            gen = self.categorical_pair
        elif method == "episode":
            gen = self.episode
        gen = partial(gen, input_shape, num_examples)
        ds = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32))
        if shuffle_buffer > 0:
            ds.shuffle(shuffle_buffer)
        return ds
