import glob
import random
from collections import Counter
from functools import wraps
from random import shuffle
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy.special import gammaln
from skimage.filters import unsharp_mask
from sklearn import preprocessing
from sklearn.decomposition import PCA, KernelPCA
import os
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import platform
from skimage import feature
from tqdm import tqdm
from scipy import ndimage, misc
__instances = {}


def singleton(cls):
    @wraps(cls)
    def getInstance(*args, **kwargs):
        instance = __instances.get(cls, None)
        if not instance:
            instance = cls(*args, **kwargs)
            __instances[cls] = instance
        return instance

    return getInstance


@singleton
class Helper():
    def __init__(self, mode = 'green', img_resize = [], img_crop = [], create_mock_test=False, aug=[False, 5], unsharp=[], crop_and_augment=None):
        self.models = []

        self.mode = mode
        self.multi_channel = False if mode == 'green' else True
        self.img_resize = img_resize
        self.img_crop = img_crop
        self.unsharp = unsharp
        self.X_train_images, self.Y_train, self.dim_x, self.dim_y = self.read_train_set()
        self.create_mock_test = create_mock_test
        if self.create_mock_test:
            self.generate_mock_indices()

        class_counts = Counter(self.Y_train)
        self.class_weights = {class_id: (len(self.Y_train) / class_counts[class_id]) for class_id in range(5)}
        print("Training Set Loaded")
        self.X_test_images = self.read_test_set()
        print("Test Set Loaded")

        if self.create_mock_test:
            self.Y_train, self.Y_mock_test = self.split_train_mock_test(self.Y_train)
            self.X_train_flattened, self.X_mock_test_flattened = self.split_train_mock_test(self.X_train_flattened)
            self.X_train_images, self.X_mock_test_images = self.split_train_mock_test(self.X_train_images)
            print("mock test set created")

        if self.mode == 'green':
            self.X_train_images = self.X_train_images[:, :, :, np.newaxis]
            self.X_test_images = self.X_test_images[:, :, :, np.newaxis]
            if self.create_mock_test:
                self.X_mock_test_images = self.X_mock_test_images[:, :, :, np.newaxis]
        self.one_hot = self.one_hot_fit(self.Y_train)


        self.aug = aug
        self.crop_and_augment = crop_and_augment
        if self.crop_and_augment is not None:
            self.augment_by_cropping()

    # def augment_by_cropping(self):
    #     X =


    def balanced_k_fold_splits(self, Y, n):
        all_label_indices = [[i for i in range(len(Y)) if Y[i] == label] for label in range(5)]
        all_label_counts = [len(label_indices) for label_indices in all_label_indices]
        required_count_per_class = int(len(Y) / 5)
        all_label_repeat_count = [
            required_count_per_class - all_label_counts[i] if required_count_per_class - all_label_counts[i] > 0 else 0
            for i in range(5)]
        all_label_indices = [self.random_augment(label_indices, all_label_repeat_count[i]) for i, label_indices in enumerate(all_label_indices)]
        splits = []
        required_count_per_split = int(required_count_per_class / n)
        for i in range(n):
            split = []
            for j in range(5):
                split.extend(all_label_indices[j][i * required_count_per_split: (i + 1) * required_count_per_split])
            random.shuffle(split)
            splits.append(split)

        k_fold = []
        indices = np.array(splits).flatten()
        for i in range(n):
            test_indices = splits[i]
            train_indices = [ind for ind in indices if ind not in test_indices]
            k_fold.append([list(set(train_indices)), test_indices])

        return np.array(k_fold)


    @staticmethod
    def random_augment(indices, augment_count):
        if len(indices) == 0: return indices
        replace = False if augment_count < len(indices) else True
        augment_indices = np.random.choice(indices, size=augment_count, replace=replace) if augment_count > 0 else []
        indices.extend(augment_indices)
        random.shuffle(indices)
        return indices


    def generate_mock_indices(self, Y_train=None, split_ratio=None):
        if Y_train is None:
            Y_train = self.Y_train
        if split_ratio is None:
            samples_per_class = 40
        else:
            samples_per_class = int(len(Y_train) * split_ratio / 5)
        all_label_indices = [[i for i in range(len(Y_train)) if Y_train[i] == label] for label in range(5)]
        self.test_mock_indices = sorted(np.array([np.random.choice(label_indices, size=samples_per_class, replace=False) for label_indices in all_label_indices]).flatten())
        self.train_mock_indices = sorted(np.array([i for i in range(len(Y_train)) if i not in self.test_mock_indices]).flatten())

    def split_train_mock_test(self, X):
        return X[self.train_mock_indices], X[self.test_mock_indices]

    def duplicate_indices(self):
        all_label_indices = [[i for i in range(len(self.Y_train)) if self.Y_train[i] == label] for label in range(1, 5)]
        duplicate_indices = sorted(np.array([np.random.choice(label_indices, size=35, replace=False) for label_indices in all_label_indices]).flatten())
        self.train_duplicate_indices=[]
        for x in range(len(self.Y_train)):
            (self.train_duplicate_indices.extend(np.repeat(x, 2, axis=0))
             if x in duplicate_indices else self.train_duplicate_indices.append(x))

    def split_train_dup(self, X):
        return X[self.train_duplicate_indices]


    def read_train_set(self):

        files = sorted(glob.glob('./data/TrainData/*.jpg'), key=lambda filename: int(filename.split('/')[-1][:-4]))
        annotations = pd.read_csv('./data/TrainAnnotations.csv', index_col=None).set_index('file_name').to_dict()['annotation']
        X_images, Y = [], []

        for file in files:

            img = cv2.imread(file)
            img = self.preprocess(img)
            if(self.mode != 'green'):
                X_images.append(img)
            else:
                X_images.append(img[:, :, 1])

            separator = '/'
            if platform.system() == 'Windows':
                separator = '\\'
            Y.append(annotations[str(file).split(separator)[-1]])

        return np.array(X_images), np.array(Y), X_images[0].shape[1], X_images[0].shape[0]

    def read_test_set(self):

        files = sorted(glob.glob('./data/TestData/*.jpg'), key=lambda filename: int(filename.split('/')[-1][:-4]))
        X_images = []
        for file in files:

            img = cv2.imread(file)

            img = self.preprocess(img)
            if(self.mode != 'green'):
                X_images.append(img)
            else:
                X_images.append(img[:, :, 1])

        return np.array(X_images)

    def train_val_split(self, X, Y, validation_split):
        num_classes = np.max(np.argmax(Y, axis=1)) + 1
        train_indices, val_indices = self.generate_split_indices(np.argmax(Y, axis=1), validation_split, num_classes)
        Y_train, Y_val = self.split_train_val(np.array(Y), train_indices, val_indices)
        if type(X).__name__ == 'list':
            X_train, X_val = [], []
            for x in X:
                train, val = self.split_train_val(np.array(x), train_indices, val_indices)
                X_train.append(train)
                X_val.append(val)
        else:
            X_train, X_val = self.split_train_val(np.array(X), train_indices, val_indices)
        return X_train, X_val, Y_train, Y_val

    @staticmethod
    def split_train_val(X, train_indices, val_indices):
        return X[train_indices], X[val_indices]

    @staticmethod
    def generate_split_indices(Y_train, split_ratio, num_classes):
        Y_train = list(Y_train)
        samples_per_class = int(len(Y_train) * split_ratio / num_classes)
        all_label_indices = [[i for i in range(len(Y_train)) if Y_train[i] == label] for label in range(num_classes)]
        val_indices = sorted(np.array([np.random.choice(label_indices, size=samples_per_class, replace=False) for label_indices in all_label_indices]).flatten())
        train_indices = sorted(np.array([i for i in range(len(Y_train)) if i not in val_indices]).flatten())
        print(set(val_indices).intersection(set(train_indices)))
        return train_indices, val_indices


    def pca_transform(self, pca, X):
        return pca.transform(X)


    def one_hot_fit(self, Y):
        labels = preprocessing.LabelEncoder()
        labels.fit(Y)
        return labels

    def one_hot_transform(self, labels, Y):
        encoded_labels = labels.transform(Y)
        one_hot_y = np_utils.to_categorical(encoded_labels)
        return np.array(one_hot_y, dtype=np.int)

    def calc_accuracy(self, Y, Y_pred):
        total = len(Y)
        if len(Y) != len(Y_pred):
            print("Shapes for predictions and ground truth don't match")
            return
        correct = sum([int(Y_pred[i] == Y[i]) for i in range(total)])
        return correct * 100 / total

    def hist_eq_2D(self, channel):
        return cv2.equalizeHist(channel)

    def hist_eq_3D(self, img):
        b, g, r = cv2.split(img)
        b = self.hist_eq_2D(b)
        g = self.hist_eq_2D(g)
        r = self.hist_eq_2D(r)
        img = cv2.merge((b, g, r))
        return img

    def preprocess(self, img):
        # implement preprocessing here


        if len(self.unsharp) == 2:
            if self.mode == 'green':

                img[:, :, 1] = (unsharp_mask(self.hist_eq_2D(img[:, :, 1]), radius=self.unsharp[0], amount=self.unsharp[1]).astype('float32') * 255).astype('uint8')
            else:
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                hsv[:, :, 2] = (unsharp_mask(self.hist_eq_2D(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)), radius=self.unsharp[0], amount=self.unsharp[1]).astype('float32') * 255).astype('uint8')
                img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        if len(self.img_resize) == 2:
            img = cv2.resize(img, (self.img_resize[0], self.img_resize[1]), cv2.INTER_CUBIC)
        if len(self.img_crop):
            img = img[-self.img_crop[1]:, -self.img_crop[0]:]
        if self.mode != 'green':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        img = img.astype('float32') / 255
        return img

    def hog_transform(self, X):
        X_hog = []
        for x in X:
            fd_hog = feature.hog(x, orientations=16, pixels_per_cell=(16, 16), cells_per_block=(5, 5), multichannel=True if self.mode!='green' else False)
            X_hog.append(fd_hog)
            # plt.imshow(x_hog)
            # plt.show()

        return np.array(X_hog)


    def showGraphs(self, history):
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

    def show_model_evaluation_bars(self):
        models = self.models
        train_accuracies = [float(model.k_train_accuracy) for model in models]
        val_accuracies = [float(model.k_val_accuracy) for model in models]
        model_names = [model.name for model in models]
        df = pd.DataFrame({'Validation accuracy': val_accuracies, 'Training accuracy': train_accuracies},
                          index=model_names)
        plt.rcParams["figure.dpi"] = 300
        ax = df.plot.bar(rot=0, title='Model Selection')
        ax.set_xlabel("Models")
        ax.set_ylabel("K-fold Cross Validation Mean Accuracy")


    def vote(self, predictions):
        voted_predictions = self.one_hot_transform(self.one_hot, predictions)
        # predictions = np.array(predictions)
        # voted_predictions = []
        # for i in range(len(predictions[0])):
        #     all_y = predictions[:, i]
        #     counts = np.bincount(all_y)
        #     y = np.argmax(counts)
        #     voted_predictions.append(y)
        # np.savetxt("predict.csv", voted_predictions, delimiter=",", fmt='%i')
        return voted_predictions

    def augment(self, data, labels, min_required_count = 0):
        if self.aug[0] == False:
            return data,  self.one_hot_transform(self.one_hot, labels)

        # data_generator = ImageDataGenerator(rotation_range=20,  brightness_range=[0.6, 1.4], horizontal_flip=True, fill_mode='reflect', zoom_range=[0.6, 1], dtype='float32', rescale=1./255)
        data_generator = ImageDataGenerator(brightness_range=[0.6, 1.4], horizontal_flip=True, zoom_range=[0.6, 1], dtype='float32', rescale=1. / 255)

        aug_data = data
        num_classes = np.max(labels) + 1
        labels = np.array(labels)
        aug_labels = np.array(labels)
        class_counts = Counter(labels)
        required_count = max(int(np.max(list(class_counts.values()))), min_required_count)
        max_count = required_count
        min_count = int(np.min(list(class_counts.values())))
        if required_count == min_count: return data,  self.one_hot_transform(self.one_hot, labels)
        for _ in tqdm(range(self.aug[1])):
            if (max_count/min_count < 1.2) and min_count > required_count: break
            if all(count == required_count for count in class_counts): break
            data_gen = data[:1]
            label_gen = labels[:1]
            for label in range(num_classes):
                if class_counts[label] < required_count:
                    old_images = data[labels == label]
                    if old_images.shape[0] > 0:
                        data_gen = np.concatenate([data_gen, old_images])
                        label_gen = np.concatenate([label_gen, [label] * old_images.shape[0]])
            for x_batch, y_batch in data_generator.flow(data_gen, label_gen, batch_size=np.array(label_gen).shape[0]):
                for label in range(num_classes):
                    if class_counts[label] < required_count and float(class_counts[label]/min_count) < 1.5:
                        required = required_count - class_counts[label]
                        new_images = x_batch[y_batch==label]
                        selected_indices = np.random.choice(new_images.shape[0], size=min(required, new_images.shape[0]), replace=False)
                        selected_images = new_images[selected_indices]
                        aug_data = np.concatenate([aug_data, selected_images])
                        aug_labels = np.concatenate([aug_labels, [label] * selected_images.shape[0]])
                        class_counts[label] += selected_images.shape[0]
                        min_count = int(np.min(list(class_counts.values())))
                        max_count = int(np.max(list(class_counts.values())))
                print(class_counts)
                # for i in range(10):
                #     plt.subplot(2, 1, 1)
                #     plt.imshow(data[i].astype('uint8'))
                #     plt.subplot(2, 1, 2)
                #     plt.imshow(x_batch[i].astype('uint8'))
                #     plt.show()
                break

        return aug_data, self.one_hot_transform(self.one_hot, aug_labels)

    def binarize_labels(self, X, Y, negative_classes, keep_classes):
        new_X = []
        new_Y = []
        for i in range(Y.shape[0]):
            if Y[i] in keep_classes:
                new_X.append(X[i])
                if Y[i] in negative_classes:
                    new_Y.append(0)
                else:
                    new_Y.append(1)

        return np.array(new_X), np.array(new_Y)

    def aug_by_crop(self, X, Y, dim, keep_counts, test=False):

        labels = []
        label_count = {}
        data = []
        original_dim_x = X[0].shape[1]
        original_dim_y = X[0].shape[0]
        mid_start_x = int((original_dim_x - dim[0])/2)
        mid_start_y = int((original_dim_y - dim[1]) / 2)
        for label in range(len(keep_counts)):
            indices_of_label = np.where(Y==label)
            data_label = []
            for x in X[indices_of_label]:
                data_label.append(x[:dim[1], :dim[0]])
                data_label.append(x[-dim[1]:, -dim[0]:])
                data_label.append(x[-dim[1]:, :dim[0]])
                data_label.append(x[:dim[1], -dim[0]:])
                data_label.append(x[mid_start_y:mid_start_y+dim[1], mid_start_x:mid_start_x+dim[0]])
            y_labels = [label] * len(data_label)
            label_count[label] = len(y_labels)
            if keep_counts[label] == -1:
                keep_counts[label] = len(data_label)
            if test == False:
                data.extend(np.array(data_label)[np.random.choice(len(data_label), size=int(keep_counts[label]), replace=False)])
                labels.extend([label] * keep_counts[label])
            else:
                data.extend(data_label)
                labels.extend([label] * len(data_label))
            del data_label

        print("Augmented to: " + str(label_count))
        if test == False:
            shuffle_zip = list(zip(data, labels))
            random.shuffle(shuffle_zip)
            data, labels = zip(*shuffle_zip)
        del X
        del Y

        print("Kept (randomly selected): " + str(Counter(labels)))
        return np.array(data), np.array(labels)

    @staticmethod
    def crop_test(X, dim):
        data = []
        for x in X:
            data.append(x[:dim[1], :dim[0]])
            data.append(x[-dim[1]:, -dim[0]:])
            data.append(x[-dim[1]:, :dim[0]])
            data.append(x[:dim[1], -dim[0]:])
            data.append(cv2.resize(x, (dim[0], dim[1]), cv2.INTER_CUBIC))
        return np.array(data)

    @staticmethod
    def predict_cropped_test(pred):
        results = []
        for i in range(0, len(pred), 4):
            votes = pred[i:i + 4]
            count = Counter(votes)
            majority = np.argmax(votes)
            maj_count = Counter(count.values())
            if 2 in maj_count.keys():
                print("confused on image: " + str(int(i / 4)) + ", votes:" + str(votes))
            results.append(int(majority))
        return results
