import collections
import random
from skimage.filters import gaussian, unsharp_mask
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import clone_model
from keras_preprocessing.image import ImageDataGenerator
from sklearn import metrics
from collections import Counter
from model_abstract import Model_Abstract
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
import tensorflow as tf
import datetime
class Keras_model(Model_Abstract):
    def __init__(self, helper, model, batch_size = 128,  verbose = 2, epochs = 50, validation_split = 0.2, loss='categorical_crossentropy', optimizer = None, weighted_metrics = None, early_stopping=False, early_stopping_patience=5, early_stopping_metric='val_loss', k_fold_splits = 5, is_cnn = False, name='Keras Model', metrics=None, show_summary=False, use_data_gen=False):
        super().__init__(helper, model, name)
        self.batch_size = batch_size
        self.verbose = verbose
        self.show_summary = show_summary
        self.validation_split = validation_split
        self.loss = loss
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping = early_stopping
        self.early_stopping_metric = early_stopping_metric
        self.k_fold_splits = k_fold_splits
        self.is_cnn = is_cnn
        self.use_data_gen = use_data_gen
        if(optimizer is None):
            self.optimizer = optimizers.Adam(learning_rate=0.001)
        else:
            self.optimizer = optimizer

        if (metrics is None):
            self.weighted_metrics = ['accuracy']
        else:
            self.weighted_metrics = weighted_metrics
        self.metrics = metrics
        if self.is_cnn:
            self.train_data_generator = ImageDataGenerator(fill_mode='reflect', rotation_range=15, channel_shift_range=0.2, brightness_range=[0.5, 1.3], horizontal_flip=True, vertical_flip=True, zoom_range=[0.5, 1], dtype='float32', rescale=1./255, samplewise_center=True, samplewise_std_normalization=True, zca_whitening=False)
            self.val_data_generator = ImageDataGenerator(dtype='float32', samplewise_center=True, samplewise_std_normalization=True, zca_whitening=False)

        # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        self.es = EarlyStopping(monitor=self.early_stopping_metric, verbose=self.verbose,
                           patience=self.early_stopping_patience, restore_best_weights=True)
        self.callbacks = [self.es] if self.early_stopping else []
        self.trained = False
        self.multi_channel = self.helper.multi_channel

    def __del__(self):
        print("deleted")

    def combined(self, X, X_unaugmented, Y, Y_unaugmented, aug_gen, train):


        original_size = Y_unaugmented.shape[0]
        un_augmented_labels = np.argmax(Y_unaugmented, axis=1)
        num_classes = int(np.max(un_augmented_labels) + 1)
        class_counts = Counter(list(un_augmented_labels))
        required_count = int(np.max(list(class_counts.values())))
        augment_by = []
        for i in range(num_classes):
            required = required_count - class_counts[i] + 1
            replace = False
            if required > class_counts[i]:
                    replace = True
            augment_by.append([required, replace])
        class_indices = [np.where(un_augmented_labels == label)[0] for label in range(num_classes)]
        selected_indices = [np.random.choice(class_indices[label], size=int(augment_by[label][0]), replace=augment_by[label][1]) for label in range(num_classes)]
        for i in range(num_classes):
            if augment_by[i][0] > 0:
                new_data_x = X_unaugmented[selected_indices[i]]
                new_data_y = Y_unaugmented[selected_indices[i]]
                X_unaugmented = np.concatenate([X_unaugmented, new_data_x])
                Y_unaugmented = np.concatenate([Y_unaugmented, new_data_y])

        selected_indices = np.random.choice(Y_unaugmented.shape[0], size=original_size, replace=False)
        blur_indices = selected_indices[:int(original_size/20)]
        sharp_indices = selected_indices[int(original_size / 20):int(2 * original_size / 20)]
        selected_indices = selected_indices[int(2 * original_size/20):]
        X_augmented = X_unaugmented[selected_indices]
        Y_augmented = Y_unaugmented[selected_indices]
        blured_images = []
        blured_labels = []
        for ind in blur_indices:
            sigma = np.random.normal(0.8, 0.15)
            filtered_img = gaussian(X_unaugmented[ind], sigma=sigma, multichannel=self.multi_channel)
            blured_images.append(filtered_img)
            blured_labels.append(Y_unaugmented[ind])
        X_augmented = np.concatenate([X_augmented, blured_images])
        Y_augmented = np.concatenate([Y_augmented, blured_labels])

        sharp_images = []
        sharp_labels = []
        for ind in sharp_indices:
            sigma = np.random.normal(0.7, 0.2)
            sharp = unsharp_mask(X_unaugmented[ind], radius=sigma, amount=1, multichannel=self.multi_channel)
            sharp_images.append(sharp)
            sharp_labels.append(Y_unaugmented[ind])
        X_augmented = np.concatenate([X_augmented, blured_images])
        Y_augmented = np.concatenate([Y_augmented, blured_labels])
        print("Augmenting partailly to: ("+ (str(train)) +") " + str(Counter(np.argmax(Y_augmented, axis=1))))
        shuffle_zip_X = list(zip(X, Y))
        random.shuffle(shuffle_zip_X)
        X, Y = zip(*shuffle_zip_X)
        X, Y = np.array(X), np.array(Y)
        shuffle_zip_X_aug = list(zip(X_augmented, Y_augmented))
        random.shuffle(shuffle_zip_X_aug)
        X_augmented, Y_augmented = zip(*shuffle_zip_X_aug)
        X_augmented, Y_augmented = np.array(X_augmented), np.array(Y_augmented)

        gen = ImageDataGenerator(dtype='float32')
        if train == True:
            gen = ImageDataGenerator(fill_mode='reflect', rotation_range=15, channel_shift_range=0.2, brightness_range=[0.5, 1.2], horizontal_flip=True, vertical_flip=True, zoom_range=[0.5, 1], dtype='float32', rescale=1./255)
            genX2 = aug_gen.flow(X_augmented, Y_augmented, batch_size=self.batch_size, shuffle=True)
        genX1 = gen.flow(X, Y, batch_size=self.batch_size, shuffle=True)


        del X_unaugmented
        del Y_unaugmented
        del X_augmented
        del Y_augmented
        del X
        del Y

        i = 0
        c = 3

        # gc.collect()
        # print("collected")
        random_list = [1, 2]
        if train == True:
            c = 2
            random_list = [1, 2, 3, 4, 6]

        while True:
            i = np.random.choice(random_list, size=1)[0]

            if i % c == 0 and train == True:
                X2 = genX2.next()
                yield X2[0], X2[1]
                del X2
            elif train ==  True and i % (c + 1) == 0:
                X2 = genX2.next()
                yield X2[0], X2[1]

                del X2
            else:
                X1 = genX1.next()
                yield X1[0], X1[1]

                del X1






    def fit(self, X, Y, crop_augment=None):

        class_counts = Counter(Y)
        class_weights = {}
        sub_total = len(list(Y))/2
        for i in range(5):
            class_weights[i] = sub_total / class_counts[i]
        print("Final weights = " + str(class_weights) )


        self.model.compile(loss=self.loss, optimizer=self.optimizer, weighted_metrics=self.weighted_metrics, metrics=self.metrics, experimental_run_tf_function=False)
        if self.show_summary: self.model.summary()
        if self.trained == False:
            Y_one_hot = self.helper.one_hot_transform(self.helper.one_hot, Y)
            self.X_train, self.X_val, self.Y_train, self.Y_val = self.helper.train_val_split(X, Y_one_hot, self.validation_split)
            if crop_augment is not None:
                self.X_train, self.Y_train = self.helper.aug_by_crop(self.X_train, np.argmax(self.Y_train, axis=1),
                                                                     dim=crop_augment[0], keep_counts=crop_augment[1])
                self.X_val, self.Y_val = self.helper.aug_by_crop(self.X_val, np.argmax(self.Y_val, axis=1),
                                                                 dim=crop_augment[0], keep_counts=crop_augment[2], test=True)

        else:

            del X
            del Y

            print("Class frequency in training set: " + str(collections.Counter(np.argmax(self.Y_train, axis=1))))
            print("Class frequency in validation set: " + str(collections.Counter(np.argmax(self.Y_val, axis=1))))
        self.Y_train, self.Y_val = self.helper.one_hot_transform(self.helper.one_hot, self.Y_train), self.helper.one_hot_transform(self.helper.one_hot, self.Y_val)
        X_train_unaugmented, X_val_unaugmented, Y_train_unaugmented, Y_val_unaugmented = self.X_train.copy(), self.X_val.copy(), self.Y_train.copy(), self.Y_val.copy()

        if (self.is_cnn ) or self.use_data_gen:
            if crop_augment is None:
                train_generator = self.combined(self.X_train, X_train_unaugmented, self.Y_train, Y_train_unaugmented, self.train_data_generator, train=True)
                val_generator = self.combined(self.X_val, X_val_unaugmented,  self.Y_val, Y_val_unaugmented, self.train_data_generator, train=False)
            else:
                # self.add_blur_sharp()
                print(self.X_train.shape)
                self.train_data_generator.fit(self.X_train)
                self.val_data_generator.fit(self.X_val)
                train_generator = self.train_data_generator.flow(self.X_train, self.Y_train, batch_size=self.batch_size, shuffle=True)
                val_generator = self.val_data_generator.flow(self.X_val, self.Y_val, batch_size=self.batch_size, shuffle=True)

            del self.helper
            self.history = self.model.fit(train_generator,
                                     verbose=self.verbose,
                                     steps_per_epoch=len(self.X_train) / self.batch_size,
                                     epochs=self.epochs,
                                     validation_data=val_generator,
                                     validation_steps=len(self.X_val) / self.batch_size,
                                     callbacks=self.callbacks,
                                     class_weight=class_weights,
                                     shuffle=True)
        else:
            del self.helper
            self.history = self.model.fit(self.X_train, self.Y_train, batch_size=self.batch_size, verbose=self.verbose, epochs=self.epochs, validation_data=(self.X_val, self.Y_val), callbacks=self.callbacks, class_weight=class_weights, shuffle=True)

        return self.X_val, self.Y_val

    def evaluate_model(self, X, Y):
        Y_one_hot = self.helper.one_hot_transform(self.helper.one_hot, Y)
        balanced_k_fold = self.helper.balanced_k_fold_splits(Y, self.k_fold_splits)
        skf = StratifiedKFold(n_splits=self.k_fold_splits, shuffle=False)
        skf.get_n_splits(X, Y)
        kf = KFold(n_splits=self.k_fold_splits, shuffle=False)
        kf.get_n_splits(X, Y)
        k_val_accuracy = []
        k_train_accuracy = []

        print("############### Running validation with Balanced K folds  ###############")
        cross_train_total, cross_val_total = self.evaluate_folds(X, Y_one_hot, self.helper.class_weights, balanced_k_fold)
        k_val_accuracy.append(cross_val_total / (self.k_fold_splits))
        k_train_accuracy.append(cross_train_total / (self.k_fold_splits))
        print("With Balanced folds: K fold validation accuracy = " + str(cross_val_total / (self.k_fold_splits)) + "%")
        print("With Balanced folds: K fold training accuracy = " + str(cross_train_total / (self.k_fold_splits)) + "%")

        print("############### Running validation with Stratified K folds  ###############")
        cross_train_total, cross_val_total = self.evaluate_folds(X, Y_one_hot, self.helper.class_weights, skf.split(X, Y))
        k_val_accuracy.append(cross_val_total / (self.k_fold_splits))
        k_train_accuracy.append(cross_train_total / (self.k_fold_splits))
        print(
            "With Stratified folds: K fold validation accuracy = " + str(cross_val_total / (self.k_fold_splits)) + "%")
        print(
            "With Stratified folds: K fold training accuracy = " + str(cross_train_total / (self.k_fold_splits)) + "%")

        print("############### Running validation with K folds  ###############")
        cross_train_total, cross_val_total = self.evaluate_folds(X, Y_one_hot, self.helper.class_weights, kf.split(X, Y))
        k_val_accuracy.append(cross_val_total / (self.k_fold_splits))
        k_train_accuracy.append(cross_train_total / (self.k_fold_splits))
        print("With K folds: K fold validation accuracy = " + str(cross_val_total / (self.k_fold_splits)) + "%")
        print("With K folds: K fold training accuracy = " + str(cross_train_total / (self.k_fold_splits)) + "%")

        self.k_val_accuracy = min(k_val_accuracy)
        self.k_train_accuracy = min(k_train_accuracy)
        print("***********************************************************************")
        print("Worst: K fold validation accuracy = " + str(self.k_val_accuracy) + "%")
        print("Worst: K fold training accuracy = " + str(self.k_val_accuracy) + "%")

    def evaluate_folds(self, X, Y_one_hot, class_weights, folds):
        val_accuracy_total = 0
        train_accuracy_total = 0
        k = 0
        for train_index, test_index in folds:
            print("################# Validation Step: " + str(k) + " #################")
            print("Intersection between validation and train set = " + str(
                set(train_index).intersection(set(test_index))))
            X_train, X_val, Y_train, Y_val = X[train_index], X[test_index], Y_one_hot[train_index], Y_one_hot[
                test_index]
            X_train_unaugmented, X_val_unaugmented, Y_train_unaugmented, Y_val_unaugmented = X_train.copy(), X_val.copy(), Y_train.copy(), Y_val.copy()
            print("Class frequency in training set: " + str(collections.Counter(np.argmax(Y_train, axis=1))))
            print("Class frequency in validation set: " + str(collections.Counter(np.argmax(Y_val, axis=1))))
            model = clone_model(self.model)
            model.compile(loss=self.loss, optimizer=self.optimizer, weighted_metrics=self.weighted_metrics, metrics=self.metrics)

            if (self.is_cnn and self.helper.aug[0] == False) or self.use_data_gen:

                train_generator = self.combined(X_train, X_train_unaugmented, Y_train, Y_train_unaugmented,
                                                self.train_data_generator, train=True)
                val_generator = self.combined(X_val, X_val_unaugmented, Y_val, Y_val_unaugmented,
                                              self.val_data_generator, train=False)
                hist = model.fit_generator(train_generator, verbose=1, epochs=self.epochs,
                                           validation_data=val_generator, callbacks=self.callbacks,
                                           steps_per_epoch=len(X_train) / self.batch_size, shuffle=True,
                                           class_weight=class_weights)
            else:
                hist = model.fit(X_train, Y_train, batch_size=self.batch_size, verbose=1, epochs=self.epochs,
                                 validation_data=(X_val, Y_val), callbacks=self.callbacks, class_weight=class_weights, shuffle=True)
            val_accuracy_total += float(hist.history['val_accuracy'][-1])
            train_accuracy_total += float(hist.history['accuracy'][-1])
            print("validation accuracy in fold: " + str(k) + " = " + str(hist.history['val_accuracy'][-1] * 100) + "%")
            k += 1
            print("############################################")
        return train_accuracy_total, val_accuracy_total

    def predict(self, X):
        pred = self.model.predict(X)
        return list(np.argmax(pred, axis=1))

    def print_classification_report(self, Y, Y_pred):
        print("Classification report for - \n{}:\n{}\n".format(self.model, metrics.classification_report(Y, Y_pred)))

    def add_blur_sharp(self):

        num_samples = self.Y_train.shape[0]
        indices = np.random.choice(num_samples, size=int(num_samples/50), replace=False)
        blur_indices = indices
        for ind in blur_indices:
            sigma = np.random.normal(0.9, 0.3)
            filtered_img = gaussian(self.X_train[ind], sigma=abs(sigma), multichannel=self.multi_channel)
            self.X_train = np.concatenate([self.X_train, [filtered_img]])
            self.Y_train = np.concatenate([self.Y_train, [self.Y_train[ind]]])

        print("Augmenting blur sharp partailly to: " + str(Counter(np.argmax(self.Y_train, axis=1))))





