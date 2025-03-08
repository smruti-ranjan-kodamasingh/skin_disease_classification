import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
import cv2
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from skin_disease_classifier.entity.config_entity import TrainingConfig
from pathlib import Path
                                                          

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
    
    def train_valid_generator(self):

        data_dir = self.config.training_data
        seed = 123

        # Splitting data and creating training and validation datasets
        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.1,
            subset="training",
            seed=seed,
            image_size=(self.config.params_image_size[0], self.config.params_image_size[1]),
            batch_size=self.config.params_batch_size
        )

        self.validation_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.1,
            subset="validation",
            seed=seed,
            image_size=(self.config.params_image_size[0], self.config.params_image_size[1]),
            batch_size=self.config.params_batch_size
        )

        # Normalizing Pixel Values for Training and Validation Datasets
        self.train_ds = self.train_ds.map(lambda x, y: (x / 255.0, y))
        self.validation_ds = self.validation_ds.map(lambda x, y: (x / 255.0, y))

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)


    def train(self):
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            verbose=1,
            mode='max',
            restore_best_weights=True,
            baseline=0.98
        )

        self.model.fit(
            self.train_ds,
            epochs=self.config.params_epochs,
            validation_data=self.validation_ds,
            callbacks=[early_stopping]
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )