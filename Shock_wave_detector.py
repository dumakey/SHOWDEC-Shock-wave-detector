import os
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
import tensorflow as tf
import cv2 as cv
import re
from shutil import rmtree

import DL_models as models
from Preprocessing import ImageTransformer
import AugmentationDataset as ADS
import reader


class ShockWaveScanner:

    def __init__(self, launch_file, check_model=False):

        class parameter_container:
            pass
        class dataset_container:
            pass
        class model_container:
            pass
        class predictions_container:
            pass

        self.parameters = parameter_container()
        self.datasets = dataset_container()
        self.model = model_container()
        self.predictions = predictions_container()

        # Setup general parameters
        casedata = reader.read_case_setup(launch_file)
        self.parameters.analysis = casedata.analysis
        self.parameters.training_parameters = casedata.training_parameters
        self.parameters.img_processing = casedata.img_processing
        self.parameters.data_augmentation = casedata.data_augmentation
        self.dataset_dir = casedata.case_dir

        # Sensitivity analysis variable identification
        sens_vars = [parameter for parameter in self.parameters.training_parameters.items() if type(parameter[1]) == list]
        if len(sens_vars) != 0:
            self.parameters.sens_variable = sens_vars[0]
        else:
            self.parameters.sens_variable = None

        # Check for model reconstruction
        if check_model:
            self.model.imported = True
            self.reconstruct_model()
        else:
            self.model.imported = False

    def __str__(self):
        class_name = type(self).__name__

        return '{}, a class to detect shockwaves on images based on Deep learning algorithms'.format(class_name)

    def launch_analysis(self):

        analysis_ID = self.parameters.analysis['type']
        analysis_list = {
                        'SINGLETRAINING': self.singletraining,
                        'SENSANALYSIS': self.sensitivity_analysis_on_training,
                        'DATAGEN': self.data_generation,
                        }
        arguments_list = {
                        'SINGLETRAINING': [bool(self.parameters.training_parameters['addaugdata'][0]),
                                           self.parameters.training_parameters['addaugdata'][1]],
                        'SENSANALYSIS': [bool(self.parameters.training_parameters['addaugdata'][0]),
                                         self.parameters.training_parameters['addaugdata'][1]],
                        'DATAGEN': [],
                        }
        for analysis in analysis_list.keys():
            F = analysis_list[analysis_ID]
            fun_args = arguments_list[analysis_ID]
            F(*fun_args)

    def sensitivity_analysis_on_training(self, add_augmented=False, augdataset_ID=1):

        # Retrieve sensitivity variable
        sens_variable = self.parameters.sens_variable

        # Perform sensitivity analysis
        self.set_datasets(add_augmented,augdataset_ID)
        X, y = sw_scanner.read_dataset()
        self.set_tensorflow_datasets()
        self.train_scanner_model(sens_variable)
        self.export_model_performance(sens_variable)
        self.export_model(sens_variable)

    def singletraining(self, add_augmented=False, augdataset_ID=1):

        self.set_datasets(add_augmented,augdataset_ID)
        self.set_tensorflow_datasets()
        self.train_scanner_model()
        self.export_model_performance()
        self.export_model()
        self.predict_on_test_set()

    def data_generation(self):

        transformations = [item[0] for item in self.parameters.img_processing if item[0] == 1]
        augdata_size = self.parameters.data_augmentation[1]
        self.generate_augmented_data(transformations,augdata_size)

    def label_image(self, frame_name):

        if 'SW=0' in os.path.splitext(frame_name)[0]:
            return 0
        elif 'SW=1' in os.path.splitext(frame_name)[0]:
            return 1

    def preprocess_data(self, im, label):
        im = tf.cast(im, tf.float32)
        im = im / 127.5
        im = im - 1

        return im, label

    def create_dataset_pipeline(self, dataset, is_train=True, num_threads=8, prefetch_buffer=100, batch_size=32):
        X, y = dataset
        y_oh = tf.one_hot(y,depth=1)
        dataset_tensor = tf.data.Dataset.from_tensor_slices((X, y_oh))

        if is_train:
            dataset_tensor = dataset_tensor.shuffle(buffer_size=X.shape[0]).repeat()
        dataset_tensor = dataset_tensor.map(self.preprocess_data, num_parallel_calls=num_threads)
        dataset_tensor = dataset_tensor.batch(batch_size)
        dataset_tensor = dataset_tensor.prefetch(prefetch_buffer)

        return dataset_tensor

    def read_dataset(self, scan=False):

        if scan:
            samples_path = []
            for (root, case_dirs, _) in os.walk(self.dataset_dir):
                for case_dir in case_dirs:
                    for (case_root, aoa_dirs, _) in os.walk(os.path.join(root,case_dir)):
                        for aoa_dir in aoa_dirs:
                            for (aoa_root, mach_dirs, _) in os.walk(os.path.join(case_root,aoa_dir)):
                                for mach_dir in mach_dirs:
                                    for (mach_root, dirs, _) in os.walk(os.path.join(aoa_root,mach_dir)):
                                        if dirs:
                                            slices_dir = dirs[0]
                                            samples = [os.path.join(mach_root,slices_dir,file) for file in
                                                       os.listdir(os.path.join(root,case_root,aoa_root,mach_root,slices_dir))]
                                            for sample in samples:
                                                samples_path.append(sample)
        else:  # If samples are directly storaged in folder
            samples_path = []
            for (root, case_dirs, _) in os.walk(self.dataset_dir):
                for case_dir in case_dirs:
                    samples = os.listdir(os.path.join(root,case_dir))
                    for sample in samples:
                        samples_path.append(os.path.join(root,case_dir,sample))

        # Generate X,y datasets
        m = len(samples_path)
        slice_dimensions = self.parameters.img_processing['slice_size']
        X = np.zeros((m,slice_dimensions[1],slice_dimensions[0],3),dtype='uint8')
        y = np.zeros((m,),dtype=int)
        for i,sample in enumerate(samples_path):
            # X-array storage
            img = cv.imread(sample)
            X[i,:,:,:] = ImageTransformer.resize(img,slice_dimensions)
            # y-label storage
            y[i] = self.label_image(os.path.basename(sample))

        return X, y

    def read_augmented_datasets(self, augdataset_ID=None):

        dataset_dir = os.path.join(os.path.dirname(self.dataset_dir),'Datasets_augmented','Dataset_{}'.format(augdataset_ID))
        if os.path.exists(dataset_dir):
            files = [os.path.join(dataset_dir,file) for file in os.listdir(dataset_dir)]
            X = []
            y = np.zeros((len(files),),dtype=int)
            for i,file in enumerate(files):
                X.append(cv.imread(file))
                y[i] = self.label_image(file)
            X = np.array(X)
        else:
            X = None
            y = None

        return X, y

    def standardize_image_size(self, X):

        img_dimensions = self.parameters.img_processing['slice_size']
        m = X.shape[0]
        if X[0].shape[0:2] != (img_dimensions[1],img_dimensions[0]):
            X_resized = np.zeros((m,img_dimensions[1],img_dimensions[0],3),dtype='uint8')
            for i in range(m):
                X_resized[i] = ImageTransformer.resize(X[i],img_dimensions)

            return X_resized
        else:
            return X

    def set_datasets(self, add_augmented=False, augdataset_ID=1):

        # Read original datasets
        X_orig, y_orig = self.read_dataset()
        # Resize images, if necessary
        X_orig = self.standardize_image_size(X_orig)

        if add_augmented:
            # Check for augmented datasets
            X_aug, y_aug = self.read_augmented_datasets(augdataset_ID)
            X_aug = self.standardize_image_size(X_aug)

            try:
                _ = X_aug.shape

                # Join both datasets
                X = np.concatenate((X_orig,X_aug),axis=0)
                y = np.concatenate((y_orig,y_aug),axis=0)
            except:
                X, y = X_orig, y_orig
        else:
            X, y = X_orig, y_orig

        X_train, X_val, y_train, y_val = train_test_split(X,y,train_size=self.parameters.training_parameters['train_size'],shuffle=True)
        X_cv, X_test, y_cv, y_test = train_test_split(X_val,y_val,train_size=0.75,shuffle=True)

        self.datasets.data_train = (X_train, y_train)
        self.datasets.data_cv = (X_cv, y_cv)
        self.datasets.data_test = (X_test, y_test)

    def set_tensorflow_datasets(self):

        self.datasets.dataset_train = self.create_dataset_pipeline(self.datasets.data_train,is_train=True)
        self.datasets.dataset_cv = self.create_dataset_pipeline(self.datasets.data_cv,is_train=False,batch_size=1)
        self.datasets.dataset_test = self.preprocess_data(self.datasets.data_test[0],tf.one_hot(self.datasets.data_test[1],depth=1))

    def generate_augmented_data(self, transformations={}, augmented_dataset_size=1):

        if self.parameters.img_processing:
            transformations = self.parameters.img_processing

        # Set storage folder for augmented dataset
        initial_dataset_folder = os.path.basename(self.dataset_dir)
        initial_dataset_root = os.path.dirname(self.dataset_dir)
        augmented_dataset_dir = os.path.join(initial_dataset_root,initial_dataset_folder + '_augmented')

        # Unpack data
        X, y = sw_scanner.read_dataset()
        # Generate new dataset
        data_augmenter = ADS.datasetAugmentationClass(X,y,transformations,augmented_dataset_size,augmented_dataset_dir)
        data_augmenter.transform_images()
        data_augmenter.export_augmented_dataset()

    def train_scanner_model(self, sens_var=None):

        # Parameters
        alpha = self.parameters.training_parameters['learning_rate']
        nepoch = self.parameters.training_parameters['epochs']
        batch_size = self.parameters.training_parameters['batch_size']
        l2_reg = self.parameters.training_parameters['l2_reg']
        dropout = self.parameters.training_parameters['dropout']
        image_shape = self.datasets.dataset_train.element_spec[0].shape[1:3]
        if self.model.imported == False:
            #Model = models.slice_scanner_lenet_model
            Model = models.slice_scanner_inception_model

        if sens_var == None:  # If it is a one-time training
            if self.model.imported == False:
                self.model.Model = Model(image_shape,l2_reg,alpha,dropout)
            self.model.History = self.model.Model.fit(self.datasets.dataset_train,epochs=nepoch,batch_size=batch_size,
                                                      steps_per_epoch=500,validation_data=self.datasets.dataset_cv,
                                                      validation_steps=None)
        else: # If it is a sensitivity analysis
            self.model.Model = []
            self.model.History = []
            if type(alpha) == list:
                for learning_rate in alpha:
                    if self.model.imported == False:
                        model = Model(image_shape,l2_reg,learning_rate,dropout)
                    self.model.Model.append(model)
                    self.model.History.append(model.fit(self.datasets.dataset_train,epochs=nepoch,batch_size=batch_size,
                                                        steps_per_epoch=500,validation_data=self.datasets.dataset_cv,
                                                        validation_steps=None))
            elif type(l2_reg) == list:
                for regularizer in l2_reg:
                    if self.model.imported == False:
                        model = Model(image_shape,regularizer,alpha,dropout)
                    self.model.Model.append(model)
                    self.model.History.append(model.fit(self.datasets.dataset_train,epochs=nepoch,batch_size=batch_size,
                                                        steps_per_epoch=500,validation_data=self.datasets.dataset_cv,
                                                        validation_steps=None))
            elif type(dropout) == list:
                for rate in dropout:
                    if self.model.imported == False:
                        model = Model(image_shape,l2_reg,alpha,rate)
                    self.model.Model.append(model)
                    self.model.History.append(model.fit(self.datasets.dataset_train,epochs=nepoch,batch_size=batch_size,
                                                        steps_per_epoch=500,validation_data=self.datasets.dataset_cv,
                                                        validation_steps=None))

    def predict_on_test_set(self, threshold=0.5):

        X_test, y_test = self.datasets.dataset_test
        logits = self.model.Model.predict(X_test)
        m_test = logits.shape[0]
        y_hat = np.array([1 if logit > threshold else 0 for logit in logits])

        metrics_functions = {
            'accuracy': tf.keras.metrics.BinaryAccuracy(),
            'recall': tf.keras.metrics.Recall(),
            'precision': tf.keras.metrics.Precision(),
            'AUC': tf.keras.metrics.AUC(),
        }

        metrics = dict.fromkeys(metrics_functions)

        for key in metrics.keys():
            metric_function = metrics_functions[key]
            metric_function.update_state(dataset_test[1],logits)
            metrics[key] = metric_function.result().numpy()

        self.predictions.predictions = y_hat
        self.predictions.score = metrics

        print()

    def export_model_performance(self, sens_var=None):

        try:
            History = self.model.History
        except:
            raise Exception('There is no evolution data for this model. Train model first.')
        else:
            if type(History) == list:
                N = len(History)
            else:
                N = 1
                History = [History]

            # Loss evolution plots #
            Nepochs = self.parameters.training_parameters['epochs']
            epochs = np.arange(1,Nepochs+1,1)

            for i,h in enumerate(History):
                loss_train = h.history['loss']
                loss_cv = h.history['val_loss']

                fig, ax = plt.subplots(1)
                ax.plot(epochs,loss_train,label='Training',color='r')
                ax.plot(epochs,loss_cv,label='Cross-validation',color='b')
                ax.grid()
                ax.set_xlabel('Epochs',size=12)
                ax.set_ylabel('Loss',size=12)
                ax.tick_params('both',labelsize=10)
                ax.legend()

                if sens_var:
                    storage_dir = os.path.join(os.path.dirname(self.dataset_dir),'Model_performance','{}={:.3f}'.format(
                                               sens_var[0],sens_var[1][i]))
                else:
                    storage_dir = os.path.join(os.path.dirname(self.dataset_dir),'Model_performance')
                if os.path.exists(storage_dir):
                    rmtree(storage_dir)
                os.makedirs(storage_dir)
                fig.savefig(os.path.join(storage_dir,'Loss_evolution.png'),dpi=200)

                # Metrics #
                metrics_name = [item for item in h.history if item not in ('loss','val_loss')]
                metrics_val = [(metric,h.history[metric][0]) for metric in metrics_name if metric.startswith('val')]
                metrics_train = [(metric,h.history[metric][0]) for metric in metrics_name if not metric.startswith('val')]

                rows = [metric[0] for metric in metrics_train]
                metric_fun = lambda L: np.array([item[1] for item in L])
                metrics_data = np.vstack((metric_fun(metrics_train),metric_fun(metrics_val))).T
                metrics = pd.DataFrame(index=rows,columns=['Training','CV'],data=metrics_data)
                metrics.to_csv(os.path.join(storage_dir,'Model_metrics.csv'),sep=';',decimal='.')

                # Loss
                loss_data = np.vstack((list(epochs), loss_train, loss_cv)).T
                loss = pd.DataFrame(columns=['Epoch', 'Training', 'CV'], data=loss_data)
                loss.to_csv(os.path.join(storage_dir, 'Model_loss.csv'), index=False, sep=';', decimal='.')

    def export_model(self, sens_var=None):

        if type(self.model.Model) == list:
            N = len(sens_var[1])
            model = self.model.Model
        else:
            N = 1
            model = [self.model.Model]

        for i in range(N):
            if sens_var:
                weights_dir = os.path.join(os.path.dirname(self.dataset_dir),'Model','{}={:.3f}'.format(sens_var[0],sens_var[1][i]))
            else:
                weights_dir = os.path.join(os.path.dirname(self.dataset_dir),'Model')
            if os.path.exists(weights_dir):
                rmtree(weights_dir)
            os.makedirs(weights_dir)

            # Export model arquitecture to JSON file
            model_json = model[i].to_json()
            with open(os.path.join(weights_dir,'SW_model_arquitecture.json'),'w') as json_file:
                json_file.write(model_json)

            # Export model weights to HDF5 file
            model[i].save_weights(os.path.join(weights_dir,'SW_model_weights.h5'))

    def reconstruct_model(self):

        weights_dir = os.path.join(os.path.dirname(self.dataset_dir),'Model')
        # Load JSON file
        json_file = open(os.path.join(weights_dir,'SW_model_arquitecture.json'),'r')
        loaded_model_json = json_file.read()
        json_file.close()

        # Build model
        self.model.Model = tf.keras.models.model_from_json(loaded_model_json)
        # Load weights into new model
        self.model.Model.load_weights(os.path.join(weights_dir,'SW_model_weights.h5'))


if __name__ == '__main__':
    launcher = r'C:\Users\juan.ramos\Shock_wave_detector\Scripts\launcher.dat'
    sw_scanner = ShockWaveScanner(launcher,check_model=True)
    sw_scanner.launch_analysis()
    print()