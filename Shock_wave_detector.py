import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import OrderedDict
import tensorflow as tf
import cv2 as cv
import pickle
from shutil import rmtree, copytree
from random import randint

import Models
import Dataset_processing as Dataprocess
import AugmentationDataset as ADS
import reader
import Postprocessing

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
        self.parameters.img_size = casedata.img_resize
        self.parameters.data_augmentation = casedata.data_augmentation
        self.parameters.activation_plotting = casedata.activation_plotting
        self.parameters.prediction = casedata.prediction
        self.case_dir = casedata.case_dir

        # Sensitivity analysis variable identification
        sens_vars = [parameter for parameter in self.parameters.training_parameters.items()
                     if type(parameter[1]) == list
                     if parameter[0] != 'addaugdata']
        if len(sens_vars) != 0:
            self.parameters.sens_variable = sens_vars[0]
        else:
            self.parameters.sens_variable = None

        # Check for model reconstruction
        if self.parameters.analysis['import'] == True:
            self.model.imported = True
            self.model.Model, self.model.History = self.reconstruct_model()
        else:
            self.model.imported = False

    def __str__(self):
        class_name = type(self).__name__

        return '{}, a class to detect shockwaves on images based on Deep learning algorithms'.format(class_name)

    def launch_analysis(self):

        analysis_ID = self.parameters.analysis['type']
        analysis_list = {
                        'singletraining': self.singletraining,
                        'sensanalysis': self.sensitivity_analysis_on_training,
                        'trainpredict': self.trainpredict,
                        'datagen': self.data_generation,
                        'plotactivations': self.plot_activations,
                        'predict': self.predict_on_test_set,
                        }
        arguments_list = {
                        'singletraining': [bool(self.parameters.training_parameters['addaugdata'][0]),
                                           self.parameters.training_parameters['addaugdata'][1]],
                        'sensanalysis': [bool(self.parameters.training_parameters['addaugdata'][0]),
                                         self.parameters.training_parameters['addaugdata'][1]],
                        'trainpredict': [bool(self.parameters.training_parameters['addaugdata'][0]),
                                          self.parameters.training_parameters['addaugdata'][1]],
                        'datagen': [],
                        'plotactivations': [],
                        'predict': [],
                        }

        F = analysis_list[analysis_ID]
        fun_args = arguments_list[analysis_ID]
        F(*fun_args)

    def sensitivity_analysis_on_training(self, add_augmented=False, augdataset_ID=1):

        case_dir = self.case_dir
        img_dims = self.parameters.img_size
        batch_size = self.parameters.training_parameters['batch_size']
        train_size = self.parameters.training_parameters['train_size']

        # Retrieve sensitivity variable
        sens_variable = self.parameters.sens_variable

        # Perform sensitivity analysis
        self.datasets.data_train, self.datasets.data_cv, self.datasets.data_test = \
        Dataprocess.get_datasets(case_dir,img_dims,train_size,add_augmented,augdataset_ID)
        
        self.datasets.dataset_train, self.datasets.dataset_cv, self.datasets.dataset_test = \
        Dataprocess.get_tensorflow_datasets(self.datasets.data_train,self.datasets.data_cv,self.datasets.data_test,batch_size)
        if self.model.imported == False:
            self.train_model(sens_variable)
        self.export_model_performance(sens_variable)
        self.export_model(sens_variable)
        self.export_nn_log()

    def singletraining(self, add_augmented=False, augdataset_ID=1):

        case_dir = self.case_dir
        img_dims = self.parameters.img_size
        train_size = self.parameters.training_parameters['train_size']
        batch_size = self.parameters.training_parameters['batch_size']

        self.datasets.data_train, self.datasets.data_cv, self.datasets.data_test = \
        Dataprocess.get_datasets(case_dir,img_dims,train_size,add_augmented,augdataset_ID)
        self.datasets.dataset_train, self.datasets.dataset_cv, self.datasets.dataset_test = \
        Dataprocess.get_tensorflow_datasets(self.datasets.data_train,self.datasets.data_cv,self.datasets.data_test,batch_size)
        if self.model.imported == False:
            self.train_model()
        self.export_model_performance()
        self.export_model()
        self.export_nn_log()
        
    def trainpredict(self, add_augmented=False, augdataset_ID=1):
        
        # Training
        case_dir = self.case_dir
        img_dims = self.parameters.img_size
        train_size = self.parameters.training_parameters['train_size']
        batch_size = self.parameters.training_parameters['batch_size']

        self.datasets.data_train, self.datasets.data_cv, self.datasets.data_test = \
        Dataprocess.get_datasets(case_dir,img_dims,train_size,add_augmented,augdataset_ID)
        self.datasets.dataset_train, self.datasets.dataset_cv, self.datasets.dataset_test = \
        Dataprocess.get_tensorflow_datasets(self.datasets.data_train,self.datasets.data_cv,self.datasets.data_test,batch_size)
        if self.model.imported == False:
            self.train_model()
        self.export_model_performance()
        self.export_model()
        self.export_nn_log()
        
        # Prediction
        model_dir = os.path.join(case_dir,'Results',str(self.parameters.analysis['case_ID']),'Model')
        generation_dir = os.path.join(case_dir,'Results','pretrained_model')
        if os.path.exists(generation_dir):
            rmtree(generation_dir)
        copytree(model_dir,generation_dir)
        self.model.imported = True
        self.predict_on_test_set()
        
        
    def data_generation(self):

        transformations = [{k:v[1:] for (k,v) in self.parameters.img_processing.items() if v[0] == 1}][0]
        augdata_size = self.parameters.data_augmentation[1]
        self.generate_augmented_data(transformations,augdata_size)

    def plot_activations(self):

        # Parameters
        case_dir = self.case_dir
        case_ID = self.parameters.analysis['case_ID']
        img_dims = self.parameters.img_size
        batch_size = self.parameters.training_parameters['batch_size']
        train_size = self.parameters.training_parameters['train_size']
        add_augmented = bool(self.parameters.training_parameters['addaugdata'][0])
        augdataset_ID = self.parameters.training_parameters['addaugdata'][1]
        n = self.parameters.activation_plotting['n_samples']
        figs_per_row = self.parameters.activation_plotting['n_cols']
        rows_to_cols_ratio = self.parameters.activation_plotting['rows2cols_ratio']
        storage_dir = os.path.join(self.case_dir,'Results','pretrained_model')
        
        # Generate datasets
        self.datasets.data_train, self.datasets.data_cv, self.datasets.data_test = \
            Dataprocess.get_datasets(case_dir,img_dims,train_size,add_augmented,augdataset_ID)
        self.datasets.dataset_train, self.datasets.dataset_cv, self.datasets.dataset_test = \
        Dataprocess.get_tensorflow_datasets(self.datasets.data_train,self.datasets.data_cv,self.datasets.data_test,batch_size)

        m_tr = self.datasets.data_train[0].shape[0]
        m_cv = self.datasets.data_cv[0].shape[0]
        m_ts = self.datasets.data_test[0].shape[0]
        m = m_tr + m_cv + m_ts

        # Read datasets
        dataset = np.zeros((m,img_dims[1],img_dims[0],3),dtype='uint8')
        dataset[:m_tr,:] = self.datasets.data_train[0]
        dataset[m_tr:m_tr+m_cv,:] = self.datasets.data_cv[0]
        dataset[m_tr+m_cv:m,:] = self.datasets.data_test[0]

        # Index image sampling
        idx = [randint(1,m) for i in range(n)]
        idx_set = set(idx)
        while len(idx) != len(idx_set):
            extra_item = randint(1,m)
            idx_set.add(extra_item)

        # Reconstruct model
        model, _ = self.reconstruct_model()
        #model = Model(image_shape,0.001,0.0,0.0,0.0,activation)
        # Load weights
        weights_filename = [file for file in os.listdir(storage_dir) if file.endswith('.h5')][0]
        model.load_weights(os.path.join(storage_dir,weights_filename))

        # Plot
        for idx in idx_set:
            img = dataset[idx,:]
            Postprocessing.monitor_hidden_layers(img,model,case_dir,figs_per_row,rows_to_cols_ratio,idx)

    def generate_augmented_data(self, transformations, augmented_dataset_size=1):

        # Set storage folder for augmented dataset
        case_dir = self.case_dir
        img_dims = self.parameters.img_size
        augmented_dataset_dir = os.path.join(case_dir,'Datasets','Datasets_augmented')

        # Unpack data
        X, y = Dataprocess.set_dataset(case_dir,img_dims)
        # Generate new dataset
        data_augmenter = ADS.datasetAugmentationClass(X,y,transformations,augmented_dataset_size,augmented_dataset_dir)
        data_augmenter.transform_images()
        data_augmenter.export_augmented_dataset()

    def train_model(self, sens_var=None):

        # Parameters
        alpha = self.parameters.training_parameters['learning_rate']
        nepoch = self.parameters.training_parameters['epochs']
        batch_size = self.parameters.training_parameters['batch_size']
        l2_reg = self.parameters.training_parameters['l2_reg']
        l1_reg = self.parameters.training_parameters['l1_reg']
        dropout = self.parameters.training_parameters['dropout']
        image_shape = self.datasets.dataset_train.element_spec[0].shape[1:3]
        activation = self.parameters.training_parameters['activation']

        #Model = Models.slice_scanner_lenet_model
        Model = Models.slice_scanner_simple_cnn_model
        #Model = Models.slice_scanner_inception_model

        self.model.Model = []
        self.model.History = []
        if sens_var == None:  # If it is a one-time training
            self.model.Model.append(Model(image_shape,alpha,l2_reg,l1_reg,dropout,activation))
            self.model.History.append(self.model.Model[-1].fit(self.datasets.dataset_train,epochs=nepoch,batch_size=batch_size,
                                                      steps_per_epoch=500,validation_data=self.datasets.dataset_cv,
                                                      validation_steps=None))
        else: # If it is a sensitivity analysis
            if type(alpha) == list:
                for learning_rate in alpha:
                    model = Model(image_shape,learning_rate,l2_reg,l1_reg,dropout,activation)
                    self.model.Model.append(model)
                    self.model.History.append(model.fit(self.datasets.dataset_train,epochs=nepoch,batch_size=batch_size,
                                                        steps_per_epoch=500,validation_data=self.datasets.dataset_cv,
                                                        validation_steps=None))
            elif type(l2_reg) == list:
                for regularizer in l2_reg:
                    model = Model(image_shape,alpha,regularizer,l1_reg,dropout,activation)
                    self.model.Model.append(model)
                    self.model.History.append(model.fit(self.datasets.dataset_train,epochs=nepoch,batch_size=batch_size,
                                                        steps_per_epoch=500,validation_data=self.datasets.dataset_cv,
                                                        validation_steps=None))
            elif type(l1_reg) == list:
                for regularizer in l1_reg:
                    model = Model(image_shape,alpha,l2_reg,regularizer,dropout,activation)
                    self.model.Model.append(model)
                    self.model.History.append(model.fit(self.datasets.dataset_train,epochs=nepoch,batch_size=batch_size,
                                                        steps_per_epoch=500,validation_data=self.datasets.dataset_cv,
                                                        validation_steps=None))
            elif type(dropout) == list:
                for rate in dropout:
                    model = Model(image_shape,alpha,l2_reg,l1_reg,rate,activation)
                    self.model.Model.append(model)
                    self.model.History.append(model.fit(self.datasets.dataset_train,epochs=nepoch,batch_size=batch_size,
                                                        steps_per_epoch=500,validation_data=self.datasets.dataset_cv,
                                                        validation_steps=None))
            elif type(activation) == list:
                for act in activation:
                    model = Model(image_shape,alpha,l2_reg,l1_reg,rate,act)
                    self.model.Model.append(model)
                    self.model.History.append(model.fit(self.datasets.dataset_train,epochs=nepoch,batch_size=batch_size,
                                                        steps_per_epoch=500,validation_data=self.datasets.dataset_cv,
                                                        validation_steps=None))

    def predict_on_test_set(self):

        img_dims = self.parameters.img_size
        pred_dir = self.parameters.prediction['dir']
        threshold = self.parameters.prediction['threshold']

        # Import model
        self.model.imported = True
        Model, History = self.reconstruct_model()

        results_dir = os.path.join(pred_dir,'Results')
        if os.path.exists(results_dir):
            rmtree(results_dir)
        os.makedirs(results_dir)

        metrics_functions = {
            'accuracy': tf.keras.metrics.BinaryAccuracy(),
            'recall': tf.keras.metrics.Recall(),
            'precision': tf.keras.metrics.Precision(),
            'tp': tf.keras.metrics.TruePositives(),
            'tn': tf.keras.metrics.TrueNegatives(),
            'fp': tf.keras.metrics.FalsePositives(),
            'fn': tf.keras.metrics.FalseNegatives(),
            'AUC': tf.keras.metrics.AUC(),
        }

        pred_cases = [folder for folder in os.listdir(pred_dir) if folder.startswith('Dataset')]
        for pred_case in pred_cases:
            metrics = dict.fromkeys(metrics_functions)

            X_test, y_test, paths_test = Dataprocess.read_preset_datasets(os.path.join(pred_dir,pred_case),return_filepaths=True)
            X_test = Dataprocess.standardize_image_size(X_test,img_dims)
            X_test, y_test = Dataprocess.preprocess_data(X_test,y_test)
            logits = Model.predict(X_test)
            m_test = logits.shape[0]
            y_hat = np.array([1 if logit > threshold else 0 for logit in logits])
            for key in metrics.keys():
                metric_function = metrics_functions[key]
                metric_function.update_state(y_test,logits)
                metrics[key] = metric_function.result().numpy()
            metrics['F1'] = 2*metrics['precision']*metrics['recall']/(metrics['precision']+metrics['recall'])

            metrics_name = list(metrics.keys())
            metrics_data = list(metrics.values())
            metrics_df = pd.DataFrame(index=metrics_name,columns=['Pred'],data=metrics_data)
            metrics_df.to_csv(os.path.join(results_dir,'%s_model_pred_metrics.csv' %pred_case),sep=';',decimal='.')

            paths_df = pd.DataFrame(index=paths_test,columns=['Ground_truth','Prediction'],data=np.array([y_test,y_hat]).T)
            paths_df.to_csv(os.path.join(results_dir,'%s_model_predictions.csv' %pred_case),sep=';',decimal='.')

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
            epochs = np.arange(1,Nepochs+1)

            case_ID = self.parameters.analysis['case_ID']
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
                plt.suptitle('Loss evolution case = {}'.format(str(case_ID)))

                if sens_var:
                    if type(sens_var[1][i]) == str:
                        storage_dir = os.path.join(self.case_dir, 'Results', str(case_ID), 'Model_performance',
                                                   '{}={}'.format(sens_var[0], sens_var[1][i]))
                    else:
                        storage_dir = os.path.join(self.case_dir, 'Results', str(case_ID), 'Model_performance',
                                                   '{}={:.3f}'.format(sens_var[0], sens_var[1][i]))
                    loss_plot_filename = 'Loss_evolution_{}_{}={}.png'.format(str(case_ID), sens_var[0],
                                                                              str(sens_var[1][i]))
                    loss_filename = 'Model_loss_{}_{}={}.csv'.format(str(case_ID), sens_var[0], str(sens_var[1][i]))
                    metrics_filename = 'Model_metrics_{}_{}={}.csv'.format(str(case_ID), sens_var[0],
                                                                           str(sens_var[1][i]))
                else:
                    storage_dir = os.path.join(self.case_dir, 'Results', str(case_ID), 'Model_performance')
                    loss_plot_filename = 'Loss_evolution_{}.png'.format(str(case_ID))
                    loss_filename = 'Model_loss_{}.csv'.format(str(case_ID))
                    metrics_filename = 'Model_metrics_{}.csv'.format(str(case_ID))

                if os.path.exists(storage_dir):
                    rmtree(storage_dir)
                os.makedirs(storage_dir)
                fig.savefig(os.path.join(storage_dir, loss_plot_filename), dpi=200)
                plt.close()

                # Metrics #
                metrics_name = [item for item in h.history if item not in ('loss', 'val_loss')]
                metrics_val = [(metric, h.history[metric][0]) for metric in metrics_name if metric.startswith('val')]
                metrics_train = [(metric, h.history[metric][0]) for metric in metrics_name if
                                 not metric.startswith('val')]

                rows = [metric[0] for metric in metrics_train]
                metric_fun = lambda L: np.array([item[1] for item in L])
                metrics_data = np.vstack((metric_fun(metrics_train), metric_fun(metrics_val))).T
                metrics = pd.DataFrame(index=rows, columns=['Training', 'CV'], data=metrics_data)
                metrics.to_csv(os.path.join(storage_dir, metrics_filename), sep=';', decimal='.')

                # Loss
                loss_data = np.vstack((list(epochs), loss_train, loss_cv)).T
                loss = pd.DataFrame(columns=['Epoch', 'Training', 'CV'], data=loss_data)
                loss.to_csv(os.path.join(storage_dir, loss_filename), index=False, sep=';', decimal='.')

    def export_model(self, sens_var=None):

        if type(self.model.Model) == list:
            N = len(self.model.Model)
        else:
            N = 1
            self.model.History = [self.model.History]
            self.model.Model = [self.model.Model]
        case_ID = self.parameters.analysis['case_ID']
        for i in range(N):
            if sens_var:
                if type(sens_var[1][i]) == str:
                    storage_dir = os.path.join(self.case_dir,'Results',str(case_ID),'Model','{}={}'
                                               .format(sens_var[0],sens_var[1][i]))
                else:
                    storage_dir = os.path.join(self.case_dir,'Results',str(case_ID),'Model','{}={:.3f}'
                                               .format(sens_var[0],sens_var[1][i]))
                model_json_name = 'SHOWDEC_model_{}_{}={}_arquitecture.json'.format(str(case_ID),sens_var[0],str(sens_var[1][i]))
                model_weights_name = 'SHOWDEC_model_{}_{}={}_weights.h5'.format(str(case_ID),sens_var[0],str(sens_var[1][i]))
                model_folder_name = 'SHOWDEC_model_{}_{}={}'.format(str(case_ID),sens_var[0],str(sens_var[1][i]))
            else:
                storage_dir = os.path.join(self.case_dir,'Results',str(case_ID),'Model')
                model_json_name = 'SHOWDEC_model_{}_arquitecture.json'.format(str(case_ID))
                model_weights_name = 'SHOWDEC_model_{}_weights.h5'.format(str(case_ID))
                model_folder_name = 'SHOWDEC_model_{}'.format(str(case_ID))

            if os.path.exists(storage_dir):
                rmtree(storage_dir)
            os.makedirs(storage_dir)

            # Export history training
            with open(os.path.join(storage_dir,'History'),'wb') as f:
                pickle.dump(self.model.History[i].history,f)

            # Save model
            # Export model arquitecture to JSON file
            model_json = self.model.Model[i].to_json()
            with open(os.path.join(storage_dir,model_json_name),'w') as json_file:
                json_file.write(model_json)
            self.model.Model[i].save(os.path.join(storage_dir,model_folder_name.format(str(case_ID))))

            # Export model weights to HDF5 file
            self.model.Model[i].save_weights(os.path.join(storage_dir,model_weights_name))

    def reconstruct_model_old(self):

        storage_dir = os.path.join(self.case_dir,'Results','pretrained_model')
        try:
            json_filename = [file for file in os.listdir(storage_dir) if file.endswith('.json')][0]
            json_file = open(os.path.join(storage_dir,json_filename),'r')
            loaded_model_json = json_file.read()
            json_file.close()

            Model = tf.keras.models.model_from_json(loaded_model_json)
        except:
            tf.config.run_functions_eagerly(True) # Enable eager execution
            try:
                model_folder = next(os.walk(storage_dir))[1][0]
            except:
                print('There is no model stored in the folder')

            Model = tf.keras.models.load_model(os.path.join(storage_dir,model_folder))
            tf.config.run_functions_eagerly(False) # Disable eager execution

        return Model

    def reconstruct_model(self, mode='train'):

        storage_dir = os.path.join(self.case_dir,'Results','pretrained_model')
        try:
            json_filename = [file for file in os.listdir(storage_dir) if file.endswith('.json')][0]
            json_file = open(os.path.join(storage_dir,json_filename),'r')
            loaded_model_json = json_file.read()
            json_file.close()

            Model = tf.keras.models.model_from_json(loaded_model_json)

        except:
            casedata = reader.read_case_logfile(os.path.join(storage_dir,'SHOWDEC.log'))
            img_dim = casedata.img_size
            alpha = casedata.training_parameters['learning_rate']
            activation = casedata.training_parameters['activation']

            # Load weights into new model
            Model = Models.slice_scanner_lenet_model(img_dim,alpha,0.0,0.0,0.0,activation)
            weights_filename = [file for file in os.listdir(storage_dir) if file.endswith('.h5')][0]
            Model.load_weights(os.path.join(storage_dir,weights_filename))

        # Reconstruct history
        class history_container:
            pass
        History = history_container()
        try:
            with open(os.path.join(storage_dir,'History'),'rb') as f:
                History.history = pickle.load(f)
            History.epoch = np.arange(1,len(History.history['loss'])+1)
            History.model = Model
        except:
            History.epoch = None
            History.model = None

        return Model, History

    def export_nn_log(self):
    
        def update_log(parameters, model):
            training = OrderedDict()
            training['TRAINING SIZE'] = parameters.training_parameters['train_size']
            training['LEARNING RATE'] = parameters.training_parameters['learning_rate']
            training['L2 REGULARIZER'] = parameters.training_parameters['l2_reg']
            training['L1 REGULARIZER'] = parameters.training_parameters['l1_reg']
            training['DROPOUT'] = parameters.training_parameters['dropout']
            training['ACTIVATION'] = parameters.training_parameters['activation']
            training['NUMBER OF EPOCHS'] = parameters.training_parameters['epochs']
            training['BATCH SIZE'] = parameters.training_parameters['batch_size']
            training['OPTIMIZER'] = [model.optimizer._name for model in model.Model]
            training['METRICS'] = [model.metrics_names[-1] if model.metrics_names != None else None for model in model.Model]
            training['DATASET_AUGMENTATION'] = bool(self.parameters.training_parameters['addaugdata'][0])
            if training['DATASET_AUGMENTATION'] == 1:
                training['DATASET_AUGMENTATION_ID'] = self.parameters.training_parameters['addaugdata'][1]
            
            analysis = OrderedDict()
            analysis['CASE ID'] = parameters.analysis['case_ID']
            analysis['ANALYSIS'] = parameters.analysis['type']
            analysis['IMPORTED MODEL'] = parameters.analysis['import']
            analysis['LAST TRAINING LOSS'] = ['{:.3f}'.format(history.history['loss'][-1]) for history in model.History]
            analysis['LAST CV LOSS'] = ['{:.3f}'.format(history.history['val_loss'][-1]) for history in model.History]

            architecture = OrderedDict()
            architecture['INPUT SHAPE'] = parameters.img_size

            return training, analysis, architecture
            
        
        parameters = self.parameters
        if parameters.analysis['type'] == 'sensanalysis':
            varname, varvalues = parameters.sens_variable
            for value in varvalues:
                parameters.training_parameters[varname] = value
                training, analysis, architecture = update_log(parameters,self.model)

                case_ID = parameters.analysis['case_ID']
                if type(value) == str:
                    storage_folder = os.path.join(self.case_dir,'Results',str(case_ID),'Model','{}={}'.format(varname,value))
                else:
                    storage_folder = os.path.join(self.case_dir,'Results',str(case_ID),'Model','{}={:.3f}'.format(varname,value))
                with open(os.path.join(storage_folder,'SHOWDEC.log'),'w') as f:
                    f.write('SHOWDEC log file\n')
                    f.write('==================================================================================================\n')
                    f.write('->ANALYSIS\n')
                    for item in analysis.items():
                        f.write(item[0] + '=' + str(item[1]) + '\n')
                    f.write('--------------------------------------------------------------------------------------------------\n')
                    f.write('->TRAINING\n')
                    for item in training.items():
                        f.write(item[0] + '=' + str(item[1]) + '\n')
                    f.write('--------------------------------------------------------------------------------------------------\n')
                    f.write('->ARCHITECTURE\n')
                    for item in architecture.items():
                        f.write(item[0] + '=' + str(item[1]) + '\n')
                    f.write('--------------------------------------------------------------------------------------------------\n')
                    f.write('->MODEL\n')
                    for model in self.model.Model:
                        model.summary(print_fn=lambda x: f.write(x + '\n'))
                    f.write('==================================================================================================\n')

        else:
            training, analysis, architecture = update_log(self.parameters,self.model)
            case_ID = parameters.analysis['case_ID']
            storage_folder = os.path.join(self.case_dir,'Results',str(case_ID))
            with open(os.path.join(storage_folder,'Model','SHOWDEC.log'),'w') as f:
                f.write('SHOWDEC log file\n')
                f.write('==================================================================================================\n')
                f.write('->ANALYSIS\n')
                for item in analysis.items():
                    f.write(item[0] + '=' + str(item[1]) + '\n')
                f.write('--------------------------------------------------------------------------------------------------\n')
                f.write('->TRAINING\n')
                for item in training.items():
                    f.write(item[0] + '=' + str(item[1]) + '\n')
                f.write('--------------------------------------------------------------------------------------------------\n')
                f.write('->ARCHITECTURE\n')
                for item in architecture.items():
                    f.write(item[0] + '=' + str(item[1]) + '\n')
                f.write('--------------------------------------------------------------------------------------------------\n')
                f.write('->MODEL\n')
                for model in self.model.Model:
                    model.summary(print_fn=lambda x: f.write(x + '\n'))
                f.write('==================================================================================================\n')


if __name__ == '__main__':
    launcher = r'C:\Users\juan.ramos\Shock_wave_detector\Scripts\launcher.dat'
    sw_scanner = ShockWaveScanner(launcher,check_model=False)
    sw_scanner.launch_analysis()