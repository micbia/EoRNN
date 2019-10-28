import numpy as np, pandas as pd, scipy as scp, os, matplotlib.pyplot as plt

from time import time
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from datetime import datetime
#from clump_functions import PercentContours

from keras import models, layers, optimizers, initializers, callbacks, regularizers
from keras.utils import multi_gpu_model, plot_model
import matplotlib as mpl
mpl.use('Agg')

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


class HistoryCheckpoint(callbacks.Callback):
    def __init__(self, filepath='./', verbose=0, save_freq=1, in_epoch=0):
        self.verbose = verbose
        self.filepath = filepath
        self.save_freq = save_freq
        self.stor_arr = []
        self.prev_epoch = 0
        self.in_epoch = in_epoch

    def on_train_begin(self, logs=None):
        if(self.in_epoch != 0):
            print('Resuming from Epoch %d...' %self.in_epoch)
            self.prev_epoch = self.in_epoch

    def on_epoch_end(self, epoch, logs=None):
        if(epoch == self.in_epoch): self.stor_arr =  [[] for i in range(len(logs))]     # initializate array
        
        fname = self.filepath+'%s_ep-%d.txt'

        if(epoch % self.save_freq == 0 and epoch != self.in_epoch): 
            for i, val in enumerate(logs):
                self.stor_arr[i] = np.append(self.stor_arr[i], logs[val])
                if(os.path.isfile(fname %(val, self.prev_epoch))):
                    chekp_arr = np.loadtxt(fname %(val, self.prev_epoch)) # load previous save
                    chekp_arr = np.append(chekp_arr, self.stor_arr[i])      # update 
                    np.savetxt(fname %(val, epoch), chekp_arr)            # save
                    os.remove(fname %(val, self.prev_epoch))              # delete old save
                else:
                    np.savetxt(fname %(val, epoch), self.stor_arr[i])
            
            self.prev_epoch = epoch
            self.stor_arr = [[] for i in range(len(logs))]          # empty storing array

            if(self.verbose): print('Updated Logs checkpoints for epoch %d.' %epoch)
        else:
            for j, val in enumerate(logs):
                self.stor_arr[j] = np.append(self.stor_arr[j], logs[val])



class NeuralNet :
    def __init__(self, INPUT_DIM     = None, OUTPUT_DIM   = None,    LOSS           = 'mse',
                       HIDDEN_DIM_2  = None, HIDDEN_DIM_3 = None,    HIDDEN_DIM_1   = None,
                       BATCH_SIZE    = None, EPOCHS       = 1000,    DROPOUT        = 0.2,
                       ACTIVATION    = layers.ELU(alpha = 0.1),      LEARNING_RATE  = 0.01,
                       METRICS       = ['mae', 'mse'],               REGULARIZER    = None,
                       DATA_USED     = '',                           VARIABLES      = '',   
                       GPU           = None,                         PATH           = './',
                       RESUME_PATH   = None,                         RESUME_EPOCH = 0):

        self.input_dim      = INPUT_DIM
        self.output_dim     = OUTPUT_DIM
        self.hidden_dim_1   = HIDDEN_DIM_1
        self.hidden_dim_2   = HIDDEN_DIM_2
        self.hidden_dim_3   = HIDDEN_DIM_3

        self.loss           = LOSS
        self.batch_size     = BATCH_SIZE
        self.epochs         = EPOCHS
        self.dropout        = DROPOUT
        self.activation     = ACTIVATION
        self.lr             = LEARNING_RATE
        self.optimizer      = optimizers.adam(lr = self.lr)
        self.metrics        = METRICS
        self.model_datetime = datetime.now().strftime('%d-%mT%H-%M-%S')
        self.data_used      = DATA_USED
        self.variables      = VARIABLES
        self.regularizers   = REGULARIZER
        self.gpu            = GPU
        self.path           = PATH
        self.resume_epoch   = RESUME_EPOCH
        self.resume_path    = RESUME_PATH


        if PATH != None:
            self.path = PATH + self.model_datetime
            if not os.path.exists(self.path):
                os.makedirs(self.path)
                os.makedirs(self.path+'/model')
                os.makedirs(self.path+'/checkpoints')
                os.makedirs(self.path+'/checkpoints/weights')
        else:
            self.path = self.model_datetime
            if not os.path.exists(self.path):
                os.makedirs(self.path)
                os.makedirs(self.path+'/model')
                os.makedirs(self.path+'/checkpoints')
                os.makedirs(self.path+'/checkpoints/weights')
        self.path += '/'



    def compile(self):
        """
        Builds/compiles model architecture for training.

        Minimum 1 hidden layer, maximum 3

        Model parameters set in class initialisation

        """

        if(self.gpu != None):
            # initialize the model on CPU in the case of GPU, so weights are hosted on CPU memory
            with tf.device("/cpu:0"):
                self.model = models.Sequential()
        else:
            self.model = models.Sequential()

        if self.regularizers != None:
            self.model.add(layers.Dense(
                      self.hidden_dim_1, input_dim=self.input_dim,
                      kernel_initializer=initializers.glorot_normal(),
                      bias_initializer=initializers.Constant(0.0),
                      kernel_regularizer=self.regularizers))
        else:
            self.model.add(layers.Dense(
                      self.hidden_dim_1, input_dim=self.input_dim,
                      kernel_initializer=initializers.glorot_normal(),
                      bias_initializer=initializers.Constant(0.0)))

        self.model.add(self.activation)
        self.model.add(layers.Dropout(self.dropout))

        if self.hidden_dim_2 != None:
            if self.regularizers != None:
                self.model.add(layers.Dense(self.hidden_dim_2, kernel_regularizer=self.regularizers))
            else:
                self.model.add(layers.Dense(self.hidden_dim_2))

            self.model.add(self.activation)
            self.model.add(layers.Dropout(self.dropout))

        if self.hidden_dim_3 != None:
            if self.regularizers != None:
                self.model.add(layers.Dense(self.hidden_dim_3, kernel_regularizer=self.regularizers))
            else:
                self.model.add(layers.Dense(self.hidden_dim_3))

            self.model.add(self.activation)
            self.model.add(layers.Dropout(self.dropout))

        self.model.add(layers.Dense(self.output_dim))

        if(self.gpu != None):
            # Replicates the model on GPUs.
            self.parallel_model = multi_gpu_model(self.model, gpus=self.gpu)
            self.parallel_model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        else:
            # Compile only on CPU
            self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)


        if(self.resume_path != None and self.resume_epoch != 0):
            # load checkpoint weights 
            self.model.load_weights('%sweights/model_weights-ep%d.h5' %(self.resume_path, self.resume_epoch))
            print('Model weights resumed from:\tmodel_weights-ep%d.h5' %(self.resume_epoch))
            
            # copy logs checkpoints
            os.system('cp %s*ep-%d.txt %scheckpoints/' %(self.resume_path, self.resume_epoch, self.path))
        else:
            if(self.gpu != None):
                print('Model Created with GPU')
            else:
                print('Model Created')

        return self.model



    def add_callbacks(self, reduce_lr=False, early_stop=False, checkpoints=0, tensorboard=False, path=None):
        """
        Adds callbacks to keras model. Set value to 1 to activate

          reduce_lr : reduces learning rate by a factor of 10 for plateau in
                      validation loss with patience of 75 epochs

         early_stop : stops model training for plateau in validation loss with
                      patience of 150 epochs

        tensorboard : writes a log to subdirectory path +/logs for visualisation of
                      model learning. More info :
                      https://www.tensorflow.org/guide/summaries_and_tensorboard

               path : directory to write tensorboard logs
        """
        self.callbacks_list = []

        if reduce_lr == True:
            reduce_lr = callbacks.ReduceLROnPlateau(monitor = 'val_loss',
                                                    factor = 0.3, patience = 75,
                                                    min_lr = 1e-4)
            self.callbacks_list.append(reduce_lr)

        if early_stop == True:
            early_stop = callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=150, restore_best_weights=True)
            self.callbacks_list.append(early_stop)

        if checkpoints != 0:
            modelcheckpointer = callbacks.ModelCheckpoint(filepath=self.path+'checkpoints/weights/model_weights-ep{epoch:d}.h5',
                                                     monitor='val_loss', save_best_only=True, 
                                                     save_weights_only=True, mode='min',
                                                     period=checkpoints)
            self.callbacks_list.append(modelcheckpointer)
            
            history = callbacks.History()
            self.callbacks_list.append(history)

            histcheckpointer = HistoryCheckpoint(filepath=self.path+'checkpoints/', verbose=0,
                                                 save_freq=checkpoints, in_epoch=self.resume_epoch)
            self.callbacks_list.append(histcheckpointer)

        if tensorboard == True:
            if path != None:
                path = path +'logs'
                if not os.path.exists(path):
                    os.makedirs(path)
            else:
                path = 'logs'
                if not os.path.exists(path):
                    os.makedirs(path)

            tensorboard = callbacks.TensorBoard(log_dir =path +"/"+self.model_datetime,
                                                batch_size=100, histogram_freq=100,
                                                update_freq='epoch')
            self.callbacks_list.append(tensorboard)

        return self.callbacks_list



    def fit(self, x_training, y_training, x_testing=None, y_testing=None, verbose=2):
        """
        Trains the model for training set x_training, y_training
        and validates on x_testing, y_testing

        Updates keras model and history objects
        """      

        if (x_testing is None or y_testing is None):
            start = time()
            if(self.gpu == None):
                self.history = self.model.fit(x_training, y_training,
                                              epochs=self.epochs, batch_size=self.batch_size,
                                              verbose=verbose, callbacks=self.callbacks_list,
                                              initial_epoch=self.resume_epoch, shuffle=True)
            else:
                self.history = self.parallel_model.fit(x_training, y_training,
                                                       epochs=self.epochs, batch_size=self.batch_size,
                                                       verbose=verbose, callbacks=self.callbacks_list,
                                                       initial_epoch=self.resume_epoch, shuffle = True)
        else:
            start = time()
            if(self.gpu == None):
                self.history = self.model.fit(x_training, y_training,
                                              epochs=self.epochs, batch_size=self.batch_size,
                                              verbose=verbose, callbacks=self.callbacks_list,
                                              validation_data=(x_testing, y_testing),
                                              initial_epoch=self.resume_epoch, shuffle=True)
            else:
                self.history = self.parallel_model.fit(x_training, y_training,
                                                       epochs=self.epochs, batch_size=self.batch_size,
                                                       verbose=verbose, callbacks=self.callbacks_list,
                                                       validation_data=(x_testing, y_testing),
                                                       initial_epoch=self.resume_epoch, shuffle=True)

        end = time()
        self.time_to_fit = end - start
        return



    def predict(self, x_testing, y_testing):
        """
        Returns trained model predicted outputs for given input array
        
        """
        self.expected_outputs    = y_testing
        self.accuracy            = self.model.evaluate(x_testing,y_testing)
        self.predictions         = self.model.predict(x_testing, verbose=1)
        print(self.accuracy)
        return



    def save_best_network(self):
        """

        """
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(self.path+"model/model.json", "w") as json_file:
            json_file.write(model_json)
        
        # serialize weights to HDF5
        self.model.save_weights(self.path+"model/model_weigths.h5")

        #Save the model to HDF5
        self.model.save(self.path+'model/model.h5')

        # Save model visualization
        plot_model(self.model, to_file=self.path+'model/model_visualization.png', show_shapes=True, show_layer_names=True)

        return



    def save_outputs(self) :
        """
        Function to save outputs produced by fit & predict functions
        saves to folder with path path + /datetime

        """

        np.savetxt(self.path + '/expected_outputs.txt', self.expected_outputs, delimiter=" ")
        np.savetxt(self.path + '/predictions.txt', self.predictions, delimiter=" ")
        for key in self.history.history.keys():
            if(self.resume_epoch != 0):
                # copy latest checkpoint
                os.system('cp %s*ep-%d.txt %scheckpoints/' %(self.resume_path, self.resume_epoch, self.path))
            
                # load oldest checkpoint logs
                prev_arr = np.loadtxt(self.resume_path+key+'_ep-'+str(self.resume_epoch)+'.txt')
            
                # update logs history
                self.history.history[key] = np.append(prev_arr, self.history.history[key])
            
            # save logs history
            np.savetxt(self.path + '/%s.txt' %key, self.history.history[key], delimiter=" ")    

        parameter_file = self.path+'/model_description.txt'
        file = open(parameter_file, 'w')
        file.write('   Data Used  : ' + str(self.data_used) + '\n')
        file.write('   INPUT_DIM  : ' + str(self.input_dim) + '\n')
        file.write('   VARIABLES  : ' + ', '.join(self.variables)  + '\n')
        file.write('  HIDDEN_DIM  : ' + str(self.hidden_dim_1) + '\n')
        file.write('HIDDEN_DIM_2  : ' + str(self.hidden_dim_2) + '\n')
        file.write('HIDDEN_DIM_3  : ' + str(self.hidden_dim_3) + '\n')
        file.write('     DROPOUT  : ' + str(self.dropout) + '\n')
        file.write('  BATCH_SIZE  : ' + str(self.batch_size) + '\n')
        file.write('      EPOCHS  : ' + str(len(self.history.history['loss'])) + '\n')
        file.write(' TIME_TO_FIT  : ' + str(self.time_to_fit) + ' [s]\n')
        file.write('      VAL_R2  : ' + str(self.accuracy[3]) + '\n')
        file.write('     VAL_MSE  : ' + str(self.accuracy[2]) + '\n')
        file.write('     VAL_MAE  : ' + str(self.accuracy[1]) + '\n')
        file.close()

        return



    def plot_predictions(self, save_fig = 'no', path = ''):
        """
        Produces/saves .png of expected/predicted parameters

        """
        axis_x = r'%s$_{i, true}$' %self.variables[-1]
        axis_y = r'%s$_{i, pred}$' %self.variables[-1]
         
        if(self.output_dim != 1):
            fig = plt.figure(figsize = (800/50, 800/96), dpi = 96)
            print(self.expected_outputs, self.expected_outputs.shape.size)
            for i in range(len(self.expected_outputs[0])):
                x = list(map(list, zip(*self.expected_outputs)))[i]
                y = list(map(list, zip(*self.predictions)))[i]

                ax = plt.subplot(2,2,i+1)
                if (i == 1) or (i==0) or (i ==2) or (i ==3):
                    ax.loglog(x, y, '+')
                    #ax.set_xlim(min(min(x) - 0.1*min(x), min(y) - 0.1*min(y)), max(max(x) + 0.4*max(x), max(y) + 0.4*max(y)))
                    #ax.set_ylim(min(min(x) - 0.1*min(x), min(y) - 0.1*min(y)), max(max(x) + 0.4*max(x), max(y) + 0.4*max(y)))
                    x_vals = np.array(plt.xlim())
                    y_vals = x_vals
                    ax.loglog(x_vals, y_vals, '--', color = 'k', linewidth = 1.0)
                else:
                    ax.plot( x, y, '+')
                    #ax.set_xlim(min(min(x) - 0.1*min(x), min(y) - 0.1*min(y)), max(max(x) + 0.1*max(x), max(y) + 0.1*max(y)))
                    #ax.set_ylim(min(min(x) - 0.1*min(x), min(y) - 0.1*min(y)), max(max(x) + 0.1*max(x), max(y) + 0.1*max(y)))
                    x_vals = np.array(plt.xlim())
                    y_vals = x_vals
                    ax.plot(x_vals, y_vals, '--', color = 'k', linewidth = 1.0)

                ax.grid()
                ax.set_xlabel(axis_x[i], size = 13)
                ax.set_ylabel(axis_y[i], size = 13)

            fig.subplots_adjust(top = 0.9, bottom = 0.1, hspace = 0.3, wspace = 0.25, right = 0.93, left = 0.08)
        else:
            fig = plt.figure(figsize = (800/50, 800/96), dpi = 96)
            x = self.expected_outputs.ravel() #.values
            y = (self.predictions).T[0]
            #plt.xlim(x.min()-0.05, x.max()+0.05), plt.ylim(y.min()-0.05, y.max()+0.05)
            plt.loglog(x, y, 'x', zorder=0)
            plt.plot(plt.xlim(), plt.xlim(), 'k--', linewidth = 1.0)
            #PercentContours(x, y, colour='red', perc_arr=[0.95, 0.68])
            plt.grid()
            plt.xlabel(axis_x, size=13), plt.ylabel(axis_y, size=13)
                   
        if (save_fig == 'yes'):
            fig.savefig(path + self.model_datetime +'/predictions.png', dpi = 100)

        return



    def plot_loss(self, save_fig = 'no', path = ''):
        """
        Plots loss metrics - Mean Square Error & Mean Absolute Error

        """
        fig1 = plt.figure(figsize = (800/50, 800/96), dpi = 96)
        fig1.subplots_adjust(top = 0.9, bottom = 0.1, hspace = 0.3, wspace = 0.25, right = 0.93, left = 0.08)

        ax1 = plt.subplot(1,2,1)
        ax1.set_ylabel('MSE'), ax1.set_xlabel('Epoch')
        ax1.plot(self.history.history['mean_squared_error'], label = 'Training Loss')
        try:
            ax1.plot(self.history.history['val_mean_squared_error'], label = 'Validation Loss')
        except KeyError:
            pass
        #ax1.set_ylim(3e-3, 5e-4)
        ax1.set_xlim([0,len(self.history.history['mean_squared_error'])-1])

        ax3 = ax1.twinx()
        ax3.semilogy(self.history.history['lr'], color = 'k', alpha = 0.4, label = 'Learning Rate')
        ax3.set_ylabel('Learning Rate')
        lns, labs   = ax1.get_legend_handles_labels()
        lns2, labs2 = ax3.get_legend_handles_labels()
        ax1.legend(lns+lns2, labs+labs2, loc=1)

        ax2 = plt.subplot(1,2,2)
        """
        ax2.set_ylabel('MAE'), ax2.set_xlabel('Epoch')
        ax2.plot(self.history.history['mean_absolute_error'], label = 'Training MAE')
        try:
            ax2.plot(self.history.history['val_mean_absolute_error'], label = 'Validation MAE')
        except KeyError:
            pass
        ax2.set_xlim([0,len(self.history.history['mean_absolute_error'])-1])
        """

        ax2.set_ylabel(r'$R^2$'), ax2.set_xlabel('Epoch')
        ax2.plot(self.history.history['R2'], color='forestgreen', label=r'Training $R^2$')
        try:
            ax2.plot(self.history.history['val_R2'], color='royalblue', label=r'Validation $R^2$')
        except KeyError:
            pass
        #ax2.set_ylim(0.84, 0.941)
        ax2.set_xlim([0,len(self.history.history['R2'])-1])#, ax2.set_ylim(0, 1)

        ax4 = ax2.twinx()
        ax4.semilogy(self.history.history['lr'], color='k', alpha=0.4, label='Learning Rate')
        ax4.set_ylabel('Learning Rate')

        lns, labs   = ax2.get_legend_handles_labels()
        lns2, labs2 = ax4.get_legend_handles_labels()

        ax2.legend(lns+lns2, labs+labs2, loc=3)
        
        if (save_fig == 'yes'):
            fig1.savefig(path + self.model_datetime +'/loss.png', dpi = 100)

        return
