# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 15:09:59 2019

@author:  George Vafeidis, Georgia Sarri, Leon Kalderon
"""

import os
import pyconll
import pyconll.util
import operator
# Used for Word2Vec glove embeddings
#from gensim.test.utils import datapath, get_tmpfile
#from gensim.models import KeyedVectors
#from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras import backend as K # Importing Keras backend (by default it is Tensorflow)
from keras.layers import Input, Dense # Layers to be used for building our model
from keras.models import Model # The class used to create a model
from keras.optimizers import Adam
from keras.utils import np_utils # Utilities to manipulate numpy arrays
from tensorflow import set_random_seed # Used for reproducible experiments
from tensorflow import keras
import tensorflow as tf
import gc
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
import seaborn as sn
from collections import defaultdict


K.set_session(tf.Session())

##############################################################################

def get_words_upos_freq(dataset):
    
    words = defaultdict(dict)
    
    for sentence in dataset:
        for j in range(len(sentence)):
            if sentence[j].form in words:
                if sentence[j].upos in words[sentence[j].form]:
                    words[sentence[j].form][sentence[j].upos] = words[sentence[j].form][sentence[j].upos] + 1
                else:
                    words[sentence[j].form][sentence[j].upos] = 1
                #words[sentence[j].form].append(sentence[j].upos)
            else:
                words[sentence[j].form][sentence[j].upos] = 1
    
    return words
    

##############################################################################
# Metrics: given in class

def recall(y_true, y_pred):
    
    """
    Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision(y_true, y_pred):
    
    """
    Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    Source
    ------
    https://github.com/fchollet/keras/issues/5400#issuecomment-314747992
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1(y_true, y_pred):
    
    """Calculate the F1 score."""
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r))


def accuracy(y_true, y_pred):
    return categorical_accuracy(y_true,y_pred)

def categorical_accuracy(y_true, y_pred):
    return K.cast(K.equal(K.argmax(y_true, axis=-1),
                          K.argmax(y_pred, axis=-1)),
                  K.floatx())

##############################################################################

MODELS_PATH = r'C:\Users\User\Desktop\MSc Courses\Natural Language Processing\assign3'
model_file_name = os.path.join(MODELS_PATH, 'temp_model.h1')
log_file_name = os.path.join(MODELS_PATH, 'temp.log')

sess = tf.Session()

def plot_history(hs, epochs, metric):
    plt.clf()
    plt.rcParams['figure.figsize'] = [6, 4]
    plt.rcParams['font.size'] = 16
    for label in hs:
        plt.plot(hs[label].history[metric], label='{0:s} train {1:s}'.format(label, metric))
        plt.plot(hs[label].history['val_{0:s}'.format(metric)], label='{0:s} validation {1:s}'.format(label, metric))
    epochs = len(hs['MLP'].history['loss'])
    x_ticks = np.arange(0, epochs + 1, 5)
    x_ticks [0] += 1
    plt.xticks(x_ticks)
    plt.ylim((0, 1))
    plt.xlabel('Epochs')
    plt.ylabel('Loss' if metric=='loss' else 'Accuracy')
    plt.legend()
    plt.show()

def clean_up(model):
    K.clear_session()
    del model
    gc.collect()  
    
def train_model(train_data,train_labels,
        validation_data,        
        optimizer,
        epochs=100,
        batch_size=128,
        hidden_layers=0,
        units =500,
        funnel = False,
        hidden_activation='relu',
        output_activation='softmax',
        dropout_rate=0.5):

    # Keras Callbacks
    lr_reducer = keras.callbacks.ReduceLROnPlateau(factor = 0.2, patience = 3, min_lr = 1e-6, verbose = 1)
    check_pointer = keras.callbacks.ModelCheckpoint(model_file_name, verbose = 1, save_best_only = True)
    early_stopper = keras.callbacks.EarlyStopping(patience = 8) # Change 4 to 8 in the final run    
    csv_logger = keras.callbacks.CSVLogger(log_file_name)

    np.random.seed(1402) # Define the seed for numpy to have reproducible experiments.
    set_random_seed(1981) # Define the seed for Tensorflow to have reproducible experiments.
    
    # Define the input layer.
    input_size = train_data.shape[1]
    input = Input(
        shape=(input_size,),
        name='Input'
    )
    x = input
    # Define the hidden layers.
    for i in range(hidden_layers):
        if funnel:
          layer_units=units // (i+1)
        else: 
          layer_units=units
        x = Dense(
           units=layer_units,
           kernel_initializer='glorot_uniform',
           activation=hidden_activation,
           name='Hidden-{0:d}'.format(i + 1)
        )(x)
    
        keras.layers.Dropout(x, dropout_rate, seed = 1231)
    # Define the output layer.
    
    output = Dense(
        units=classes,
        kernel_initializer='uniform',
        activation=output_activation,
        name='Output'
    )(x)
    # Define the model and train it.
    model = Model(inputs=input, outputs=output)
    #model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001),
                  metrics=[precision, recall, f1, accuracy])
    
    keras.backend.get_session().run(tf.global_variables_initializer())
    hs = model.fit(
        x=train_data,
        y=train_labels,
        validation_data = validation_data,
#        validation_split=0.1, # use 10% of the training data as validation data
        epochs=epochs,
        shuffle = True,
        verbose=1,
        batch_size=batch_size,
#        callbacks = [early_stopper, check_pointer,  csv_logger]
        callbacks = [early_stopper, check_pointer, lr_reducer,  csv_logger]
#        callbacks = [early_stopper, nan_terminator, check_pointer, lr_reducer, csv_logger]
        )  
    print('Finished training.')
    print('------------------')
    model.summary() # Print a description of the model.
    return model, hs

# GLOVE was also considered
######################################################################
#file1 = r'C:\MSc Data Science\Second Year 2nd Quarter\Text Analytics\Assignments\glove.6B.50d.txt'
#file2 = r'C:\MSc Data Science\Second Year 2nd Quarter\Text Analytics\Assignments\gensim_glove_vectors.txt'
##file1 = r'C:\temp\glove.6B\glove.6B.50d.txt'
##file2 = r'C:\temp\glove.6B\gensim_glove_vectors.txt'
#
#glove2word2vec(glove_input_file=file1, word2vec_output_file=file2)
#glove_model = KeyedVectors.load_word2vec_format(file2, binary=False)
######################################################################

#fasttext
######################################################################

idx = 0
vocab = {}

FASSTEX_FILE = r'C:\Users\User\Desktop\MSc Courses\Natural Language Processing\assign3\cc.en.300.vec\cc.en.300.vec'
FASTEX_OUTPUT = r'C:\Users\User\Desktop\MSc Courses\Natural Language Processing\assign3\fasttext.npy'

with open(FASSTEX_FILE, 'r', encoding="utf-8", newline='\n',errors='ignore') as f:
    for l in f:
        line = l.rstrip().split(' ')
        if idx == 0:
            vocab_size = int(line[0]) + 1
            dim = int(line[1])
            vecs = np.zeros(vocab_size*dim).reshape(vocab_size,dim)
            vocab["__PADDING__"] = 0
            idx = 1
        else:
            vocab[line[0]] = idx
            emb = np.array(line[1:]).astype(np.float)
            if (emb.shape[0] == dim):
                vecs[idx,:] = emb
                idx+=1
            else:
                continue

    pickle.dump(vocab,open("fasttext_voc",'wb'))
    np.save(FASTEX_OUTPUT,vecs)

fasttext_embed = np.load(FASTEX_OUTPUT)
fasttext_word_to_index = pickle.load(open("fasttext_voc", 'rb'))

#fasttext_word_to_index['w']
fasttext_embed[12]
a=np.vstack((np.zeros(300),fasttext_embed[12]))
MAX_WORDS =20000
MAX_SEQUENCE_LENGTH = 1000
EMBEDDING_DIM = 300

def data_to_embeddings(dataset, fasttext_embed, fasttext_word_to_index, window_size=3):
  x_set = []
  y_set = []
  window_pad = (window_size - 1) / 2
  l = 0
  for sentence in dataset:
      #print(sentence.text)
      for j in range(len(sentence)):          
          try:
              embedding = fasttext_embed[fasttext_word_to_index[sentence[j].form]]     
              #print(sentence[j].form)
              for i in range(1, int(window_pad+1)):                                    
                  if j-i < 0:
                      embedding = np.hstack((np.zeros(300), embedding))                                         
                      try:
                          embedding = np.hstack((embedding, fasttext_embed[fasttext_word_to_index[sentence[j+i].form]]))                   
                      except:                          
                          embedding = np.hstack(embedding, np.zeros(300))        
                  else:
                      try:
                          embedding = np.hstack((fasttext_embed[fasttext_word_to_index[sentence[j-i].form]], embedding))
                      except:
                          embedding = np.hstack((np.zeros(300), embedding))
                      try:
                          embedding = np.hstack((embedding, fasttext_embed[fasttext_word_to_index[sentence[j+i].form]]))
                      except:
                          embedding = np.hstack((embedding, np.zeros(300)))                           
                        
              upos = sentence[j].upos
              x_set.append(embedding)
              y_set.append(upos)              
          except:
              l +=1
              pass
              
  print(len(x_set))
  print(len(set(y_set)))
  print(l)
  
  return x_set, y_set

UD_ENGLISH_TRAIN = r'C:\Users\User\Desktop\MSc Courses\Natural Language Processing\assign3\data\corrected\en_cesl-ud-train.conllu'
UD_ENGLISH_VALIDATION = r'C:\Users\User\Desktop\MSc Courses\Natural Language Processing\assign3\data\corrected\en_cesl-ud-dev.conllu'
UD_ENGLISH_TEST = r'C:\Users\User\Desktop\MSc Courses\Natural Language Processing\assign3\data\corrected\en_cesl-ud-test.conllu'

train = pyconll.load_from_file(UD_ENGLISH_TRAIN)
validation =  pyconll.load_from_file(UD_ENGLISH_VALIDATION)
test =  pyconll.load_from_file(UD_ENGLISH_TEST)

x_train, y_train_raw = data_to_embeddings(train, fasttext_embed, fasttext_word_to_index)
x_valid, y_valid_raw = data_to_embeddings(validation, fasttext_embed, fasttext_word_to_index)
x_test, y_test_raw = data_to_embeddings(test, fasttext_embed, fasttext_word_to_index)

# MLP
#Categorize outputs
le = preprocessing.LabelEncoder()
#train
le.fit(y_train_raw)
y_train = le.transform(y_train_raw)

#Validation
le.fit(y_valid_raw)
y_valid = le.transform(y_valid_raw)

#Test
le.fit(y_test_raw)
y_test = le.transform(y_test_raw)

classes = len(set(y_train_raw))
Y_trainMLP = np_utils.to_categorical(y_train, classes)
Y_validMLP = np_utils.to_categorical(y_valid, classes)
X_train_final_MLP = np.array(x_train)
X_valid_final_MLP = np.array(x_valid)

#np.take(X_train_final_MLP,np.random.permutation(X_train_final_MLP.shape[1]),axis=1,out=X_train_final_MLP)
batch_size = 1024
epochs = 100

# Using Adam
optimizer = Adam()

mlp_model_adam, mlp_hs_adam = train_model(
    train_data=X_train_final_MLP,
    train_labels=Y_trainMLP,
    validation_data = (X_valid_final_MLP,Y_validMLP),
    optimizer=optimizer,
    epochs=epochs,
    batch_size=batch_size,
    funnel=True,
    hidden_layers=2,
    units = 220,
    hidden_activation='tanh',
    output_activation='softmax',
    dropout_rate=0.5
)

Y_test_final = np_utils.to_categorical(y_test, classes)
X_test_final_MLP = np.array(x_test)
mlp_eval_adam = mlp_model_adam.evaluate(X_test_final_MLP, Y_test_final, verbose=1)
#clean_up(model=mlp_model_adam)


print("Train Loss     : {0:.5f}".format(mlp_hs_adam.history['loss'][-1]))
print("Validation Loss: {0:.5f}".format(mlp_hs_adam.history['val_loss'][-1]))
print("Test Loss      : {0:.5f}".format(mlp_eval_adam[0]))
print("---")

print("Train Accuracy     : {0:.5f}".format(mlp_hs_adam.history['accuracy'][-1]))
print("Validation Accuracy: {0:.5f}".format(mlp_hs_adam.history['val_accuracy'][-1]))

print("Test categorical_cross_entropy:{0:.5f}".format(mlp_eval_adam[0]))
print("Test precision      : {0:.5f}".format(mlp_eval_adam[1]))
print("Test recall      : {0:.5f}".format(mlp_eval_adam[2]))
print("Test f1      : {0:.5f}".format(mlp_eval_adam[3]))
print("Test accuracy      : {0:.5f}".format(mlp_eval_adam[4]))

report_model = classification_report(np.argmax(mlp_model_adam.predict(X_test_final_MLP),axis=1),
                                           np.argmax(Y_test_final,axis=1),
                                           target_names = le.classes_,digits = 4)
print(report_model) 

# mlp_models.append(('Model9',mlp_hs_adam,mlp_eval_adam))

# Plot train and validation error per epoch.
plot_history(hs={'MLP': mlp_hs_adam}, epochs=epochs, metric='loss')
plot_history(hs={'MLP': mlp_hs_adam}, epochs=epochs, metric='accuracy')


# plot_history(hs={mlp_models[0][0]: mlp_models[0][1]}, epochs=epochs, metric='loss')
# plot_history(hs={mlp_models[0][0]: mlp_models[0][1]}, epochs=epochs, metric='acc')

cm1 = confusion_matrix(np.argmax(Y_test_final,axis=1),np.argmax(mlp_model_adam.predict(X_test_final_MLP),axis=1))

df_cm = pd.DataFrame(cm1, index = [i for i in np.linspace(0,classes-1,classes,dtype=int)],
                  columns = [i for i in np.linspace(0,classes-1,classes,dtype=int)])
plt.figure(figsize = (10,18))
sn.set(font_scale=1.0)
sn.heatmap(df_cm, annot=True, fmt='d',cmap='Blues',xticklabels=le.classes_,yticklabels=le.classes_)

########################################################################################################

train_freqs = get_words_upos_freq(train)
test_freqs = get_words_upos_freq(test)

def get_baseline_tag(train_freqs, word, classes = le.classes_):    
    try:
        tag_dict = train_freqs[word.form]
        baseline_tag = max(tag_dict.items(), key=operator.itemgetter(1))[0]
        
        return baseline_tag
    except Exception as ex:
        if word.upos == 'SYM':
            return le.classes_[np.random.randint(len(le.classes_)) - 1]
            
        return np.nan

y_baseline = []
y_true = []
exception_list = []
for sentence in test:
    for j in range(len(sentence)):
        word_dict = sentence[j]
        baseline_tag = get_baseline_tag(train_freqs, word_dict)
        true_tag = word_dict.upos
        if baseline_tag is not np.nan:
            y_baseline.append(baseline_tag)
            y_true.append(true_tag)
        else:            
            exception_list.append(word_dict.form)     

cm2 = cm1
np.fill_diagonal(cm2,0)

df_cm = pd.DataFrame(cm2, index = [i for i in np.linspace(0,classes-1,classes,dtype=int)],
                  columns = [i for i in np.linspace(0,classes-1,classes,dtype=int)])
plt.figure(figsize = (10,18))
sn.set(font_scale=1.0)
sn.heatmap(df_cm, annot=True, fmt='d',cmap='Blues',xticklabels=le.classes_,yticklabels=le.classes_)
       
report_baseline = classification_report(np.argmax(mlp_model_adam.predict(X_test_final_MLP),axis=1),
                                           np.argmax(Y_test_final,axis=1),
                                           target_names = le.classes_,digits = 4)

report_baseline = classification_report(y_baseline, y_true, digits = 4)
print(report_baseline)

cm_baseline = confusion_matrix(np.argmax(Y_test_final,axis=1),np.argmax(mlp_model_adam.predict(X_test_final_MLP),axis=1))
delta_cm = cm_baseline-cm1


df_cm = pd.DataFrame(delta_cm, index = [i for i in np.linspace(0,classes-1,classes,dtype=int)],
                  columns = [i for i in np.linspace(0,classes-1,classes,dtype=int)])
plt.figure(figsize = (10,18))
sn.set(font_scale=1.0)
sn.heatmap(df_cm, annot=True, fmt='d',cmap='Blues',xticklabels=le.classes_,yticklabels=le.classes_)
