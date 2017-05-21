import numpy as np

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.wrappers import TimeDistributed
from keras.layers import Conv1D

import progressbar
from conlleval import conlleval

# Preprocess Toolbox
def extract_nested_column(file_path, col_idx):
    s = open(file_path, 'r').read().split('\n\n')
        # 一定要检查语料结尾是否合乎规范，有没有多余的回车，需手工去除，这次栽这里了。
    nested_col = []

    for i in range(len(s)):
        nested_col.append([pair.split('\t')[col_idx] for pair in s[i].split('\n')])

    return nested_col

def unnest_list(nested_list):
    unnested_list = []
    for i in range(len(nested_list)):
        for j in range(len(nested_list[i])):
            unnested_list.append(nested_list[i][j])
    return unnested_list

def allocate_index(item_list):
    idx_dict = {}
    idx = 0
    for each_item in item_list:
        if each_item not in idx_dict:
            idx_dict[each_item] = idx
            idx += 1
    return idx_dict

def map_index(item_nested_list, idx_dict):
    idx_nested_list = item_nested_list  # create a new list of the same size
    for i in range(len(item_nested_list)):
        for j in range(len(item_nested_list[i])):
            idx_nested_list[i][j] = idx_dict[item_nested_list[i][j]]
    return idx_nested_list

def preprocess(file_path, idx_dict=None):
    words = extract_nested_column(file_path, 0)
    labels = extract_nested_column(file_path, 1)

    if idx_dict == None:
        words_idx_dict = allocate_index(unnest_list(words))
        labels_idx_dict = allocate_index(unnest_list(labels))
    else:
        words_idx_dict = idx_dict[0]
        labels_idx_dict = idx_dict[1]

    words_idx = map_index(words, words_idx_dict)
    labels_idx = map_index(labels, labels_idx_dict)
    
    words = extract_nested_column(file_path, 0)
    labels = extract_nested_column(file_path, 1)
    
    return list2ndarray(words), list2ndarray(labels), list2ndarray(words_idx), list2ndarray(labels_idx), words_idx_dict, labels_idx_dict

def list2ndarray(nested_list):
    for i in range(len(nested_list)):
        nested_list[i] = np.array(nested_list[i])
    return nested_list

# Define model
def model_1(input_dim, output_dim):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=100))
        # Turns positive integers (indexes) into dense vectors of fixed size.
    # model.add(Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
        # This layer creates a convolution kernel over a single spatial (or temporal) dimension to produce a tensor of outputs.
    model.add(Dropout(rate=0.25))
    model.add(SimpleRNN(units=100, return_sequences=True))
        # Fully-connected RNN where the output is to be fed back to input.
    model.add(TimeDistributed(layer=Dense(units=output_dim, activation='softmax')))
        # This wrapper allows to apply a layer to every temporal slice of an input.
        # 该包装器可以把一个层应用到输入的每一个时间步上
        # layer: a layer instance.
    model.compile('rmsprop', loss='categorical_crossentropy')
        # RMSProp optimizer is usually a good choice for recurrent neural networks.
    return model

def model_2(input_dim, output_dim):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=100))
        # Turns positive integers (indexes) into dense vectors of fixed size.
    model.add(Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
        # This layer creates a convolution kernel over a single spatial (or temporal) dimension to produce a tensor of outputs.
    model.add(Dropout(rate=0.25))
    model.add(SimpleRNN(units=100, return_sequences=True))
        # Fully-connected RNN where the output is to be fed back to input.
    model.add(TimeDistributed(layer=Dense(units=output_dim, activation='softmax')))
        # This wrapper allows to apply a layer to every temporal slice of an input.
        # 该包装器可以把一个层应用到输入的每一个时间步上
        # layer: a layer instance.
    model.compile('rmsprop', loss='categorical_crossentropy')
        # RMSProp optimizer is usually a good choice for recurrent neural networks.
    return model

def model_3(input_dim, output_dim):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=100))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dense(units=output_dim))
    model.add(Activation('softmax'))
    model.compile('rmsprop', loss='categorical_crossentropy')

    return model

def model_4(input_dim, output_dim):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=100))
    model.add(Bidirectional(LSTM(units=100, return_sequences=True)))
    model.add(Dense(units=output_dim))
    model.add(Activation('softmax'))
    model.compile('rmsprop', loss='categorical_crossentropy')

    return model

# model = rnn_model(n_vocab, n_classes)

# # Training
# n_epochs = 2

# train_f1 = []
# valid_f1 = []
# best_valid_f1 = 0

# for i in range(n_epochs):
#     print("Epoch {}".format(i))

#     print("Training =>")
#     train_pred_label = []
#     avg_loss = 0

#     bar = ProgressBar(maxval=len(train_x)) #?
#     for n_batch, sent in bar(enumerate(train_x)):
#         label = train_label[n_batch]


def main():
    # Load data
    words_idx_dict, labels_idx_dict = preprocess('./data/atis.all.txt')[4:6]
    # print(words_idx_dict)
    # print(labels_idx_dict)
    train_words, train_labels, train_x, train_label = preprocess('./data/atis.train.txt', 
        (words_idx_dict, labels_idx_dict))[0:4]
    # print(train_words[0])
    # print(train_x[0])
    # print(train_labels[0])
    # print(train_label[0])
    val_words, val_labels, val_x, val_label = preprocess('./data/atis.test.txt', 
        (words_idx_dict, labels_idx_dict))[0:4]

    idx2w = {words_idx_dict[k]:k for k in words_idx_dict}
    idx2la = {labels_idx_dict[k]:k for k in labels_idx_dict}

    n_vocab = len(words_idx_dict)
    n_classes = len(labels_idx_dict)
    print(n_vocab, n_classes)

    model = model_3(input_dim=n_vocab, output_dim=n_classes)


    words_val = [ list(map(lambda x: idx2w[x], w)) for w in val_x]
    groundtruth_val = [ list(map(lambda x: idx2la[x], y)) for y in val_label]
    words_train = [ list(map(lambda x: idx2w[x], w)) for w in train_x]
    groundtruth_train = [ list(map(lambda x: idx2la[x], y)) for y in train_label]


    ### Training
    n_epochs = 100

    train_f_scores = []
    val_f_scores = []
    best_val_f1 = 0

    for i in range(n_epochs):
        print("Epoch {}".format(i))
        
        print("Training =>")
        train_pred_label = []
        avgLoss = 0
            
        bar = progressbar.ProgressBar(maxval=len(train_x))
        for n_batch, sent in bar(enumerate(train_x)):
            label = train_label[n_batch]
            label = np.eye(n_classes)[label][np.newaxis,:]
            sent = sent[np.newaxis,:]

            # print(label, sent, label.shape, sent.shape)
            
            if sent.shape[1] > 1: #some bug in keras
                loss = model.train_on_batch(sent, label)
                avgLoss += loss

            pred = model.predict_on_batch(sent)
            pred = np.argmax(pred,-1)[0]
            train_pred_label.append(pred)

        avgLoss = avgLoss/n_batch
        
        predword_train = [ list(map(lambda x: idx2la[x], y)) for y in train_pred_label]
        con_dict = conlleval(predword_train, groundtruth_train, words_train, 'r.txt')
        train_f_scores.append(con_dict['f1'])
        print('Loss = {}, Precision = {}, Recall = {}, F1 = {}'.format(avgLoss, con_dict['r'], con_dict['p'], con_dict['f1']))
        
        
        print("Validating =>")
        
        val_pred_label = []
        avgLoss = 0
        
        bar = progressbar.ProgressBar(maxval=len(val_x))
        for n_batch, sent in bar(enumerate(val_x)):
            label = val_label[n_batch]
            label = np.eye(n_classes)[label][np.newaxis,:]
            sent = sent[np.newaxis,:]
            
            if sent.shape[1] > 1: #some bug in keras
                loss = model.test_on_batch(sent, label)
                avgLoss += loss

            pred = model.predict_on_batch(sent)
            pred = np.argmax(pred,-1)[0]
            val_pred_label.append(pred)

        avgLoss = avgLoss/n_batch
        
        predword_val = [ list(map(lambda x: idx2la[x], y)) for y in val_pred_label]
        con_dict = conlleval(predword_val, groundtruth_val, words_val, 'r.txt')
        val_f_scores.append(con_dict['f1'])
        
        print('Loss = {}, Precision = {}, Recall = {}, F1 = {}'.format(avgLoss, con_dict['r'], con_dict['p'], con_dict['f1']))

        if con_dict['f1'] > best_val_f1:
            best_val_f1 = con_dict['f1']
            open('model_architecture.json','w').write(model.to_json())
            model.save_weights('best_model_weights.h5',overwrite=True)
            print("Best validation F1 score = {}".format(best_val_f1))
        print()

def show():
    # Load data
    words_idx_dict, labels_idx_dict = preprocess('./data/atis.all.txt')[4:6]
    train_words, train_labels, train_x, train_label = preprocess('./data/atis.train.txt', 
        (words_idx_dict, labels_idx_dict))[0:4]
    val_words, val_labels, val_x, val_label = preprocess('./data/atis.test.txt', 
        (words_idx_dict, labels_idx_dict))[0:4]

    idx2w = {words_idx_dict[k]:k for k in words_idx_dict}
    idx2la = {labels_idx_dict[k]:k for k in labels_idx_dict}

    n_vocab = len(words_idx_dict)
    n_classes = len(labels_idx_dict)
    # print(train_words[0], train_labels[0], train_x[0], train_label[0])
    print(len(train_words), len(val_words))


if __name__ == '__main__':
    main()