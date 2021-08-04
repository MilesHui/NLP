from sklearn.model_selection import StratifiedKFold
import numpy as np
import sys
import random
import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
from sklearn.model_selection import train_test_split
from bert import train_transformer
from bert import test_transformer
from parameters import output_dir, weights_dir
from data import load_data_fin, load_data_twit, load_data_ticker, load_data_reddit

# Set Random Seed
random.seed(42)
np.random.seed(42)
rand_seed = 42
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Hyperparameters
learning_rate = 2e-5
num_epochs = 5
batch_size = 32
patience = 3
warm_up_proportion = 0.1
max_grad_norm = 1.0
max_seq_length = 96


def train_bert(train):
    # Input Data
    x_train = np.array(train['sentence'])
    y_train = np.array(train['label'].astype(int).values)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.33, random_state=42)


    # (train_indices, valid_indices) in enumerate(skf.split(y_train, y_train)
    # Make sure to load the pre-trained model every time
    bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

    acc, f1, valid_best = train_transformer(bert_model, x_train, y_train, x_valid, y_valid,
                                            learning_rate, num_epochs, batch_size, patience, warm_up_proportion,
                                            max_grad_norm, max_seq_length)

    torch.save(bert_model, output_dir + 'bert_model.pt')


    print("Average Accuracy: %.8f, Average F1: %.8f" % (acc, f1))




# def train_bert(train, skf):
#     # Input Data
#     x_train = np.array(train['sentence'])
#     y_train = np.array(train['label'].astype(int).values)
#     oof_train = np.zeros((len(train), 3), dtype=np.float32)
#     acc_list, f1_list = [], []
#
#
#     for n_fold, (train_indices, valid_indices) in enumerate(skf.split(y_train, y_train)):
#         # logger.info('================     fold {}    ==============='.format(n_fold+1))
#         print('================     fold {}    ==============='.format(n_fold + 1))
#         # Input train data for this fold
#         x_train_fold = x_train[train_indices]
#         y_train_fold = y_train[train_indices]
#         # Input validation data for this fold
#         x_valid_fold = x_train[valid_indices]
#         y_valid_fold = y_train[valid_indices]
#
#         # Make sure to load the pre-trained model every time
#         bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
#
#         acc, f1, valid_best = train_transformer(bert_model, x_train_fold, y_train_fold, x_valid_fold, y_valid_fold,
#                                                 learning_rate, num_epochs, batch_size, patience, warm_up_proportion,
#                                                 max_grad_norm, max_seq_length)
#         oof_train[valid_indices] = valid_best
#
#         torch.save(bert_model, output_dir + 'bert_model_{}.pt'.format(n_fold))
#
#         acc_list.append(acc)
#         f1_list.append(f1)
#
#         # Number of folds to iterrate
#         if n_fold == 10:
#             break
#
#     print("#################### FINAL RESULT ####################")
#
#     avg_acc = np.mean(acc_list)
#     avg_f1 = np.mean(f1_list)
#
#     print("Average Accuracy: %.8f, Average F1: %.8f" % (avg_acc, avg_f1))


def retrain_test_bert(train, test):

    # test and result
    # Re-train the model with all Train set and validate on Test set

    # Input Data
    x_train = np.array(train['sentence'])
    y_train = np.array(train['label'].astype(int).values)

    # Input Data
    x_test = np.array(test['sentence'])
    y_test = np.array(test['label'].astype(int).values)

    # Make sure to load the pre-trained model every time
    bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

    acc, f1, valid_best = \
        train_transformer(bert_model, x_train, y_train, x_test, y_test, \
                          learning_rate, num_epochs, batch_size, patience, warm_up_proportion, max_grad_norm,
                          max_seq_length)
    torch.save(bert_model, output_dir + 'bert_model_test.pt')

    print(('Accuracy: %.8f, f1: %.8f\n' % (acc, f1)))


def test_bert(test, weight):

    x_test = np.array(test['sentence'])
    y_test = np.array(test['label'].astype(int).values)

    # load data
    if torch.cuda.is_available():
        bert_model = torch.load(weights_dir+weight)
    else:
        bert_model = torch.load(weights_dir + weight, map_location=torch.device('cpu'))
    acc, f1= test_transformer(bert_model, x_test, y_test, learning_rate, num_epochs, batch_size, patience, warm_up_proportion, max_grad_norm, max_seq_length)

    print("Best result")
    print("Accuracy: %.8f, F1: %.8f" % (acc, f1))
    return acc, f1


if __name__ == "__main__":

    # cross validation
    # fold = StratifiedKFold(n_splits=7)
    # Run the training with Stratified KFold
    # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rand_seed)


    # SELECT data
    if sys.argv[1] == 'fin':
        train_fin = load_data_fin()

        # Split the data
        msk_fin = np.random.rand(len(train_fin)) < 0.8
        train = train_fin[msk_fin]
        test = train_fin[~msk_fin]

    elif sys.argv[1] == 'twit':
        train_twit = load_data_twit()
        # Split the data
        msk_twit = np.random.rand(len(train_twit)) < 0.8
        train = train_twit[msk_twit]
        test = train_twit[~msk_twit]

    elif sys.argv[1] == 'ticker':
        test = load_data_ticker()

    elif sys.argv[1] == 'reddit':
        train, test = load_data_reddit()

    # select weight
    try:
        if sys.argv[3] == 'fin':
            weight = 'fin_bert_model_4.pt'
        elif sys.argv[3] == 'twit':
            weight = 'twit_bert_model_0.pt'
    except:
        pass


    # select train or test
    if sys.argv[2] == 'train':
        # train_bert(train, skf)
        train_bert(train)
    elif sys.argv[2] == 'retrain':
        retrain_test_bert(train, test)
    elif sys.argv[2] == 'test':
        acc, f1 = test_bert(test, weight)