import logging
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from transformers import AdamW as AdamW_HF, get_linear_schedule_with_warmup

# Python libraries

from tqdm.notebook import tqdm

# Data Science modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
plt.style.use('ggplot')

import scikitplot as skplt
import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from utils import set_logger, metric
from parameters import output_dir

logger = set_logger('sa_model_comparison_finphrase', logging.DEBUG)


# Check the distribution to decide the maximum length
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Define a DataSet Class which simply return (x, y) pair instead
class SimpleDataset(Dataset):
    def __init__(self, x, y):
        self.datalist=[(x[i], y[i]) for i in range(len(y))]
    def __len__(self):
        return len(self.datalist)
    def __getitem__(self,idx):
        return(self.datalist[idx])


class BertTextClassifier(nn.Module):
    def __init__(self, hidden_size, dense_size, output_size, dropout=0.1):
        """
        Initialize the model
        """
        super().__init__()
        self.output_size = output_size
        self.dropout = dropout

        self.bert = BertModel.from_pretrained('bert-base-uncased',
                                              output_hidden_states=True,
                                              output_attentions=True)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.weights = nn.Parameter(torch.rand(13, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, dense_size)
        self.fc2 = nn.Linear(dense_size, output_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids):
        """
        Perform a forward pass of the model on nn_input
        """

        all_hidden_states, all_attentions = self.bert(input_ids)[-2:]
        batch_size = input_ids.shape[0]
        ht_cls = torch.cat(all_hidden_states)[:, :1, :].view(13, batch_size, 1, 768)
        atten = torch.sum(ht_cls * self.weights.view(13, 1, 1, 1), dim=[1, 3])
        atten = F.softmax(atten.view(-1), dim=0)
        feature = torch.sum(ht_cls * atten.view(13, 1, 1, 1), dim=[0, 2])
        # Dense layer
        dense_out = self.fc1(self.dropout(feature))
        # Concatinate the dense output and meta inputs
        #  concat_layer = torch.cat((dense_out, nn_input_meta.float()), 1)
        out = self.fc2(dense_out)
        # out = self.fc(self.dropout(feature))

        return out



# function to train Bert model
def train_transformer(model, x_train, y_train, x_valid, y_valid, learning_rate, num_epochs, batch_size, patience,
                      warm_up_proportion, max_grad_norm, max_seq_length):
    # Move model to GUP/CPU device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Load data into SimpleDataset
    train_ds = SimpleDataset(x_train, y_train)
    valid_ds = SimpleDataset(x_valid, y_valid)

    # Use DataLoader to load data from Dataset in batches
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    num_total_opt_steps = int(len(train_loader) * num_epochs)
    print(
        'Total Training Steps: {} ({} batches x {} epochs)'.format(num_total_opt_steps, len(train_loader), num_epochs))

    # Instead of AdamW from torch.optim, use the one from Huggingface with scheduler for learning curve decay
    # To reproduce BertAdam specific behavior set correct_bias=False
    optimizer = AdamW_HF(model.parameters(), lr=learning_rate, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_total_opt_steps * warm_up_proportion,
                                                num_training_steps=num_total_opt_steps)  # PyTorch scheduler

    # Set Train Mode
    model.train()

    # Tokenizer Parameter
    param_tk = {
        'return_tensors': "pt",
        'padding': 'max_length',
        'max_length': max_seq_length,
        'add_special_tokens': True,
        'truncation': True
    }

    # Initialize
    best_f1 = 0.
    valid_best = np.zeros((len(y_valid), 2))
    early_stop = 0
    train_losses = []
    valid_losses = []
    total_steps = 0
    train_loss_set = []

    for epoch in tqdm(range(num_epochs), desc="Epoch"):
        # print('================     epoch {}     ==============='.format(epoch+1))
        train_loss = 0.

        for i, batch in enumerate(train_loader):
            # Input features and labels from batch and move to device
            x_train_bt, y_train_bt = batch
            x_train_bt = tokenizer(x_train_bt, **param_tk).to(device)
            y_train_bt = torch.tensor(y_train_bt, dtype=torch.long).to(device)

            # Reset gradient
            optimizer.zero_grad()

            # Feedforward prediction
            # y_pred = bert_model(x_ids, x_mask, x_sids)
            # loss, logits = model(**x_train_bt, labels=y_train_bt)
            output = model(**x_train_bt, labels=y_train_bt)  # todo
            loss = output.loss
            logits = output.logits

            # Calculate Loss - no longer required.
            # loss = loss_fn(y_pred[0], y_truth)

            # Backward Propagation
            loss.backward()

            # Training Loss
            train_loss += loss.item() / len(train_loader)
            train_loss_set.append(loss.item())

            # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # Update Weights and Learning Rate
            optimizer.step()
            scheduler.step()

            logger.debug('train batch: %d, train_loss: %8f' % (i, loss.item() / len(train_loader)))
            total_steps += 1

        train_losses.append(train_loss)

        # Move to Evaluation Mode
        model.eval()

        # Initialize for Validation
        val_loss = 0.
        # valid_preds_fold = np.zeros((y_valid.size(0), 3))
        y_valid_pred = np.zeros((len(y_valid), 3))

        with torch.no_grad():
            for i, batch in enumerate(valid_loader):
                # Input features and labels from batch and move to device
                x_valid_bt, y_valid_bt = batch
                x_valid_bt = tokenizer(x_valid_bt, **param_tk).to(device)
                y_valid_bt = torch.tensor(y_valid_bt, dtype=torch.long).to(device)

                # loss, logits = model(**x_valid_bt, labels=y_valid_bt)
                # loss, logits = model(**x_valid_bt, labels=y_valid_bt)
                output = model(**x_valid_bt, labels=y_valid_bt)  # todo
                loss = output.loss
                logits = output.logits

                val_loss += loss.item() / len(valid_loader)
                y_valid_pred[i * batch_size:(i + 1) * batch_size] = F.softmax(logits, dim=1).cpu().numpy()
                logger.debug('validation batch: {}, val_loss: {}'.format(i, loss.item() / len(valid_loader)))
        valid_losses.append(val_loss)

        # Calculate metrics
        acc, f1 = metric(y_valid, np.argmax(y_valid_pred, axis=1))

        # If improving, save the model. If not, count up for early stopping
        if best_f1 < f1:
            early_stop = 0
            best_f1 = f1
            valid_best = y_valid_pred
            # torch.save(model.state_dict(), output_dir + out_model_name)
        else:
            early_stop += 1

        print('epoch: %d, train loss: %.4f, valid loss: %.4f, acc: %.4f, f1: %.4f, best_f1: %.4f, last lr: %.6f' %
              (epoch + 1, train_loss, val_loss, acc, f1, best_f1, scheduler.get_last_lr()[0]))

        logger.debug('valid_best: {}'.format(np.argmax(valid_best, axis=1)))

        if device == 'cuda:0':
            torch.cuda.empty_cache()

        # Early stop if it reaches patience number
        if early_stop >= patience:
            break

        # Back to Train Mode
        model.train()

    # Once all epochs are done, output summaries
    print('================ Training Completed: Starting Post Process ===============')

    # Draw training/validation losses
    plt.figure(figsize=(15, 6))
    plt.plot(train_losses, 'b-o', label='Training Loss')
    plt.plot(valid_losses, 'r-o', label='Validation Loss')
    plt.title("Training/Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.show()


    # Check the metrics for the validation set
    acc, f1 = metric(y_valid, np.argmax(valid_best, axis=1))
    logger.info('epoch: best, acc: %.8f, f1: %.8f, best_f1: %.8f\n' % (acc, f1, best_f1))

    # Convert to class names from 0, 1, 2
    class_names = ['Negative', 'Neutral', 'Positive']
    y_valid_class = [class_names[int(idx)] for idx in y_valid]
    pred_valid_class = [class_names[int(idx)] for idx in np.argmax(valid_best, axis=1)]

    titles_options = [("Confusion matrix, without normalization", None), ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = skplt.metrics.plot_confusion_matrix(y_valid_class, pred_valid_class, normalize=normalize, title=title)
    plt.show()

    # plot training performance
    plt.figure(figsize=(15, 8))
    plt.title("Training loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.plot(train_loss_set)
    plt.show()

    return acc, f1, valid_best

# function to test Bert model
def test_transformer(model,  x_valid, y_valid, learning_rate, num_epochs, batch_size, patience,
                     warm_up_proportion, max_grad_norm, max_seq_length):
    # Move model to GUP/CPU device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Load data into SimpleDataset
    valid_ds = SimpleDataset(x_valid, y_valid)


    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, shuffle=False)


    # Instead of AdamW from torch.optim, use the one from Huggingface with scheduler for learning curve decay
    # To reproduce BertAdam specific behavior set correct_bias=False
    optimizer = AdamW_HF(model.parameters(), lr=learning_rate, correct_bias=False)

    # Tokenizer Parameter
    param_tk = {
        'return_tensors': "pt",
        'padding': 'max_length',
        'max_length': max_seq_length,
        'add_special_tokens': True,
        'truncation': True
    }

    # Initialize
    valid_losses = []

    # Move to Evaluation Mode
    model.eval()

    # Initialize for Validation
    val_loss = 0.
    # valid_preds_fold = np.zeros((y_valid.size(0), 3))
    y_valid_pred = np.zeros((len(y_valid), 3))

    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            # Input features and labels from batch and move to device
            x_valid_bt, y_valid_bt = batch
            x_valid_bt = tokenizer(x_valid_bt, **param_tk).to(device)
            y_valid_bt = torch.tensor(y_valid_bt, dtype=torch.long).to(device)

            # loss, logits = model(**x_valid_bt, labels=y_valid_bt)
            # loss, logits = model(**x_valid_bt, labels=y_valid_bt)
            output = model(**x_valid_bt, labels=y_valid_bt)  # todo
            loss = output.loss
            logits = output.logits

            val_loss += loss.item() / len(valid_loader)
            y_valid_pred[i * batch_size:(i + 1) * batch_size] = F.softmax(logits, dim=1).cpu().numpy()
            logger.debug('validation batch: {}, val_loss: {}'.format(i, loss.item() / len(valid_loader)))
    valid_losses.append(val_loss)
    # Calculate metrics
    acc, f1 = metric(y_valid, np.argmax(y_valid_pred, axis=1))
    # print('shape', y_valid_pred.shape)
    # np.savetxt(output_dir + "y_pred.csv", y_valid_pred, delimiter=",")
    # np.savetxt(output_dir + "y_pred_label.csv", np.argmax(y_valid_pred, axis=1), delimiter=",")
    print('acc is:', acc, 'f1 is:', f1)

    # Convert to class names from 0, 1, 2
    class_names = ['Negative', 'Neutral', 'Positive']
    y_valid_class = [class_names[int(idx)] for idx in y_valid]
    pred_valid_class = [class_names[int(idx)] for idx in np.argmax(y_valid_pred, axis=1)]

    titles_options = [("Confusion matrix, without normalization", None), ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = skplt.metrics.plot_confusion_matrix(y_valid_class, pred_valid_class, normalize=normalize, title=title)
    plt.show()


    return acc, f1



# function to predict Bert model
def predict_transformer(model,  x_valid, batch_size, max_seq_length):
    # Move model to GUP/CPU device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Load data into SimpleDataset
    y_fake = list(np.zeros(len(x_valid)))
    valid_ds = SimpleDataset(x_valid,y_fake)

    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    # Tokenizer Parameter
    param_tk = {
        'return_tensors': "pt",
        'padding': 'max_length',
        'max_length': max_seq_length,
        'add_special_tokens': True,
        'truncation': True
    }

    # Move to Evaluation Mode
    model.eval()

    # Initialize for Validation
    val_loss = 0.
    # valid_preds_fold = np.zeros((y_valid.size(0), 3))
    y_valid_pred = np.zeros((len(x_valid), 3))

    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            # Input features and labels from batch and move to device
            x_valid_bt, y_valid_bt = batch
            x_valid_bt = tokenizer(x_valid_bt, **param_tk).to(device)
            y_valid_bt = torch.tensor(y_valid_bt, dtype=torch.long).to(device)

            output = model(**x_valid_bt, labels=y_valid_bt)  # todo
            loss = output.loss
            logits = output.logits

            val_loss += loss.item() / len(valid_loader)
            y_valid_pred[i * batch_size:(i + 1) * batch_size] = F.softmax(logits, dim=1).cpu().numpy()

    class_names = ['Negative', 'Neutral', 'Positive']
    pred_valid_class = [class_names[int(idx)] for idx in np.argmax(y_valid_pred, axis=1)]


    return pred_valid_class