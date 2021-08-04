# Python libraries

import sys
import time
import logging

# Data Science modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
plt.style.use('ggplot')

# Import Scikit-learn moduels
from sklearn.metrics import accuracy_score, f1_score

# # Import nltk modules and download dataset
# import nltk
# from nltk.corpus import stopwords
#
#
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
#
# stop = set(stopwords.words('english'))


# Set logger
loggers = {}


def set_logger(name, level):
    global loggers

    if loggers.get(name):
        return loggers.get(name)
    else:
        logger = logging.getLogger(name)
        if (logger.hasHandlers()):
            logger.handlers.clear()

        logger.setLevel(level)

        timestamp = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())
        formatter = logging.Formatter('[%(asctime)s][%(levelname)s] ## %(message)s')

        fh = logging.FileHandler(name + '.log')
        # fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        ch = logging.StreamHandler(sys.stdout)
        # ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        loggers[name] = logger
        return logger





# Set Seaborn Style
sns.set(style='white', context='notebook', palette='deep')

# # evaluation config
# result_df = pd.DataFrame(columns=['Accuracy', 'F1'], index=['A: Lexicon', 'B: Tfidf', 'C1: LSTM', 'C2: LSTM+GloVe', 'D1: BERT', 'D2: ALBERT'])

# Define metrics
# Here, use F1 Macro to evaluate the model.
def metric(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    return acc, f1
#
# scoring = {'Accuracy': 'accuracy', 'F1': 'f1_macro'}
# refit = 'F1'


