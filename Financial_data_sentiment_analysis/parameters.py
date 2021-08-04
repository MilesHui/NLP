import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


finphrase_dir = parentdir + '/data/external/FinancialPhraseBank-v1.0/'
tweet_dir = parentdir + '/data/external/'
twit_dir = parentdir + '/data/external/'
output_dir = parentdir + '/models/'
weights_dir = parentdir + '/models/'
