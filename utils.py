import numpy as np
import matplotlib.pyplot as plt
import os

import torch 

device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda"  

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Change directory to wherever you store your datasets
DATA_DIR = "/home/nathgoh/Github/CS224U-NMT/datasets"

def get_sentences(file):
    filename = os.path.join(DATA_DIR, file)

    sentences = []
    with open(filename, "r") as f:
        for line in f:
            sentences.append(line.strip().split())
    return sentences

def get_vocabs(file):
    filename = os.path.join(DATA_DIR, file)
    
    with open(filename, "r") as f:
        return ['<pad>'] + [line.strip() for line in f]

def get_max_len_sentences(source_sentences, target_sentences, sentence_length = 48):
    new_source_sentences, new_target_sentences = [], []
    for source, target in zip(source_sentences, target_sentences):
        if len(source) <= sentence_length and len(target) <= sentence_length and len(source) > 0 and len(target) > 0:
            new_source_sentences.append(source)
            new_target_sentences.append(target)
        
    return new_source_sentences, new_target_sentences

def lookup_words(x, vocab):
    return [vocab[j] for j in x]

def plot_perplexity(perplexities):
    plt.title("Perplexity per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.plot(perplexities)

def print_examples(model, source_vocab, target_vocab, data_loader, decoder, attention = False, n = 3, EOS_INDEX = 3, max_len = 50):

    model.eval()
        
    for j, (source_ids, source_lengths, target_ids, target_lengths) in enumerate(data_loader):
        if not attention:
            result = decoder(model, source_ids.to(device), source_lengths.to(device), max_len = max_len)
        else:
            result, _ = decoder(model, source_ids.to(device), source_lengths.to(device), max_len = max_len)

        # Remove <s>
        source_ids = source_ids[0, 1:]
        target_ids = target_ids[0, 1:]

        # Remove </s> and <pad>
        source_ids = source_ids[:np.where(source_ids == EOS_INDEX)[0][0]]
        target_ids = target_ids[:np.where(target_ids == EOS_INDEX)[0][0]]

        prediction = " ".join(utils.lookup_words(result, vocab = target_vocab))
        target = " ".join(utils.lookup_words(target_ids, vocab = target_vocab))
   
        print("Example {}".format(j + 1))
        print("Source : ", " ".join(lookup_words(source_ids, vocab = source_vocab)))
        print("Target : ", " ".join(lookup_words(target_ids, vocab = target_vocab)))
        print("Prediction: ", " ".join(lookup_words(result, vocab = target_vocab)))
        print()
        
        if j == n - 1:
            break