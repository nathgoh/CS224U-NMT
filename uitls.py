import numpy as np
import os

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

def plot_perplexity(perplexities):
    plt.title("Perplexity per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.plot(perplexities)