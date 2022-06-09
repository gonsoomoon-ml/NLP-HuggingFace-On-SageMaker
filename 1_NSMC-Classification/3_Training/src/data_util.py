import torch
import config

from pathlib import Path
import re


def read_nsmc_split(corpus_path):
    corpus_path = Path(corpus_path)
    texts = []
    labels = []

    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus_path = Path(corpus_path)
        texts = []
        labels = []
    
        next(f)
        for line in f:
            # Remove tab
            label, sentence = line.strip().split(',') 
            # Remove punctuations
            sentence = re.sub('[\.\,\(\)\{\}\[\]\`\'\!\?\:\;\-\=]', ' ', sentence)
            # Remove non-Korean characters
            sentence = re.sub('[^가-힣ㄱ-하-ㅣ\\s]', '', sentence)
            
            texts.append(sentence)
            labels.append(int(label))
            
    return texts, labels




class NSMCDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        self.label_names = ['negative', 'positive']

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

