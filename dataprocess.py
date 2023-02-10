import json
import torch
import random
from torch.utils.data import Dataset
from transformers import AutoTokenizer, logging
from torch.nn.utils.rnn import pad_sequence
from utils import *

def dataRead(mode):
    logging.set_verbosity_error()
    portion = 4000
    assert mode in ['train', 'dev']
    prompt5List = [0, 1, 2, 3, 4]
    question = []
    signal = []
    trainSet = []
    testSet = []
    promptTrain = []
    prompt_config = {}
    
    with open('./data/Config.jsonl','r') as fin:
        prompt_config = json.loads(fin.read())
        
    with open('./data/train.json','r') as fin:
        for line in fin:
            jsn = json.loads(line)
            trainSet.append(jsn)
    
    with open('./data/test.json','r') as fin:
        for line in fin:
            jsn = json.loads(line)
            testSet.append(jsn)
    
    with open('./data/prompt5_allTrain_ChatGPT.jsonl','r') as fin:
        for line in fin:
            jsn = json.loads(line)
            promptTrain.append(jsn)
    
    for index in range(7473):
        temp = 0 
        sig = []
        for i in range(5):
            sig.append(promptTrain[i]['answerList'][index])
            temp += promptTrain[i]['answerList'][index]
        if temp not in [5]:
            question.append(trainSet[index]['question'])
            signal.append(sig)
    
    # if mode == 'train':
    #     return question[:portion], signal[:portion]
    # elif mode == 'dev':
    #     return question[portion:], signal[portion:]
    
    return question, signal
            
    
class PromptDataset(Dataset):
    def __init__(self, fromdata, model_name='bert'):
        super().__init__()
        question, signal = fromdata
        self.question = question
        self.signal = signal
        
        self.len = len(signal)
        if model_name == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        elif model_name == 'deberta':
            self.tokenizer = AutoTokenizer.from_pretrained("../T2_prompt/PLM/deberta-v3-large")
        elif model_name == 'roberta':
            self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        
    def __len__(self):
        return self.len
        
    def __getitem__(self, idx):
        word_pieces = self.tokenizer.tokenize("[CLS]" + self.question[idx] + "[SEP]")
        word_pieces = word_pieces[:512] if len(word_pieces) > 511 else word_pieces
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        
        tokens_tensors = torch.tensor(ids, dtype=torch.long)
        masks_tensors = torch.tensor([1] * len(tokens_tensors), dtype=torch.long)
        return tokens_tensors, masks_tensors, self.signal[idx]


def create_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    masks_tensors = [s[1] for s in samples]
    signal = [s[2] for s in samples]
    
    tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)
    masks_tensors = pad_sequence(masks_tensors, batch_first=True)
    signals_tensors = torch.tensor(signal, dtype=torch.float)
    
    return tokens_tensors, masks_tensors , signals_tensors
        
            

def dataReadForTest():
    testSet = []
    question = []
    signal = []
            
    with open('./data/test.json','r') as fin:
        for line in fin:
            jsn = json.loads(line)
            testSet.append(jsn)
    
    for jsn in testSet:
        question.append(jsn['question'])
        signal.append(0)
        
    return (question, signal)
    
        
            
            
    
    
    