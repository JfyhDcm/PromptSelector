import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSequenceClassification, logging

DEBERTA_HIDDEN_SIZE = 1024
BERT_HIDDEN_SIZE = 768
class PModel(nn.Module):
    def __init__(self, model_name='bert', HIDDEN_SIZE=None):
        super(PModel, self).__init__()
        logging.set_verbosity_error()
        self.num_labels = 5
        if model_name =='bert':
            self.plm = AutoModel.from_pretrained("bert-base-uncased")
            self.layer_norm = nn.LayerNorm(BERT_HIDDEN_SIZE)
            self.classifier = nn.Linear(BERT_HIDDEN_SIZE, self.num_labels)
        elif model_name == 'deberta':
            self.plm = AutoModel.from_pretrained("../T2_prompt/PLM/deberta-v3-large")
            self.layer_norm = nn.LayerNorm(DEBERTA_HIDDEN_SIZE)
            self.classifier = nn.Linear(DEBERTA_HIDDEN_SIZE, self.num_labels)
        elif model_name == 'roberta':
            self.plm = AutoModel.from_pretrained("xlm-roberta-large")
            self.layer_norm = nn.LayerNorm(DEBERTA_HIDDEN_SIZE)
            self.classifier = nn.Linear(DEBERTA_HIDDEN_SIZE, self.num_labels)
        else:
            self.plm = AutoModel.from_pretrained(model_name) 
            self.layer_norm = nn.LayerNorm(HIDDEN_SIZE)
            self.classifier = nn.Linear(HIDDEN_SIZE, self.num_labels)
        self.dropout = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, tokens_tensors,  masks_tensors):
        outputs = self.plm(input_ids=tokens_tensors, attention_mask=masks_tensors)
        output = self.layer_norm(outputs['last_hidden_state'][:, 0, :])
        output = self.dropout(output)
        output = self.classifier(output)
        output = self.sigmoid(output)
        return output
    

class PModel2(nn.Module):
    def __init__(self, model_name='bert', HIDDEN_SIZE=None):
        super(PModel2, self).__init__()
        logging.set_verbosity_error()
        self.num_labels = 5
        self.n_hidden = 96
        if model_name =='bert':
            self.plm = AutoModel.from_pretrained("bert-base-uncased")
            self.layer_norm = nn.LayerNorm(BERT_HIDDEN_SIZE)
            self.hidden = nn.Linear(BERT_HIDDEN_SIZE, self.n_hidden)
            self.classifier = nn.Linear(self.n_hidden, self.num_labels)
        elif model_name == 'deberta':
            self.plm = AutoModel.from_pretrained("../T2_prompt/PLM/deberta-v3-large")
            self.layer_norm = nn.LayerNorm(DEBERTA_HIDDEN_SIZE)
            self.hidden = nn.Linear(DEBERTA_HIDDEN_SIZE, self.n_hidden)
            self.classifier = nn.Linear(self.n_hidden, self.num_labels)
        elif model_name == 'roberta':
            self.plm = AutoModel.from_pretrained("xlm-roberta-large")
            self.layer_norm = nn.LayerNorm(DEBERTA_HIDDEN_SIZE)
            self.hidden = nn.Linear(DEBERTA_HIDDEN_SIZE, self.n_hidden)
            self.classifier = nn.Linear(self.n_hidden, self.num_labels)
        else:
            self.plm = AutoModel.from_pretrained(model_name) 
            self.layer_norm = nn.LayerNorm(HIDDEN_SIZE)
            self.classifier = nn.Linear(HIDDEN_SIZE, self.num_labels)
        self.dropout = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()
        self.lrelu = nn.LeakyReLU(0.1)
        

    def forward(self, tokens_tensors,  masks_tensors):
        outputs = self.plm(input_ids=tokens_tensors, attention_mask=masks_tensors)
        output = self.layer_norm(outputs['last_hidden_state'][:, 0, :])
        output = self.dropout(output)
        output = self.hidden(output)
        output = self.lrelu(output)
        output = self.classifier(output)
        output = self.sigmoid(output)
        return output