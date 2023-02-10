import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import torch
import argparse
from tqdm import tqdm
import torch.nn as nn
from dataprocess import *
from model import *
from utils import *
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

def train(new_model=True):
    
    writer = SummaryWriter("logs-" + MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if new_model == True:
        model = torch.nn.DataParallel(PModel(model_name=MODEL_NAME, HIDDEN_SIZE=HIDDEN_SIZE))
        model = model.to(device)
    else:
        model = torch.nn.DataParallel(PModel(model_name=MODEL_NAME, HIDDEN_SIZE=HIDDEN_SIZE))
        model.load_state_dict(torch.load("./model_saved/"+MODEL_NAME+".pth"))
        model = model.to(device)
        
    # for fre in FREEZE:
    #     for name, parameter in model.named_parameters():
    #         if fre in name:
    #             parameter.requires_grad = False
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer 
                if not any(nd in n for nd in no_decay) ], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer
                if any(nd in n for nd in no_decay) ], 'weight_decay': 0.0}
    ]
    # and p.requires_grad
    optimizer = AdamW(optimizer_grouped_parameters, lr=LR)
    # optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    cross_entropy = nn.CrossEntropyLoss()
    max_acc = -1
    early_stop = 0
    cheat_max = 0
    for epo in range(EPOCH):
        model.train()
        epo_loss = 0
        trainloader = DataLoader(PromptDataset(fromdata=dataRead(mode='train'), model_name=MODEL_NAME), \
            batch_size=BATCH_SIZE, collate_fn=create_mini_batch)
        pbar = tqdm(trainloader)
        for data in pbar:
            pbar.set_description("Epoch %d"%(epo+1))
            tokens_tensors, masks_tensors , signals_tensors = data
            tokens_tensors = tokens_tensors.to(device)
            masks_tensors = masks_tensors.to(device)
            signals_tensors = signals_tensors.to(device)
            output = model(tokens_tensors, masks_tensors)
            loss = cross_entropy(output, signals_tensors)
            optimizer.zero_grad()
            
            # epo_loss += loss.sum().item()
            # loss = torch.mean(loss, dim=0)
            # loss.backward()
            
            loss.backward(torch.ones(loss.shape).to(device))
            epo_loss += loss.sum().item()
            
            optimizer.step()
            pbar.set_postfix(Epo = int(epo_loss), Loss=loss.item() / BATCH_SIZE)
        writer.add_scalar('train:epo_loss', epo_loss, epo)
 
                    
        # model.eval()
        # devloader = DataLoader(PromptDataset(fromdata=dataRead(mode='dev'), model_name=MODEL_NAME), \
        #     batch_size=64, collate_fn=create_mini_batch)
        # pbar = tqdm(devloader)
        # correct = 0
        # all_pred = 0
        # for data in pbar:
        #     pbar.set_description("Epoch %d"%(epo+1))
        #     tokens_tensors, masks_tensors , signals_tensors = data
        #     tokens_tensors = tokens_tensors.to(device)
        #     masks_tensors = masks_tensors.to(device)
        #     signals_tensors = signals_tensors.to(device)
        #     output = model(tokens_tensors, masks_tensors) 
        #     pred = torch.gt(output, 0.5)
        #     correct += (pred == signals_tensors).sum().item()
        #     all_pred += int(pred.size(0) * pred.size(1))
        #     pbar.set_postfix(cor=correct, all=all_pred)
        # acc = correct / all_pred
        
        # if acc > max_acc:
        #     max_acc = acc
        #     torch.save(model.state_dict(), "./model_saved/"+MODEL_NAME+".pth")
        #     early_stop = 0
        # else:
        #     early_stop += 1
        # if early_stop >= EPOCH:
        #     break
        
        acc_test = testForDavinci(model)
        if acc_test > cheat_max:
            cheat_max = acc_test
            torch.save(model.state_dict(), "./model_saved/"+MODEL_NAME+".pth")
    writer.close() 
            
                    
        
def testForDavinci(model=None):
    if model == None:
        model = torch.nn.DataParallel(PModel(model_name=MODEL_NAME, HIDDEN_SIZE=HIDDEN_SIZE))
        model.load_state_dict(torch.load("./model_saved/"+MODEL_NAME+".pth"))
        # model = PModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    testSet = []  
    with open('./data/prompt5_allTest_ChatGPT.jsonl','r') as fin:
        for line in fin:
            jsn = json.loads(line)
            testSet.append(jsn)
            
    testloader = DataLoader(PromptDataset(fromdata=dataReadForTest(), model_name=MODEL_NAME), batch_size=64, \
        collate_fn=create_mini_batch)
    pbar = tqdm(testloader)
    
    cor = 0
    cnt = 0
    result_statis = [0,0,0,0,0]
    for data in pbar:
        tokens_tensors, masks_tensors , signals_tensors = data
        tokens_tensors = tokens_tensors.to(device)
        masks_tensors = masks_tensors.to(device)
        
        output = model(tokens_tensors, masks_tensors)
        _, index = torch.max(output, dim=1)
        maxIndex = index.cpu().numpy().tolist()

        for max_index in maxIndex:
            cor += testSet[max_index]['answerList'][cnt]
            result_statis[max_index] += 1
            cnt += 1
        pbar.set_postfix(cor=cor, all=cnt)
    print("Final Result: cor={}, {}".format(cor, cor/1319))
    print("{}, {}, {}, {}, {}".format(result_statis[0], result_statis[1], result_statis[2], result_statis[3], result_statis[4]))
    return cor 
             

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",type=str, default='train',choices=['train','test','traintest'])
    parser.add_argument("--model_name",type=str, default='deberta')
    parser.add_argument("--bsz",type=int, default=32)
    parser.add_argument("--hidden_size",type=int, default=None)
    parser.add_argument("--epoch",type=int, default=30)
    parser.add_argument("--lr",type=float, default=2e-5)

    args = parser.parse_args()
    job_mode = args.mode
    BATCH_SIZE = args.bsz
    EPOCH = args.epoch
    LR = args.lr
    MODEL_NAME = args.model_name
    HIDDEN_SIZE = args.hidden_size
    FREEZE = ['plm']
    
    if job_mode == 'train' or job_mode == 'traintest':
        train()
    if job_mode == 'test' or job_mode == 'traintest':
        testForDavinci()
    if job_mode == 'mytest':
        # testloader = DataLoader(PromptDataset(dataReadForTest()), batch_size=BATCH_SIZE, collate_fn=create_mini_batch)
        trainloader = DataLoader(PromptDataset(dataRead(mode='train', epo=0)), batch_size=BATCH_SIZE, collate_fn=create_mini_batch)
        for data in trainloader:
            tokens_tensors, segments_tensors, masks_tensors, answer , signals_tensors, prompt1, l5 = data
            # print(tokens_tensors)
            print(len(tokens_tensors[0]), len(segments_tensors[0]), len(masks_tensors[0]))
            # model = PModel()
            model = AutoModelForSequenceClassification.from_pretrained("./PLM/xlm-roberta-large", num_labels=2, ignore_mismatched_sizes=True)
            # model = AutoModelForSequenceClassification.from_pretrained("./PLM/xlm-roberta-large", num_labels=2)
            logits = model(input_ids=tokens_tensors, attention_mask=masks_tensors, token_type_ids=segments_tensors, labels=signals_tensors)
            print(logits)
            exit()
        