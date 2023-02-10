import random
import json
import numpy as np

def rdmMakeList(n,m,l):
    """
    n:范围从0到n-1
    m:共m个
    l:回避序列l
    """
    now = []
    for _ in range(m):
        x = random.randint(0, n-1)
        while x in now or x in l:
            x = random.randint(0, n-1)
        now.append(x)
    return now


def makePrompt1(jsn, l5):
    prompt1 = "Question:[Ques]\nAnswer:[Ans]\n\n"
    prompt = ""
    for l in l5:
        prompt += prompt1.replace('[Ques]', jsn[l]['question']).replace('[Ans]', jsn[l]['answer'])
    return prompt
        
        
def makePrompt2(jsn, q):
    prompt2 = "Question:[Ques]\nAnswer:"
    prompt = ""
    prompt += prompt2.replace('[Ques]', jsn[q]['question'])
    return prompt


def writeJsonl(data, path):
    with open(path,'w+') as fout:
        for d in data:
            fout.write(json.dumps(d) + '\n')    
    

def anaAnswer(str):
    str = str.replace('\n','')
    x = str.find('####')
    x += 5
    ans = ""
    while x<len(str) and (str[x]=='.' or str[x].isdigit() or str[x]=='-'):
        ans += str[x]
        x += 1
    return ans
            
            
