import torch,logging,random
from pathlib import Path
from transformers import LayoutLMTokenizer,LayoutLMForTokenClassification
from torch.optim import AdamW
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

def train():
    # Generate synthetic training data
    samples=[]
    banks=['Maybank','CIMB','Public Bank','RHB','HSBC','UOB','Standard Chartered','DuitNow']
    for b in banks:
        for i in range(5):
            date,number="251031",f"{random.randint(100000,999999)}"
            if b=="Maybank":tid=f"MYCN{date}{number}"
            elif b=="CIMB":tid=f"B10-{date[:4]}-{number}"
            elif b=="Public Bank":tid=f"PBB{date}{number}"
            elif b=="RHB":tid=f"RHB{date}{number}"
            elif b=="HSBC":tid=f"HSBC{date}{number}"
            elif b=="UOB":tid=f"UOB{date}{number}"
            elif b=="Standard Chartered":tid=f"SCB{date}{number}"
            else:tid=f"DN{date}{number}"
            text=f"{b} Transfer Reference: {tid} Status: Successful"
            tokens=[];words=text.split();pos=0
            for w in words:
                s=text.find(w,pos);e=s+len(w)
                tokens.append({'text':w,'start':s,'end':e,'bbox':[s,0,e,20]})
                pos=e
            labels=[0]*len(tokens)
            for j,tk in enumerate(tokens):
                if tk['text']==tid:
                    labels[j]=1;break
            bboxes=[[max(0,min(1000,int(t['bbox'][0]*0.1))),0,1000,1000] for t in tokens]
            samples.append({'text':text,'labels':labels,'bboxes':bboxes,'tokens':tokens,'id':f'{b}_{i}','transaction_id':tid})
            logger.info(f"Synth {b} -> {tid}")
    
    logger.info(f"Training with {len(samples)} samples")
    tokenizer=LayoutLMTokenizer.from_pretrained('microsoft/layoutlm-base-uncased',use_fast=False)
    model=LayoutLMForTokenClassification.from_pretrained('microsoft/layoutlm-base-uncased',num_labels=3)
    device=torch.device('cuda'if torch.cuda.is_available()else'cpu')
    model.to(device);model.train()
    optimizer=AdamW(model.parameters(),lr=2e-5,weight_decay=0.01)
    for epoch in range(20):
        random.shuffle(samples)
        total_loss=0
        for sample in samples:
            encoding=tokenizer(sample['text'],max_length=512,padding='max_length',truncation=True,return_tensors='pt')
            labels=torch.tensor([sample['labels']+[0]*(512-len(sample['labels']))]).to(device)
            bboxes=torch.tensor([sample['bboxes']+[[0,0,1000,1000]]*(512-len(sample['bboxes']))]).to(device)
            outputs=model(input_ids=encoding['input_ids'].to(device),attention_mask=encoding['attention_mask'].to(device),bbox=bboxes,labels=labels)
            loss=outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss+=loss.item()
        logger.info(f"Epoch {epoch+1}/20 loss={total_loss/len(samples):.4f}")
        if total_loss/len(samples)<0.01:break
    Path('app/models').mkdir(exist_ok=True)
    torch.save({'model_state_dict':model.state_dict(),'tokenizer_name':'microsoft/layoutlm-base-uncased'},'app/models/transaction_extractor_comprehensive.pt')
    logger.info("Model saved")

if __name__=="__main__":train()