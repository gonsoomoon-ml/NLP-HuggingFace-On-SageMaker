from torch.utils.data import SubsetRandomSampler
import numpy as np
import random
import torch
import os

from torch.utils.data import DataLoader, SubsetRandomSampler

from data_util import read_nsmc_split, NSMCDataset
import config

# from datasets import load_dataset
from transformers import (
    ElectraModel, 
    ElectraTokenizer, 
    ElectraForSequenceClassification, 
    Trainer, 
    TrainingArguments, 
    set_seed
)



def create_train_meta(train_dataset):    
    '''
    레이블 갯수, label2id, id2label 생성
    '''
    labels = train_dataset.label_names
    num_labels = len(labels)
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label
        
    return num_labels, label2id, id2label


def _get_train_data_loader(args, train_texts, train_labels, logger):        
    '''
    train data loader 생성
    '''
    # Electra Model 입력 인코딩 생성    
    tokenizer = ElectraTokenizer.from_pretrained(args.tokenizer_id)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    train_dataset = NSMCDataset(train_encodings, train_labels)        
    logger.info(f"size of train_dataset : {len(train_dataset)}")        

    # 일부 데이터만 사용할지 결정
    if args.use_subset_train_sampler:
        subset_train_sampler = create_random_sampler(train_dataset, frac=0.01, is_shuffle=True, logger=logger)
        train_loader = DataLoader(dataset=train_dataset, 
                                  batch_size=args.train_batch_size, 
                                  sampler=subset_train_sampler)    
    else:
        train_sampler = create_random_sampler(train_dataset, frac=1, is_shuffle=True, logger=logger)
        train_loader = DataLoader(dataset=train_dataset, 
                                  batch_size=args.train_batch_size, 
                                  sampler=train_sampler)    

    return train_loader, train_dataset

        
def _get_val_data_loader(args, val_texts, val_labels, logger):        
    '''
    eval data loader 생성
    '''
    # Electra Model 입력 인코딩 생성    
    tokenizer = ElectraTokenizer.from_pretrained(args.tokenizer_id)

    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    val_dataset = NSMCDataset(val_encodings, val_labels)
    logger.info(f"size of val_dataset : {len(val_dataset)}")

    eval_sampler = create_random_sampler(val_dataset, frac=1, is_shuffle=False, logger=logger)
    eval_loader = DataLoader(dataset=val_dataset, 
                                  shuffle=False, 
                                  batch_size=args.eval_batch_size, 
                                  sampler=eval_sampler)    

    return eval_loader, val_dataset        


def _get_test_data_loader(args, test_texts, test_labels, logger):        
    '''
    test data loader 생성
    '''
    # Electra Model 입력 인코딩 생성    
    tokenizer = ElectraTokenizer.from_pretrained(args.tokenizer_id)

    test_encodings = tokenizer(test_texts, truncation=True, padding=True)
    test_dataset = NSMCDataset(test_encodings, test_labels)
    logger.info(f"size of test_dataset : {len(test_dataset)}")

    test_sampler = create_random_sampler(test_dataset, frac=1, is_shuffle=False, logger=logger)
    test_loader = DataLoader(dataset=test_dataset, 
                                  shuffle=False, 
                                  batch_size=args.test_batch_size, 
                                  sampler=test_sampler)    

    return test_loader , test_dataset       





def create_random_sampler(dataset, frac, is_shuffle, logger):
    '''
    일부 데이터만 일부 랜덤 샘플링
    '''
    dataset_size = len(dataset)
    dataset_indices = list(range(dataset_size))    
    sample_size = int(dataset_size * frac)
    if is_shuffle:
        sample_indices = random.sample(dataset_indices, k=sample_size)    
    else:
        sample_indices = np.arange(sample_size).tolist()
    
    logger.info(f'dataset size with frac: {frac} ==> {len(sample_indices)}')
    #print(sample_indices)
    sampler = SubsetRandomSampler(sample_indices)    
    
    return sampler



def load_model_network(args, train_dataset, device, logger):        
    '''
    모델 네트워크를 로딩
    '''
    num_labels, label2id, id2label = create_train_meta(train_dataset)    
    logger.info(f"num_labels: {num_labels}")
    logger.info(f"label2id: {label2id}")    
    logger.info(f"id2label: {id2label}")        
    
    model = ElectraForSequenceClassification.from_pretrained(
        args.model_id, num_labels=num_labels, label2id=label2id, id2label=id2label
    )

    model.to(device)
    model.train()

    return model



def train_epoch(args, model, train_loader, optimizer, epoch, device, logger, sampler=None):
    if sampler:
        sampler.set_epoch(epoch)
    
    model.train()

    for batch_idx, batch in enumerate(train_loader,1):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        #print("output: \n", outputs)
        loss = outputs[0]
        loss.backward()
        optimizer.step()

#         print("batch_idx: ", batch_idx)
#         print("args.log_interval: ", args.log_interval)
        if batch_idx % args.log_interval == 0:
            logger.info(
                "Train Epoch: {} [{}/{} ({:.0f}%)] Loss={:.6f};".format(
                    epoch,
                    batch_idx * len(batch),
                    len(train_loader.sampler),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
        
def eval_epoch(args, model, epoch, device, logger, eval_loader):
    '''
    테스트 데이타로 추론하여 평가
    '''
    acc_list=[]
    model.eval()
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#             logger.debug(f"outputs: \n {outputs[1]}")
#             logger.debug(f"labels:  \n {labels})")
            acc = accuracy(outputs[1], labels)
            acc_list.append(acc)
            

        logger.info(
            "Train Epoch: {} Acc={:.6f};".format(
            epoch,
            np.mean(acc_list),
                )
            )  
    return np.mean(acc_list)

def test_model(args, model, device, logger, test_loader):
    '''
    테스트 데이타로 추론하여 평가
    '''
    acc_list=[]
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            
            acc = accuracy(outputs[1], labels)
            acc_list.append(acc)
            

        logger.info(
            "Test Accuracy: Acc={:.6f};".format(
            np.mean(acc_list))
            )  
        
    return np.mean(acc_list)

def test_model_with_predicton(model, device, logger, test_loader):
    '''
    테스트 데이타로 추론하여 평가
    '''
    acc_list=[]
    prediction_list = []
    label_list = []
    
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            
            acc = accuracy(outputs[1], labels)
            acc_list.append(acc)
            prediction_list.append(outputs[1])
            label_list.append(labels)

        logger.info(
            "Test Accuracy: Test-Acc : {:.6f};".format(
            np.mean(acc_list))
            )  
        
    return np.mean(acc_list), prediction_list, label_list


        
def save_best_model(model, acc, epoch, best_acc, model_dir, logger):        
    if acc > best_acc:
        best_acc, best_epoch = acc, epoch

        if not os.path.exists(config.model_dir):
            os.makedirs(config.model_dir, exist_ok=True)
            
        ### Save Model 을 다른 곳에 저장
        _save_model(model, model_dir, f'{config.model_name}', logger)  
    else:
        pass

    return best_acc


#     return best_hr, best_ndcg, best_epoch
def accuracy(out, labels):
    # outputs = np.argmax(out, axis=1)
    outputs = torch.argmax(out, axis=1)
    outputs = outputs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()    
    
    return np.sum(outputs==labels)/float(labels.size)

def _save_model(model, model_dir, model_weight_file_name, logger):
    '''
    model_dir (예: /opt/ml/model) 에 모델 저장
    '''
    path = os.path.join(model_dir, model_weight_file_name)
    logger.info(f"the model is saved at {path}")    
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.state_dict(), path)


######################################
# Trainer
######################################

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# compute metrics function for binary classification
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

def compute_metrics_with_label(preds, labels):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": round(acc,3), 
            "f1": round(f1,3), 
            "precision": round(precision,3), 
            "recall": round(recall,3)}

    