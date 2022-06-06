import os
import time
import numpy as np
import logging
import sys
import time

import torch
from transformers import AdamW

from glob import glob
from sklearn.model_selection import train_test_split
 
# Custom Module

from data_util import read_nsmc_split
from train_util import create_train_meta, create_random_sampler, train_epoch, eval_epoch, save_best_model
from train_util import _get_train_data_loader, _get_val_data_loader, load_model_network

import config

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)
# logger.setLevel(logging.WARNING)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler(sys.stdout))    


def train(args):
    '''
    1. args 를 받아서 입력 데이터 로딩
    2. 데이터 세트 생성
    3. 모델 네트워크 생성
    4. 훈련 푸프 실행
    5. 모델 저장
    '''
    #######################################
    ## 커맨드 인자 확인     
    #######################################
    
    logger.info("=====> Load Input Arguemtn <===========")            
    logger.info(f"##### Args: \n {vars(args)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.debug(f"device: {device}")    
    
    
    #######################################
    ## 데이터 로딩 및 데이터 세트 생성 
    #######################################

    logger.info("=====> data loading <===========")        

    train_data_filenames = glob(os.path.join(args.train_data_dir, '*_train.txt'))
    logger.info(f'train_data_filenames {train_data_filenames}')
    
    # 1개의 파일만 지정 함. 
    # --> 복수개의 파일시 코드 수정 필요
    train_file_path = train_data_filenames[0]
    
    
    # 훈련 Text, Label 로딩    
    train_texts, train_labels = read_nsmc_split(train_file_path)
    
    logger.info(f'train_file_path {train_file_path}')    
    logger.debug(f"len: {len(train_texts)} \nSample: {train_texts[0:5]}")
    logger.debug(f"len: {len(train_labels)} \nSample: {train_labels[0:5]}")

    # evaluation == True 이면 train 데이타를 다시 train, val 로 분리
    if args.is_evaluation:
        train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, 
                                                                        train_labels, 
                                                                        test_size=args.eval_ratio)
    train_loader, train_dataset = _get_train_data_loader(args, train_texts, train_labels, logger)        
    

    #######################################
    ## 모델 네트워크 생성
    #######################################
    logger.info("=====> model loading <===========")        
    
    model = load_model_network(args, train_dataset, device, logger)

    #######################################
    ## 옵티마이저 정의
    #######################################
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)


    #######################################
    ## 훈련 루프 실행
    #######################################
    logger.info("=====> Training Loop <===========")        
    
    best_acc = 0
    for epoch in range(args.epochs):
        start_time = time.time()

        train_epoch(args, 
                    model, 
                    train_loader, 
                    optimizer, 
                    epoch, 
                    device, 
                    logger,
                    sampler=None, 
                    )            

        eval_loader = elapsed_time = time.time() - start_time    
        print("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
                    time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))

        if args.is_evaluation:
            
            eval_loader = _get_val_data_loader(args, val_texts, val_labels, logger)        
            acc = eval_epoch(args, 
                       model, 
                       epoch, 
                       device, 
                       logger,
                       eval_loader)

            best_acc = save_best_model(model, 
                                       acc, 
                                       epoch, 
                                       best_acc,
                                       args.model_dir,
                                       logger)            

    
    
