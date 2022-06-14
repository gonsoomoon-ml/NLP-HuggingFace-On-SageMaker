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
 
from transformers import (
    Trainer, 
    TrainingArguments, 
    set_seed
)

    
    
# Custom Module

from data_util import read_nsmc_split
from train_util import _get_train_data_loader, _get_val_data_loader, _get_test_data_loader, load_model_network
from train_util import create_train_meta, create_random_sampler, train_epoch, eval_epoch, test_model, save_best_model, _save_model


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
        
    logger.info("=====> Loading Train Dataset <===========")                        
    train_loader, train_dataset = _get_train_data_loader(args, train_texts, train_labels, logger)        
        
    # 테스트 데이터 셋으로 검증 여부
    if args.is_test:
        logger.info("=====> Loading Test Dataset <===========")                
        test_data_filenames = glob(os.path.join(args.test_data_dir, '*_test.txt'))
        logger.info(f'test_data_filenames {test_data_filenames}')

        # 1개의 파일만 지정 함. 
        # --> 복수개의 파일시 코드 수정 필요
        test_file_path = test_data_filenames[0]
        
        # 훈련 Text, Label 로딩    
        test_texts, test_labels = read_nsmc_split(test_file_path)

        logger.info(f'test_file_path {test_file_path}')    
        logger.debug(f"len: {len(test_texts)} \nSample: {test_texts[0:5]}")
        logger.debug(f"len: {len(test_labels)} \nSample: {test_labels[0:5]}")
        
        test_loader = _get_test_data_loader(args, test_texts, test_labels, logger)                


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
    
    # 검증 셋 성능평가
    if args.is_evaluation:
        logger.info("=====> Loading Validation Dataset <===========")                                    
        eval_loader = _get_val_data_loader(args, val_texts, val_labels, logger)        

    
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

        elapsed_time = time.time() - start_time    
        print("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
                    time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))

        # 검증 셋 성능평가
        if args.is_evaluation:
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
        else:
            ### Save Model 을 다른 곳에 저장
            _save_model(model, args.model_dir, f'{config.model_name}.pth', logger)  

            
    # 테스트 셋 검증
    if args.is_test:
        logger.info("=====> test model performance <===========")                
        test_loader = _get_test_data_loader(args, test_texts, test_labels, logger)        
        acc = test_model(args, 
                        model, 
                        device, 
                        logger,
                        test_loader)

from transformers.trainer_utils import get_last_checkpoint
from train_util import compute_metrics, compute_metrics_with_label

            
def train_Trainer(args):
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
    
    # checkpoint_dir 생성
    os.makedirs(args.checkpoint_dir, exist_ok=True)        
    
    #######################################
    ## 훈련 데이터 로딩 및 데이터 세트 생성 
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
        
    logger.info("=====> Loading Train Dataset <===========")                        
    train_loader, train_dataset = _get_train_data_loader(args, train_texts, train_labels, logger)        
        
    #######################################
    ## 모델 네트워크 생성
    #######################################
    logger.info("=====> model loading <===========")        
    
    model = load_model_network(args, train_dataset, device, logger)

    
    #######################################
    ## Trainer 정의
    #######################################
    logger.info("=====> Define Trainer <===========")        
    
    # 검증 셋 성능평가
    if args.is_evaluation:
        logger.info("=====> Loading Validation Dataset <===========")                                    
        eval_loader, val_dataset = _get_val_data_loader(args, val_texts, val_labels, logger)        

        training_args = TrainingArguments(
            output_dir= args.checkpoint_dir,
            overwrite_output_dir=True if get_last_checkpoint(args.checkpoint_dir) is not None else False,    
            num_train_epochs=  args.epochs,              # total number of training epochs
            per_device_train_batch_size= args.train_batch_size,  # batch size per device during training
            per_device_eval_batch_size= args.eval_batch_size,   # batch size for evaluation
            learning_rate=float(args.learning_rate),    
            warmup_steps= args.warmup_steps,                # number of warmup steps for learning rate scheduler
            weight_decay= args.weight_decay,               # strength of weight decay    
            fp16=args.fp16,    
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            disable_tqdm=args.disable_tqdm,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            logging_dir=f"{args.output_data_dir}/logs",
            logging_steps=  args.logging_steps,
        )

        
        trainer = Trainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics    
        )
    else:
        training_args = TrainingArguments(
            output_dir= args.checkpoint_dir,
            overwrite_output_dir=True if get_last_checkpoint(args.checkpoint_dir) is not None else False,    
            num_train_epochs=  args.epochs,              # total number of training epochs
            per_device_train_batch_size= args.train_batch_size,  # batch size per device during training
            learning_rate=float(args.learning_rate),    
            warmup_steps= args.warmup_steps,                # number of warmup steps for learning rate scheduler
            weight_decay= args.weight_decay,               # strength of weight decay    
            fp16=args.fp16,    
            save_strategy="epoch",
            save_total_limit=1,
            disable_tqdm=args.disable_tqdm,
            metric_for_best_model="accuracy",
            logging_dir=f"{args.output_data_dir}/logs",
            logging_steps=  args.logging_steps,
        )

        
        trainer = Trainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
        )

    #######################################
    ## Training 실행
    ## 기존의 가중치 (checkpoint 파일) 가 있으면 그 가중치를 로딩하여 훈련 시작
    #######################################
        
    # Set seed before initializing model
    set_seed(args.seed)
        
    # train model
    if get_last_checkpoint(args.checkpoint_dir) is not None:
        logger.info("***** Continue Training *****")
        last_checkpoint = get_last_checkpoint(args.checkpoint_dir)
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()

    #######################################
    ## Test Dataset 평가
    #######################################
        
    if args.is_test:    

        logger.info("=====> Loading Test Files <===========")                
        test_data_filenames = glob(os.path.join(args.test_data_dir, '*_test.txt'))
        logger.info(f'test_data_filenames {test_data_filenames}')

        # 1개의 파일만 지정 함. 
        # --> 복수개의 파일시 코드 수정 필요
        test_file_path = test_data_filenames[0]
        
        # 훈련 Text, Label 로딩    
        test_texts, test_labels = read_nsmc_split(test_file_path)

        logger.info(f'test_file_path {test_file_path}')    
        logger.debug(f"len: {len(test_texts)} \nSample: {test_texts[0:5]}")
        logger.debug(f"len: {len(test_labels)} \nSample: {test_labels[0:5]}")
        
        logger.info("=====> Loading Test Dataset <===========")                        
        test_loader, test_dataset = _get_test_data_loader(args, test_texts, test_labels, logger)         
        
        
        logger.info("=====> Prediction on Test Dataset <===========")                                
        # Use the model to get predictions
        test_predictions = trainer.predict(test_dataset)
        # 각각의 예측에 대해서 가장 높은 취함.
        test_predictions_argmax = np.argmax(test_predictions[0], axis=1)

        
        test_references = test_dataset.labels
        # Compute accuracy
        metrics = compute_metrics_with_label(preds=test_predictions_argmax, labels=test_references)       
        logger.info(f"Test Metrics: {metrics}")
        
            
    #######################################
    ## 모델 저장
    #######################################
            
    # Saves the model to s3 uses os.environ["SM_MODEL_DIR"] to make sure checkpointing works
    trainer.save_model(args.model_dir)

        


    
    
