import argparse
import os
import json
import sys

from train_lib import train

def parser_args():
    parser = argparse.ArgumentParser()

    # Default Setting
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=0)    
#    parser.add_argument("--disable_tqdm", type=bool, default=True)
    parser.add_argument("--fp16", type=bool, default=True)    
    parser.add_argument("--tokenizer_id", type=str, default='monologg/koelectra-small-v3-discriminator')
    parser.add_argument("--model_id", type=str, default='monologg/koelectra-small-v3-discriminator')
    
    # SageMaker Container environment
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train_data_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    #parser.add_argument("--test_data_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument('--checkpoint_dir', type=str, default='/opt/ml/checkpoints')   
    parser.add_argument("--is_evaluation", type=bool, default=True)    
    parser.add_argument("--eval_ratio", type=float, default=0.2)       
    parser.add_argument("--use_subset_train_sampler", type=bool, default=True)        
    parser.add_argument("--log_interval", type=int, default=50)      
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"]) 
    parser.add_argument("--seed", type=int, default=42)  

    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = parser_args()
    train(args)    



