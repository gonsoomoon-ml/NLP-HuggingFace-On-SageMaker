import os

# project prefix
project_prefix = 'sm_hugging_naver_toxic'

# dataset name 
dataset = 'naver_toxic_review'
labels = ['toxic','obscene','threat','insult','identity_hate']

# data location
train_data_dir = f'data/{dataset}'
test_data_dir = f'data/{dataset}'

# model
tokenizer_id = 'monologg/koelectra-small-v3-discriminator'
model_id = 'monologg/koelectra-small-v3-discriminator'
num_cls_vector = 256

# model artifacts
model_dir = f'models/{dataset}'
os.makedirs(model_dir, exist_ok=True)






