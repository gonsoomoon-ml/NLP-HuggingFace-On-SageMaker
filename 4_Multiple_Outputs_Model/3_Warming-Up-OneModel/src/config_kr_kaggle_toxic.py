import os

# project prefix
project_prefix = 'sm_hugging_translated_kaggle_toxic'

# dataset name 
dataset = 'translated_kaggle_toxic_review'
labels = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

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






