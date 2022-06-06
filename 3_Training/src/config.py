# project prefix
project_prefix = 'sm_hugging_nsmc'

# dataset name 
dataset = 'nsmc'
assert dataset in ['nsmc']

# data location
raw_data_dir = f'data/{dataset}/rawdata/'
train_data_dir = f'data/{dataset}/train'
test_data_dir = f'data/{dataset}/test'

# data loader
use_subset_train_sampler = "True"

# model
tokenizer_id = 'monologg/koelectra-small-v3-discriminator'
model_id = 'monologg/koelectra-small-v3-discriminator'

# model artifacts
output_data_dir = f'output/{dataset}'
model_dir = f'models/{dataset}'
checkpoint_dir = f'checkpoint/{dataset}'

# Evaluation 
is_evaluation = "True"
is_test = "True"
eval_ratio = 0.2

model_name = 'sentimental-electro-hf'




