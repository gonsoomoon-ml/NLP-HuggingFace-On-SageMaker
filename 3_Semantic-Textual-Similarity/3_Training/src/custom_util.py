import boto3, os
import pandas as pd
import numpy as np


####################
# 데이터 준비 함수
####################

import progressbar

class MyProgressBar():
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar=progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()

import zipfile
def extract_news_data(args, news_path, logger): 
    '''
    ZIP 파일 압축 해제하고 Text 데이터 리턴
    '''
    # with zipfile.ZipFile(news_path, 'r') as zip_ref:
    #     zip_ref.extractall(train_dir)

    # !rm -rf {news_path}    

    news_data = []
    f = open(f'{args.train_dir}/KCCq28_Korean_sentences_EUCKR_v2.txt', 'rt', encoding='cp949')
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        news_data.append(line)
    f.close()
    
    logger.info(f"news_data length:  {len(news_data)}")
    
    return news_data

            
            
####################
# 데이터 전처리 함수
####################

stop_words = ['가다','있다']
def Okt_tokenizer(raw, pos=["Noun","Adjective"], stopword=stop_words):
#def Okt_tokenizer(raw, pos=["Noun"], stopword=stop_words):
    '''
    문장의 토큰나이징 함수
    '''
    from konlpy.tag import Okt
    Okt = Okt()
    

    
    word_list = [
        word for word, tag in Okt.pos(
            raw, 
            norm=True,   # normalize 그랰ㅋㅋ -> 그래ㅋㅋ
            stem=True    # stemming 바뀌나->바뀌다
            )
            if len(word) > 1 and tag in pos \
            and word not in stopword 
        ]
    
    #  중복 제거
    #  중복 제거가 유사도에는 좋은 영향을 안줌. 몇 개 샘플로 확인
    # word_list = list(dict.fromkeys(word_list))
    
    return ' '.join(word_list)
    

import re    
def remove_spec_chars(s):
    '''
    특수 문자 제거.
    '''
    
    if pd.isnull(s):
        return s
    s = re.sub(r'\W+', ' ', s)   # 알파벳, 숫자 이외 모두 제거

    
    
    remove_ptns = ['\r', '\n', '>','<','{','}','[',']',']'
                   '$','/', ',','-','_','=','+','.','!','@','^','%','#','*','(',')',
                   '₩','`','&','|',':',';','<','>','?','\\','\'','"'
                  ]
    
    dic = dict(zip(remove_ptns, [' '] * len(remove_ptns)))
    regex = re.compile("|".join(map(re.escape, dic.keys())), 0)
    s = regex.sub(lambda match: dic[match.group(0)], s)
    s = re.sub('\s+',' ', s)

    return s
    
def encode_multi_gpu_embedding(model, corpus):
    corpus_embeddings = model.encode(corpus, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
#     model.start_multi_process_pool
#     pool = model.start_multi_process_pool()

#     #Compute the embeddings using the multi-process pool
#     corpus_embeddings = model.encode_multi_process(sentences, pool)
#     print("Embeddings computed. Shape:", corpus_embeddings.shape)

#     #Optional: Stop the proccesses in the pool
#     model.stop_multi_process_pool(pool)
    
    return corpus_embeddings

import traceback
def preprocess_doc(df):
    '''
    특수문자를 제거한 후에 Tokenization 을 함.
    '''
    try: 
        # ddf["doc_cl"] = ddf["doc"].apply(preproces_doc, meta=('doc', 'object')).compute()   
        pd_df = df.copy()
        
        pd_df["doc_cl"] = df["doc"].apply(remove_spec_chars)
        pd_df["doc_cl"] = pd_df["doc_cl"].apply(Okt_tokenizer)
        pd_df["doc_cl"]
        
        return pd_df
    
    except Exception:
        print(traceback.format_exc())    




####################
# 추론 도움 함수
####################
from sentence_transformers import util
import torch

def semantic_search(model, corpus, corpus_embeddings, queries, top_k):
    # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
    top_k = min(top_k, len(corpus))
    for query in queries:
        query_embedding = model.encode(query, convert_to_tensor=False)

        # We use cosine-similarity and torch.topk to find the highest 5 scores
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        print("\n======================")
        print("Query:", query)
        print("======================")        
        print(f"\nTop {top_k} most similar sentences in corpus:\n")

        for score, idx in zip(top_results[0], top_results[1]):
            print("(Score: {:.4f})".format(score), corpus[idx] )






