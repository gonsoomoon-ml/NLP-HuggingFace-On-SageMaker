Multi-Lingual Models
- https://www.sbert.net/docs/pretrained_models.html

The following models generate aligned vector spaces, i.e., similar inputs in different languages are mapped close in vector space. You do not need to specify the input language. Details are in our publication Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation. We used the following 50+ languages: ar, bg, ca, cs, da, de, el, en, es, et, fa, fi, fr, fr-ca, gl, gu, he, hi, hr, hu, hy, id, it, ja, ka, ko, ku, lt, lv, mk, mn, mr, ms, my, nb, nl, pl, pt, pt-br, ro, ru, sk, sl, sq, sr, sv, th, tr, uk, ur, vi, zh-cn, zh-tw.

Semantic Similarity

These models find semantically similar sentences within one language or across languages:

distiluse-base-multilingual-cased-v1: Multilingual knowledge distilled version of multilingual Universal Sentence Encoder. Supports 15 languages: Arabic, Chinese, Dutch, English, French, German, Italian, Korean, Polish, Portuguese, Russian, Spanish, Turkish.
distiluse-base-multilingual-cased-v2: Multilingual knowledge distilled version of multilingual Universal Sentence Encoder. This version supports 50+ languages, but performs a bit weaker than the v1 model.
paraphrase-multilingual-MiniLM-L12-v2 - Multilingual version of paraphrase-MiniLM-L12-v2, trained on parallel data for 50+ languages.
paraphrase-multilingual-mpnet-base-v2 - Multilingual version of paraphrase-mpnet-base-v2, trained on parallel data for 50+ languages.


## 참고 자료

- Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
    - https://arxiv.org/pdf/1908.10084.pdf
    

* 논문 설명 - Sentence-BERT : Sentence Embeddings using Siamese BERT-Networks
    * https://mlgalaxy.blogspot.com/2020/09/sentence-bert-sentence-embeddings-using.html


- fastText 공식 사이트
    - 텍스트 표현 및 텍스트 분류를 위한 가볍고 효율적인 NLP 라이브러리
    - https://fasttext.cc


- Gensim: NLP 라이브러리
    - 이 워크샵에서는 한국어 Pre-trained FastText Model을 로딩하기 위해 사용 함
    - https://radimrehurek.com/gensim/index.html


- GluonNLP의 튜토리얼:  [Using Pre-trained Word Embeddings](https://nlp.gluon.ai/examples/word_embedding/word_embedding.html) 


- Daekeun Kim, [Fine Tuning Naver Movie Review Sentiment Classification with KoBERT using GluonNLP](https://github.com/daekeun-ml/kobert-workshop)


* Fine-tuning Pre-trained BERT Models
    * GluonNLP 의 문장 유사성 파인 튜닝 튜토리얼
    * https://nlp.gluon.ai/examples/sentence_embedding/bert.html


* Jay Alammar. A Visual Guide to Using BERT for the First Time, 
    * http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time
    
    
* How the Embedding Layers in BERT Were Implemented
    * 입력이 입베팅으로 변환하는 것을 직관적으로 보여줌
    * https://medium.com/@_init_/why-bert-has-3-embedding-layers-and-their-implementation-details-9c261108e28a
    
    
* Understanding BERT — Word Embeddings
    * 버트의 워드 임베팅에 대한 결과 임
    * https://medium.com/@dhartidhami/understanding-bert-word-embeddings-7dc4d2ea54ca
    
    
* FastText Pre-trained 한국어 모델 사용기
    * https://inahjeon.github.io/fasttext/
    
    
* 문서 유사도 측정 구현하기
    * https://velog.io/@jakeseo_me/문서-유사도-측정-구현하기-2 (https://velog.io/@jakeseo_me/%EB%AC%B8%EC%84%9C-%EC%9C%A0%EC%82%AC%EB%8F%84-%EC%B8%A1%EC%A0%95-%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B0-2)
    
    
* BERT 톺아보기
    * http://docs.likejazz.com/bert/
    
    
* 2018-2020 NLU 연구 동향을 소개합니다
    * https://www.kakaobrain.com/blog/118
    
    
* 딥러닝으로 동네생활 게시글 필터링하기
    * https://medium.com/daangn/딥러닝으로-동네생활-게시글-필터링하기-263cfe4bc58d (https://medium.com/daangn/%EB%94%A5%EB%9F%AC%EB%8B%9D%EC%9C%BC%EB%A1%9C-%EB%8F%99%EB%84%A4%EC%83%9D%ED%99%9C-%EA%B2%8C%EC%8B%9C%EA%B8%80-%ED%95%84%ED%84%B0%EB%A7%81%ED%95%98%EA%B8%B0-263cfe4bc58d)
    
    
* NLP 실습 텍스트 유사도 - 01 (데이터 EDA 및 전처리)
    * https://heung-bae-lee.github.io/2020/02/10/NLP_10/
    
    
* PapersWithCode의 한국어 데이터셋
    * https://smilegate.ai/2021/02/10/paperswithcode-korean-dataset/
    
    
* KorNLU Datasets
    * https://github.com/kakaobrain/KorNLUDatasets
    
    
* product-matching-model
    * https://github.com/jahyeha/product-matching-model
    
    
* Product Matching in eCommerce using deep learning
    * https://medium.com/walmartglobaltech/product-matching-in-ecommerce-4f19b6aebaca
    
    
- KoNLPy (한국어 NLP 파이썬 패키지))
    - https://konlpy.org/en/latest/install/
    - 관련 참조
        - https://yuddomack.tistory.com/entry/처음부터-시작하는-EC2-konlpy-mecab-설치하기ubuntu
    
    


---

## 기타

- Mecab 설치 방법
    - ```! curl https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh | bash  ```
    
    
- NLTK
    - https://www.nltk.org
    - 설치 방법
        - ``` ! pip install nltk
        import nltk
        nltk.download('punkt')
        ```

- sent2vect 설치
     - ``` pip install sent2vec ```
     
- Globe 
    - https://nlp.stanford.edu/projects/glove/
    - 설치 방법
    -  ```    
%%sh
export PATH=$PATH:/home/ec2-user/.local/bin
export PATH=$PATH:x/root/.local/bin # 스튜디오
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d danielwillgeorge/glove6b100dtxt     
! unzip glove6b100dtxt.zip -d glove6b/
```