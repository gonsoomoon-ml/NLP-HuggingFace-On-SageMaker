import pandas as pd
import re    
def remove_spec_chars(s):
    '''
    특수 문자 제거.
    '''
    
    if pd.isnull(s):
        return s
    
    ##########################    
    # html 태그 제거
    ##########################    
    re_html = re.compile(r'<[^>]+>')
    s = re_html.sub('', s)    
    
    
    ##########################    
    # 알파벳, 숫자 이외 모두 제거   
    ##########################    
    s = re.sub(r'\W+', ' ', s) 
    
    ##########################    
    # 다른 문자와 분리된 숫자  제거   
    ##########################        
    s = re.sub(r'\b\d+\b', ' ', s) # match only those digits that are not part of another word.  
        
    ##########################
    # 특수 문자 제거 
    ##########################    
    remove_ptns = ['&gt','&lt','gt','lt'
                   '\r', '\n', '>','<','{','}','[',']',']',
                   '$','/', ',','-','_','=','+','.','!','@','^','%','#','*','(',')',
                   '₩','`','&','|',':',';','<','>','?','\\','\'','"',
                   'ㅠ','ㅜ','ㅎ','ㅋ','ㅡ','ㅂ','ㅈ','ㅗ','ㄷ','ㄱ','ㅇ','ㅊ'
                  ]
    dic = dict(zip(remove_ptns, [' '] * len(remove_ptns)))
    regex = re.compile("|".join(map(re.escape, dic.keys())), 0)
    
    s = regex.sub(lambda match: dic[match.group(0)], s)

    ##########################        
    # 스페이스 제거
    ##########################    
    s = re.sub('\s+',' ', s)    

    return s

####################
# 데이터 전처리 함수
####################
from konlpy.tag import Okt
Okt = Okt()


stop_words = ['가다','있다']
pos_labels = ["Noun","Adjective","Adverb","Modifier","Verb","Determiner"]
def Okt_tokenizer(raw, pos=pos_labels, stopword=stop_words):
#def Okt_tokenizer(raw, pos=["Noun"], stopword=stop_words):
    '''
    문장의 토큰나이징 함수
    '''
    

    
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


from konlpy.tag import Mecab
mecab = Mecab()


stop_words = ['컬리']
pos_list= ['NNG','MAG','VV','MM','VA','VV+EC']
def Mecab_tokenizer(raw, pos=pos_list, stopword=stop_words):
    '''
    문장의 토큰나이징 함수
    '''
    
    word_list = [
        word for word, tag in mecab.pos(
            raw
            )
            if len(word) > 1 and tag in pos \
            and word not in stopword 
        ]
    
    #  중복 제거
    #  중복 제거가 유사도에는 좋은 영향을 안줌. 몇 개 샘플로 확인
    # word_list = list(dict.fromkeys(word_list))
    
    return ' '.join(word_list)




import traceback
def preprocess_doc(df, target_col, new_col):
    '''
    특수문자를 제거한 후에 Tokenization 을 함.
    '''
    try: 
        pd_df = df.copy()
        
        clean_data = df[target_col].apply(remove_spec_chars)
        pd_df.insert(0, column=new_col, value=clean_data)    
        # pd_df[new_col] = pd_df[new_col].apply(Mecab_tokenizer)    
        # pd_df[new_col] = pd_df[new_col].apply(Okt_tokenizer)            

        return pd_df
    
    except Exception:
        print(traceback.format_exc())    

