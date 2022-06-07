# NLP 작업을 위한 Hugging Face on SageMaker 워크샵

---

# 1. 배경
[Hugging Face](https://huggingface.co/)는 [Transformer](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)#:~:text=A%20transformer%20is%20a%20deep,and%20computer%20vision%20(CV).) 를 이용하여 자연어 처리, 이미지 등의 작업을 쉽게 하기 위해서 만들어진 오픈 소스 라이브러리 입니다.
Hugging Face 는 SageMaker 와 협업을 통하여 조금 더 쉽게 실무에 적용할 수 있게 해주고 있습니다.
이 워크샵에서는 네이버 영화 리뷰 데이터 셋을 통한 감성 분석을 박장원님이 만드신 [KoELECTRA Pre-Trained Model](https://github.com/monologg/KoELECTRA) 를 통해서 Downstream Task 의 하나인 감성 분석을 하겠습니다.
또한 김대근님이 만든신 [모두를 위한 클라우드 네이티브 한국어 자연어 처리 모델 훈련 및 활용법 (부제: 허깅페이스(Hugging Face)와 Amazon SageMaker가 만났다!)](https://github.com/daekeun-ml/sm-huggingface-kornlp) 워크샵의 많은 내용을 참조 하였습니다.

# 2. 주요 사용 기술
- Hugging Face 의 한국어 감성(Sentiment) 분석에 대해서 "스크래치 코드" 가지고 파인 튜닝을 배움.
    - 사용자 데이터 셋을 생성하여 Pytorch, HF Trainer 를 통해 훈련
- 세이지 메이커에서 "스크래치 코드" 를 훈련 할 수 있게 로컬 모드, 호스트 모드로 훈련 함. (로컬 모드, 호스트 모드는 하단의 "참고" 섹션 참조)
    - Pytorch SubsetSampler 등을 통하여 개발의 용이성을 높임.
    - 세이지 메이커 "실험" 을 통한 훈련 잡 추적
- Hugging Face 세이지 메이커 빌트인 도커를 사용하여 세이지 메이커 앤드포인트 생성 및  추론
    - 네이버 영화 리뷰 테스트 데이터에 대한  (약 50,000 개) 감성 분류 (약 89% 정확도 보임.)


# 3. 실습

## 3.1. 실습 환경
- 세이지 메이커 노트북 인스턴스 ml.p3.2xlarge , ml.p3.8xlarge 의 conda_python3 에서 테스트 됨.

## 3.2. 실습 단계

- 1_Setup (필수)
    - -0.0.Setup-Environment.ipynb
    
    
- 2_WarmingUp (옵션)
    - 이 폴더는 생략 가능합니다. Hugging Face 에 익숙해지기 위한 연습 노트북이 있습니다.
    
    
- 3_Training (필수)
    - 1.1.Prepare_Data_Naver_Review.ipynb
        - 네이버 영화 감성 리뷰 데이터 
    - 2.1.Train_Scratch.ipynb
        - 감성 분류 (긍정, 부정) 을 위한 파인 튜닝 스케치
    - 2.2.Train_Local_Script_Mode.ipynb
        - 세이지 메이커를 이용한 감성 분류 (긍정, 부정) 을 위한 파인 튜닝 (로컬 모드, 호스트 모드)


- 4_Serving (필수)
    - 1.1.Rreal_time_endpoint.ipynb
        - 세이지 메이커 앤드 포인트를 생성하여 추론


# 4. 주요 파일 구조

```
 |-1_Setup
 | |-0.0.Setup-Environment.ipynb
 |-2_WarmingUp
 | |-0.1.warming_up_yelp_review.ipynb
 | |-0.2.warming_up_imdb_custom_dataset.ipynb
 | |-0.3.warming_up_naver_review.ipynb
 |-3_Training
 | |-1.1.Prepare_Data_Naver_Review.ipynb
 | |-2.1.Train_Scratch.ipynb
 | |-2.2.Train_Local_Script_Mode.ipynb
 | |-src
 | | |-data_util.py
 | | |-train.py
 | | |-train_util.py
 | | |-requirements.txt
 | | |-config.py
 |-4_Serving
 | |-1.1.Rreal_time_endpoint.ipynb
 | |-src
 | | |-inference_utils.py
```


# A. 참고

- Use Hugging Face with Amazon SageMaker
    - 세이지 메이커 개발자 문서
    - https://docs.aws.amazon.com/sagemaker/latest/dg/hugging-face.html
    

- SageMaker Python SDK : Hugging Face
    - https://sagemaker.readthedocs.io/en/stable/frameworks/huggingface/index.html    


- Hugging Face Site: Deploy models to Amazon SageMaker
    - https://huggingface.co/docs/sagemaker/inference#deploy-a-🤗-transformers-model-trained-in-sagemaker
    

- Hugging Face: Fine Tuning
    - https://huggingface.co/docs/transformers/training
    
    
- On-Side Workshop: Hugging Face Transformers on Amazon SageMaker
    - https://github.com/philschmid/huggingface-sagemaker-workshop-series/tree/main/on-side-event
    
    
- Hugging Face Official Repo
    - https://github.com/huggingface/transformers


- 세이지 메이커로 파이토치 사용 
    - [Use PyTorch with the SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html)


- Use PyTorch with the SageMaker Python SDK
    - https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html


- Amazon SageMaker Local Mode Examples
    - TF, Pytorch, SKLean, SKLearn Processing JOb에 대한 로컬 모드 샘플
        - https://github.com/aws-samples/amazon-sagemaker-local-mode
    - Pytorch 로컬 모드
        - https://github.com/aws-samples/amazon-sagemaker-local-mode/blob/main/pytorch_script_mode_local_training_and_serving/pytorch_script_mode_local_training_and_serving.py    



- pytorch dataset 정리
    - https://hulk89.github.io/pytorch/2019/09/30/pytorch_dataset/


    