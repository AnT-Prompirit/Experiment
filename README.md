# MAIN PROCESS Experiment
![fig3](https://github.com/AnT-Prompirit/Experiment/assets/77625287/20a25a71-d19a-469e-aef3-6563cb384ba9)

#### (a) Emotion Labels
1. 모델이 제시하는 확률에 따른 **강조 형용사 추가** ( Carrillo et al. )
    - 각 형용사가 감정 강조에 미치는 영향을 나타낸 가중치 이용
    - e.g. “extremely”: 95%, “very”: 75%
2. **동의어 추가**를 이용한 감정 레이블 반복 및 강조 ( Wang et al. )
    - FAISS와 AnglE를 이용해 사용자 감정 동의어 중 프롬프트와 유사도가 높은 감정 3개를 선정하고 프롬프트 뒤에 추가
        - 감정 label는 앞에, 유의어는 뒤에 붙임으로써 image-emotion alignment는 높이고 image-text alignment의 loss는 최소화한다.
    - WordNet과 SenticNet에서 중복된 단어와 문맥별 단어(예: revolt, repel, four-star, first class)를 제거한 감정 동의어를 수집해 corpus를 구축
    - cosine similarity를 고려하고 성능 저하를 피하는 corpus에서 상위 3개의 동의어를 선택
        - AnglE를 사용해 입력으로 들어온 원본 텍스트와 감정 동의어 간의 유사도를 계산
- 최종적으로 감정을 강조하는 네가지 방식을 비교함
    - E1: 감정 레이블
    - E2: 강조 형용사 + 감정 레이블
    - E3: 감정 레이블 + 동의어 추가
    - E4: 강조 형용사 + 감정 레이블 + 동의어 추가

#### (b) Style Modifiers

- 감정을 고려하면서 Style keywords를 적용하는 두가지 방식을 고안하고 비교
    1. **사용자 감정과 일치하는 Artemis 데이터셋 캡션에서 키워드 추출**
        - 클러스터 생성
            - LLaMA-2로 데이터 셋의 캡션에서 스타일 관련 키워드 추출
            - 추출한 키워드가 포함된 텍스트 데이터를 BERT 임베딩 시킨 후 t-SNE 차원 축소 진행
            - 계층적 클러스터링 진행 (자식 노드 최대 30개)
        - 입력 텍스트와 가장 가까운 클러스터를 찾아, 해당 클러스터의 스타일 키워드들 중 TF (Term Frequency) 점수가 높은 상위 3개의 키워드 추출
    2. **RAG** 기능으로 **스타일 수식어 리스트**에서 키워드 선택
        - MidJourney-Styles-and-Keywords-Reference에서 수집한 스타일 키워드를 color, dimensionality, style, light, perspective로 분류해 리스트로 정리
            - 내용에 급격한 스타일 변화가 일어나는 수정자보다는,이미지의 시각적 요소에 변화를 주는 수정자들을 선택
        - RAG: LangChain을 통해 LLaMA-2에 입력 텍스트와 감정 레이블을 query로, 스타일 수식어 리스트를 context로 제공
            - vector space에서 text chunk를 나타내는 임베딩은 사전 훈련된 모델 sentence-transformers/all-MiniLM-L6-v2를 사용하여 다운로드
            - 이 임베딩을 저장하고 retrieve할 vector store로는 ChromaDB를 사용
        - emotion, aesthetic, context를 모두 고려하여 3개 키워드 선택
- 최종적으로 style modifier를 적용하는 두가지 방식을 비교함
    - **S1: 사용자 감정과 일치하는 Artemis 데이터셋 캡션에서 키워드 추출**
    - S2: **RAG** 기능으로 **스타일 수식어 리스트**에서 키워드 선택

## Fitting the LMER Model
  - 고정 효과: 프롬프트 엔지니어링 방법
  - 랜덤 효과: 각 프롬프트의 ID
  - 독립 변수: IEA, ITA, Aesthetic Score

이후 fixed 및 interaction effect에 대한 분산 분석 (ANOVA)과 사후 검정 (post-hoc test) 수행  
    → 각 방법이 통계적으로 유의미한 차이를 보였음을 입증함
  
### Image-Emotion Alignment 
이미지와 감정 레이블 간의 코사인 유사도, CLIP Score 기반
![그림6](https://github.com/AnT-Prompirit/Experiment/assets/77625287/8be4616f-7790-4b41-ad68-7b3c135c3195)

### Image-Text Alignment 
이미지와 input 텍스트 간의 코사인 유사도, CLIP Score 기반
![그림7](https://github.com/AnT-Prompirit/Experiment/assets/77625287/8636c6ba-c9d2-4ec9-9021-bfd2f0d17084)

### Aesthetics
이미지의 심미적 품질 평가
![그림8](https://github.com/AnT-Prompirit/Experiment/assets/77625287/904c616d-0cf1-48b7-9d22-3b40b2b69d51)

### 정규화 진행
<img width="477" alt="Untitled" src="https://github.com/AnT-Prompirit/Experiment/assets/77625287/3c89e34a-5074-47ea-a71b-3b5461667ff0">

- Avg. score는 모든 점수를 [0, 1]로 정규화하여 계산함
    - IEA와 ITA는 CLIP Score 기반으로 하기 때문에 Aesthetic Score와 스케일이 다름
    - 세 가지 측면을 고려한 종합적 평가를 위해 정규화 진행

## 분석 결과

- 선형혼합회귀모델 (LMER) 피팅: E3+S2 가 IEA, Aesthetic Score 에서 가장 높은 점수를 보임
- 정규화: E3+﻿S2 가 가장 좋은 결과를 보임

**⇒ E3+﻿S2을 Main Process 파이프라인으로 채택**

: 원본 텍스트 → 감정 레이블 + 토큰화 된 텍스트 + 감정 동의어 3개 + 스타일 키워드

## 실험 코드 실행 방법
### Emotion Labels
#### E1 ()
### Style Modifiers
#### S1 (Caption)
1. 랭체인_스타일키워드_레이블링.ipynb 실행: 아르테미스 데이터셋을 라마를 이용해 레이블링한다.
2. 클러스터링_임베딩_차원축소.ipynb 실행: 레이블링 한 데이터를 이용해 클러스터링 진행
3. 최종실험데이터_만들기.ipynb 실행: 클러스트를 만든 후 입력 데이터에 따라 스타일 키워드 추출하는 코드
#### S2 (RAG)
1. S2_get_answer.ipynb 실행: 'styleDB' 폴더를 RAG DB로 연결, LLaMA-2 답변 생성 후 'S2_answer.csv' 파일로 저장
2. S2_extract_keyword_from_answer.ipynb 실행: LLaMA-2 답변에서 Style Modifiers만 추출하여 'S2_modifiers.csv' 파일로 저장
### Data Analysis
1. 정규화_평균_비교.ipynb : ITA, IEA, Aethetic Score의 정규화 코드
2. ImageGenerater_ClipScore_PickScore_AstheticScore.ipynb
   - Overview:  This project leverages the power of Stable Diffusion for image generation, enhanced with evaluation metrics including CLIP Score, PickScore, and Aesthetic Score. These metrics provide insights into the similarity between generated images and prompts, aesthetic quality, and alignment with human preferences.
   - Key Components:
       - **Stable Diffusion Image Generation**: Utilizes [Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-2-1) for generating high-quality images based on textual descriptions.
       - **CLIP Score**: Measures the similarity between generated images and prompts, as proposed by Radford et al., 2021.
       - **Aesthetic Score**: Evaluates the aesthetic quality of individual images following the methodology of Schuhmann et al., 2022. Implementation details are available on [GitHub](https://github.com/LAION-AI/aesthetic-predictor).
3. 
