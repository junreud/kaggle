# Kaggle Titanic - 생존자 예측 프로젝트

## 📌 프로젝트 소개
Kaggle의 입문용 머신러닝 대회인 **Titanic - Machine Learning from Disaster** 문제입니다.  
타이타닉호 승객 데이터를 기반으로 생존 여부를 예측하는 이진 분류(Binary Classification) 문제입니다.

🔗 [Kaggle Competition Link](https://www.kaggle.com/competitions/titanic/overview)

## 📊 데이터 설명

### Raw Data (`Titanic/data/raw/`)
- **train.csv**: 학습용 데이터 (891명의 승객 정보 + 생존 여부)
- **test.csv**: 테스트용 데이터 (418명의 승객 정보)
- **gender_submission.csv**: 제출 예시 파일

### 주요 Feature
- `PassengerId`: 승객 ID
- `Survived`: 생존 여부 (0 = 사망, 1 = 생존) - **Target Variable**
- `Pclass`: 객실 등급 (1 = 1등석, 2 = 2등석, 3 = 3등석)
- `Name`: 승객 이름
- `Sex`: 성별
- `Age`: 나이
- `SibSp`: 함께 탑승한 형제자매/배우자 수
- `Parch`: 함께 탑승한 부모/자녀 수
- `Ticket`: 티켓 번호
- `Fare`: 운임
- `Cabin`: 객실 번호
- `Embarked`: 탑승 항구 (C = Cherbourg, Q = Queenstown, S = Southampton)

## 🚀 시작하기

### 1. 가상환경 생성 및 활성화

#### Windows
```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화
venv\Scripts\activate
```

#### macOS / Linux
```bash
# 가상환경 생성
python3 -m venv venv

# 가상환경 활성화
source venv/bin/activate
```

### 2. 필요 라이브러리 설치
```bash
pip install pandas scikit-learn seaborn matplotlib numpy xgboost lightgbm jupyter
```

### 3. Jupyter Notebook 실행
```bash
jupyter notebook
```

## 📁 프로젝트 구조
```
Titanic/
├── data/
│   ├── raw/              # 원본 데이터
│   ├── processed/        # 전처리된 데이터
│   └── submissions/      # 제출용 파일
└── notebooks/
    ├── 01_eda_and_baseline.ipynb              # 데이터 탐색 & 베이스라인 모델
    ├── 02_data_preprocessing.ipynb            # 데이터 전처리
    ├── 03_modeling_and_evaluation.ipynb       # 모델링 & 평가
    └── 04_advanced_modeling_and_ensemble.ipynb # 고급 모델링 & 앙상블
```

## 📚 학습 순서
노트북을 **01번부터 04번까지 순서대로** 실행하며 다음을 경험할 수 있습니다:

1. **EDA (탐색적 데이터 분석)** - 데이터의 특성과 패턴 파악
2. **데이터 전처리** - 결측치 처리, 피처 엔지니어링
3. **모델링 & 평가** - 다양한 머신러닝 알고리즘 비교
4. **앙상블 기법** - 모델 성능 향상 전략

실무에서 머신러닝 프로젝트를 어떻게 진행하는지, 각 알고리즘과 전처리 방식이 성능에 어떤 영향을 미치는지 직접 확인할 수 있습니다.

---

**선형대수학 동아리의 선물입니다. 화이팅! 🎓**