"""
타이타닉 생존 예측 - 전체 파이프라인
Kaggle Titanic Competition - Complete ML Pipeline

이 파일은 1~4단계 노트북을 하나로 통합한 전체 파이프라인입니다.
실행 순서:
1. 데이터 로딩 및 EDA
2. 데이터 전처리 및 특성 엔지니어링
3. 기본 모델링 및 평가
4. 고급 모델링 및 앙상블
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# 머신러닝 모델들
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# 고급 모델
import xgboost as xgb
import lightgbm as lgb

# 평가 및 전처리
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# 시각화 설정
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)


class TitanicPipeline:
    """타이타닉 생존 예측 전체 파이프라인"""
    
    def __init__(self, data_path='data/raw/', processed_path='data/processed/', 
                 submission_path='data/submissions/'):
        """
        초기화
        
        Parameters:
        -----------
        data_path : str
            원본 데이터 경로
        processed_path : str
            전처리된 데이터 저장 경로
        submission_path : str
            제출 파일 저장 경로
        """
        self.data_path = data_path
        self.processed_path = processed_path
        self.submission_path = submission_path
        
        # 폴더 생성
        os.makedirs(processed_path, exist_ok=True)
        os.makedirs(submission_path, exist_ok=True)
        
        # 데이터 저장 변수
        self.train_df = None
        self.test_df = None
        self.all_data = None
        
        # 전처리된 데이터
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.X_train_full = None
        self.y_train_full = None
        self.X_test = None
        self.test_ids = None
        
        # 특성 이름
        self.feature_columns = None
        
        # 모델 저장
        self.models = {}
        self.results = {}
        
        print("="*80)
        print("🚢 타이타닉 생존 예측 파이프라인 초기화 완료!")
        print("="*80)
    
    
    # ========================================================================
    # 1단계: 데이터 로딩 및 EDA
    # ========================================================================
    
    def load_data(self):
        """원본 데이터 로딩"""
        print("\n📂 1단계: 데이터 로딩")
        print("-" * 80)
        
        self.train_df = pd.read_csv(self.data_path + 'train.csv')
        self.test_df = pd.read_csv(self.data_path + 'test.csv')
        
        print(f"✅ 훈련 데이터: {self.train_df.shape}")
        print(f"✅ 테스트 데이터: {self.test_df.shape}")
        print(f"💡 총 {len(self.train_df) + len(self.test_df)}명의 승객 데이터")
    
        
    def explore_data(self):
        """기본적인 EDA 수행"""
        print("\n📊 1단계: 탐색적 데이터 분석 (EDA)")
        print("-" * 80)
        
        print("\n[기본 정보]")
        print(self.train_df.info())
        
        print("\n[기술 통계]")
        print(self.train_df.describe())
        
        print("\n[누락값 확인]")
        missing = self.train_df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        for col, count in missing.items():
            pct = (count / len(self.train_df)) * 100
            print(f"- {col}: {count}개 ({pct:.1f}%)")
        
        print("\n[생존율 분석]")
        survival_rate = self.train_df['Survived'].mean()
        print(f"전체 생존율: {survival_rate:.1%}")
        
        print("\n성별 생존율:")
        print(self.train_df.groupby('Sex')['Survived'].agg(['count', 'mean']))
        
        print("\n객실등급별 생존율:")
        print(self.train_df.groupby('Pclass')['Survived'].agg(['count', 'mean']))
    
    
    # ========================================================================
    # 2단계: 데이터 전처리 및 특성 엔지니어링
    # ========================================================================
    
    def preprocess_data(self):
        """데이터 전처리 및 특성 엔지니어링"""
        print("\n🔧 2단계: 데이터 전처리 및 특성 엔지니어링")
        print("-" * 80)
        
        # 훈련 + 테스트 데이터 결합 (일관된 전처리)
        print("\n[1] 데이터 결합")
        train_len = len(self.train_df)
        self.train_df['is_train'] = 1
        self.test_df['is_train'] = 0
        self.test_df['Survived'] = -1  # 임시값
        
        self.all_data = pd.concat([self.train_df, self.test_df], ignore_index=True)
        print(f"✅ 결합된 데이터: {self.all_data.shape}")
        
        # Age 누락값 처리
        print("\n[2] Age 누락값 처리 (성별+객실등급+탑승지+Cabin유무별 중앙값)")
        self.all_data['Has_Cabin_temp'] = self.all_data['Cabin'].notna().astype(int)
        
        age_medians_4 = self.all_data.groupby(['Sex', 'Pclass', 'Embarked', 'Has_Cabin_temp'])['Age'].median()
        age_medians_3 = self.all_data.groupby(['Sex', 'Pclass', 'Embarked'])['Age'].median()
        age_medians_2 = self.all_data.groupby(['Sex', 'Pclass'])['Age'].median()
        
        def fill_age(row):
            if pd.isna(row['Age']):
                key4 = (row['Sex'], row['Pclass'], row['Embarked'], row['Has_Cabin_temp'])
                if key4 in age_medians_4.index and not pd.isna(age_medians_4[key4]):
                    return age_medians_4[key4]
                
                key3 = (row['Sex'], row['Pclass'], row['Embarked'])
                if key3 in age_medians_3.index and not pd.isna(age_medians_3[key3]):
                    return age_medians_3[key3]
                
                key2 = (row['Sex'], row['Pclass'])
                if key2 in age_medians_2.index and not pd.isna(age_medians_2[key2]):
                    return age_medians_2[key2]
                
                return self.all_data['Age'].median()
            return row['Age']
        
        self.all_data['Age'] = self.all_data.apply(fill_age, axis=1)
        self.all_data.drop('Has_Cabin_temp', axis=1, inplace=True)
        print(f"✅ Age 누락값 처리 완료")
        
        # Embarked 누락값 처리
        print("\n[3] Embarked 누락값 처리 (최빈값)")
        most_common = self.all_data['Embarked'].mode()[0]
        self.all_data['Embarked'].fillna(most_common, inplace=True)
        print(f"✅ Embarked 누락값 '{most_common}'로 대체")
        
        # Fare 누락값 처리
        print("\n[4] Fare 누락값 처리 (Pclass별 중앙값)")
        fare_medians = self.all_data.groupby('Pclass')['Fare'].median()
        self.all_data['Fare'] = self.all_data.apply(
            lambda row: fare_medians[row['Pclass']] if pd.isna(row['Fare']) else row['Fare'],
            axis=1
        )
        print(f"✅ Fare 누락값 처리 완료")
        
        # Cabin 처리
        print("\n[5] Cabin 처리 (누락값 정보를 특성으로 활용)")
        self.all_data['Has_Cabin'] = self.all_data['Cabin'].notna().astype(int)
        self.all_data['Cabin_Deck'] = self.all_data['Cabin'].str[0]
        self.all_data['Cabin_Deck'].fillna('Unknown', inplace=True)
        self.all_data.drop('Cabin', axis=1, inplace=True)
        print(f"✅ Has_Cabin, Cabin_Deck 특성 생성")
        
        # Ticket 패턴 추출
        print("\n[6] Ticket 패턴 특성 생성")
        
        def extract_ticket_prefix(ticket):
            if pd.isna(ticket):
                return 'NO_TICKET'
            ticket_clean = str(ticket).replace(' ', '').replace('.', '').upper()
            if ticket_clean.isdigit():
                return 'NUMERIC_ONLY'
            prefix = ''
            for char in ticket_clean:
                if char.isalpha() or char == '/':
                    prefix += char
                else:
                    break
            return prefix if prefix else 'NUMERIC_ONLY'
        
        self.all_data['Ticket_Prefix'] = self.all_data['Ticket'].apply(extract_ticket_prefix)
        
        def categorize_ticket_prefix(prefix):
            if prefix in ['PC', 'C']:
                return 'Premium'
            elif prefix in ['CA', 'SOTON', 'STON']:
                return 'Regional'
            elif prefix.startswith('A/'):
                return 'A_Series'
            elif prefix in ['W/C', 'SC', 'PP', 'WE/P', 'SOC', 'SOP']:
                return 'Special'
            elif prefix == 'NUMERIC_ONLY':
                return 'Standard'
            else:
                return 'Other'
        
        self.all_data['Ticket_Group'] = self.all_data['Ticket_Prefix'].apply(categorize_ticket_prefix)
        
        def get_ticket_number_length(ticket):
            if pd.isna(ticket):
                return 0
            numbers = ''.join(filter(str.isdigit, str(ticket)))
            return len(numbers)
        
        self.all_data['Ticket_Number_Length'] = self.all_data['Ticket'].apply(get_ticket_number_length)
        
        ticket_counts = self.all_data['Ticket'].value_counts()
        self.all_data['Ticket_Group_Size'] = self.all_data['Ticket'].map(ticket_counts)
        
        def categorize_ticket_group_size(size):
            if size == 1:
                return 'Solo'
            elif size <= 3:
                return 'Small_Group'
            else:
                return 'Large_Group'
        
        self.all_data['Ticket_Share_Type'] = self.all_data['Ticket_Group_Size'].apply(categorize_ticket_group_size)
        print(f"✅ Ticket 관련 특성 4개 생성")
        
        # 가족 관련 특성
        print("\n[7] 가족 관련 특성 생성")
        self.all_data['Family_Size'] = self.all_data['SibSp'] + self.all_data['Parch'] + 1
        
        def categorize_family_size(size):
            if size == 1:
                return 'Alone'
            elif size <= 4:
                return 'Small'
            else:
                return 'Large'
        
        self.all_data['Family_Size_Group'] = self.all_data['Family_Size'].apply(categorize_family_size)
        print(f"✅ Family_Size, Family_Size_Group 특성 생성")
        
        # 나이대 그룹
        print("\n[8] 나이대 그룹 생성")
        def categorize_age(age):
            if age < 18:
                return 'Child'
            elif age < 30:
                return 'Young_Adult'
            elif age < 50:
                return 'Middle_Age'
            else:
                return 'Senior'
        
        self.all_data['Age_Group'] = self.all_data['Age'].apply(categorize_age)
        print(f"✅ Age_Group 특성 생성")
        
        # 요금 범위
        print("\n[9] 요금 범위 생성")
        self.all_data['Fare_Range'] = pd.qcut(self.all_data['Fare'], q=4, labels=['Low', 'Medium', 'High', 'Very_High'])
        print(f"✅ Fare_Range 특성 생성")
        
        # Title 추출
        print("\n[10] Title 추출")
        self.all_data['Title'] = self.all_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        title_mapping = {'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master'}
        self.all_data['Title'] = self.all_data['Title'].map(title_mapping).fillna('Other')
        print(f"✅ Title 특성 생성")
        
        # 고급 상호작용 특성
        print("\n[11] 고급 상호작용 특성 생성")
        self.all_data['Pclass_Sex'] = self.all_data['Pclass'].astype(str) + '_' + self.all_data['Sex']
        
        def age_sex_category(row):
            if row['Age'] < 16:
                return 'Child'
            else:
                return row['Sex']
        
        self.all_data['Age_Sex_Group'] = self.all_data.apply(age_sex_category, axis=1)
        self.all_data['Fare_Per_Person'] = self.all_data['Fare'] / self.all_data['Family_Size']
        self.all_data['Name_Length'] = self.all_data['Name'].str.len()
        self.all_data['Has_Parentheses'] = self.all_data['Name'].str.contains('\(').astype(int)
        self.all_data['Age_Sex_Interaction'] = self.all_data['Age_Group'].astype(str) + '_' + self.all_data['Sex']
        
        # 1등급 + 성별 상호작용 특성 (오류 분석 결과 반영)
        self.all_data['Pclass1_Male'] = ((self.all_data['Pclass'] == 1) & (self.all_data['Sex'] == 'male')).astype(int)
        self.all_data['Pclass1_Female'] = ((self.all_data['Pclass'] == 1) & (self.all_data['Sex'] == 'female')).astype(int)
        
        # 청년층 세분화 특성 (오류 분석 결과 반영 - Young_Adult 오류율 9.2%)
        self.all_data['YoungAdult_Male_LowClass'] = (
            (self.all_data['Age_Group'] == 'Young_Adult') & 
            (self.all_data['Sex'] == 'male') & 
            (self.all_data['Pclass'] >= 2)
        ).astype(int)
        print(f"✅ 고급 상호작용 특성 9개 생성")
        
        # 범주형 변수 인코딩
        print("\n[12] 범주형 변수 인코딩")
        categorical_columns = ['Sex', 'Embarked', 'Family_Size_Group', 'Age_Group', 
                              'Fare_Range', 'Title', 'Cabin_Deck',
                              'Ticket_Group', 'Ticket_Share_Type',
                              'Pclass_Sex', 'Age_Sex_Group', 'Age_Sex_Interaction']
        
        label_encoders = {}
        for col in categorical_columns:
            le = LabelEncoder()
            self.all_data[col + '_encoded'] = le.fit_transform(self.all_data[col])
            label_encoders[col] = le
        
        print(f"✅ {len(categorical_columns)}개 범주형 변수 인코딩 완료")
        
        # 최종 특성 선택
        print("\n[13] 최종 특성 선택")
        self.feature_columns = [
            'Pclass', 'Age', 'Fare', 'Family_Size', 'Has_Cabin',
            'Sex_encoded', 'Embarked_encoded', 'Family_Size_Group_encoded',
            'Age_Group_encoded', 'Fare_Range_encoded', 'Title_encoded', 'Cabin_Deck_encoded',
            'Ticket_Group_encoded', 'Ticket_Share_Type_encoded', 'Ticket_Number_Length', 'Ticket_Group_Size',
            'Pclass_Sex_encoded', 'Age_Sex_Group_encoded', 'Fare_Per_Person', 'Name_Length', 'Has_Parentheses',
            'Age_Sex_Interaction_encoded', 'Pclass1_Male', 'Pclass1_Female', 'YoungAdult_Male_LowClass'
        ]
        
        print(f"✅ 최종 특성: {len(self.feature_columns)}개")
        
        # 데이터 분할
        print("\n[14] 데이터 분할")
        train_processed = self.all_data[self.all_data['is_train'] == 1].copy()
        test_processed = self.all_data[self.all_data['is_train'] == 0].copy()
        
        X_train_full = train_processed[self.feature_columns]
        y_train_full = train_processed['Survived']
        X_test_final = test_processed[self.feature_columns]
        
        # 훈련/검증 분할
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
        )
        
        # 전체 훈련 데이터 (최종 모델용)
        self.X_train_full = X_train_full
        self.y_train_full = y_train_full
        self.X_test = X_test_final
        self.test_ids = test_processed['PassengerId']
        
        print(f"✅ 훈련 데이터: {self.X_train.shape}")
        print(f"✅ 검증 데이터: {self.X_val.shape}")
        print(f"✅ 전체 훈련: {self.X_train_full.shape}")
        print(f"✅ 테스트 데이터: {self.X_test.shape}")
        
        # 저장
        print("\n[15] 전처리 데이터 저장")
        self.X_train.to_csv(self.processed_path + 'X_train.csv', index=False)
        self.X_val.to_csv(self.processed_path + 'X_val.csv', index=False)
        self.y_train.to_csv(self.processed_path + 'y_train.csv', index=False)
        self.y_val.to_csv(self.processed_path + 'y_val.csv', index=False)
        self.X_test.to_csv(self.processed_path + 'X_test.csv', index=False)
        self.test_ids.to_csv(self.processed_path + 'test_ids.csv', index=False)
        pd.Series(self.feature_columns).to_csv(self.processed_path + 'feature_names.csv', index=False)
        print(f"✅ 전처리 데이터 저장 완료: {self.processed_path}")
    
    
    # ========================================================================
    # 3단계: 기본 모델링 및 평가
    # ========================================================================
    
    def train_basic_models(self):
        """기본 ML 모델들 훈련 및 평가"""
        print("\n🤖 3단계: 기본 모델 훈련 및 평가")
        print("-" * 80)
        
        basic_models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(random_state=42),
            'KNN': KNeighborsClassifier(),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42)
        }
        
        print("\n[기본 모델 성능 비교]")
        print(f"{'모델':<25} {'훈련 정확도':<15} {'검증 정확도':<15}")
        print("-" * 55)
        
        for name, model in basic_models.items():
            model.fit(self.X_train, self.y_train)
            train_acc = accuracy_score(self.y_train, model.predict(self.X_train))
            val_acc = accuracy_score(self.y_val, model.predict(self.X_val))
            
            self.models[name] = model
            self.results[name] = {'train_acc': train_acc, 'val_acc': val_acc}
            
            print(f"{name:<25} {train_acc:.4f} ({train_acc*100:.2f}%)  {val_acc:.4f} ({val_acc*100:.2f}%)")
        
        # 최고 성능 모델
        best_model_name = max(self.results, key=lambda k: self.results[k]['val_acc'])
        best_val_acc = self.results[best_model_name]['val_acc']
        print("\n" + "="*55)
        print(f"🏆 최고 성능: {best_model_name} (검증 정확도: {best_val_acc:.4f})")
        print("="*55)
    
    
    # ========================================================================
    # 4단계: 고급 모델링 및 앙상블
    # ========================================================================
    
    def train_advanced_models(self):
        """XGBoost, LightGBM 훈련"""
        print("\n🚀 4단계: 고급 모델 훈련")
        print("-" * 80)
        
        advanced_models = {
            'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
            'LightGBM': lgb.LGBMClassifier(
                random_state=42,
                n_estimators=100,
                max_depth=5,
                min_child_samples=10,
                verbose=-1
            )
        }
        
        print("\n[고급 모델 성능]")
        print(f"{'모델':<25} {'훈련 정확도':<15} {'검증 정확도':<15}")
        print("-" * 55)
        
        for name, model in advanced_models.items():
            model.fit(self.X_train, self.y_train)
            train_acc = accuracy_score(self.y_train, model.predict(self.X_train))
            val_acc = accuracy_score(self.y_val, model.predict(self.X_val))
            
            self.models[name] = model
            self.results[name] = {'train_acc': train_acc, 'val_acc': val_acc}
            
            print(f"{name:<25} {train_acc:.4f} ({train_acc*100:.2f}%)  {val_acc:.4f} ({val_acc*100:.2f}%)")
    
    
    def hyperparameter_tuning_xgboost(self):
        """XGBoost 하이퍼파라미터 튜닝 (GridSearchCV)"""
        print("\n⚙️ XGBoost 하이퍼파라미터 튜닝 (GridSearchCV)")
        print("-" * 80)
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5]
        }
        
        total_combinations = np.prod([len(v) for v in param_grid.values()])
        print(f"📊 총 {total_combinations}개의 파라미터 조합 탐색")
        print(f"💡 5-Fold CV 사용 → 총 {total_combinations * 5}번의 모델 훈련")
        
        xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        print("\n🔄 GridSearchCV 시작... (시간이 걸릴 수 있습니다)")
        grid_search.fit(self.X_train_full, self.y_train_full)
        
        print("\n✅ GridSearchCV 완료!")
        print(f"\n🏆 최적 파라미터:")
        for param, value in grid_search.best_params_.items():
            print(f"  - {param}: {value}")
        
        print(f"\n📊 최고 CV 점수: {grid_search.best_score_:.4f} ({grid_search.best_score_*100:.2f}%)")
        
        # 최적 모델 저장
        self.models['XGBoost_Tuned'] = grid_search.best_estimator_
        
        # 검증 데이터로 평가
        val_pred = grid_search.best_estimator_.predict(self.X_val)
        val_acc = accuracy_score(self.y_val, val_pred)
        print(f"🎯 검증 정확도: {val_acc:.4f} ({val_acc*100:.2f}%)")
        
        self.results['XGBoost_Tuned'] = {
            'train_acc': grid_search.best_score_,
            'val_acc': val_acc,
            'best_params': grid_search.best_params_
        }
        
        return grid_search.best_estimator_
    
    
    def train_ensemble(self):
        """앙상블 모델 (Voting Classifier) 훈련"""
        print("\n🎭 앙상블 모델 훈련 (Voting Classifier)")
        print("-" * 80)
        
        # 앙상블에 사용할 모델들
        # 튜닝된 XGBoost 모델이 있으면 사용하고, 없으면 기본 모델을 사용합니다.
        xgb_tuned_model = self.models.get('XGBoost_Tuned', self.models.get('XGBoost'))

        estimators = [
            ('rf', self.models.get('Random Forest', RandomForestClassifier(n_estimators=200, random_state=42))),
            ('gb', self.models.get('Gradient Boosting', GradientBoostingClassifier(random_state=42))),
            ('xgb_tuned', xgb_tuned_model),
            ('lgb', self.models.get('LightGBM', lgb.LGBMClassifier(random_state=42, n_estimators=100, max_depth=5, 
                                      min_child_samples=10, verbose=-1)))
        ]
        
        # Soft Voting (확률 평균)
        voting_clf = VotingClassifier(estimators=estimators, voting='soft')
        
        print("📦 앙상블 구성:")
        for name, model in estimators:
            print(f"  - {name}: {model.__class__.__name__}")
        
        print("\n🔄 앙상블 모델 훈련 중...")
        voting_clf.fit(self.X_train_full, self.y_train_full)
        
        # 검증 데이터로 평가
        val_pred = voting_clf.predict(self.X_val)
        val_acc = accuracy_score(self.y_val, val_pred)
        
        self.models['Ensemble_Voting'] = voting_clf
        self.results['Ensemble_Voting'] = {'val_acc': val_acc}
        
        print(f"\n✅ 앙상블 모델 훈련 완료!")
        print(f"🎯 검증 정확도: {val_acc:.4f} ({val_acc*100:.2f}%)")
    
    
    # ========================================================================
    # 최종: 제출 파일 생성
    # ========================================================================
    
    def create_submission(self, model_name='XGBoost_Tuned'):
        """
        제출 파일 생성
        
        Parameters:
        -----------
        model_name : str
            사용할 모델 이름
        """
        print(f"\n📝 제출 파일 생성 ({model_name})")
        print("-" * 80)
        
        if model_name not in self.models:
            print(f"❌ 모델 '{model_name}'을 찾을 수 없습니다.")
            print(f"사용 가능한 모델: {list(self.models.keys())}")
            return
        
        model = self.models[model_name]
        
        # 전체 훈련 데이터로 재훈련
        print(f"🔄 전체 훈련 데이터로 {model_name} 재훈련 중...")
        model.fit(self.X_train_full, self.y_train_full)
        
        # 테스트 데이터 예측
        print("🎯 테스트 데이터 예측 중...")
        test_pred = model.predict(self.X_test)
        
        # 제출 파일 생성
        submission = pd.DataFrame({
            'PassengerId': self.test_ids,
            'Survived': test_pred
        })
        
        filename = f'submission_{model_name.lower()}.csv'
        filepath = self.submission_path + filename
        submission.to_csv(filepath, index=False)
        
        print(f"\n✅ 제출 파일 생성 완료!")
        print(f"📁 저장 위치: {filepath}")
        print(f"📊 예측 결과:")
        print(f"  - 생존 예측: {(test_pred == 1).sum()}명")
        print(f"  - 사망 예측: {(test_pred == 0).sum()}명")
        print(f"  - 생존율: {test_pred.mean():.1%}")
    
    
    def analyze_errors(self, model_name='XGBoost_Tuned', top_n=10):
        """
        오류 분석: 잘못 예측한 샘플 분석
        
        Parameters:
        -----------
        model_name : str
            분석할 모델 이름
        top_n : int
            표시할 잘못된 예측 샘플 수
        """
        print(f"\n🔍 오류 분석 ({model_name})")
        print("-" * 80)
        
        if model_name not in self.models:
            print(f"❌ 모델 '{model_name}'을 찾을 수 없습니다.")
            return
        
        model = self.models[model_name]
        
        # 검증 데이터로 예측
        y_pred = model.predict(self.X_val)
        
        # 잘못 예측한 샘플 찾기
        wrong_mask = (self.y_val.values != y_pred)
        wrong_indices = self.y_val.index[wrong_mask]
        
        print(f"\n📊 전체 검증 샘플: {len(self.y_val)}개")
        print(f"❌ 잘못 예측한 샘플: {wrong_mask.sum()}개 ({wrong_mask.sum()/len(self.y_val)*100:.1f}%)")
        print(f"✅ 정확히 예측한 샘플: {(~wrong_mask).sum()}개 ({(~wrong_mask).sum()/len(self.y_val)*100:.1f}%)")
        
        if wrong_mask.sum() == 0:
            print("\n🎉 모든 샘플을 정확히 예측했습니다!")
            return
        
        # 잘못 예측한 샘플 데이터 추출
        X_wrong = self.X_val.loc[wrong_indices]
        y_true_wrong = self.y_val.loc[wrong_indices]
        y_pred_wrong = pd.Series(y_pred[wrong_mask], index=wrong_indices)
        
        # 원본 데이터에서 해당 샘플의 상세 정보 가져오기
        # (전처리 전 데이터에서 이해하기 쉬운 정보 추출)
        train_processed = self.all_data[self.all_data['is_train'] == 1].copy()
        wrong_samples_detail = train_processed.loc[wrong_indices, 
            ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Family_Size', 'Title', 'Has_Cabin']
        ].copy()
        wrong_samples_detail['실제값'] = y_true_wrong.values
        wrong_samples_detail['예측값'] = y_pred_wrong.values
        
        print(f"\n[잘못 예측한 샘플 상위 {min(top_n, len(wrong_samples_detail))}개]")
        print(wrong_samples_detail.head(top_n).to_string())
        
        # 오류 패턴 분석
        print("\n" + "="*80)
        print("📈 오류 패턴 분석")
        print("="*80)
        
        print("\n1️⃣ 성별별 오류율:")
        if 'Sex' in train_processed.columns:
            for sex in train_processed.loc[wrong_indices, 'Sex'].unique():
                sex_mask = train_processed.loc[wrong_indices, 'Sex'] == sex
                sex_count = sex_mask.sum()
                sex_total = (train_processed.loc[self.y_val.index, 'Sex'] == sex).sum()
                print(f"   {sex}: {sex_count}/{sex_total} ({sex_count/sex_total*100:.1f}%)")
        
        print("\n2️⃣ 객실등급별 오류율:")
        for pclass in sorted(train_processed.loc[wrong_indices, 'Pclass'].unique()):
            pclass_mask = train_processed.loc[wrong_indices, 'Pclass'] == pclass
            pclass_count = pclass_mask.sum()
            pclass_total = (train_processed.loc[self.y_val.index, 'Pclass'] == pclass).sum()
            print(f"   {pclass}등급: {pclass_count}/{pclass_total} ({pclass_count/pclass_total*100:.1f}%)")
        
        print("\n3️⃣ 나이대별 오류율:")
        if 'Age_Group' in train_processed.columns:
            for age_group in train_processed.loc[wrong_indices, 'Age_Group'].unique():
                age_mask = train_processed.loc[wrong_indices, 'Age_Group'] == age_group
                age_count = age_mask.sum()
                age_total = (train_processed.loc[self.y_val.index, 'Age_Group'] == age_group).sum()
                if age_total > 0:
                    print(f"   {age_group}: {age_count}/{age_total} ({age_count/age_total*100:.1f}%)")
        
        print("\n4️⃣ False Positive vs False Negative:")
        fp = ((y_true_wrong == 0) & (y_pred_wrong == 1)).sum()  # 사망인데 생존 예측
        fn = ((y_true_wrong == 1) & (y_pred_wrong == 0)).sum()  # 생존인데 사망 예측
        print(f"   False Positive (사망→생존 오예측): {fp}개")
        print(f"   False Negative (생존→사망 오예측): {fn}개")
    
    
    def print_summary(self):
        """전체 결과 요약 출력"""
        print("\n" + "="*80)
        print("📊 전체 결과 요약")
        print("="*80)
        
        if not self.results:
            print("아직 모델이 훈련되지 않았습니다.")
            return
        
        print(f"\n{'모델':<30} {'검증 정확도':<15}")
        print("-" * 45)
        
        # 검증 정확도 기준 정렬
        sorted_results = sorted(
            [(name, res.get('val_acc', 0)) for name, res in self.results.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        for name, val_acc in sorted_results:
            print(f"{name:<30} {val_acc:.4f} ({val_acc*100:.2f}%)")
        
        best_model = sorted_results[0]
        print("\n" + "="*45)
        print(f"🏆 최고 성능: {best_model[0]}")
        print(f"📈 검증 정확도: {best_model[1]:.4f} ({best_model[1]*100:.2f}%)")
        print("="*45)


# ========================================================================
# 메인 실행
# ========================================================================

def main():
    """전체 파이프라인 실행"""
    
    print("\n" + "="*80)
    print("🚢 타이타닉 생존 예측 - 전체 파이프라인 실행")
    print("="*80)
    
    # 파이프라인 초기화
    pipeline = TitanicPipeline(
        data_path='data/raw/',
        processed_path='data/processed/',
        submission_path='data/submissions/'
    )
    
    # 1단계: 데이터 로딩 및 EDA
    pipeline.load_data()
    pipeline.explore_data()
    
    # 2단계: 데이터 전처리
    pipeline.preprocess_data()
    
    # 3단계: 기본 모델 훈련
    pipeline.train_basic_models()
    
    # 4단계: 고급 모델 훈련
    pipeline.train_advanced_models()
    
    # XGBoost 하이퍼파라미터 튜닝
    pipeline.hyperparameter_tuning_xgboost()
    
    # 앙상블 모델
    pipeline.train_ensemble()
    
    # 전체 결과 요약
    pipeline.print_summary()
    
    # 오류 분석 (최고 성능 모델)
    pipeline.analyze_errors('XGBoost_Tuned', top_n=15)
    
    # 제출 파일 생성
    print("\n" + "="*80)
    print("📝 제출 파일 생성")
    print("="*80)
    pipeline.create_submission('XGBoost_Tuned')
    pipeline.create_submission('Ensemble_Voting')
    
    print("\n" + "="*80)
    print("✅ 전체 파이프라인 실행 완료!")
    print("="*80)


if __name__ == "__main__":
    main()
