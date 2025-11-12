# ✅ Kaggle 제출 체크리스트 (Parquet)

## 🎯 제출 전 필수 확인사항

### 1️⃣ 파일명
```
/kaggle/working/submission.parquet
```

### 2️⃣ 컬럼 스키마
| 컬럼명 | 타입 | 범위 | 설명 |
|--------|------|------|------|
| `date_id` | int64 | - | 날짜 ID |
| `allocation` | float64 | [0, 2] | S&P500 배분 비율 |

⚠️ **중요**: `forward_returns`가 아닌 `allocation` 사용!

### 3️⃣ 데이터 품질
- ✅ NaN 없음 (`.isna().sum() == 0`)
- ✅ inf 없음 (`.isinf().sum() == 0`)
- ✅ allocation 범위 [0, 2] (`.clip(0, 2)`)
- ✅ date_id 정렬 (오름차순 권장)

### 4️⃣ 파일 형식
- ✅ Parquet 포맷 (CSV ❌)
- ✅ PyArrow 엔진 사용
- ✅ 인덱스 제외 (`index=False`)

### 5️⃣ Evaluation API 사용
```python
# Kaggle 노트북에서 반드시 이 방식 사용
from kaggle_evaluation.default_gateway import default_gateway

gateway = default_gateway()
for date_id in test_dates:
    features = get_features(date_id)
    allocation = model.predict(features)
    gateway.send_prediction(date_id, allocation)
```

## 🔍 노트북에서 확인할 내용

### Cell 9 (최종 제출 파일 생성)
```python
# ✅ 올바른 컬럼명
submission = pd.DataFrame({
    'date_id': test_df['date_id'].astype('int64'),
    'allocation': predictions.astype('float64')  # ← allocation!
})

# ✅ 범위 제한
submission['allocation'] = submission['allocation'].clip(0, 2)

# ✅ 검증
assert list(submission.columns) == ['date_id', 'allocation']
assert submission['allocation'].isna().sum() == 0
assert (submission['allocation'] >= 0).all()
assert (submission['allocation'] <= 2).all()

# ✅ Parquet 저장
submission.to_parquet(
    '/kaggle/working/submission.parquet',
    index=False,
    engine='pyarrow'
)
```

## 📊 Allocation 값의 의미

| 값 | 의미 |
|----|------|
| 0.0 | 전액 현금 (S&P500 0%) |
| 1.0 | 전액 투자 (S&P500 100%) |
| 2.0 | 레버리지 (S&P500 200%, 최대) |

## 🚨 흔한 실수들

### ❌ 틀린 예시들
```python
# 1. 틀린 컬럼명
submission = pd.DataFrame({
    'date_id': ids,
    'forward_returns': preds  # ❌ 학습 타깃이지 제출 컬럼 아님!
})

# 2. 범위 검증 없음
submission['allocation'] = predictions  # ❌ 음수나 2 초과 가능

# 3. CSV 저장
submission.to_csv('submission.csv')  # ❌ Parquet 필수!

# 4. 인덱스 포함
submission.to_parquet(path, index=True)  # ❌ index=False 필수
```

### ✅ 올바른 예시
```python
submission = pd.DataFrame({
    'date_id': test_df['date_id'].astype('int64'),
    'allocation': predictions.astype('float64')
})
submission['allocation'] = submission['allocation'].clip(0, 2)
submission.to_parquet('/kaggle/working/submission.parquet', 
                     index=False, engine='pyarrow')
```

## 🧪 로컬 테스트

```bash
# 제출 형식 검증 스크립트 실행
python scripts/check_submission_format.py
```

예상 출력:
```
✓ Columns: ['date_id', 'allocation']
✓ date_id: int64
✓ allocation: float64
✓ allocation range: [0.0000, 2.0000]
✓ No NaN, No inf
✓ Parquet format

✅ All checks passed!
```

## 📝 제출 절차

1. **Dataset 업로드**
   - `src/`, `conf/` 폴더를 Kaggle Dataset으로 업로드
   - Dataset 이름: `prediction-market-modules`

2. **노트북 업로드**
   - `kaggle_submission_with_modules.ipynb` 업로드
   - Add Data → Competition Data (hull-tactical-market-prediction)
   - Add Data → Your Dataset (prediction-market-modules)

3. **실행 및 검증**
   - Run All 실행
   - Cell 9 출력에서 형식 확인
   - `submission.parquet` 생성 확인

4. **제출**
   - Submit to Competition 클릭
   - 스코어 확인

## ✅ 최종 체크리스트

제출 전 모두 체크:
- [ ] 컬럼명: `date_id`, `allocation`
- [ ] 타입: int64, float64
- [ ] allocation 범위: [0, 2]
- [ ] NaN/inf 없음
- [ ] Parquet 포맷 (pyarrow)
- [ ] 인덱스 제외
- [ ] 파일 경로: `/kaggle/working/submission.parquet`
- [ ] Evaluation API 사용 (선택사항)

## 🎉 준비 완료!

모든 항목이 체크되었다면 Kaggle에 제출하세요! 
