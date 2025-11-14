# 🪵 Logging Guide

## 로깅 시스템 구조

프로젝트의 모든 로그는 **`src/utils.py`의 `get_logger()`**를 통해 중앙 집중식으로 관리됩니다.

---

## 📝 로그 파일 위치

```
logs/
└── prediction_market.log  ← 모든 모듈의 통합 로그
```

---

## 🔧 사용법

### **1. 모듈에서 로거 가져오기**

```python
from src.utils import get_logger

# 첫 호출 시에만 초기화 (log_file, level 설정)
logger = get_logger(log_file="logs/prediction_market.log", level="INFO")

# 이후 다른 모듈에서는 그냥 호출 (이미 초기화된 logger 반환)
logger = get_logger()
```

### **2. 로그 레벨**

- `DEBUG`: 상세한 디버깅 정보
- `INFO`: 일반적인 정보 메시지 (기본값)
- `WARNING`: 경고 메시지
- `ERROR`: 에러 메시지

```python
logger.debug("Detailed debug information")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error message")
```

### **3. Timer와 함께 사용**

```python
from src.utils import get_logger, Timer

logger = get_logger()

with Timer("Data loading", logger):
    # 시간 측정이 필요한 코드
    df = pd.read_csv("data.csv")
```

---

## 🎯 모범 사례

### ✅ **권장**

```python
# src/data.py
from src.utils import get_logger

logger = get_logger(log_file="logs/prediction_market.log", level="INFO")
```

```python
# src/features.py
from src.utils import get_logger

logger = get_logger()  # 이미 초기화된 logger 사용
```

### ❌ **비권장**

```python
# 각 모듈에서 개별 로거 생성 (핸들러 중복 발생)
from src.utils import setup_logging

logger = setup_logging(log_file="logs/data.log")  # ❌
```

---

## 📊 로그 출력 예시

```
2025-11-11 12:30:15,264 - prediction_market - INFO - Starting: Loading data
2025-11-11 12:30:15,264 - prediction_market - INFO - Loading train data from data/raw/train.csv
2025-11-11 12:30:15,359 - prediction_market - INFO - Train shape: (9021, 98)
2025-11-11 12:30:15,364 - prediction_market - INFO - Completed: Loading data - Time: 0.10s
```

**포맷**: `{timestamp} - {logger_name} - {level} - {message}`

---

## 🔍 로그 분석

### **터미널에서 실시간 모니터링**

```bash
tail -f logs/prediction_market.log
```

### **특정 레벨만 필터링**

```bash
grep "ERROR" logs/prediction_market.log
grep "WARNING" logs/prediction_market.log
```

### **시간대별 필터링**

```bash
grep "2025-11-11 12:30" logs/prediction_market.log
```

---

## 🧹 로그 파일 관리

### **로그 파일 정리**

```bash
# 오래된 로그 삭제
rm logs/*.log

# 또는 .gitignore에 추가되어 있으므로 git에서 자동 제외됨
```

### **로그 로테이션 (선택사항)**

나중에 필요하면 `logging.handlers.RotatingFileHandler` 사용:

```python
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    'logs/prediction_market.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
```

---

## 💡 팁

1. **개발 시**: `level="DEBUG"`로 상세 로그 확인
2. **프로덕션**: `level="INFO"`로 중요 정보만 기록
3. **디버깅**: 로그 파일에서 타임스탬프로 성능 병목 지점 파악
4. **에러 추적**: ERROR 레벨 로그를 검색해서 문제 빠르게 식별

---

**요약**: 모든 모듈에서 `get_logger()`를 사용하면 통합된 로그 관리가 가능하고, 핸들러 중복 문제가 해결됩니다! 🚀
