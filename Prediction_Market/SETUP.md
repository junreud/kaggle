# Hull Tactical Market Prediction - Setup Guide

## Quick Start

### 1. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Set Up Kaggle API

1. Go to https://www.kaggle.com/settings/account
2. Click "Create New API Token"
3. Place `kaggle.json` in `~/.kaggle/`
4. Set permissions:
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

### 4. Download Data

```bash
python scripts/download_data.py
```

### 5. Run Tests

```bash
pytest tests/ -v
```

## Project Structure

```
Prediction_Market/
├── conf/                   # Configuration files
│   └── params.yaml        # Main configuration
├── data/
│   ├── raw/               # Raw data from Kaggle
│   └── processed/         # Processed data
├── src/                   # Source code
│   ├── __init__.py
│   ├── utils.py           # Utility functions
│   ├── data.py            # Data loading and preprocessing
│   ├── features.py        # Feature engineering
│   ├── models.py          # Model definitions
│   ├── cv.py              # Cross-validation
│   ├── metric.py          # Custom metrics
│   └── position.py        # Position mapping strategies
├── notebooks/             # Jupyter notebooks
├── scripts/               # Utility scripts
│   └── download_data.py   # Data download script
├── tests/                 # Unit tests
├── artifacts/             # Model artifacts (gitignored)
├── models/                # Saved models (gitignored)
├── logs/                  # Log files (gitignored)
└── submissions/           # Submission files

```

## Development Workflow

1. **Phase 0**: Setup (✅ Complete)
2. **Phase 1**: Data exploration & EDA
3. **Phase 2**: Cross-validation & metrics
4. **Phase 3**: Return prediction model
5. **Phase 4**: Risk forecasting model
6. **Phase 5**: Position mapping & risk control
7. **Phase 6**: Ensemble & stability
8. **Phase 7**: Submission pipeline

## Useful Commands

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_utils.py -v

# Format code
black src/ tests/

# Check code style
flake8 src/ tests/

# Sort imports
isort src/ tests/
```

## Notes

- All random seeds are set to 42 for reproducibility
- Data files are gitignored to save space
- Use `params.yaml` for all configuration changes
- Check README.md for detailed project plan
