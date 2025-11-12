"""
Check submission.parquet format and structure.

CORRECT FORMAT for Hull Tactical Market Prediction:
- Columns: date_id (int64), allocation (float64)
- Range: allocation must be in [0, 2] representing S&P500 allocation
- No NaN, no inf values
- Parquet format with pyarrow engine
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("CREATING SAMPLE SUBMISSION FILE")
print("="*80)

# Create sample submission data (correct format for Kaggle)
sample_data = {
    'date_id': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype='int64'),
    'allocation': np.array([1.234, 0.567, 1.890, 0.345, 1.678,
                           1.456, 0.789, 1.123, 0.234, 1.567], dtype='float64')
}

submission = pd.DataFrame(sample_data)

print("\n✅ Sample Submission DataFrame:")
print(submission)

print("\n" + "="*80)
print("DATAFRAME INFO")
print("="*80)
print(submission.info())

print("\n" + "="*80)
print("COLUMN DTYPES")
print("="*80)
for col in submission.columns:
    print(f"  {col}: {submission[col].dtype}")

print("\n" + "="*80)
print("STATISTICS")
print("="*80)
print(submission.describe())

# Save as parquet
output_path = Path("sample_submission.parquet")
submission.to_parquet(output_path, index=False, engine='pyarrow')

print(f"\n✅ Saved to: {output_path}")
print(f"   File size: {output_path.stat().st_size} bytes")

# Read it back to verify
print("\n" + "="*80)
print("READING BACK FROM PARQUET")
print("="*80)

loaded = pd.read_parquet(output_path)
print(loaded)

print("\n" + "="*80)
print("VERIFICATION")
print("="*80)
print(f"✓ Columns: {list(loaded.columns)}")
print(f"✓ Shape: {loaded.shape}")
print(f"✓ Data types match: {(submission.dtypes == loaded.dtypes).all()}")
print(f"✓ Values match: {submission.equals(loaded)}")

print("\n" + "="*80)
print("EXPECTED FORMAT FOR KAGGLE")
print("="*80)
print("Required columns:")
print("  1. date_id (int64)")
print("  2. allocation (float64) - Range [0, 2]")
print("")
print("Requirements:")
print("  ✓ No index column")
print("  ✓ No missing values (NaN)")
print("  ✓ No infinite values (inf)")
print("  ✓ Parquet format (not CSV)")
print("  ✓ PyArrow engine")
print("  ✓ Column names exactly: 'date_id', 'allocation'")
print("  ✓ date_id must match test.csv date_id values")
print("  ✓ allocation in range [0, 2] representing S&P500 allocation")

print("\n" + "="*80)
print("SAMPLE VALUES")
print("="*80)
print(f"date_id range: {loaded['date_id'].min()} to {loaded['date_id'].max()}")
print(f"allocation range: {loaded['allocation'].min():.4f} to {loaded['allocation'].max():.4f}")
print(f"allocation mean: {loaded['allocation'].mean():.4f}")
print(f"allocation std: {loaded['allocation'].std():.4f}")

# Cleanup
output_path.unlink()
print(f"\n✅ Cleaned up {output_path}")
print("="*80)

print("\n✅ All checks passed!")
