"""
Test the submission notebook locally before uploading to Kaggle.
"""
import subprocess
import sys

print("="*80)
print("TESTING SUBMISSION NOTEBOOK LOCALLY")
print("="*80)

# Convert notebook to Python script and execute
print("\nConverting notebook to Python script...")
result = subprocess.run([
    'jupyter', 'nbconvert', 
    '--to', 'script',
    '--output', 'submission_test',
    'submission.ipynb'
], capture_output=True, text=True)

if result.returncode != 0:
    print("Error converting notebook:")
    print(result.stderr)
    sys.exit(1)

print("✓ Notebook converted to submission_test.py")

# Modify the script to use local data paths
print("\nModifying data paths for local testing...")
with open('submission_test.py', 'r') as f:
    content = f.read()

# Replace Kaggle paths with local paths
content = content.replace('/kaggle/input/train.csv', 'data/raw/train.csv')
content = content.replace('/kaggle/input/test.csv', 'data/raw/test.csv')

with open('submission_test.py', 'w') as f:
    f.write(content)

print("✓ Paths updated for local testing")

# Execute the script
print("\n" + "="*80)
print("EXECUTING SUBMISSION SCRIPT")
print("="*80 + "\n")

result = subprocess.run(['python', 'submission_test.py'], capture_output=True, text=True)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

if result.returncode == 0:
    print("\n" + "="*80)
    print("✓ TEST SUCCESSFUL!")
    print("="*80)
    print("\nYour notebook is ready to upload to Kaggle!")
    print("\nNext steps:")
    print("1. Open submission.ipynb")
    print("2. Upload to Kaggle competition")
    print("3. Update data paths to Kaggle format:")
    print("   - /kaggle/input/train.csv")
    print("   - /kaggle/input/test.csv")
    print("4. Run the notebook on Kaggle")
else:
    print("\n" + "="*80)
    print("✗ TEST FAILED")
    print("="*80)
    print(f"Exit code: {result.returncode}")
