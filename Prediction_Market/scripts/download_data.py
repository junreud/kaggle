"""
Download data from Kaggle competition.

This script requires Kaggle API credentials to be set up.
See: https://github.com/Kaggle/kaggle-api#api-credentials
"""

import os
import sys
from pathlib import Path

def download_data(competition_name: str = "hull-tactical-market-prediction"):
    """
    Download competition data from Kaggle.
    
    Args:
        competition_name: Name of the Kaggle competition
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("Error: kaggle package not installed.")
        print("Install it with: pip install kaggle")
        sys.exit(1)
    
    # Check if Kaggle credentials exist
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("Error: Kaggle credentials not found.")
        print("Please set up your Kaggle API credentials:")
        print("1. Go to https://www.kaggle.com/settings/account")
        print("2. Click 'Create New API Token'")
        print("3. Place kaggle.json in ~/.kaggle/")
        print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        sys.exit(1)
    
    # Create data directory
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize API
    api = KaggleApi()
    api.authenticate()
    
    print(f"Downloading data from competition: {competition_name}")
    
    try:
        # Download all competition files
        api.competition_download_files(
            competition_name,
            path=str(data_dir),
            quiet=False
        )
        
        # Unzip if needed
        import zipfile
        zip_file = data_dir / f"{competition_name}.zip"
        if zip_file.exists():
            print(f"Unzipping {zip_file}...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            zip_file.unlink()  # Remove zip file
            print("Unzipping complete!")
        
        print("Data download complete!")
        print(f"Files saved to: {data_dir.absolute()}")
        
        # List downloaded files
        files = list(data_dir.glob("*"))
        print(f"\nDownloaded {len(files)} files:")
        for f in files:
            print(f"  - {f.name}")
            
    except Exception as e:
        print(f"Error downloading data: {e}")
        print("\nPossible solutions:")
        print("1. Make sure you've accepted the competition rules on Kaggle")
        print("2. Check that the competition name is correct")
        print("3. Verify your Kaggle API credentials are valid")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Kaggle competition data")
    parser.add_argument(
        "--competition",
        type=str,
        default="hull-tactical-market-prediction",
        help="Kaggle competition name"
    )
    
    args = parser.parse_args()
    download_data(args.competition)
