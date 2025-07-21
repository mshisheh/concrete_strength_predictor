#!/usr/bin/env python3
"""
Simple script to check the current best model file.
"""

import joblib
from pathlib import Path
import datetime

def check_model():
    """Check the current best model file."""
    model_path = Path("models/best_model.joblib")
    
    if not model_path.exists():
        print("❌ No best_model.joblib found")
        return
    
    print(f"✅ Model file exists: {model_path}")
    
    # Get file stats
    stat = model_path.stat()
    mod_time = datetime.datetime.fromtimestamp(stat.st_mtime)
    print(f"📅 Last modified: {mod_time}")
    print(f"📊 File size: {stat.st_size:,} bytes")
    
    try:
        # Load model info
        model_info = joblib.load(model_path)
        
        if isinstance(model_info, dict):
            print(f"🤖 Model name: {model_info.get('model_name', 'Unknown')}")
            print(f"🕐 Timestamp: {model_info.get('timestamp', 'Unknown')}")
            print(f"🎯 Model type: {type(model_info.get('model', 'Unknown')).__name__}")
        else:
            print(f"⚠️  Model is not a dict: {type(model_info)}")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")

if __name__ == "__main__":
    check_model()
