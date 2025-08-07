#!/usr/bin/env python3
"""
Quick start script for Face Recognition API
"""
import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.11+"""
    if sys.version_info < (3, 11):
        print("❌ Python 3.11+ is required")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]} detected")

def create_directories():
    """Create necessary directories"""
    directories = [
        "storage/uploads",
        "storage/database", 
        "storage/temp",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    print("✅ Created directory structure")

def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        sys.exit(1)

def check_gpu():
    """Check for GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("⚠️ No GPU detected, will use CPU")
            return False
    except ImportError:
        print("⚠️ PyTorch not found, cannot check GPU")
        return False

def update_config_for_cpu():
    """Update config.yaml to use CPU if no GPU"""
    import yaml
    
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        config['face_recognition']['device'] = -1  # Use CPU
        
        with open('config.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(config, f, indent=2)
        
        print("✅ Updated config for CPU usage")
    except Exception as e:
        print(f"⚠️ Could not update config: {e}")

def main():
    print("🚀 Face Recognition API Quick Start")
    print("=" * 40)
    
    # Check requirements
    check_python_version()
    create_directories()
    
    # Install dependencies
    install_dependencies()
    
    # Check GPU and update config if needed
    has_gpu = check_gpu()
    if not has_gpu:
        update_config_for_cpu()
    
    print("\n" + "=" * 40)
    print("🎉 Setup complete!")
    print("\nTo start the API:")
    print("  python main.py")
    print("\nTo use Docker:")
    print("  docker-compose up --build")
    print("\nWeb interface will be available at:")
    print("  http://localhost:8000/web")
    print("\nAPI documentation:")
    print("  http://localhost:8000/docs")
    print("\n📋 GDPR Compliance Reminders:")
    print("  • Ensure explicit consent before enrolling faces")
    print("  • Regularly review and delete old data")
    print("  • Document data processing purposes")
    print("  • Provide data export/deletion on request")

if __name__ == "__main__":
    main()
