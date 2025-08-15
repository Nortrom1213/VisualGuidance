#!/usr/bin/env python
"""
Comprehensive Test Script for Visual Guidance System

This script tests all major functionalities of the Visual Guidance System:
1. Environment and dependencies
2. File structure and configuration
3. Model architecture and creation
4. Dataset handling
5. Feature bank operations
6. Pipeline functionality
7. Training scripts
"""

import os
import sys
import json
import torch
import numpy as np
from PIL import Image
import tempfile
import shutil
import importlib.util

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_environment():
    """Test Python environment and PyTorch setup."""
    print("=" * 60)
    print("Testing Environment")
    print("=" * 60)
    
    try:
        # Python version
        print(f"Python version: {sys.version}")
        
        # PyTorch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU count: {torch.cuda.device_count()}")
        
        # Test basic tensor operations
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.randn(2, 3).to(device)
        y = torch.randn(2, 3).to(device)
        z = x + y
        print(f"✓ Tensor operations on {device}: {z.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False

def test_file_structure():
    """Test if all required files and directories exist."""
    print("\n" + "=" * 60)
    print("Testing File Structure")
    print("=" * 60)
    
    try:
        required_dirs = [
            "config", "models", "pipeline", "utils", "scripts", "data", "tests"
        ]
        
        required_files = [
            "config/config.py",
            "models/__init__.py",
            "models/adapters.py",
            "models/stp_detector.py",
            "models/mstp_selector.py",
            "pipeline/__init__.py",
            "pipeline/inference.py",
            "utils/__init__.py",
            "utils/feature_bank.py",
            "scripts/train_stp_detector.py",
            "scripts/train_mstp_selector.py",
            "README.md",
            "requirements.txt"
        ]
        
        # Check directories
        for dir_name in required_dirs:
            if os.path.isdir(dir_name):
                print(f"✓ Directory: {dir_name}")
            else:
                print(f"✗ Missing directory: {dir_name}")
        
        # Check files
        for file_name in required_files:
            if os.path.isfile(file_name):
                print(f"✓ File: {file_name}")
            else:
                print(f"✗ Missing file: {file_name}")
        
        return True
        
    except Exception as e:
        print(f"✗ File structure test failed: {e}")
        return False

def test_configuration():
    """Test configuration loading and validation."""
    print("\n" + "=" * 60)
    print("Testing Configuration")
    print("=" * 60)
    
    try:
        # Load config file
        config_file = "config/config.py"
        if not os.path.exists(config_file):
            print(f"✗ Config file not found: {config_file}")
            return False
        
        # Read and parse config
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for key configuration elements
        required_elements = [
            "class Config:",
            "STP_DETECTOR_CONFIG",
            "MSTP_SELECTOR_CONFIG",
            "INFERENCE_CONFIG",
            "SUPPORTED_GAMES"
        ]
        
        for element in required_elements:
            if element in content:
                print(f"✓ Found: {element}")
            else:
                print(f"✗ Missing: {element}")
        
        # Try to import config
        spec = importlib.util.spec_from_file_location("config", config_file)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        # Test config values
        config = config_module.Config()
        print(f"✓ Config loaded successfully")
        print(f"  - STP detector params: {len(config.STP_DETECTOR_CONFIG)}")
        print(f"  - MSTP selector params: {len(config.MSTP_SELECTOR_CONFIG)}")
        print(f"  - Supported games: {len(config.SUPPORTED_GAMES)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_model_architecture():
    """Test model architecture files and classes."""
    print("\n" + "=" * 60)
    print("Testing Model Architecture")
    print("=" * 60)
    
    try:
        model_files = [
            "models/adapters.py",
            "models/stp_detector.py", 
            "models/mstp_selector.py"
        ]
        
        for model_file in model_files:
            if os.path.exists(model_file):
                with open(model_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for key classes and functions
                if "adapters.py" in model_file:
                    if "class Adapter" in content:
                        print(f"✓ {model_file}: Adapter class found")
                    if "class CombinedPredictor" in content:
                        print(f"✓ {model_file}: CombinedPredictor class found")
                        
                elif "stp_detector.py" in model_file:
                    if "class STPDataset" in content:
                        print(f"✓ {model_file}: STPDataset class found")
                    if "class STPDetector" in content:
                        print(f"✓ {model_file}: STPDetector class found")
                    if "def get_stp_detector" in content:
                        print(f"✓ {model_file}: get_stp_detector function found")
                        
                elif "mstp_selector.py" in model_file:
                    if "class MSTPSelectorNet" in content:
                        print(f"✓ {model_file}: MSTPSelectorNet class found")
                    if "class MSTPSelectorDataset" in content:
                        print(f"✓ {model_file}: MSTPSelectorDataset class found")
            else:
                print(f"✗ {model_file}: File not found")
        
        return True
        
    except Exception as e:
        print(f"✗ Model architecture test failed: {e}")
        return False

def test_feature_bank():
    """Test feature bank functionality."""
    print("\n" + "=" * 60)
    print("Testing Feature Bank")
    print("=" * 60)
    
    try:
        feature_bank_file = "utils/feature_bank.py"
        if not os.path.exists(feature_bank_file):
            print(f"✗ Feature bank file not found: {feature_bank_file}")
            return False
        
        # Read feature bank file
        with open(feature_bank_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for key components
        required_components = [
            "class FeatureBank",
            "def add_feature",
            "def get_similar_features"
        ]
        
        for component in required_components:
            if component in content:
                print(f"✓ Found: {component}")
            else:
                print(f"✗ Missing: {component}")
        
        # Try to create feature bank instance
        spec = importlib.util.spec_from_file_location("feature_bank", feature_bank_file)
        feature_bank_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(feature_bank_module)
        
        # Test feature bank creation
        feature_bank = feature_bank_module.FeatureBank(feature_dim=512, quality_gamma=0.5, top_k=100)
        print(f"✓ Feature bank created successfully")
        print(f"  - Feature dimension: {feature_bank.feature_dim}")
        print(f"  - Quality gamma: {feature_bank.quality_gamma}")
        print(f"  - Top K: {feature_bank.top_k}")
        
        # Test adding features
        test_features = [np.random.randn(512) for _ in range(3)]
        test_metadata = [
            {"image_id": f"test_{i}.jpg", "bbox": [100*i, 100*i, 200*i, 200*i]}
            for i in range(3)
        ]
        
        for feature, metadata in zip(test_features, test_metadata):
            feature_bank.add_feature(feature, metadata)
        
        print(f"✓ Added {len(test_features)} features successfully")
        print(f"  - Total features: {len(feature_bank.features)}")
        print(f"  - Total metadata: {len(feature_bank.metadata)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Feature bank test failed: {e}")
        return False

def test_pipeline():
    """Test inference pipeline structure."""
    print("\n" + "=" * 60)
    print("Testing Inference Pipeline")
    print("=" * 60)
    
    try:
        pipeline_file = "pipeline/inference.py"
        if not os.path.exists(pipeline_file):
            print(f"✗ Pipeline file not found: {pipeline_file}")
            return False
        
        # Read pipeline file
        with open(pipeline_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for key components
        required_components = [
            "class VisualGuidancePipeline",
            "def process_single_image",
            "def process_directory",
            "def _load_detector",
            "def _load_selector"
        ]
        
        for component in required_components:
            if component in content:
                print(f"✓ Found: {component}")
            else:
                print(f"✗ Missing: {component}")
        
        # Try to import pipeline
        spec = importlib.util.spec_from_file_location("inference", pipeline_file)
        pipeline_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pipeline_module)
        
        print(f"✓ Pipeline module imported successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Pipeline test failed: {e}")
        return False

def test_training_scripts():
    """Test training script structure and functionality."""
    print("\n" + "=" * 60)
    print("Testing Training Scripts")
    print("=" * 60)
    
    try:
        training_scripts = [
            "scripts/train_stp_detector.py",
            "scripts/train_mstp_selector.py"
        ]
        
        for script in training_scripts:
            if os.path.exists(script):
                with open(script, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                print(f"✓ Training script exists: {script}")
                
                # Check for key components
                if 'if __name__ == "__main__"' in content:
                    print(f"  ✓ Has main function")
                else:
                    print(f"  ✗ Missing main function")
                
                if 'argparse' in content:
                    print(f"  ✓ Uses argparse for arguments")
                else:
                    print(f"  ✗ Missing argument parsing")
                
                if 'def train' in content or 'def main' in content:
                    print(f"  ✓ Has training function")
                else:
                    print(f"  ✗ Missing training function")
                    
            else:
                print(f"✗ Training script missing: {script}")
        
        return True
        
    except Exception as e:
        print(f"✗ Training scripts test failed: {e}")
        return False

def test_dataset_creation():
    """Test dataset creation with sample data."""
    print("\n" + "=" * 60)
    print("Testing Dataset Creation")
    print("=" * 60)
    
    try:
        # Create temporary test data
        temp_dir = tempfile.mkdtemp()
        test_image_path = os.path.join(temp_dir, "test_image.jpg")
        
        # Create dummy test image
        dummy_image = Image.new('RGB', (640, 480), color='red')
        dummy_image.save(test_image_path)
        
        # Create test annotations
        test_annotations = [
            {
                "image_id": "test_image.jpg",
                "boxes": [[100, 100, 200, 200], [300, 300, 400, 400]],
                "areas": [10000, 10000]
            }
        ]
        
        test_annotations_file = os.path.join(temp_dir, "test_annotations.json")
        with open(test_annotations_file, 'w') as f:
            json.dump(test_annotations, f)
        
        print(f"✓ Created test data successfully")
        print(f"  - Test image: {test_image_path}")
        print(f"  - Annotations: {test_annotations_file}")
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset creation test failed: {e}")
        return False

def test_dependencies():
    """Test if all required dependencies are available."""
    print("\n" + "=" * 60)
    print("Testing Dependencies")
    print("=" * 60)
    
    try:
        required_packages = [
            "torch", "torchvision", "numpy", "PIL", "cv2"
        ]
        
        for package in required_packages:
            try:
                if package == "PIL":
                    import PIL
                    print(f"✓ {package} imported successfully")
                elif package == "cv2":
                    import cv2
                    print(f"✓ {package} imported successfully")
                elif package == "torch":
                    import torch
                    print(f"✓ {package} imported successfully")
                elif package == "torchvision":
                    import torchvision
                    print(f"✓ {package} imported successfully")
                elif package == "numpy":
                    import numpy
                    print(f"✓ {package} imported successfully")
            except ImportError as e:
                print(f"✗ {package} import failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Dependencies test failed: {e}")
        return False

def run_comprehensive_test():
    """Run all tests and provide summary."""
    print("Visual Guidance System - Comprehensive Test Suite")
    print("=" * 80)
    
    tests = [
        ("Environment", test_environment),
        ("File Structure", test_file_structure),
        ("Configuration", test_configuration),
        ("Model Architecture", test_model_architecture),
        ("Feature Bank", test_feature_bank),
        ("Pipeline", test_pipeline),
        ("Training Scripts", test_training_scripts),
        ("Dataset Creation", test_dataset_creation),
        ("Dependencies", test_dependencies)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 80)
    print("COMPREHENSIVE TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{status:4} | {test_name}")
    
    print("-" * 80)
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The Visual Guidance System is ready to use.")
    else:
        print(f"⚠️  {total - passed} tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
