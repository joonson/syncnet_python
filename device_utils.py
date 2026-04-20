#!/usr/bin/python
#-*- coding: utf-8 -*-

import torch

def get_device(prefer_cuda=True):
    """
    Get the best available device (CUDA if available and preferred, otherwise CPU)
    
    Args:
        prefer_cuda (bool): Whether to prefer CUDA over CPU if available
        
    Returns:
        str: Device string ('cuda' or 'cpu')
    """
    if prefer_cuda and torch.cuda.is_available():
        device = 'cuda'
        print(f'[INFO] Using CUDA GPU: {torch.cuda.get_device_name(0)}')
    else:
        device = 'cpu'
        if prefer_cuda and not torch.cuda.is_available():
            print('[INFO] CUDA not available, falling back to CPU')
        else:
            print('[INFO] Using CPU')
    
    return device

def print_device_info():
    """Print information about available devices"""
    print("=" * 50)
    print("Device Information:")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    print(f"PyTorch Version: {torch.__version__}")
    print("=" * 50)
