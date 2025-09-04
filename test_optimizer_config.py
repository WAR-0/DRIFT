#!/usr/bin/env python3
"""
Quick test to debug optimizer configuration issues
"""

import sys
sys.path.append('/home/aaron/projects/DRIFT')

from core.config import get_config
import json

def test_config_structure():
    print("Testing configuration structure...")
    
    config = get_config()
    print(f"Config type: {type(config)}")
    
    # Print config structure
    if hasattr(config, 'drift'):
        print("✅ config.drift exists")
        if hasattr(config.drift, 'resonance'):
            print("✅ config.drift.resonance exists")
            if hasattr(config.drift.resonance, 'amplification'):
                print("✅ config.drift.resonance.amplification exists")
                print(f"Amplification type: {type(config.drift.resonance.amplification)}")
                print(f"Amplification content: {config.drift.resonance.amplification}")
            else:
                print("❌ config.drift.resonance.amplification missing")
                print(f"Resonance attrs: {dir(config.drift.resonance)}")
        else:
            print("❌ config.drift.resonance missing")
    else:
        print("❌ config.drift missing")
        print(f"Config top-level keys: {list(config.__dict__.keys()) if hasattr(config, '__dict__') else 'No __dict__'}")

if __name__ == "__main__":
    test_config_structure()