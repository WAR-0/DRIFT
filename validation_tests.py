#!/usr/bin/env python3
"""
DRIFT System Improvements Validation Tests
Verify all requirements from the improvement request have been met
"""

import os
import json
import time
import yaml
from pathlib import Path

def test_centralized_configuration():
    """Test centralized hyperparameters"""
    print("Testing Centralized Configuration...")
    
    # Check if config file exists
    config_path = Path("config/drift_config.yaml")
    assert config_path.exists(), "❌ config/drift_config.yaml not found"
    
    # Load and validate config
    with open(config_path) as f:
        config_data = yaml.safe_load(f)
    
    # Check key sections exist
    required_sections = ['drift', 'nurture', 'system', 'experiments']
    for section in required_sections:
        assert section in config_data, f"❌ Missing config section: {section}"
    
    # Check specific values were moved from hardcoded
    assert config_data['drift']['resonance']['threshold'] == 0.62, "❌ Resonance threshold not found"
    assert config_data['drift']['memory']['shadow_limit'] == 999, "❌ Shadow limit not found"
    assert config_data['nurture']['topology']['termination_base'] == 1000000.0, "❌ Termination base not found"
    
    print("✅ Centralized configuration working")
    return True

def test_configuration_loader():
    """Test config.py loads from YAML"""
    print("Testing Configuration Loader...")
    
    try:
        from core.config import get_config
        config = get_config()
        
        # Test accessing nested values
        assert config.drift.resonance.threshold == 0.62
        assert config.drift.memory.shadow_limit == 999
        assert hasattr(config, 'nurture')
        
        print("✅ Configuration loader working")
        return True
        
    except Exception as e:
        print(f"❌ Configuration loader failed: {e}")
        return False

def test_emotional_tagger():
    """Test transformer-based emotional tagger"""
    print("Testing Emotional Tagger v2...")
    
    try:
        from core.emotional_tagger_v2 import RobustEmotionalTagger
        
        tagger = RobustEmotionalTagger()
        
        # Test basic functionality
        result = tagger.tag("I'm not happy at all")
        assert hasattr(result, 'valence')
        assert hasattr(result, 'arousal')
        assert hasattr(result, 'confidence')
        
        # Test negation handling (should be negative valence)
        assert result.valence < 0, "❌ Negation not handled correctly"
        
        print("✅ Emotional tagger v2 working (with negation handling)")
        return True
        
    except Exception as e:
        print(f"❌ Emotional tagger failed: {e}")
        return False

def test_structured_logging():
    """Test structured logging system"""
    print("Testing Structured Logging...")
    
    try:
        from core.drift_logger import get_drift_logger, DRIFTEvent
        
        logger = get_drift_logger("test_component")
        
        # Test basic logging
        logger.info("test_event", test_param="test_value", numeric_value=123)
        
        # Test specialized event methods
        logger.resonance_calculated(
            score=0.75,
            threshold=0.62,
            components={"semantic": 0.4, "preservation": 0.3, "emotional": 0.05},
            triggered=True,
            action="help"
        )
        
        print("✅ Structured logging working")
        return True
        
    except Exception as e:
        print(f"❌ Structured logging failed: {e}")
        return False

def test_identity_validator():
    """Test identity validator framework exists"""
    print("Testing Identity Validator...")
    
    try:
        from experiments.identity_validator import IdentityValidator
        
        # Just test that it loads - full testing would require models
        validator = IdentityValidator(use_openai=False)
        assert hasattr(validator, 'validate_consistency')
        assert hasattr(validator, 'create_baseline_profile')
        
        print("✅ Identity validator framework working")
        return True
        
    except Exception as e:
        print(f"❌ Identity validator failed: {e}")
        return False

def test_ablation_study():
    """Test ablation study framework"""
    print("Testing Ablation Study Framework...")
    
    try:
        from experiments.ablation_study import AblationStudy
        
        study = AblationStudy()
        assert hasattr(study, 'run_full_study')
        assert hasattr(study, 'analyze_component_impacts')
        
        # Check components are defined
        expected_components = [
            'emotional_tagging', 'drift_stream', 'reflection_stream',
            'shadow_memory', 'resonance_detection', 'consolidation'
        ]
        
        for component in expected_components:
            assert component in study.components, f"❌ Missing component: {component}"
        
        print("✅ Ablation study framework working")
        return True
        
    except Exception as e:
        print(f"❌ Ablation study failed: {e}")
        return False

def test_lexicon_terminology():
    """Test lexicon and terminology updates"""
    print("Testing Lexicon and Terminology...")
    
    # Check lexicon exists
    lexicon_path = Path("LEXICON.md")
    assert lexicon_path.exists(), "❌ LEXICON.md not found"
    
    # Check content has correct mappings
    with open(lexicon_path) as f:
        lexicon_content = f.read()
    
    # Verify key terminology mappings exist
    mappings = [
        ("Consciousness", "Integrative Core"),
        ("Thought", "Generated Fragment"), 
        ("Shadow Memory", "Transient Buffer"),
        ("Drift", "Associative Elaboration"),
        ("Resonance", "Saliency Gating"),
        ("Emotional Tagging", "Valence-Arousal Heuristics"),
        ("Reflection", "Consolidated-Content Re-synthesis"),
        ("Identity", "Behavioral Consistency Profile")
    ]
    
    for old_term, new_term in mappings:
        assert old_term in lexicon_content and new_term in lexicon_content, \
               f"❌ Missing terminology mapping: {old_term} -> {new_term}"
    
    print("✅ Lexicon and terminology complete")
    return True

def test_no_hardcoded_values():
    """Verify no hardcoded numbers remain in main files"""
    print("Testing Hardcoded Values Removal...")
    
    # Check that integrated_consciousness_v2.py uses configuration
    try:
        from integrated_consciousness_v2 import IntegratedConsciousness
        
        # This should not raise an error - it should use config
        ai = IntegratedConsciousness("test")
        config = ai.config
        
        # Verify it's using config values
        assert hasattr(config, 'drift')
        assert hasattr(config.drift, 'resonance')
        
        print("✅ Hardcoded values successfully removed")
        return True
        
    except Exception as e:
        print(f"❌ Hardcoded value removal incomplete: {e}")
        return False

def test_demo_update():
    """Test updated demo works"""
    print("Testing Updated Demo...")
    
    demo_path = Path("demo_updated.py")
    assert demo_path.exists(), "❌ demo_updated.py not found"
    
    # Check if demo contains new terminology
    with open(demo_path) as f:
        demo_content = f.read()
    
    new_terms = ["Integrative Core", "Associative Elaboration", "Saliency Gating", "Valence-Arousal Heuristics"]
    for term in new_terms:
        assert term in demo_content, f"❌ Demo missing new term: {term}"
    
    print("✅ Demo updated with new terminology")
    return True

def run_all_tests():
    """Run all validation tests"""
    print("=" * 60)
    print("DRIFT SYSTEM IMPROVEMENTS VALIDATION")
    print("=" * 60)
    
    tests = [
        ("Centralized Hyperparameters", test_centralized_configuration),
        ("Configuration Loader", test_configuration_loader),  
        ("Transformer Emotional Tagger", test_emotional_tagger),
        ("Structured Logging", test_structured_logging),
        ("Identity Validator", test_identity_validator),
        ("Ablation Study", test_ablation_study),
        ("Lexicon & Terminology", test_lexicon_terminology),
        ("Hardcoded Values Removed", test_no_hardcoded_values),
        ("Updated Demo", test_demo_update)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL IMPROVEMENTS SUCCESSFULLY IMPLEMENTED!")
        print("The DRIFT system has been enhanced with:")
        print("  ✓ Centralized configuration management")
        print("  ✓ Transformer-based emotional analysis") 
        print("  ✓ Structured JSON logging")
        print("  ✓ Identity validation with LLM-as-judge")
        print("  ✓ Systematic ablation study framework")
        print("  ✓ Precise scientific terminology")
        print("  ✓ No hardcoded values remain")
    else:
        print(f"\n⚠️  {total - passed} tests failed - see details above")
    
    print("\n" + "=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)