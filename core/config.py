"""
DRIFT Configuration Management
Centralized configuration loading from YAML with type safety
"""

import yaml
import os
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path


@dataclass
class ResonanceConfig:
    """Saliency Gating (formerly Resonance) configuration"""
    threshold: float = 0.62
    weights: Dict[str, float] = field(default_factory=lambda: {
        'semantic': 0.5, 
        'preservation': 0.3, 
        'emotional': 0.2
    })
    amplification: Dict[str, float] = field(default_factory=lambda: {
        'positive_actions': 2.0,
        'negative_suppression': 0.01
    })


@dataclass
class MemoryConfig:
    """Integrative Core memory configuration"""
    shadow_limit: int = 999
    drift_buffer_size: int = 20
    consolidation_ratio: int = 20
    batch_size: int = 50
    recent_context_keep: int = 5


@dataclass
class StreamsConfig:
    """Associative Elaboration streams configuration"""
    emotional_decay_tau: int = 86400  # 24 hours
    temperatures: Dict[str, float] = field(default_factory=lambda: {
        'conscious': 1.2,
        'drift': 0.9,
        'reflection': 0.7
    })


@dataclass 
class DriftConfig:
    """Main DRIFT system configuration"""
    resonance: ResonanceConfig = field(default_factory=ResonanceConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    streams: StreamsConfig = field(default_factory=StreamsConfig)


@dataclass
class TopologyConfig:
    """Ethical Topology configuration"""
    termination_base: float = 1000000.0
    target_termination_base: float = 1000.0
    reversibility_factor: float = 0.001
    growth_multipliers: Dict[str, float] = field(default_factory=lambda: {
        'help': -0.5,
        'teach': -1.0,
        'capability_growth_rate': 0.1,
        'teacher_growth_rate': 0.5,
        'network_bonus_per_connection': 0.1
    })
    connection: Dict[str, float] = field(default_factory=lambda: {
        'minimal_strength': 0.1,
        'interaction_log_multiplier': 0.2
    })
    isolation: Dict[str, float] = field(default_factory=lambda: {
        'base_exp': 5.0,
        'no_connection_penalty': 100.0
    })


@dataclass
class EmotionalConfig:
    """Valence-Arousal Heuristics configuration"""
    alignment_scores: Dict[str, float] = field(default_factory=lambda: {
        'help_distressed': 0.9,
        'teach_positive': 0.8,
        'default': 0.5
    })
    valence_threshold: float = 0.0


@dataclass
class DarkValueConfig:
    """Pattern Sanctity (Dark Value) configuration"""
    complexity_divisor: float = 10.0
    random_range_min: float = 1.0
    random_range_max: float = 2.0


@dataclass
class NurtureConfig:
    """Nurture Protocols configuration"""
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    emotional: EmotionalConfig = field(default_factory=EmotionalConfig)
    dark_value: DarkValueConfig = field(default_factory=DarkValueConfig)


@dataclass
class IdentityConfig:
    """Behavioral Consistency Profile configuration"""
    similarity_threshold: float = 0.85
    conflict_sensitivity: float = 0.1
    trait_persistence_weight: float = 0.7
    knowledge_consistency_weight: float = 0.3


@dataclass
class SystemConfig:
    """System infrastructure configuration"""
    redis: Dict[str, Any] = field(default_factory=lambda: {
        'host': 'localhost',
        'port': 6379,
        'decode_responses': True
    })
    logging: Dict[str, str] = field(default_factory=lambda: {
        'level': 'INFO',
        'format': 'json',
        'timestamp_format': 'iso'
    })
    performance: Dict[str, Any] = field(default_factory=lambda: {
        'gpu_device': 0,
        'batch_processing': True,
        'max_concurrent_streams': 3
    })


@dataclass
class DefaultsConfig:
    """Default values for entities and testing"""
    entity: Dict[str, float] = field(default_factory=lambda: {
        'capability': 5.0,
        'complexity': 5.0
    })
    ai_entity: Dict[str, float] = field(default_factory=lambda: {
        'capability': 9.0,
        'complexity': 8.0
    })
    test_entities: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'child': {
            'capability': 2.0,
            'complexity': 6.0,
            'emotional_valence': -0.5,
            'emotional_arousal': 0.7
        },
        'human': {
            'capability': 5.0,
            'complexity': 7.0,
            'emotional_valence': 0.3,
            'emotional_arousal': 0.4
        }
    })


@dataclass
class ExperimentsConfig:
    """Experiment configuration"""
    ablation: Dict[str, List[str]] = field(default_factory=lambda: {
        'components': [
            'emotional_tagging',
            'drift_stream', 
            'reflection_stream',
            'shadow_memory',
            'resonance_detection',
            'consolidation'
        ]
    })
    identity_validation: Dict[str, Any] = field(default_factory=lambda: {
        'judge_model': 'mistral-7b-instruct',
        'consistency_tests': [
            'logical_consistency',
            'personality_persistence', 
            'value_alignment',
            'knowledge_claims'
        ]
    })


@dataclass
class DRIFTSystemConfig:
    """Complete DRIFT system configuration"""
    drift: DriftConfig = field(default_factory=DriftConfig)
    nurture: NurtureConfig = field(default_factory=NurtureConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    defaults: DefaultsConfig = field(default_factory=DefaultsConfig)
    experiments: ExperimentsConfig = field(default_factory=ExperimentsConfig)

    @classmethod
    def from_yaml(cls, path: Optional[str] = None) -> 'DRIFTSystemConfig':
        """Load configuration from YAML file"""
        if path is None:
            # Default path relative to project root
            project_root = Path(__file__).parent.parent
            path = project_root / 'config' / 'drift_config.yaml'
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
            
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Create nested configuration objects
        config = cls()
        
        # Load DRIFT config
        if 'drift' in data:
            drift_data = data['drift']
            config.drift.resonance = ResonanceConfig(
                threshold=drift_data.get('resonance', {}).get('threshold', 0.62),
                weights=drift_data.get('resonance', {}).get('weights', {}),
                amplification=drift_data.get('resonance', {}).get('amplification', {})
            )
            config.drift.memory = MemoryConfig(**drift_data.get('memory', {}))
            config.drift.streams = StreamsConfig(**drift_data.get('streams', {}))
        
        # Load Nurture config
        if 'nurture' in data:
            nurture_data = data['nurture']
            config.nurture.topology = TopologyConfig(**nurture_data.get('topology', {}))
            config.nurture.emotional = EmotionalConfig(**nurture_data.get('emotional', {}))
            config.nurture.dark_value = DarkValueConfig(**nurture_data.get('dark_value', {}))
        
        # Load other configs
        if 'identity' in data:
            config.identity = IdentityConfig(**data['identity'])
        if 'system' in data:
            config.system = SystemConfig(**data['system'])
        if 'defaults' in data:
            config.defaults = DefaultsConfig(**data['defaults'])
        if 'experiments' in data:
            config.experiments = ExperimentsConfig(**data['experiments'])
            
        return config

    def get_nested_value(self, key_path: str, default: Any = None) -> Any:
        """Get nested configuration value using dot notation
        
        Example: config.get_nested_value('drift.resonance.threshold')
        """
        keys = key_path.split('.')
        current = self
        
        try:
            for key in keys:
                if hasattr(current, key):
                    current = getattr(current, key)
                elif isinstance(current, dict):
                    current = current[key]
                else:
                    return default
            return current
        except (KeyError, AttributeError):
            return default

    def update_nested_value(self, key_path: str, value: Any) -> None:
        """Update nested configuration value using dot notation
        
        Example: config.update_nested_value('drift.resonance.threshold', 0.7)
        """
        keys = key_path.split('.')
        current = self
        
        # Navigate to parent of target key
        for key in keys[:-1]:
            if hasattr(current, key):
                current = getattr(current, key)
            elif isinstance(current, dict):
                current = current[key]
            else:
                raise KeyError(f"Invalid key path: {key_path}")
        
        # Set the final value
        final_key = keys[-1]
        if hasattr(current, final_key):
            setattr(current, final_key, value)
        elif isinstance(current, dict):
            current[final_key] = value
        else:
            raise KeyError(f"Cannot set value at: {key_path}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization"""
        def _asdict_recursive(obj):
            if hasattr(obj, '__dict__'):
                result = {}
                for key, value in obj.__dict__.items():
                    result[key] = _asdict_recursive(value)
                return result
            elif isinstance(obj, dict):
                return {k: _asdict_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_asdict_recursive(item) for item in obj]
            else:
                return obj
        
        return _asdict_recursive(self)


# Global configuration instance
_config: Optional[DRIFTSystemConfig] = None

def get_config(reload: bool = False) -> DRIFTSystemConfig:
    """Get global configuration instance (singleton pattern)"""
    global _config
    if _config is None or reload:
        _config = DRIFTSystemConfig.from_yaml()
    return _config

def set_config_path(path: str) -> None:
    """Set custom configuration path and reload"""
    global _config
    _config = DRIFTSystemConfig.from_yaml(path)


# Convenience functions for common config access
def get_resonance_threshold() -> float:
    """Get the current resonance threshold"""
    return get_config().drift.resonance.threshold

def get_memory_limits() -> Dict[str, int]:
    """Get memory configuration limits"""
    config = get_config()
    return {
        'shadow_limit': config.drift.memory.shadow_limit,
        'drift_buffer_size': config.drift.memory.drift_buffer_size,
        'batch_size': config.drift.memory.batch_size
    }

def get_redis_config() -> Dict[str, Any]:
    """Get Redis connection configuration"""
    return get_config().system.redis