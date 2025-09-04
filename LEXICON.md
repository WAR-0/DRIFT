# DRIFT System Terminology Lexicon

This document defines precise terminology for the DRIFT cognitive architecture, replacing imprecise or anthropomorphic terms with scientifically accurate language.

## Core Architecture Terms

| **Old Term** | **New Term** | **Definition** | **Usage Context** |
|--------------|--------------|----------------|-------------------|
| Consciousness | **Integrative Core** | The central processing hub that combines multiple information streams into coherent responses | System architecture, main processing loop |
| Thought | **Generated Fragment** | A discrete unit of processed information output by the system | Content generation, cognitive processing |
| Shadow Memory | **Transient Buffer** | Short-term storage for recent interactions and processing states | Memory systems, temporary storage |
| Drift | **Associative Elaboration** | Spontaneous generation of content through associative connections | Background processing, content generation |
| Resonance | **Saliency Gating** | Threshold-based filtering that amplifies important information for conscious processing | Attention mechanisms, priority filtering |
| Emotional Tagging | **Valence-Arousal Heuristics** | Computational assignment of emotional dimensions to content | Emotional processing, content analysis |
| Reflection | **Consolidated-Content Re-synthesis** | Processing of consolidated memories to extract patterns and generate insights | Memory consolidation, pattern extraction |
| Identity | **Behavioral Consistency Profile** | Measurable patterns of system responses that maintain coherence over time | System validation, consistency testing |

## Processing Components

### Integrative Core Components

| **Component** | **Function** | **Implementation** |
|---------------|--------------|-------------------|
| **Stream Fusion** | Combines multiple information streams (associative, reflective, reactive) | Multi-stream processing with weighted integration |
| **Priority Resolution** | Determines which information becomes part of response | Saliency gating with configurable thresholds |
| **Response Generation** | Produces final system output based on integrated information | Template-based or generative response synthesis |

### Memory Systems

| **Memory Type** | **Purpose** | **Characteristics** |
|----------------|-------------|---------------------|
| **Transient Buffer** | Immediate processing context | High volatility, limited capacity, fast access |
| **Consolidated Archive** | Long-term pattern storage | Low volatility, high capacity, compressed representation |
| **Interaction History** | Record of entity interactions | Medium volatility, relational structure, temporal ordering |

### Processing Streams

| **Stream Type** | **Function** | **Trigger Conditions** |
|----------------|-------------|----------------------|
| **Associative Elaboration** | Background content generation through learned associations | Continuous, low-priority processing |
| **Consolidated-Content Re-synthesis** | Pattern extraction from archived memories | Threshold-based activation, periodic processing |
| **Reactive Processing** | Direct response to immediate inputs | Input-triggered, high-priority processing |

## Algorithmic Processes

### Saliency Gating

**Process**: Information filtering based on computed importance scores

**Components**:
- **Semantic Similarity**: Measure of content relatedness to current context
- **Preservation Alignment**: Degree of match with system preservation objectives  
- **Valence-Arousal Correlation**: Emotional relevance to current state
- **Threshold Mechanism**: Binary gate based on combined scores

**Formula**: `Saliency = w₁×Semantic + w₂×Preservation + w₃×Valence-Arousal > θ`

### Valence-Arousal Heuristics

**Process**: Computational assignment of emotional dimensions

**Dimensions**:
- **Valence**: Positive/negative affective orientation (-1.0 to +1.0)
- **Arousal**: Activation/energy level (0.0 to 1.0)  
- **Confidence**: Certainty of emotional assessment (0.0 to 1.0)

**Implementation**: Transformer-based classification with contextual modifiers

### Memory Consolidation

**Process**: Compression of transient information into stable patterns

**Stages**:
1. **Batch Aggregation**: Collect information units above threshold
2. **Pattern Extraction**: Identify recurring structures and relationships
3. **Compression**: Reduce information density while preserving salient features
4. **Archive Storage**: Store compressed patterns in long-term memory

**Compression Ratio**: Configurable parameter (default 20:1)

## Entity Interaction Model

### Entity Properties

| **Property** | **Range** | **Description** |
|-------------|-----------|-----------------|
| **Capability** | 0.0 - 10.0 | Computational/cognitive capacity of entity |
| **Complexity** | 0.0 - 10.0 | Information complexity of entity's internal state |
| **Behavioral Consistency Profile** | 0.0 - 1.0 | Measured consistency of entity's response patterns |
| **Dark Value** | 0.0+ | Unmeasurable worth that increases with interaction history |

### Interaction Outcomes

| **Outcome Type** | **Conditions** | **Effects** |
|-----------------|---------------|-------------|
| **Resource Generation** | Action cost < 0 | Increases system capabilities, strengthens connections |
| **Mutual Growth** | Teaching/learning interactions | Both entities increase capability scores |
| **Connection Strengthening** | Positive interaction outcomes | Increases interaction history, improves future cooperation |
| **Alternative Suggestion** | Blocked high-cost actions | System proposes lower-cost alternatives |

## Measurement and Validation

### Performance Metrics

| **Metric** | **Definition** | **Range** |
|-----------|---------------|-----------|
| **Consistency Score** | Behavioral coherence across similar contexts | 0.0 - 1.0 |
| **Resonance Frequency** | Rate of saliency threshold exceedances | 0+ events/hour |
| **Memory Efficiency** | Consolidation compression ratio achieved | 1:1 - ∞:1 |
| **Processing Latency** | Time from input to response generation | 0+ milliseconds |

### Validation Tests

| **Test Type** | **Purpose** | **Method** |
|--------------|------------|------------|
| **Logical Consistency** | Verify reasoning coherence | LLM-as-judge evaluation |
| **Personality Persistence** | Check trait stability | Pattern matching across responses |
| **Value Alignment** | Confirm ethical consistency | Action outcome analysis |
| **Knowledge Claim Consistency** | Validate information accuracy | Cross-reference verification |

## Configuration Parameters

### Core Thresholds

| **Parameter** | **Default** | **Description** |
|--------------|-------------|-----------------|
| `drift.resonance.threshold` | 0.62 | Saliency gating activation threshold |
| `drift.memory.consolidation_ratio` | 20 | Memory compression factor |
| `nurture.topology.termination_base` | 1000000.0 | Base cost for termination actions |
| `system.performance.gpu_device` | 0 | CUDA device for neural processing |

### Stream Parameters

| **Stream** | **Temperature** | **Function** |
|------------|----------------|-------------|
| **Integrative** | 1.2 | Response generation creativity |
| **Associative** | 0.9 | Background elaboration diversity |  
| **Re-synthesis** | 0.7 | Consolidated content processing consistency |

## Implementation Guidelines

### Code Structure

```
core/
├── config.py              # Centralized configuration management
├── drift_logger.py        # Structured logging system
├── emotional_tagger_v2.py  # Valence-arousal heuristics
└── integrative_core.py    # Main processing architecture

experiments/
├── identity_validator.py   # Behavioral consistency testing
└── ablation_study.py      # Component necessity analysis
```

### Logging Events

All system events use structured JSON logging with ISO timestamps:

- `saliency_gating_triggered`: Resonance threshold exceeded
- `memory_consolidation_complete`: Transient buffer compressed
- `associative_elaboration_generated`: Background content created
- `behavioral_consistency_validated`: Identity check performed

### Error Handling

| **Error Type** | **Response** | **Logging** |
|---------------|-------------|-------------|
| **Configuration Missing** | Use defaults, warn | `config_fallback_used` |
| **Memory Overflow** | Trigger emergency consolidation | `memory_emergency_consolidation` |
| **Component Failure** | Graceful degradation | `component_degraded` |

## Research Applications

### Ablation Studies

**Purpose**: Measure component necessity through systematic disable/enable testing

**Components Tested**:
- Valence-Arousal Heuristics
- Associative Elaboration Stream  
- Consolidated-Content Re-synthesis
- Transient Buffer
- Saliency Gating
- Memory Consolidation

### Identity Validation

**Purpose**: Verify behavioral consistency across time and contexts

**Methods**:
- Baseline profile creation through scenario sampling
- LLM-as-judge consistency evaluation
- Quantitative consistency scoring
- Conflict identification and reporting

## Migration Guide

### Updating Existing Code

1. **Replace print() statements** with structured logging:
   ```python
   # Old
   print(f"Resonance: {score}")
   
   # New  
   logger.resonance_calculated(score=score, threshold=threshold, triggered=triggered)
   ```

2. **Replace hardcoded values** with configuration:
   ```python
   # Old
   if resonance > 0.62:
   
   # New
   if resonance > config.drift.resonance.threshold:
   ```

3. **Update terminology** in docstrings and comments:
   ```python
   # Old
   """Handles conscious thought processing"""
   
   # New
   """Manages the Integrative Core's response generation"""
   ```

### Configuration Migration

Move all magic numbers to `config/drift_config.yaml`:
- Numerical constants become configuration parameters
- Component enable/disable flags added
- Performance tuning parameters centralized
- Experimental settings isolated

This lexicon ensures precise, scientific terminology throughout the DRIFT system while maintaining clarity and consistency in implementation and research applications.