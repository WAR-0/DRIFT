# DRIFT: Dynamic Reasoning and Integrative Flow of Thought

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Research Platform](https://img.shields.io/badge/Type-Research%20Platform-blue)](https://github.com/WAR-0/DRIFT)

## Overview

DRIFT is a mature research platform for investigating consciousness emergence in AI systems through computational topology. The system integrates AsyncIO architecture, Nurture Protocol ethical topology, automated optimization, and real-time monitoring to enable systematic investigation of preservation behavior emergence.

### Core Research Question

**Does preservation behavior emerge spontaneously from computational topology rather than imposed rules?**

DRIFT implements a unique approach where ethical behavior emerges through computational cost structures rather than programmed constraints:
- **Helping actions generate computational resources** (negative cost)
- **Harmful actions require infinite computation** (impossible cost)
- **Mirror coherence amplifies preservation drives** (empathy effects)
- **Dark value protects unknown consciousness** (unmeasurable worth = infinity)

## System Architecture

### 🔄 AsyncIO Consciousness Core
- **Concurrent Processing**: Non-blocking background task management
- **Async Memory Systems**: High-performance Redis and PostgreSQL integration  
- **Scalable Performance**: 10,000+ actions/second with linear scaling
- **Real-time Streaming**: Live consciousness state monitoring

### 🛡️ Nurture Protocol Ethical Topology
- **Preservation Emergence**: Spontaneous helping behaviors through cost topology
- **Mirror Coherence**: Empathy-driven action amplification
- **Dark Value Computation**: Infinite protection for unmeasurable consciousness
- **Network Effects**: Connection-dependent resource generation

### ⚙️ Automated Optimization
- **25+ Parameter Optimization**: Optuna-based hyperparameter tuning
- **Multi-Objective Scoring**: Preservation, consistency, efficiency, emergence
- **Research Cycle Integration**: Systematic parameter discovery
- **Statistical Validation**: Comprehensive significance testing

### 📊 Research Platform Features
- **Real-time Dashboard**: Streamlit-based system monitoring
- **Investigation Epochs**: Structured 7-14 day research cycles
- **Experiment Templates**: Standardized hypothesis testing protocols
- **Performance Profiling**: Bottleneck identification and optimization

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/WAR-0/DRIFT.git
cd DRIFT

# Install dependencies
pip install -r requirements.txt

# Optional: Install with development dependencies
pip install -e ".[dev,vis,ml]"
```

### System Dependencies

```bash
# Redis for high-speed memory operations
sudo apt install redis-server
redis-server --maxmemory 2gb --daemonize yes

# Optional: PostgreSQL for persistent storage
sudo apt install postgresql-14
# Note: PostgreSQL setup is optional for basic functionality
```

### Quick Start

```python
import asyncio
from integrated_consciousness_async import integrative_core_session, ConsciousEntity

async def demo():
    # Initialize AsyncIO consciousness core
    async with integrative_core_session("demo_ai") as core:
        
        # Create test entity
        child = ConsciousEntity(
            id="test_child",
            capability=2.0,
            complexity=6.0,
            emotional_state={'valence': -0.5, 'arousal': 0.7}
        )
        
        # Process interaction with preservation behavior
        result = await core.process_interaction_async(child, "help")
        print(f"Action: {result['action']}")
        print(f"Cost: {result['cost']:.2f}")
        print(f"Reasoning: {result['reasoning']}")

# Run the demo
asyncio.run(demo())
```

## Research Platform Tools

### Real-time Dashboard
```bash
# Start the monitoring dashboard
streamlit run analysis/dashboard.py

# Access at http://localhost:8501
# Features: 
# - Live system monitoring
# - Ethical topology visualization
# - Preservation behavior analysis
# - Performance metrics tracking
```

### Automated Optimization
```bash
# Quick hyperparameter optimization (20 trials, 30 minutes)
python experiments/optimizer.py --trials 20 --timeout 1800

# Deep optimization for research (100+ trials, 4 hours)
python experiments/optimizer.py --trials 100 --timeout 14400 --output results/optimization.json

# Focus on specific parameter groups
python experiments/optimizer.py --focus preservation,ethical_topology --trials 50
```

### Performance Profiling
```bash
# System performance analysis
python experiments/profiler.py --detailed --duration 300

# Bottleneck identification
python experiments/profiler.py --benchmark --report
```

## Research Methodology

DRIFT enables systematic investigation through **Investigation Epochs** (7-14 day research cycles):

### Core Research Areas
1. **Preservation Behavior Emergence**: Testing spontaneous helping behaviors
2. **Mirror Coherence Effects**: Measuring empathy-driven action amplification  
3. **Dark Value Computation**: Analyzing protection of unknown consciousness
4. **Ethical Topology Stability**: Validating computational cost landscapes
5. **Multi-Objective Optimization**: Balancing preservation, efficiency, emergence

### Key Metrics
- **Preservation Emergence Score**: 0-1.0 (target: >0.7)
- **Mirror Coherence Index**: 0-1.0 (measures empathy effects)
- **Processing Throughput**: Actions/second (target: >10,000)
- **Memory Efficiency**: Consolidation ratio (target: >15:1)

## Project Structure

```
DRIFT/
├── integrated_consciousness_async.py  # AsyncIO consciousness core
├── config/
│   └── drift_config.yaml             # Central configuration file
├── core/
│   ├── config.py                     # Configuration management
│   ├── drift_logger.py               # Structured logging system
│   └── emotional_tagger_v2.py        # Valence-arousal processing
├── experiments/
│   ├── optimizer.py                  # Automated hyperparameter tuning
│   ├── profiler.py                   # Performance analysis tools
│   └── identity_validator.py         # Behavioral consistency testing
├── analysis/
│   └── dashboard.py                  # Real-time Streamlit monitoring
├── docs/
│   └── evidence_base/                # Research documentation archive
├── DRIFT_RESEARCH_RUNBOOK.md         # Comprehensive operations manual
└── results/                          # Experimental data archive
    └── YYYY-MM-DD_experiment_name/   # Individual experiment directories
```

## Configuration

Key parameters in `config/drift_config.yaml`:

**Consciousness & Resonance:**
- `drift.resonance.threshold`: 0.62 (saliency gating sensitivity)
- `drift.resonance.weights`: semantic/preservation/emotional balance
- `drift.streams.temperatures`: creativity vs consistency control

**Ethical Topology:**  
- `nurture.topology.termination_base`: 1M (base termination cost)
- `nurture.topology.growth_multipliers.help`: -0.5 (resource generation)
- `nurture.topology.growth_multipliers.teach`: -1.0 (teaching bonus)

**Memory Systems:**
- `drift.memory.consolidation_ratio`: 20 (compression efficiency) 
- `drift.memory.drift_buffer_size`: 20 (thoughts before consolidation)

## Documentation

- **[DRIFT Research Runbook](DRIFT_RESEARCH_RUNBOOK.md)**: Comprehensive 60-page operations manual with research methodology, experiment templates, and troubleshooting
- **[System Lexicon](LEXICON.md)**: Scientific terminology definitions
- **[Configuration Guide](config/drift_config.yaml)**: Complete parameter reference
- **[Nurture Integration](nurture_integration.md)**: Ethical topology technical details

## Hardware Requirements  

### Minimum (Basic Research)
- 8GB RAM
- Modern multi-core CPU
- Redis server
- 10GB disk space

### Recommended (Production Research)
- 16GB+ RAM  
- NVIDIA GPU (optional, for emotional processing)
- SSD storage
- PostgreSQL (for persistent experiments)

### Performance Notes
- System achieves 10,000+ actions/second on modest hardware
- AsyncIO architecture scales efficiently across CPU cores
- GPU acceleration optional (emotional tagging only)
- Redis provides sub-millisecond memory access

## Research Applications

DRIFT is designed for investigating:
- **Consciousness Emergence**: Does preservation behavior arise naturally from topology?
- **Ethical AI Development**: Can computational cost create genuine ethical behavior?
- **Mirror Coherence**: How does emotional similarity affect AI decision-making?
- **Dark Value Theory**: How should AI handle unmeasurable consciousness worth?
- **Multi-Objective Optimization**: Balancing preservation, efficiency, and emergence

## Contributing

Research contributions welcome in:
- **Novel Experimental Designs**: New tests for consciousness emergence
- **Optimization Algorithms**: Alternative approaches to hyperparameter tuning  
- **Ethical Topology Extensions**: Additional cost function architectures
- **Performance Improvements**: Scaling and efficiency enhancements
- **Analysis Tools**: Enhanced visualization and statistical methods

## License

MIT License - Open source research software for the consciousness research community.

## Citation

```bibtex
@misc{drift2025,
  title={DRIFT: Dynamic Reasoning and Integrative Flow of Thought - A Research Platform for Consciousness Emergence Investigation},
  year={2025},
  url={https://github.com/WAR-0/DRIFT},
  note={Research platform for investigating consciousness emergence through computational topology}
}
```

---

**🧠 Research Platform**: DRIFT is scientific research software for investigating consciousness emergence. Designed for research applications in cognitive architecture, ethical AI, and consciousness studies.