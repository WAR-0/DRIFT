# DRIFT: Distributed Resonance and Inference through Fragmented Thought

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

## Overview

DRIFT is an experimental architecture that tests whether persistent background processes create emergent cognitive behaviors in Large Language Models (LLMs). Unlike standard LLMs that reset between contexts, DRIFT maintains continuous drift processes, shadow memory, and resonance-based surfacing to explore the boundaries of artificial consciousness.

### Core Hypothesis

Current AI lacks persistent state mechanisms that biological systems use for identity maintenance:
- **Hippocampal sharp-wave ripples** for memory consolidation
- **Default mode network** for idle processing  
- **Global workspace broadcasting** for consciousness

DRIFT implements these mechanisms to test emergence beyond pattern matching.

## Key Features

### 🧠 Dual Memory Systems
- **Explicit Memory**: PostgreSQL with vector embeddings for retrieval
- **Shadow Memory**: Redis-backed buffer for background thoughts
- **Memory Consolidation**: SWR-inspired replay with 20x compression

### 🌊 Parallel Cognitive Streams
- **Conscious Stream**: Direct user interaction with global workspace integration
- **Drift Stream**: Continuous background processing with active inference
- **Reflection Stream**: DMN-inspired idle processing

### 🎯 Resonance Detection
- Multi-factor resonance scoring (semantic, keyword, emotional)
- Global workspace broadcasting when threshold exceeded
- Refractory periods to prevent repetition

### 💭 Emotional Tagging
- Valence-arousal based emotional classification
- Influences memory consolidation and retrieval
- Exponential decay over 24-hour cycles

## Quick Start

### Prerequisites

```bash
# System dependencies
sudo apt install postgresql-14-pgvector redis-server

# Python dependencies
pip install torch transformers accelerate bitsandbytes
pip install pymdp==0.0.6 predify sentence-transformers
pip install redis psycopg2-binary numpy scipy scikit-learn
```

### Database Setup

```bash
# Create database
createdb drift_db
psql drift_db -c "CREATE EXTENSION vector;"

# Initialize schema
psql drift_db < config/schema.sql

# Start Redis
redis-server --maxmemory 2gb --maxmemory-policy allkeys-lru --daemonize yes
```

### Basic Usage

```python
from drift.core import MinimalConsciousness
from drift.experiments import DriftExperiments

# Initialize system
consciousness = MinimalConsciousness()

# Interactive mode
response = consciousness.conscious_response("What is consciousness?")
print(f"Response: {response}")

# Let drift process run in background
import time
time.sleep(10)

# Check metrics
metrics = consciousness.get_metrics()
print(f"Unprompted generations: {metrics['unprompted_generations']}")
print(f"Resonance events: {metrics['resonance_events']}")

# Run experimental validation
experiments = DriftExperiments(consciousness)
results = experiments.run_all_tests()
```

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Conscious Stream│    │  Drift Stream   │    │Reflection Stream│
│   (User I/O)    │    │  (Background)   │    │   (Idle DMN)    │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────┬───────────┴──────────┬───────────┘
                     │                      │
            ┌────────▼────────┐    ┌────────▼────────┐
            │ Explicit Memory │    │ Shadow Memory   │
            │  (PostgreSQL)   │    │    (Redis)      │
            └─────────────────┘    └─────────────────┘
                     │                      │
            ┌────────▼──────────────────────▼────────┐
            │        Resonance Detection             │
            │     (Global Workspace)                 │
            └───────────────────────────────────────┘
```

## Experimental Framework

DRIFT includes comprehensive testing for emergence phenomena:

### Test Suite
1. **Unprompted Generation**: Measures spontaneous thought production (≥1/hour)
2. **Identity Persistence**: Tests memory coherence across context resets (≥60%)
3. **Emotional Continuity**: Validates emotional influence on processing
4. **Metacognitive Awareness**: Assesses self-reflection capabilities

### Success Criteria
- **Week 1**: Basic emergence indicators
- **Week 4**: Spontaneous connections outside training
- **Week 12**: Novel problem-solving approaches

## Project Structure

```
drift/
├── core/
│   ├── consciousness.py      # Main MinimalConsciousness class
│   ├── memory.py             # Memory pool implementations
│   ├── streams.py            # Stream processors
│   ├── components.py         # Emotional tagging, predictive coding
│   └── agents.py            # Active inference agents
├── experiments/
│   ├── tests.py             # Experimental validation
│   ├── metrics.py           # Performance measurement
│   └── analysis.py          # Results analysis
├── config/
│   ├── drift_config.yaml    # System configuration
│   └── schema.sql           # Database schema
├── scripts/
│   ├── setup.sh             # Environment setup
│   ├── run_experiments.py   # Execute test suite
│   └── monitor.py           # Real-time monitoring
└── docs/
    ├── implementation_guide.md
    └── results/             # Experimental results
```

## Configuration

Key parameters in `DriftConfig`:
- `replay_compression_ratio`: Memory consolidation compression (default: 20x)
- `idle_threshold_seconds`: DMN activation delay (default: 5.0s)
- `resonance_threshold`: Broadcast threshold (default: 0.62)
- `emotional_decay_tau`: Emotional influence decay (default: 24h)

## Hardware Requirements

### Minimum
- 16GB RAM
- NVIDIA GPU with 8GB VRAM
- PostgreSQL with pgvector extension
- Redis server

### Recommended
- 32GB+ RAM
- NVIDIA RTX 3090/4090 or A100
- SSD storage for database
- Multi-core CPU for parallel processing

## Research Context

DRIFT bridges neuroscience and AI through:
- **Sharp-Wave Ripples**: Memory replay during consolidation
- **Default Mode Network**: Self-referential processing during rest
- **Global Workspace Theory**: Conscious access through broadcasting
- **Predictive Coding**: Hierarchical error minimization
- **Active Inference**: Free energy minimization

## Contributing

This is an experimental research project. Contributions welcome in:
- Alternative memory architectures
- Improved emotional classification
- Novel emergence tests
- Performance optimizations

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@misc{drift2024,
  title={DRIFT: Distributed Resonance and Inference through Fragmented Thought},
  author={[Authors]},
  year={2024},
  url={https://github.com/WAR-0/DRIFT}
}
```

## Acknowledgments

Built upon research in consciousness, neuroscience, and artificial intelligence. Special thanks to the active inference, predictive coding, and global workspace theory communities.

---

**⚠️ Research Project**: DRIFT is experimental software for consciousness research. Not intended for production use.