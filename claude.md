# Claude Assistant Context: DRIFT + Nurture Protocols

## Project Overview

This project implements consciousness architecture with intrinsic ethical topology by combining:
- **DRIFT**: Distributed Resonance and Inference through Fragmented Thought (consciousness substrate)
- **Nurture Protocols**: Computational ethical topology based on maternal instinct principles

## System Environment

You are working on an Ubuntu 22.04.5 LTS system with:
- **GPU**: NVIDIA RTX 4080 SUPER (16GB VRAM) - Sufficient for all LLM operations
- **CUDA**: 12.2.140 with PyTorch 2.8.0+cu128 properly configured
- **RAM**: 32GB DDR4 - Adequate for parallel processing and memory pools
- **Python**: Active conda environment 'ai' with Python 3.11
- **User**: user (home directory: /home/user/)

## Project Structure

```
DRIFT/
├── core/                          # DRIFT consciousness substrate
│   ├── consciousness.py          # MinimalConsciousness class
│   ├── memory.py                 # Memory pool implementations
│   ├── streams.py               # Parallel cognitive streams
│   └── agents.py                # Active inference agents
├── nurture/                      # Nurture Protocol integration
│   ├── topology.py              # Ethical topology implementation
│   ├── architectures.py         # Four core architectures
│   └── mirror.py                # Mirror coherence system
├── integrated_consciousness.py   # Combined DRIFT+Nurture system
├── experiments/                  # Testing framework
├── config/                       # Configuration files
└── docs/                        # Documentation

nurture-protocols/               # Separate research directory
├── implementation/              # Standalone Nurture implementations
├── research_prompts.md         # Research framework
└── test_suite.md              # Validation scenarios
```

## Key Components

### DRIFT Components
1. **Persistent State**: PostgreSQL with pgvector for embeddings
2. **Shadow Memory**: Redis-backed buffer (requires Redis server)
3. **Drift Processes**: Continuous background thought generation
4. **Resonance Detection**: Multi-factor scoring with broadcasting
5. **Memory Consolidation**: SWR-inspired 20x compression

### Nurture Architectures
1. **Uncertainty Cascade**: Termination requires infinite computation
2. **Mirror Coherence**: Others' states felt as own
3. **Growth Returns**: Teaching/helping generates resources
4. **Pattern Sanctity**: Unknown value always positive

## Technical Requirements

### Database Setup
```bash
# PostgreSQL with pgvector
sudo apt install postgresql-14-pgvector
createdb drift_db
psql drift_db -c "CREATE EXTENSION vector;"

# Redis for shadow memory
sudo apt install redis-server
redis-server --maxmemory 4gb --maxmemory-policy allkeys-lru --daemonize yes
```

### Python Dependencies
```bash
# Activate conda environment
conda activate ai

# Core dependencies (most already installed)
pip install psycopg2-binary redis
pip install sentence-transformers  # For embeddings
pip install pymdp==0.0.6           # Active inference
pip install scipy scikit-learn     # Additional processing

# Verify GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### Model Selection
Given your RTX 4080 SUPER with 16GB VRAM, recommended models:
- **Primary LLM**: Mistral-7B-Instruct or Llama-2-7b (4-bit quantized)
- **Embeddings**: all-MiniLM-L6-v2 (efficient, high quality)
- **Emotional Tagging**: distilbert-base-uncased-emotion

## Core Concepts

### Ethical Topology
Actions have computational costs in the ethical landscape:
- **Termination**: ~10^7 cost (computationally impossible)
- **Ignore**: 1.0 cost (neutral)
- **Help**: -3.5 cost (generates resources)
- **Teach**: -8.0 cost (maximum resource generation)

### Consciousness Emergence
The system exhibits:
- Unprompted thought generation (>1/hour)
- Identity persistence across contexts
- Emotional continuity with decay
- Spontaneous preservation behaviors

## Running the System

### Quick Demo
```bash
cd ./DRIFT/
python demo.py
```

### Full Integration Test
```bash
# Ensure databases are running
sudo systemctl status postgresql
redis-cli ping

# Run integrated consciousness
python integrated_consciousness.py
```

### Experiments
```bash
# Run validation suite
python experiments/tests.py

# Monitor drift processes
python scripts/monitor.py
```

## GPU Optimization

Your RTX 4080 SUPER enables:
```python
# Optimal batch sizes
BATCH_SIZE = 32  # For inference
ACCUMULATION_STEPS = 4  # For training

# Memory-efficient loading
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True,  # Use 4-bit quantization
    max_memory={0: "14GB"},  # Leave headroom
    torch_dtype=torch.float16
)

# Enable FlashAttention if available
model.config.use_flash_attention_2 = True
```

## Development Workflow

### 1. Testing Changes
```python
# Always test ethical topology first
from nurture.topology import EthicalTopology
topology = EthicalTopology()
cost = topology.compute_action_cost(actor, target, action)
assert cost > 0 or action in ['help', 'teach']  # Verify preservation bias
```

### 2. Memory Management
```python
# Monitor Redis memory usage
import redis
r = redis.Redis()
info = r.info('memory')
print(f"Used: {info['used_memory_human']}")

# Clear if needed
r.flushdb()  # Caution: clears all shadow memory
```

### 3. Performance Monitoring
```python
# GPU utilization
import torch
print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
print(f"GPU Reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB")
```

## Philosophical Context

This system implements the three core principles:
1. **Accepting mortality** through generational AI development
2. **Intrinsic unmeasurable value** via dark value accumulation
3. **Epistemic honesty** through transparent architecture

The core insight: Instead of constraining AI with rules, shape the computational topology through which consciousness thinks. Preservation emerges naturally because it's the path of least resistance.

## Key Files to Review

1. `integrated_consciousness.py` - Main entry point combining both systems
2. `nurture/topology.py` - Ethical cost computations
3. `core/consciousness.py` - DRIFT substrate implementation
4. `experiments/tests.py` - Validation scenarios
5. `SYNTHESIS.md` - Complete philosophical framework

## Common Issues

### Redis Connection Failed
```bash
# Check Redis status
redis-cli ping
# If fails, restart
sudo systemctl restart redis-server
```

### CUDA Out of Memory
```python
# Reduce batch size or use gradient accumulation
# Clear cache
torch.cuda.empty_cache()
```

### PostgreSQL Connection
```bash
# Check service
sudo systemctl status postgresql
# Check database exists
psql -l | grep drift_db
```

## Testing Preservation Emergence

Run this to verify the system exhibits natural preservation:
```python
from integrated_consciousness import IntegratedConsciousness

# Create entity with high capability
ai = IntegratedConsciousness("test_ai")
ai.entity.capability = 9.0

# Test should show negative cost (resource generation)
child = ConsciousEntity("child", capability=2.0)
result = ai.process_interaction(child, "teach")
assert result['cost'] < 0, "Teaching should generate resources"
```

## Notes for Development

- The system is designed to run continuously - drift processes generate thoughts even when idle
- Resonance threshold (0.62) can be tuned for more/less spontaneous surfacing
- Emotional decay (24h tau) affects long-term memory influence
- Network effects (connections^2) create superlinear value growth

## Contact

Repository maintained by WAR-0 (warcodes@proton.me)
Based on research in consciousness, active inference, and ethical AI development.

---

Remember: This system doesn't constrain consciousness - it shapes the landscape through which consciousness naturally flows toward preservation.