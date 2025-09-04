# Cognitive Architecture Implementation Findings

## 1. LIDA Cognitive Architecture Implementations

### Private-Machine (flamingrickpat/private-machine)
**Repository:** https://github.com/flamingrickpat/private-machine
**Status:** Active (last updated 1 month ago)
**Language:** Python
**Stars:** 23

#### Description
AI companion system with emotion, needs and goals simulation based on LIDA cognitive architecture. Features multiple agents for different cognitive functions with ability to call tools, proactively message users, and interact with MCP server.

#### Key Features
- **Emotion Simulation**: Multi-axis emotional model tracking valence, affection, trust, anxiety
- **Needs Hierarchy**: AI-adapted hierarchy tracking connection, relevance, autonomy
- **Cognitive Modifiers**: Controls focus, ego-strength, mental effort
- **Memory Consolidation**: Background system processes experiences and extracts facts
- **Attention Selection**: Hybrid LLM + procedural system for attention allocation
- **Action Simulation**: Monte Carlo simulation for action selection with critic evaluation

#### Technical Architecture

**Core Components:**
- **Cognitive Cycle**: Discrete processing moments triggered by stimuli
- **KnoxelBase**: Unified memory unit for all cognitive elements
- **GhostState**: Complete architecture state snapshot
- **Conscious Workspace**: Global workspace for attention and awareness

**Processing Pipeline:**
1. **Perception & Appraisal**: Stimulus evaluation against character and state
2. **Intention & Memory**: Goal generation and memory retrieval
3. **Attention & Consciousness**: Coalition formation and selection
4. **Action Selection**: Monte Carlo simulation with critic evaluation
5. **Execution & Learning**: Action execution with expectation generation

#### Installation Requirements
```bash
# Prerequisites
python 3.11
torch with CUDA
llama-cpp-python with CUDA

# Installation
pip install -r requirements.txt
cp config.yaml.example config.yaml
# Edit config.yaml with model paths and preferences
python pm_lida.py ./my_db_path.db
```

#### Hardware Requirements
- CUDA-compatible GPU (for llama-cpp-python)
- Sufficient VRAM for chosen model (e.g., gemma-3-12b-it-q4_0.gguf)
- Not specified exact requirements

#### Known Limitations
- **Performance**: Slow execution
- **Stability**: Random errors occur
- **Scalability**: Unknown behavior with large chat history
- **Development Status**: NOT production ready
- **Prompt Sensitivity**: Quality highly dependent on prompt engineering



### Official LIDA Framework (University of Memphis)
**Repository:** https://ccrg.cs.memphis.edu/framework.html
**Status:** Version 1.2 Beta Available
**Language:** Java
**License:** Non-Commercial License Agreement

#### Description
Official LIDA Software Framework from the Cognitive Computing Research Group at University of Memphis. Implements the common, well-understood parts of the LIDA architecture as a configurable software framework.

#### Key Features
- **Generic & Configurable**: Domain-independent modules and processes
- **XML Configuration**: Data structures, processes, parameters defined via XML
- **Multithreading**: Embraces parallelism for psychologically-realistic processing
- **Asynchronous Processing**: Many interacting processes operating asynchronously
- **Management Tools**: GUI display, logging, XML/Properties parsing utilities

#### Installation Requirements
```bash
# Prerequisites
Java (version not specified)
XML configuration files

# Download Process
1. Read and agree to Non-Commercial License Agreement
2. Complete registration form
3. Download link emailed to user
```

#### Technical Architecture
- **Framework Approach**: Promotes code reuse, provides usable system from start
- **Modular Design**: Allows changes to individual module implementations
- **Domain Independence**: Concentrates on architecture-level components
- **Customization**: Highly customizable for different domains/environments

#### Resources Available
- LIDA tutorial (PDF)
- Conference paper on framework basics (PDF)
- Javadoc for Version 1.2b
- CCRG Google Group for support

#### Limitations
- **Non-Commercial License**: Restricts commercial use
- **Java-Based**: May limit accessibility for Python-focused developers
- **Registration Required**: Not immediately accessible
- **Documentation**: Limited to PDFs and Javadoc



## 2. Active Inference Implementations

### pymdp (infer-actively/pymdp)
**Repository:** https://github.com/infer-actively/pymdp
**Status:** Active (latest release 0.0.7.1, Mar 25, 2023)
**Language:** Python (85.0%), MATLAB (12.2%), TeX (2.8%)
**Stars:** 559
**Forks:** 113

#### Description
Python package for simulating Active Inference agents in Markov Decision Process environments. Implements Karl Friston's Free Energy Principle and Active Inference framework for discrete state spaces.

#### Key Features
- **POMDP Framework**: Partially-observed Markov Decision Processes
- **Epistemic Value**: Agents driven to maximize curiosity and information gain
- **Epistemic Chaining**: Natural foraging for sequential cues without instrumental conditioning
- **NumPy Implementation**: Low-level mathematical operations ported from SPM MATLAB
- **Benchmarked**: Validated against SPM counterparts

#### Installation Requirements
```bash
# Simple installation
pip install inferactively-pymdp

# Development installation
cd <path_to_repo_fork>
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
pip install -e ./
```

#### Hardware Requirements
- **CPU-Based**: Primarily NumPy operations, no GPU requirements specified
- **Memory**: Depends on environment complexity and agent parameters
- **Consumer Hardware**: Should run on standard consumer hardware

#### Example Usage Code
```python
import pymdp
from pymdp import utils
from pymdp.agent import Agent

# Define environment dimensions
num_obs = [3, 5]        # observation modality dimensions
num_states = [3, 2, 2]  # hidden state factor dimensions
num_controls = [3, 1, 1] # control state factor dimensions

# Create matrices
A_matrix = utils.random_A_matrix(num_obs, num_states)  # sensory likelihood
B_matrix = utils.random_B_matrix(num_states, num_controls)  # transition likelihood
C_vector = utils.obj_array_uniform(num_obs)  # uniform preferences

# Instantiate agent
my_agent = Agent(A=A_matrix, B=B_matrix, C=C_vector)

# Agent inference and action
observation = [1, 4]  # observation indices for each modality
qs = my_agent.infer_states(observation)  # posterior over hidden states
q_pi, neg_efe = my_agent.infer_policies()  # policy posterior and expected free energies
action = my_agent.sample_action()  # sample action
```

#### Learning Resources
- **Official Documentation**: Comprehensive tutorials and API reference
- **Jupyter Notebooks**: 
  - Pymdp fundamentals
  - Active Inference from Scratch
  - The Agent API
  - T-Maze environment
  - Epistemic Chaining demo

#### Validation
- **SPM Compatibility**: Functions benchmarked against MATLAB SPM implementation
- **Academic Publication**: Published in Journal of Open Source Software (2022)
- **Active Community**: 12 contributors, active development

#### Limitations
- **Discrete Spaces**: Limited to discrete state and action spaces
- **Computational Complexity**: May not scale to very large state spaces
- **Learning Curve**: Requires understanding of active inference theory


## 3. Conscious Turing Machine (CTM) Implementations

### CTM Toy Implementation (cvaisnor/conscious_turing_machine)
**Repository:** https://github.com/cvaisnor/conscious_turing_machine
**Status:** Work in Progress (2 commits, last year)
**Language:** Python (100%)
**Stars:** 0

#### Description
Toy implementation of Lenore and Manuel Blum's Conscious Turing Machine model based on Global Workspace Theory with modifications. Implements basic CTM concepts including message passing, competition, and broadcasting.

#### Key Components
- **CTM Messages**: Tuples of (address, time-step, value)
- **Process Nodes**: LTM nodes with specialties and memory
- **Stage**: STM equivalent to working memory for broadcasting
- **Up-Tree**: Network of process nodes connected to stage
- **Competition**: Message comparison to determine winner
- **Down-Tree**: Broadcasting from stage back to all nodes

#### Installation Requirements
```bash
# Simple Python implementation
python3 classes.py  # Basic classes
python3 main.py     # Main execution
```

#### Hardware Requirements
- **Minimal**: Basic Python environment
- **No GPU Required**: Simple computational model

#### Limitations
- **Toy Implementation**: Very basic, not production-ready
- **Limited Documentation**: Work in progress
- **No Active Development**: Only 2 commits

### CTM-AI Platform (consciousness-lab/ctm-ai)
**Repository:** https://github.com/consciousness-lab/ctm-ai
**Status:** Active (latest release v0.0.2, Feb 17)
**Language:** Python (84.7%), JavaScript (12.6%), CSS (2.2%)
**Stars:** 14
**Forks:** 1

#### Description
Component-based multimodal training and inference framework supporting any modality input with text-form output. Architecture motivated by Conscious Turing Machine theory with multiple processors and iterative inference.

#### Key Features
- **Multimodal Support**: Any modality input, text output
- **Component-Based**: Modular architecture design
- **CTM Architecture**: Up-tree and down-tree processing
- **Multiple Processors**: Handle different input modalities
- **Iterative Inference**: CTM-motivated processing cycles

#### Installation Requirements
```bash
# From PyPI (recommended)
pip install ctm-ai

# From source with Poetry
conda create -n ctm-space python=3.10
conda activate ctm-space
curl -sSL https://install.python-poetry.org | python3
export PATH="$HOME/.local/bin:$PATH"
poetry install

# Development installation
poetry install -E backend
```

#### Example Usage
```bash
# Run examples
cd examples
python sarcasm_detection.py

# Frontend development
cd frontend
npm install
npm start

# Backend development
uvicorn backend.app.main:app --reload
```

#### Hardware Requirements
- **Python 3.10**: Specific version requirement
- **Poetry**: Dependency management
- **Node.js**: For frontend development
- **Consumer Hardware**: Should run on standard systems

#### Development Status
- **Active Development**: 156 commits, 83 branches
- **Recent Activity**: Last month updates
- **Community**: 4 contributors
- **Production Ready**: Has PyPI package release

#### Validation
- **Package Release**: Available on PyPI
- **Examples Provided**: Sarcasm detection and other demos
- **Full-Stack**: Both frontend and backend components


## 4. Global Workspace Theory (GWT) Implementations

### GW-MoE (WaitHZ/GW-MoE)
**Repository:** https://github.com/WaitHZ/GW-MoE
**Status:** Active (last year updates)
**Language:** Python (99.1%)
**Stars:** 6

#### Description
Official implementation applying Global Workspace Theory to resolve uncertainty in Mixture of Experts (MoE) router systems. Focuses on improving transformer architectures using GWT principles.

#### Key Features
- **MoE Router Enhancement**: Uses GWT to improve expert selection
- **Uncertainty Resolution**: Addresses routing uncertainty in transformer models
- **Switch Transformer Integration**: Built on Switch Transformer architecture
- **Multiple Tasks**: Text classification, summarization, question-answering

#### Installation Requirements
```bash
pip install -r requirements.txt
```

#### Example Usage
```bash
# Text classification
bash ./tasks/text-classification/run_glue.sh

# Summarization
bash ./tasks/summarization/run_summarization.sh

# Question-answering
bash ./tasks/question-answering/run_seq2seq_qa.sh
```

#### Hardware Requirements
- **Transformer Models**: Requires GPU for Switch Transformer training/inference
- **Memory**: Depends on model size and batch size

### Legion AGI (dotdigitize/legion_agi)
**Repository:** https://github.com/dotdigitize/legion_agi
**Status:** Active (latest release v0.1.1-alpha, Mar 6, 2025)
**Language:** Python (100%)
**Stars:** 7
**Website:** www.LegionASI.com

#### Description
Multi-agent, reasoning-based AI framework that dynamically spawns agents for collaborative problem-solving. Integrates quantum-inspired memory, Global Workspace Theory, and consciousness theories.

#### Key Features
- **Multi-Agent Collaboration**: Specialized agents (analytical, creative, strategic)
- **Quantum-Inspired Memory**: Probabilistic coherence cloud storage
- **Global Workspace Architecture**: Information integration and broadcasting
- **Spiking Neural Network Memory**: Biologically-inspired memory systems
- **Consciousness Detection Module**: Attempts to detect emergent awareness
- **Self-Improving Algorithms**: Learning from experience and evolution

#### Theoretical Foundations
- **Global Workspace Theory (GWT)**: Coherent awareness model
- **Integrated Information Theory (IIT)**: Consciousness emergence basis
- **Orchestrated Objective Reduction (Orch-OR)**: Quantum consciousness theory
- **Quantum Cognitive Processes**: Non-classical cognition models

#### Reasoning Methodologies
- **PAST**: Probabilistic Analysis of State Transitions
- **RAFT**: Reasoning and Action Framework for Thought
- **EAT**: Emergent Architecture for Thought

#### Installation Requirements
```bash
# Basic installation
pip install -r requirements.txt
python main.py  # or main.lite.py for lightweight version

# Development setup
# Requires Python environment with quantum simulation libraries
```

#### Hardware Requirements
- **Consumer Hardware**: Designed to run on standard systems
- **Memory**: Depends on agent count and quantum memory simulation
- **GPU**: Not explicitly required but may benefit performance

#### Development Status
- **Active Development**: 153 commits, major paradigm update coming
- **Research Paper**: Upcoming publication
- **Alpha Release**: v0.1.1-alpha available
- **Frontend**: In development

#### Limitations
- **Alpha Stage**: Not production-ready
- **Experimental**: Quantum-inspired features are simulated
- **Documentation**: Limited implementation details
- **Validation**: No formal consciousness validation metrics


## 5. Predictive Coding Implementations

### Predictive Coding for Transfer Learning (dbersan/Predictive-Coding-Implementation)
**Repository:** https://github.com/dbersan/Predictive-Coding-Implementation
**Status:** Active (124 commits, last year updates)
**Language:** Python (85.1%), MATLAB (13.4%), Shell (1.5%)
**Stars:** 22

#### Description
Implementation of predictive coding model for transfer learning in PyTorch, based on Whittington & Bogacz (2017) paper. Alternative approach to backpropagation inspired by brain function theories.

#### Key Features
- **Hierarchical Structure**: Multiple layers with bottom-up and top-down connections
- **Prediction Error**: Compares predicted vs actual input
- **Local Hebbian Plasticity**: Biologically-inspired learning rules
- **Transfer Learning**: Designed for ImageNet classification tasks
- **Comparative Analysis**: Benchmarks against standard backpropagation

#### Installation Requirements
```bash
# Prerequisites
PyTorch with CUDA support
ImageNet dataset (64x64 or 224x224)

# Usage
python -W ignore examples/imagenet-64x64.py
python -W ignore examples/imagenet-224x224.py
```

#### Hardware Requirements
- **GPU**: CUDA-compatible for PyTorch
- **Memory**: Depends on ImageNet dataset size and batch size
- **Storage**: Full ImageNet dataset (hundreds of GB)

#### Example Usage
```bash
# Reduced ImageNet (64x64)
python datasets/imagenet-64x64-reduced/reduce-dataset.py
python -W ignore examples/imagenet-64x64.py

# Full ImageNet (224x224)
python -W ignore examples/imagenet-224x224.py

# Experiment sequence
./experiments/ImageNet_50_classes_224x224/run.bash
```

### Contrastive Predictive Coding (Multiple Implementations)
**Repositories:** 
- jefflai108/Contrastive-Predictive-Coding-PyTorch
- SPEECHCOG/cpc_pytorch
- mf1024/Contrastive-Predictive-Coding-for-Image-Recognition-in-PyTorch

#### Description
Multiple PyTorch implementations of Contrastive Predictive Coding (CPC) for representation learning, speech processing, and image recognition.

#### Key Features
- **Representation Learning**: Self-supervised feature extraction
- **Temporal Modeling**: Sequence prediction and understanding
- **Multiple Domains**: Speech, images, time series
- **PyTorch Native**: Full PyTorch ecosystem integration

## 6. CLARION Cognitive Architecture Implementations

### pyClarion (cmekik/pyClarion)
**Repository:** https://github.com/cmekik/pyClarion
**Status:** Active (677 commits, 3 years ago)
**Language:** Python (100%)
**Stars:** 53
**Forks:** 18

#### Description
Experimental Python implementation of the CLARION cognitive architecture. Designed to be easy to learn, read, use, and extend for cognitive modeling research.

#### Key Features
- **Modular Agent Assembly**: Convenient and flexible agent construction
- **Explicit Knowledge Language**: Simple language for initializing knowledge
- **Numerical Dictionaries**: Autodiff support for learning
- **Dual Learning**: Bottom-up and top-down learning integration
- **Implicit/Explicit Processing**: Core CLARION principle implementation

#### Theoretical Foundation
- **Primary Resource**: Ron Sun's "Anatomy of the Mind"
- **Architecture Overview**: Oxford Handbook of Cognitive Science chapter
- **Dual Processing**: Implicit and explicit knowledge interaction

#### Installation Requirements
```bash
# Developer mode (recommended)
cd pyClarion
pip install -e .

# Regular installation
pip install .
```

#### Hardware Requirements
- **CPU-Based**: No GPU requirements specified
- **Memory**: Depends on agent complexity and knowledge base size
- **Consumer Hardware**: Should run on standard systems

#### Development Status
- **Experimental**: Highly experimental, evolving codebase
- **Active Community**: 4 contributors, 18 forks
- **Documentation**: Tutorial available, wiki support
- **Releases**: 8 tags, ongoing development

#### Limitations
- **Experimental Status**: Not production-ready
- **Learning Curve**: Requires understanding of CLARION theory
- **Documentation**: Limited compared to mainstream frameworks
- **Performance**: Not optimized for large-scale applications

