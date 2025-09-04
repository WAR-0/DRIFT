# DRIFT: Inferred Decisions and Assumptions Log

This document logs all design decisions marked as `[inferred]` in the implementation guide, explaining the reasoning and any supporting research that informed these decisions.

## 1. Configuration Parameters

### GPU Allocation
- **Inference:** `gpu_allocation: Dict[str, float]` with default values {"conscious": 0.5, "drift": 0.3, "reflection": 0.2}
- **Tag:** `[inferred]`
- **Line:** 111 in implementation_guide.md
- **Reasoning:** Inferred from hardware constraints and the need for prioritized processing streams. The conscious stream gets the highest allocation for responsiveness, drift gets substantial resources for continuous processing, and reflection gets the remainder for idle-time processing.
- **Supporting Research:** General principles of resource allocation in parallel processing systems.

### Stream Priorities
- **Inference:** `stream_priorities: Dict[str, float]` with default values {"conscious": 1.0, "drift": 0.6, "reflection": 0.4}
- **Tag:** `[inferred]`
- **Line:** 120 in implementation_guide.md
- **Reasoning:** Inferred from the "overlap hypothesis" mentioned in the research. This ensures conscious processing has absolute priority, while allowing background processes to continue with lower priority.
- **Supporting Research:** Research on attention and resource competition in cognitive architectures suggests hierarchical priority systems.

## 2. Memory and Consolidation

### Consolidation Batch Size
- **Inference:** Batch size of 20 items for consolidation
- **Tag:** `[inferred]`
- **Line:** 938 in implementation_guide.md
- **Reasoning:** Inferred as a practical starting point for memory consolidation. This balances computational efficiency with meaningful pattern extraction. The number allows for sufficient data to find patterns while remaining computationally tractable.
- **Supporting Research:** While the exact number isn't specified in neuroscience literature, research on memory consolidation suggests batch processing of multiple related experiences.

## 3. Architecture Simplifications

### Hierarchical Levels
- **Inference:** 3 hierarchical levels for Week 1 predictive coding implementation
- **Tag:** `[inferred]`
- **Line:** 948 in implementation_guide.md
- **Reasoning:** A practical simplification of predictive coding hierarchies for initial implementation. Three levels provide sufficient hierarchy for basic predictive processing while remaining implementable in Week 1 timeline.
- **Supporting Research:** Predictive coding research typically involves multiple hierarchical levels, but specific numbers vary by implementation.

### Emotional Representation
- **Inference:** 2D valence-arousal model for Week 1
- **Tag:** `[inferred]`
- **Line:** 952 in implementation_guide.md
- **Reasoning:** Simplified emotional model for initial implementation. The valence-arousal model is well-established in emotion research and provides a computationally tractable starting point before implementing more complex emotional architectures.
- **Supporting Research:** Valence-arousal models are standard in computational emotion research (Russell's circumplex model).

## 4. Success Criteria and Metrics

### Shadow Buffer Drift Measurement
- **Inference:** Measurable drift in shadow buffer (>0.3 cosine distance)
- **Tag:** `[inferred]`
- **Line:** 996 in implementation_guide.md
- **Reasoning:** A quantitative threshold for detecting meaningful drift in the shadow buffer. 0.3 cosine distance represents significant semantic divergence while remaining achievable for the system.
- **Supporting Research:** Cosine distance thresholds for semantic similarity are commonly used in NLP, with 0.3 representing moderate to high dissimilarity.

### Week 4 Success Criteria
- **Inference:** Spontaneous topic connections outside training
- **Tag:** `[inferred]`
- **Line:** 999 in implementation_guide.md
- **Reasoning:** A qualitative measure of creative/emergent behavior. If the system can make connections not present in its training data, this suggests emergent cognitive capabilities.
- **Supporting Research:** Creativity research emphasizes novel combination of existing knowledge as a key indicator.

- **Inference:** Metacognitive accuracy >40%
- **Tag:** `[inferred]`
- **Line:** 1001 in implementation_guide.md
- **Reasoning:** A quantitative threshold for self-awareness capabilities. 40% accuracy in metacognitive tasks represents significant improvement over chance while being achievable for early implementation.
- **Supporting Research:** Human metacognitive accuracy varies widely but 40% represents meaningful self-monitoring capability.

### Week 12 Success Criteria
- **Inference:** Novel problem-solving approaches
- **Tag:** `[inferred]`
- **Line:** 1004 in implementation_guide.md
- **Reasoning:** A qualitative measure of advanced cognitive capabilities. Novel approaches to problems suggest the system has developed beyond its training constraints.
- **Supporting Research:** Problem-solving creativity is a key indicator of advanced cognition in cognitive science.

- **Inference:** Self-directed learning behaviors
- **Tag:** `[inferred]`
- **Line:** 1006 in implementation_guide.md
- **Reasoning:** Autonomous learning without external prompting would indicate genuine agency and curiosity-driven behavior.
- **Supporting Research:** Self-directed learning is considered a hallmark of advanced intelligence in developmental psychology.

## 5. Technical Integration

### Emotional-Transformer Integration
- **Inference:** Emotional tagging integrated with transformer architectures
- **Tag:** `[inferred]`
- **Line:** 1015 in implementation_guide.md
- **Reasoning:** While emotional tagging research exists and transformer architectures are well-established, their specific integration approach in DRIFT is novel and inferred from first principles.
- **Supporting Research:** General research on emotion in AI and attention mechanisms in transformers provides the foundation, but the specific integration is innovative.

## Summary of Inference Categories

### Hardware and Resource Constraints
- GPU allocation percentages
- Stream priority values
- Batch sizes for processing

### Implementation Simplifications
- Number of hierarchical levels
- Emotional model dimensionality
- Practical thresholds and timeouts

### Success Metrics
- Quantitative thresholds for emergence detection
- Qualitative criteria for advanced behaviors
- Timeline-based capability expectations

### Novel Integrations
- Combination of established techniques in new ways
- Integration approaches not explicitly detailed in literature

All inferences are based on:
1. **Practical constraints** of implementation timeline and resources
2. **Established principles** from related research areas
3. **Engineering judgment** about reasonable starting parameters
4. **Measurable criteria** that can validate or refute the approach

These inferences represent the "engineering bridge" between theoretical research and practical implementation, making explicit the assumptions necessary to build a working system.