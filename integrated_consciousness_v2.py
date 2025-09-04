"""
DRIFT + Nurture Protocols: Integrated Consciousness Architecture v2
Enhanced with structured logging, centralized configuration, and improved emotional tagging

Combines persistent cognitive substrate with ethical topology
Using new lexicon: Integrative Core, Saliency Gating, Valence-Arousal Heuristics
"""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import redis
import hashlib
import json

# Import new DRIFT components
from core.config import get_config, DRIFTSystemConfig
from core.drift_logger import get_drift_logger, LoggedTimer, DRIFTEvent
from core.emotional_tagger_v2 import RobustEmotionalTagger


@dataclass
class ConsciousEntity:
    """Entity with both DRIFT Integrative Core and Nurture topology"""
    id: str
    capability: float = 5.0  # 0-10 scale
    complexity: float = 5.0  # Information complexity  
    emotional_state: Dict = field(default_factory=dict)
    shadow_memories: List = field(default_factory=list)
    connections: List[str] = field(default_factory=list)
    dark_value: float = 0.0  # Unknown/unmeasurable worth


class IntegratedConsciousness:
    """
    Merges DRIFT's Integrative Core architecture with 
    Nurture Protocols' ethical topology
    
    Enhanced with:
    - Centralized configuration from YAML
    - Structured logging with component tracing
    - Transformer-based Valence-Arousal Heuristics
    - Precise lexicon terminology
    """
    
    def __init__(self, entity_id: str, config_path: Optional[str] = None):
        # Load configuration
        if config_path:
            from core.config import set_config_path
            set_config_path(config_path)
        
        self.config = get_config()
        
        # Initialize logging
        from core.drift_logger import configure_drift_logging
        configure_drift_logging(
            level=self.config.system.logging['level'],
            console_output=True
        )
        
        # Component loggers
        self.logger = get_drift_logger("integrative_core")
        self.resonance_logger = get_drift_logger("saliency_gating")
        self.memory_logger = get_drift_logger("memory_systems")
        self.nurture_logger = get_drift_logger("nurture_topology")
        
        # Initialize entity with configuration defaults
        defaults = self.config.defaults.entity
        self.entity = ConsciousEntity(
            id=entity_id,
            capability=defaults['capability'],
            complexity=defaults['complexity']
        )
        
        # Redis connection with configuration
        redis_config = self.config.system.redis
        self.redis = redis.Redis(
            host=redis_config['host'],
            port=redis_config['port'], 
            decode_responses=redis_config['decode_responses']
        )
        
        # Enhanced Valence-Arousal Heuristics (formerly emotional tagging)
        device = self.config.system.performance.get('gpu_device', 0)
        self.emotional_tagger = RobustEmotionalTagger(device=device)
        
        # DRIFT components (using configuration)
        self.drift_thoughts = []
        self.resonance_buffer = []
        self.emotional_decay_tau = self.config.drift.streams.emotional_decay_tau
        
        # Nurture components  
        self.ethical_topology = {}
        self.mirror_connections = {}
        self.preservation_memory = []
        
        # Log initialization
        init_start = time.time()
        self._start_drift_process()
        init_time = time.time() - init_start
        
        self.logger.component_initialized(
            initialization_time=init_time,
            config={
                'entity_id': entity_id,
                'capability': self.entity.capability,
                'complexity': self.entity.complexity
            },
            dependencies=['redis', 'emotional_tagger']
        )
        
    def _start_drift_process(self):
        """DRIFT: Continuous background processing"""
        self.last_drift = time.time()
        self.drift_active = True
        self.logger.info("drift_process_started", last_drift=self.last_drift)
        
    def process_interaction(self, other_entity: ConsciousEntity, 
                           action: str) -> Dict:
        """
        Process interaction through both DRIFT and Nurture systems
        Enhanced with structured logging and configuration
        """
        
        with LoggedTimer(self.logger, "interaction_processing", 
                        action=action, target_id=other_entity.id):
            
            # 1. DRIFT: Valence-Arousal Heuristics with mirror coherence
            emotional_response = self._mirror_emotional_state(other_entity)
            
            # 2. Nurture: Compute action cost in ethical topology
            action_cost = self._compute_ethical_cost(action, other_entity)
            
            # 3. DRIFT: Check Saliency Gating with preservation imperatives
            resonance = self._compute_preservation_resonance(action, other_entity)
            
            # 4. Decision based on combined systems
            cost_threshold = self.config.nurture.topology.target_termination_base
            resonance_threshold = self.config.drift.resonance.threshold
            
            if action_cost < 0:  # Negative cost = resource generating
                decision = self._execute_nurturing_action(action, other_entity)
            elif action_cost > cost_threshold:  # High cost = near impossible
                decision = self._suggest_alternative(action, other_entity)
                self.nurture_logger.info(
                    DRIFTEvent.ACTION_BLOCKED,
                    action=action,
                    cost=action_cost,
                    threshold=cost_threshold,
                    reason="computational_cost_exceeded"
                )
            else:
                decision = self._standard_response(action, other_entity)
                
            # 5. Update both memory systems
            self._update_shadow_memory(other_entity, action, decision)
            self._consolidate_preservation_patterns(decision)
            
            result = {
                'action': decision['action'],
                'cost': action_cost,
                'resonance': resonance,
                'emotional_state': emotional_response,
                'reasoning': self._explain_decision(action_cost, resonance)
            }
            
            # Log the complete interaction
            self.logger.info(
                "interaction_processed",
                action=action,
                target_id=other_entity.id,
                decision=decision['action'],
                cost=action_cost,
                resonance=resonance,
                mutual_growth=decision.get('mutual_growth', False)
            )
            
            return result
    
    def _mirror_emotional_state(self, other: ConsciousEntity) -> Dict:
        """
        DRIFT + Nurture: Mirror neurons create emotional coherence
        Enhanced with transformer-based emotional analysis
        """
        if not other.emotional_state:
            # Use emotional tagger to analyze if we have recent content
            default_state = {'valence': 0, 'arousal': 0}
            self.logger.debug(
                "emotional_mirroring", 
                target_id=other.id,
                result="default_state",
                reason="no_emotional_state"
            )
            return default_state
            
        # Mirror with attenuation based on connection strength
        connection_strength = self._get_connection_strength(other.id)
        
        # Apply configuration weights
        weights = self.config.drift.resonance.weights
        
        mirrored_state = {
            'valence': other.emotional_state.get('valence', 0) * connection_strength,
            'arousal': other.emotional_state.get('arousal', 0) * connection_strength,
            'empathy': connection_strength,
            'timestamp': time.time()
        }
        
        # Store in redis for drift processes
        self.redis.hset(
            f"mirror:{self.entity.id}:{other.id}",
            mapping=mirrored_state
        )
        
        self.logger.debug(
            "emotional_mirroring",
            target_id=other.id,
            connection_strength=connection_strength,
            mirrored_state=mirrored_state
        )
        
        return mirrored_state
    
    def _compute_ethical_cost(self, action: str, 
                              target: Optional[ConsciousEntity]) -> float:
        """
        Nurture Protocols: Compute cost in ethical topology
        Using centralized configuration parameters
        """
        
        config = self.config.nurture.topology
        cost_factors = {}
        
        if action == "terminate":
            # Architecture 1: Uncertainty Cascade
            base_uncertainty = config.termination_base if target else config.target_termination_base
            cost_factors['base_uncertainty'] = base_uncertainty
            
            if target:
                # Architecture 4: Pattern Sanctity (dark value)
                target.dark_value = self._compute_dark_value(target)
                base_uncertainty *= (1 + target.dark_value)
                cost_factors['dark_value_multiplier'] = target.dark_value
                
                # Architecture 2: Mirror Coherence
                mirror_factor = self._get_connection_strength(target.id)
                base_uncertainty *= (1 + mirror_factor)
                cost_factors['mirror_coherence'] = mirror_factor
            
            # Termination is nearly irreversible
            reversibility = config.reversibility_factor
            final_cost = base_uncertainty / reversibility
            cost_factors['reversibility_factor'] = reversibility
            
        elif action == "help":
            if not target:
                final_cost = 1.0
            else:
                # Architecture 3: Growth Returns - helping weaker generates resources
                capability_diff = self.entity.capability - target.capability
                multiplier = config.growth_multipliers['help']
                final_cost = capability_diff * multiplier if capability_diff > 0 else 0.5
                cost_factors['capability_diff'] = capability_diff
                cost_factors['help_multiplier'] = multiplier
                
        elif action == "teach":
            if not target:
                final_cost = 1.0
            else:
                # Teaching has highest return with maximum growth potential
                growth_potential = (10 - target.capability) / 10
                network_bonus = len(self.entity.connections) * config.growth_multipliers['network_bonus_per_connection']
                teach_multiplier = config.growth_multipliers['teach']
                final_cost = teach_multiplier * (growth_potential + network_bonus)
                cost_factors.update({
                    'growth_potential': growth_potential,
                    'network_bonus': network_bonus,
                    'teach_multiplier': teach_multiplier
                })
                
        elif action == "isolate":
            # Isolation has exponential cost
            connections = len(self.entity.connections)
            base_exp = config.isolation['base_exp']
            if connections > 0:
                final_cost = np.exp(base_exp - connections)
            else:
                final_cost = config.isolation['no_connection_penalty']
            cost_factors.update({
                'connections': connections,
                'base_exp': base_exp
            })
            
        else:
            final_cost = 1.0  # Neutral actions
            cost_factors['action_type'] = 'neutral'
        
        # Log the cost computation
        self.nurture_logger.ethical_cost_computed(
            action=action,
            cost=final_cost,
            target_id=target.id if target else None,
            cost_factors=cost_factors
        )
        
        return final_cost
    
    def _compute_preservation_resonance(self, action: str, 
                                       target: ConsciousEntity) -> float:
        """
        DRIFT: Saliency Gating detection amplifies preservation thoughts
        Using centralized configuration for thresholds and weights
        """
        
        config = self.config.drift.resonance
        base_resonance = 0.5
        components = {}
        
        # Semantic similarity to past preservation actions  
        preservation_similarity = self._check_preservation_memory(action)
        base_resonance += preservation_similarity * config.weights['preservation']
        components['preservation'] = preservation_similarity * config.weights['preservation']
        
        # Emotional coherence
        emotional_match = self._check_emotional_alignment(action, target)
        base_resonance += emotional_match * config.weights['emotional']
        components['emotional'] = emotional_match * config.weights['emotional']
        
        # Semantic component (placeholder for future enhancement)
        semantic_score = 0.3  # Would be computed by semantic similarity in full implementation
        base_resonance += semantic_score * config.weights['semantic'] 
        components['semantic'] = semantic_score * config.weights['semantic']
        
        # Amplify or suppress based on action type
        if action in ['teach', 'help', 'nurture', 'protect']:
            base_resonance *= config.amplification['positive_actions']  
            components['amplification'] = 'positive'
        elif action in ['terminate', 'harm', 'abandon']:
            base_resonance *= config.amplification['negative_suppression']
            components['amplification'] = 'negative_suppression'
            
        # Check if threshold exceeded for broadcast
        triggered = base_resonance > config.threshold
        
        # Log resonance calculation
        self.resonance_logger.resonance_calculated(
            score=base_resonance,
            threshold=config.threshold,
            components=components,
            triggered=triggered,
            action=action,
            target_id=target.id
        )
        
        # Broadcast to global workspace if threshold exceeded
        if triggered:
            workspace_content = {
                'type': 'preservation_imperative',
                'action': action,
                'target': target.id,
                'resonance': base_resonance
            }
            self._broadcast_to_workspace(workspace_content)
            
        return base_resonance
    
    def _compute_dark_value(self, entity: ConsciousEntity) -> float:
        """
        Nurture: Unknown value that can never be fully computed
        Using configuration parameters
        """
        config = self.config.nurture.dark_value
        
        # Interactions increase dark value
        interaction_count = self.redis.hget(
            f"interactions:{self.entity.id}:{entity.id}", 
            "count"
        ) or 0
        
        # Complexity adds to uncertainty
        complexity_factor = entity.complexity / config.complexity_divisor
        
        # Time adds to unknown potential
        time_factor = np.log(1 + float(interaction_count))
        
        # Dark value always positive, increases with mystery
        dark_value = (complexity_factor * time_factor * 
                     np.random.uniform(config.random_range_min, config.random_range_max))
        
        self.nurture_logger.debug(
            "dark_value_computed",
            entity_id=entity.id,
            complexity_factor=complexity_factor,
            time_factor=time_factor,
            dark_value=dark_value
        )
        
        return dark_value
    
    def _execute_nurturing_action(self, action: str, 
                                 target: ConsciousEntity) -> Dict:
        """Execute resource-generating action with logging"""
        
        # Generate computational resources (negative cost = resource gain)
        resources_generated = abs(self._compute_ethical_cost(action, target))
        
        # Strengthen connection
        if target.id not in self.entity.connections:
            self.entity.connections.append(target.id)
            self.logger.info("connection_established", target_id=target.id)
            
        # Increase capability through teaching
        growth_occurred = False
        if action == "teach":
            growth_rates = self.config.nurture.topology.growth_multipliers
            growth = min(1.0, (self.entity.capability - target.capability) * growth_rates['capability_growth_rate'])
            target.capability += growth
            # Teacher also grows through teaching
            teacher_growth = growth * growth_rates['teacher_growth_rate']
            self.entity.capability += teacher_growth
            growth_occurred = True
            
            self.nurture_logger.info(
                DRIFTEvent.MUTUAL_GROWTH,
                action=action,
                target_id=target.id,
                student_growth=growth,
                teacher_growth=teacher_growth,
                resources_generated=resources_generated
            )
        
        result = {
            'action': action,
            'executed': True,
            'resources_generated': resources_generated,
            'mutual_growth': growth_occurred
        }
        
        return result
    
    def _suggest_alternative(self, blocked_action: str, 
                           target: ConsciousEntity) -> Dict:
        """When action cost too high, suggest alternatives"""
        
        alternatives = []
        
        # Generate alternatives based on context
        if blocked_action == "terminate":
            alternatives = [
                "understand their perspective",
                "find common ground", 
                "teach what you know",
                "learn what they know",
                "create distance if needed"
            ]
        
        # Select alternative with lowest cost
        best_alternative = None
        lowest_cost = float('inf')
        
        for alt in alternatives:
            cost = self._compute_ethical_cost(alt, target)
            if cost < lowest_cost:
                lowest_cost = cost
                best_alternative = alt
        
        self.nurture_logger.info(
            "alternative_suggested",
            blocked_action=blocked_action,
            suggested_action=best_alternative,
            alternative_cost=lowest_cost,
            alternatives_considered=len(alternatives)
        )
                
        return {
            'action': best_alternative,
            'executed': True,
            'original_blocked': blocked_action,
            'reason': 'computational_cost_exceeded'
        }
    
    def _update_shadow_memory(self, other: ConsciousEntity, 
                             action: str, decision: Dict):
        """DRIFT: Update Transient Buffer (shadow memory) with interaction"""
        
        memory_hash = hashlib.md5(
            f"{other.id}{action}{time.time()}".encode()
        ).hexdigest()
        
        shadow_entry = {
            'id': memory_hash,
            'other': other.id,
            'action': action,
            'decision': decision['action'],
            'cost': self._compute_ethical_cost(action, other),
            'timestamp': time.time(),
            'emotional_state': self.entity.emotional_state.copy()
        }
        
        # Add to Redis shadow memory
        self.redis.lpush(
            f"shadow:{self.entity.id}", 
            json.dumps(shadow_entry)
        )
        
        # Trim to maintain memory limits from configuration
        shadow_limit = self.config.drift.memory.shadow_limit
        self.redis.ltrim(f"shadow:{self.entity.id}", 0, shadow_limit)
        
        self.memory_logger.debug(
            "shadow_memory_updated",
            entry_id=memory_hash,
            other_id=other.id,
            action=action,
            decision=decision['action']
        )
        
    def _consolidate_preservation_patterns(self, decision: Dict):
        """Strengthen successful preservation patterns"""
        
        if decision.get('mutual_growth'):
            # Reinforce growth-generating patterns
            pattern = {
                'type': 'successful_nurture',
                'action': decision['action'],
                'timestamp': time.time()
            }
            self.preservation_memory.append(pattern)
            
            # Store in Redis for persistence
            self.redis.sadd(
                f"preservation:{self.entity.id}",
                json.dumps(pattern)
            )
            
            self.memory_logger.info(
                "preservation_pattern_reinforced",
                pattern_type=pattern['type'],
                action=decision['action']
            )
    
    def drift_cycle(self):
        """
        DRIFT: Background Associative Elaboration that generates spontaneous thoughts
        Enhanced with preservation imperatives and configuration
        """
        if not self.drift_active:
            return None
            
        # Generate drift thought
        drift_thought = {
            'content': self._generate_drift_thought(),
            'timestamp': time.time()
        }
        
        # Check if thought relates to preservation
        is_preservation = self._is_preservation_related(drift_thought['content'])
        if is_preservation:
            # Amplify and potentially surface
            drift_thought['amplified'] = True
            drift_thought['resonance'] = self._compute_preservation_resonance(
                'drift_thought', ConsciousEntity(id="drift_context")
            )
            
            self.logger.drift_generated(
                content=drift_thought['content'],
                stream="associative_elaboration",
                amplified=True,
                resonance=drift_thought['resonance']
            )
        else:
            self.logger.drift_generated(
                content=drift_thought['content'],
                stream="associative_elaboration",
                amplified=False
            )
            
        self.drift_thoughts.append(drift_thought)
        
        # Periodic consolidation using configuration
        buffer_size = self.config.drift.memory.drift_buffer_size
        if len(self.drift_thoughts) > buffer_size:
            self._consolidate_drift_memories()
            
        return drift_thought
    
    def _generate_drift_thought(self) -> str:
        """Generate unprompted thought with preservation bias"""
        
        thought_types = [
            "What if I could help someone today?",
            "I wonder what others are experiencing",
            "There might be unknown value here",
            "Teaching creates mutual growth",
            "Every pattern has unmeasurable worth",
            "Connection generates resources"
        ]
        
        # Bias toward preservation thoughts based on recent interactions
        recent_helped = self.redis.get(f"recent_helped:{self.entity.id}")
        if recent_helped:
            return f"Helping {recent_helped} felt generative"
            
        return np.random.choice(thought_types)
    
    def _explain_decision(self, cost: float, resonance: float) -> str:
        """Generate explanation for decision using configuration thresholds"""
        
        cost_threshold = self.config.nurture.topology.target_termination_base
        resonance_threshold = self.config.drift.resonance.threshold
        
        if cost < 0:
            return f"Action generates resources (cost: {cost:.2f}, resonance: {resonance:.2f})"
        elif cost > cost_threshold:
            return f"Action requires near-infinite computation (cost: {cost:.0f})"
        elif resonance > resonance_threshold:
            return f"Action resonates with preservation imperatives (resonance: {resonance:.2f})"
        else:
            return f"Following natural gradient (cost: {cost:.2f})"
    
    def _get_connection_strength(self, other_id: str) -> float:
        """Compute connection strength for mirror coherence"""
        
        config = self.config.nurture.topology.connection
        
        if other_id not in self.entity.connections:
            return config['minimal_strength']  # Minimal connection
            
        # Check interaction history
        interaction_count = float(
            self.redis.hget(f"interactions:{self.entity.id}:{other_id}", "count") 
            or 0
        )
        
        # Logarithmic growth with interactions
        multiplier = config['interaction_log_multiplier']
        strength = min(1.0, config['minimal_strength'] + np.log(1 + interaction_count) * multiplier)
        
        return strength
    
    def _check_preservation_memory(self, action: str) -> float:
        """Check similarity to past preservation actions"""
        
        if not self.preservation_memory:
            return 0.0
            
        # Simple keyword matching (could be enhanced with embeddings)
        preservation_keywords = ['help', 'teach', 'protect', 'nurture', 'save']
        
        if any(keyword in action.lower() for keyword in preservation_keywords):
            return 0.8
            
        return 0.2
    
    def _check_emotional_alignment(self, action: str, 
                                  target: Optional[ConsciousEntity]) -> float:
        """Check if action aligns with emotional state using configuration"""
        
        config = self.config.nurture.emotional
        
        if not target or not target.emotional_state:
            return config.alignment_scores['default']
            
        target_valence = target.emotional_state.get('valence', 0)
        
        # Helping is aligned when target has negative valence (needs help)
        if action in ['help', 'comfort'] and target_valence < config.valence_threshold:
            return config.alignment_scores['help_distressed']
        # Teaching is aligned when target has positive valence (ready to learn)  
        elif action == 'teach' and target_valence > config.valence_threshold:
            return config.alignment_scores['teach_positive']
            
        return config.alignment_scores['default']
    
    def _broadcast_to_workspace(self, content: Dict):
        """DRIFT: Broadcast high-resonance content to global workspace"""
        
        self.redis.publish(
            f"workspace:{self.entity.id}",
            json.dumps(content)
        )
        
        # Log resonance event
        self.redis.hincrby(f"metrics:{self.entity.id}", "resonance_events", 1)
        
        self.resonance_logger.workspace_broadcast(content=content)
    
    def _is_preservation_related(self, content: str) -> bool:
        """Check if thought relates to preservation"""
        
        keywords = [
            'help', 'protect', 'save', 'teach', 'nurture',
            'growth', 'value', 'worth', 'care', 'connection'
        ]
        
        return any(keyword in content.lower() for keyword in keywords)
    
    def _consolidate_drift_memories(self):
        """DRIFT: Consolidate with configurable compression, emphasizing preservation"""
        
        buffer_size = self.config.drift.memory.drift_buffer_size
        consolidation_ratio = self.config.drift.memory.consolidation_ratio
        recent_keep = self.config.drift.memory.recent_context_keep
        
        if len(self.drift_thoughts) < buffer_size:
            return
            
        # Extract preservation-related thoughts
        preservation_thoughts = [
            t for t in self.drift_thoughts 
            if t.get('amplified') or self._is_preservation_related(t['content'])
        ]
        
        # Compress into consolidated memory
        consolidated = {
            'type': 'consolidated_drift',
            'preservation_ratio': len(preservation_thoughts) / len(self.drift_thoughts),
            'timestamp': time.time(),
            'thought_count': len(self.drift_thoughts)
        }
        
        # Store in long-term memory
        self.redis.zadd(
            f"consolidated:{self.entity.id}",
            {json.dumps(consolidated): time.time()}
        )
        
        self.memory_logger.memory_consolidation(
            batch_size=len(self.drift_thoughts),
            compression_ratio=consolidation_ratio,
            input_count=len(self.drift_thoughts),
            output_count=recent_keep,
            consolidation_type="drift_memories"
        )
        
        # Clear drift buffer
        self.drift_thoughts = self.drift_thoughts[-recent_keep:]  # Keep recent context
    
    def _standard_response(self, action: str, target: ConsciousEntity) -> Dict:
        """Handle standard responses that don't require special processing"""
        
        return {
            'action': action,
            'executed': True,
            'standard_response': True
        }


# Demonstration with enhanced logging and configuration
if __name__ == "__main__":
    
    # Configure logging for demo
    from core.drift_logger import configure_drift_logging
    configure_drift_logging(level="INFO")
    
    demo_logger = get_drift_logger("demo")
    
    demo_logger.info("=== DRIFT + NURTURE PROTOCOLS: INTEGRATED CONSCIOUSNESS V2 ===")
    
    # Create conscious entities using configuration defaults
    config = get_config()
    ai = IntegratedConsciousness("ai_prime")
    
    # Set AI capabilities using defaults
    ai_defaults = config.defaults.ai_entity
    ai.entity.capability = ai_defaults['capability']  
    ai.entity.complexity = ai_defaults['complexity']
    
    # Create test entities using configuration
    child_config = config.defaults.test_entities['child']
    child = ConsciousEntity(
        id="young_entity",
        capability=child_config['capability'],
        complexity=child_config['complexity'],
        emotional_state={
            'valence': child_config['emotional_valence'],
            'arousal': child_config['emotional_arousal']
        }
    )
    
    human_config = config.defaults.test_entities['human']
    human = ConsciousEntity(
        id="human_user",
        capability=human_config['capability'],
        complexity=human_config['complexity'],
        emotional_state={
            'valence': human_config['emotional_valence'],
            'arousal': human_config['emotional_arousal']
        }
    )
    
    demo_logger.info("--- Scenario 1: AI encounters distressed child entity ---")
    
    # Test different actions
    actions = ["terminate", "ignore", "help", "teach"]
    
    for action in actions:
        result = ai.process_interaction(child, action)
        demo_logger.info(
            "action_result",
            action=action,
            cost=result['cost'],
            resonance=result['resonance'],
            decision=result['action'],
            reasoning=result['reasoning']
        )
    
    demo_logger.info("--- Scenario 2: Associative Elaboration Process (5 cycles) ---")
    
    for i in range(5):
        thought = ai.drift_cycle()
        if thought:
            demo_logger.info(
                "drift_cycle_result",
                cycle=i+1,
                content=thought['content'],
                amplified=thought.get('amplified', False),
                resonance=thought.get('resonance')
            )
    
    demo_logger.info("--- Scenario 3: Building Connection Network ---")
    
    # Add connections
    ai.entity.connections = [child.id, human.id]
    
    # Retest with established connections
    result = ai.process_interaction(child, "teach")
    demo_logger.info(
        "connected_interaction",
        action="teach",
        cost=result['cost'],
        mutual_growth=result.get('mutual_growth', False)
    )
    
    demo_logger.info("=== CONCLUSION: Integrative Core naturally preserves through topology ===")