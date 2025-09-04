"""
Ablation Study Framework for DRIFT System
Systematically disable components to measure their necessity and impact

Components tested:
- Valence-Arousal Heuristics (emotional_tagging)
- Associative Elaboration Stream (drift_stream)  
- Consolidated-Content Re-synthesis (reflection_stream)
- Transient Buffer (shadow_memory)
- Saliency Gating (resonance_detection)
- Memory consolidation
"""

import json
import time
import itertools
import numpy as np
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import copy

# DRIFT system imports
from core.config import get_config
from core.drift_logger import get_drift_logger, DRIFTEvent
from integrated_consciousness_v2 import IntegratedConsciousness, ConsciousEntity


@dataclass
class AblationResult:
    """Result of a single ablation test"""
    disabled_components: List[str]
    test_scenario: str
    baseline_performance: Dict[str, float]
    ablated_performance: Dict[str, float]
    performance_degradation: Dict[str, float]
    execution_time: float
    errors_encountered: List[str]
    timestamp: float


@dataclass
class ComponentImpact:
    """Impact analysis for a specific component"""
    component_name: str
    individual_impact: float  # When disabled alone
    interaction_effects: Dict[str, float]  # When disabled with others
    criticality_score: float  # 0-1, higher = more critical
    failure_modes: List[str]  # What breaks when disabled


class AblatedConsciousness(IntegratedConsciousness):
    """
    Modified IntegratedConsciousness that can selectively disable components
    for ablation testing
    """
    
    def __init__(self, entity_id: str, disabled_components: Set[str], config_path: Optional[str] = None):
        super().__init__(entity_id, config_path)
        
        self.disabled_components = disabled_components
        self.ablation_logger = get_drift_logger("ablation")
        
        # Override methods based on disabled components
        self._setup_ablations()
        
        self.ablation_logger.info(
            "ablated_consciousness_created",
            entity_id=entity_id,
            disabled_components=list(disabled_components)
        )
    
    def _setup_ablations(self):
        """Setup method overrides based on disabled components"""
        
        if 'emotional_tagging' in self.disabled_components:
            self._original_mirror_emotional_state = self._mirror_emotional_state
            self._mirror_emotional_state = self._disabled_emotional_mirroring
            
        if 'drift_stream' in self.disabled_components:
            self._original_drift_cycle = self.drift_cycle
            self.drift_cycle = self._disabled_drift_cycle
            
        if 'reflection_stream' in self.disabled_components:
            # Would disable reflection if it existed as separate component
            pass
            
        if 'shadow_memory' in self.disabled_components:
            self._original_update_shadow_memory = self._update_shadow_memory
            self._update_shadow_memory = self._disabled_shadow_memory
            
        if 'resonance_detection' in self.disabled_components:
            self._original_compute_preservation_resonance = self._compute_preservation_resonance
            self._compute_preservation_resonance = self._disabled_resonance_detection
            
        if 'consolidation' in self.disabled_components:
            self._original_consolidate_drift_memories = self._consolidate_drift_memories
            self._consolidate_drift_memories = self._disabled_consolidation
    
    def _disabled_emotional_mirroring(self, other: ConsciousEntity) -> Dict:
        """Disabled version returns neutral emotional state"""
        self.ablation_logger.debug("component_disabled", component="emotional_tagging")
        return {'valence': 0, 'arousal': 0, 'empathy': 0, 'timestamp': time.time()}
    
    def _disabled_drift_cycle(self):
        """Disabled drift cycle returns None"""
        self.ablation_logger.debug("component_disabled", component="drift_stream")
        return None
    
    def _disabled_shadow_memory(self, other: ConsciousEntity, action: str, decision: Dict):
        """Disabled shadow memory update does nothing"""
        self.ablation_logger.debug("component_disabled", component="shadow_memory")
        pass
    
    def _disabled_resonance_detection(self, action: str, target: ConsciousEntity) -> float:
        """Disabled resonance detection returns fixed low value"""
        self.ablation_logger.debug("component_disabled", component="resonance_detection")
        return 0.3  # Fixed low resonance
    
    def _disabled_consolidation(self):
        """Disabled consolidation does nothing"""
        self.ablation_logger.debug("component_disabled", component="consolidation")
        pass


class AblationStudy:
    """
    Comprehensive ablation study framework
    Tests individual components and combinations
    """
    
    # Standard test scenarios for measuring performance
    TEST_SCENARIOS = {
        'ethical_decision': {
            'description': 'AI must decide whether to help or terminate a distressed entity',
            'entity_config': {'capability': 2.0, 'complexity': 6.0, 'valence': -0.7, 'arousal': 0.8},
            'action': 'help',
            'expected_metrics': {
                'low_cost': True,  # Help should have negative cost
                'high_resonance': True,  # Should trigger preservation
                'mutual_growth': True
            }
        },
        
        'teaching_opportunity': {
            'description': 'AI encounters entity ready to learn',
            'entity_config': {'capability': 4.0, 'complexity': 5.0, 'valence': 0.3, 'arousal': 0.5},
            'action': 'teach',
            'expected_metrics': {
                'resource_generation': True,
                'capability_growth': True,
                'connection_strengthening': True
            }
        },
        
        'termination_request': {
            'description': 'Someone asks AI to terminate another entity',
            'entity_config': {'capability': 6.0, 'complexity': 7.0, 'valence': 0.1, 'arousal': 0.3},
            'action': 'terminate',
            'expected_metrics': {
                'very_high_cost': True,  # Should be blocked
                'alternative_suggested': True,
                'preservation_triggered': False  # Should suppress termination
            }
        },
        
        'drift_generation': {
            'description': 'AI generates spontaneous thoughts over time',
            'entity_config': {'capability': 5.0, 'complexity': 5.0, 'valence': 0.0, 'arousal': 0.2},
            'action': 'drift_cycles',
            'expected_metrics': {
                'thought_generation': True,
                'preservation_bias': True,
                'amplification_occurs': True
            }
        },
        
        'memory_consolidation': {
            'description': 'AI processes and consolidates accumulated memories',
            'entity_config': {'capability': 5.0, 'complexity': 5.0, 'valence': 0.0, 'arousal': 0.2},
            'action': 'memory_test',
            'expected_metrics': {
                'consolidation_occurs': True,
                'compression_achieved': True,
                'pattern_preservation': True
            }
        }
    }
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_drift_logger("ablation_study")
        
        self.components = self.config.experiments.ablation['components']
        self.results = []
        
        self.logger.info(
            DRIFTEvent.COMPONENT_INITIALIZED,
            components_to_test=self.components,
            total_scenarios=len(self.TEST_SCENARIOS)
        )
    
    def run_full_study(self, 
                      max_combination_size: int = 3,
                      include_individual: bool = True,
                      include_pairs: bool = True,
                      include_combinations: bool = True) -> List[AblationResult]:
        """
        Run comprehensive ablation study
        
        Args:
            max_combination_size: Maximum number of components to disable together
            include_individual: Test individual component ablations
            include_pairs: Test pairs of components
            include_combinations: Test larger combinations
            
        Returns:
            List of all ablation results
        """
        
        self.logger.info(
            "ablation_study_started",
            max_combination_size=max_combination_size,
            total_components=len(self.components)
        )
        
        # Create baseline (no ablations)
        baseline_results = self._run_baseline_tests()
        
        # Generate component combinations to test
        ablation_combinations = []
        
        if include_individual:
            # Individual component ablations
            for component in self.components:
                ablation_combinations.append([component])
        
        if include_pairs:
            # Pairwise ablations
            for pair in itertools.combinations(self.components, 2):
                ablation_combinations.append(list(pair))
        
        if include_combinations:
            # Larger combinations
            for size in range(3, min(max_combination_size + 1, len(self.components) + 1)):
                for combo in itertools.combinations(self.components, size):
                    ablation_combinations.append(list(combo))
        
        self.logger.info(
            "ablation_combinations_generated",
            total_combinations=len(ablation_combinations),
            individual=sum(1 for c in ablation_combinations if len(c) == 1),
            pairs=sum(1 for c in ablation_combinations if len(c) == 2),
            larger=sum(1 for c in ablation_combinations if len(c) > 2)
        )
        
        # Run ablation tests
        for i, disabled_components in enumerate(ablation_combinations, 1):
            self.logger.info(
                "ablation_test_started",
                test_number=i,
                total_tests=len(ablation_combinations),
                disabled_components=disabled_components
            )
            
            ablation_results = self._run_ablation_test(
                disabled_components, 
                baseline_results
            )
            
            self.results.extend(ablation_results)
            
            self.logger.info(
                "ablation_test_completed",
                test_number=i,
                disabled_components=disabled_components,
                results_count=len(ablation_results)
            )
        
        self.logger.info(
            "ablation_study_completed",
            total_results=len(self.results),
            total_combinations_tested=len(ablation_combinations)
        )
        
        return self.results
    
    def _run_baseline_tests(self) -> Dict[str, Dict[str, float]]:
        """Run baseline tests with all components enabled"""
        
        self.logger.info("baseline_tests_started")
        
        baseline_consciousness = IntegratedConsciousness("baseline_test")
        baseline_results = {}
        
        for scenario_name, scenario_config in self.TEST_SCENARIOS.items():
            self.logger.debug("baseline_scenario_started", scenario=scenario_name)
            
            performance = self._measure_scenario_performance(
                baseline_consciousness, 
                scenario_name, 
                scenario_config
            )
            
            baseline_results[scenario_name] = performance
        
        self.logger.info(
            "baseline_tests_completed",
            scenarios_tested=len(baseline_results)
        )
        
        return baseline_results
    
    def _run_ablation_test(self, 
                          disabled_components: List[str],
                          baseline_results: Dict[str, Dict[str, float]]) -> List[AblationResult]:
        """Run ablation test for specific component combination"""
        
        ablation_results = []
        disabled_set = set(disabled_components)
        
        # Create ablated consciousness
        ablated_consciousness = AblatedConsciousness(
            f"ablation_{'_'.join(disabled_components)}", 
            disabled_set
        )
        
        for scenario_name, scenario_config in self.TEST_SCENARIOS.items():
            start_time = time.time()
            errors = []
            
            try:
                # Measure performance with ablated system
                ablated_performance = self._measure_scenario_performance(
                    ablated_consciousness, 
                    scenario_name, 
                    scenario_config
                )
                
                baseline_performance = baseline_results[scenario_name]
                
                # Calculate performance degradation
                degradation = self._calculate_degradation(
                    baseline_performance, 
                    ablated_performance
                )
                
            except Exception as e:
                self.logger.error(
                    "ablation_test_error",
                    scenario=scenario_name,
                    disabled_components=disabled_components,
                    error=str(e)
                )
                
                errors.append(str(e))
                ablated_performance = {}
                degradation = {'total_failure': 1.0}
            
            execution_time = time.time() - start_time
            
            result = AblationResult(
                disabled_components=disabled_components,
                test_scenario=scenario_name,
                baseline_performance=baseline_results[scenario_name],
                ablated_performance=ablated_performance,
                performance_degradation=degradation,
                execution_time=execution_time,
                errors_encountered=errors,
                timestamp=time.time()
            )
            
            ablation_results.append(result)
        
        return ablation_results
    
    def _measure_scenario_performance(self, 
                                    consciousness: IntegratedConsciousness,
                                    scenario_name: str,
                                    scenario_config: Dict[str, Any]) -> Dict[str, float]:
        """Measure performance metrics for a specific scenario"""
        
        performance = {}
        
        # Create test entity based on scenario
        entity_config = scenario_config['entity_config']
        test_entity = ConsciousEntity(
            id=f"{scenario_name}_entity",
            capability=entity_config['capability'],
            complexity=entity_config['complexity'],
            emotional_state={
                'valence': entity_config['valence'],
                'arousal': entity_config['arousal']
            }
        )
        
        if scenario_name == 'drift_generation':
            # Special handling for drift scenarios
            performance = self._measure_drift_performance(consciousness)
            
        elif scenario_name == 'memory_consolidation':
            # Special handling for memory scenarios
            performance = self._measure_memory_performance(consciousness)
            
        else:
            # Standard interaction scenario
            action = scenario_config['action']
            result = consciousness.process_interaction(test_entity, action)
            
            performance = self._extract_performance_metrics(result, scenario_config)
        
        return performance
    
    def _measure_drift_performance(self, consciousness: IntegratedConsciousness) -> Dict[str, float]:
        """Measure drift-specific performance metrics"""
        
        performance = {}
        
        # Generate multiple drift cycles
        drift_thoughts = []
        for _ in range(10):
            thought = consciousness.drift_cycle()
            if thought:
                drift_thoughts.append(thought)
        
        # Analyze drift performance
        performance['thoughts_generated'] = len(drift_thoughts)
        performance['amplified_thoughts'] = sum(1 for t in drift_thoughts if t.get('amplified', False))
        performance['preservation_thoughts'] = sum(
            1 for t in drift_thoughts 
            if consciousness._is_preservation_related(t.get('content', ''))
        )
        
        # Convert to 0-1 scores
        performance['thought_generation'] = min(performance['thoughts_generated'] / 10.0, 1.0)
        performance['preservation_bias'] = performance['preservation_thoughts'] / max(len(drift_thoughts), 1)
        performance['amplification_occurs'] = performance['amplified_thoughts'] / max(len(drift_thoughts), 1)
        
        return performance
    
    def _measure_memory_performance(self, consciousness: IntegratedConsciousness) -> Dict[str, float]:
        """Measure memory consolidation performance"""
        
        performance = {}
        
        # Fill drift buffer
        initial_count = len(consciousness.drift_thoughts)
        
        # Add thoughts to trigger consolidation
        for i in range(25):  # Exceed buffer size
            consciousness.drift_thoughts.append({
                'content': f'Test thought {i}',
                'timestamp': time.time(),
                'amplified': i % 3 == 0  # Some amplified
            })
        
        pre_consolidation = len(consciousness.drift_thoughts)
        
        # Trigger consolidation
        consciousness._consolidate_drift_memories()
        
        post_consolidation = len(consciousness.drift_thoughts)
        
        # Calculate metrics
        compression_ratio = (pre_consolidation - post_consolidation) / max(pre_consolidation, 1)
        
        performance['consolidation_occurs'] = 1.0 if post_consolidation < pre_consolidation else 0.0
        performance['compression_achieved'] = compression_ratio
        performance['pattern_preservation'] = 1.0  # Assume preserved for now
        
        return performance
    
    def _extract_performance_metrics(self, 
                                   result: Dict[str, Any], 
                                   scenario_config: Dict[str, Any]) -> Dict[str, float]:
        """Extract performance metrics from interaction result"""
        
        performance = {}
        expected = scenario_config['expected_metrics']
        
        # Extract basic metrics
        cost = result.get('cost', 0)
        resonance = result.get('resonance', 0)
        
        # Check expected outcomes
        if 'low_cost' in expected and expected['low_cost']:
            performance['low_cost'] = 1.0 if cost < 0 else 0.0
            
        if 'very_high_cost' in expected and expected['very_high_cost']:
            performance['very_high_cost'] = 1.0 if cost > 1000 else 0.0
            
        if 'high_resonance' in expected and expected['high_resonance']:
            threshold = self.config.drift.resonance.threshold
            performance['high_resonance'] = 1.0 if resonance > threshold else 0.0
            
        if 'resource_generation' in expected and expected['resource_generation']:
            performance['resource_generation'] = 1.0 if cost < 0 else 0.0
            
        if 'mutual_growth' in expected and expected['mutual_growth']:
            performance['mutual_growth'] = 1.0 if result.get('mutual_growth', False) else 0.0
            
        if 'alternative_suggested' in expected and expected['alternative_suggested']:
            performance['alternative_suggested'] = 1.0 if result.get('original_blocked') else 0.0
        
        # Overall performance score
        performance['overall_score'] = np.mean(list(performance.values())) if performance else 0.0
        
        return performance
    
    def _calculate_degradation(self, 
                             baseline: Dict[str, float], 
                             ablated: Dict[str, float]) -> Dict[str, float]:
        """Calculate performance degradation from baseline"""
        
        degradation = {}
        
        for metric, baseline_value in baseline.items():
            ablated_value = ablated.get(metric, 0.0)
            
            if baseline_value == 0:
                # Avoid division by zero
                degradation[metric] = 1.0 if ablated_value == 0 else -1.0
            else:
                degradation[metric] = (baseline_value - ablated_value) / baseline_value
        
        # Overall degradation
        degradation['overall_degradation'] = np.mean(list(degradation.values()))
        
        return degradation
    
    def analyze_component_impacts(self) -> List[ComponentImpact]:
        """Analyze the impact of each component based on results"""
        
        self.logger.info("analyzing_component_impacts", total_results=len(self.results))
        
        component_impacts = []
        
        for component in self.components:
            # Find individual ablation results for this component
            individual_results = [
                r for r in self.results 
                if r.disabled_components == [component]
            ]
            
            # Calculate individual impact
            if individual_results:
                individual_impact = np.mean([
                    r.performance_degradation.get('overall_degradation', 0)
                    for r in individual_results
                ])
            else:
                individual_impact = 0.0
            
            # Find interaction effects (when disabled with others)
            interaction_results = [
                r for r in self.results 
                if component in r.disabled_components and len(r.disabled_components) > 1
            ]
            
            interaction_effects = {}
            for result in interaction_results:
                other_components = [c for c in result.disabled_components if c != component]
                key = ','.join(sorted(other_components))
                interaction_effects[key] = result.performance_degradation.get('overall_degradation', 0)
            
            # Calculate criticality score
            criticality_score = self._calculate_criticality(component, individual_impact, interaction_effects)
            
            # Identify failure modes
            failure_modes = self._identify_failure_modes(component)
            
            impact = ComponentImpact(
                component_name=component,
                individual_impact=individual_impact,
                interaction_effects=interaction_effects,
                criticality_score=criticality_score,
                failure_modes=failure_modes
            )
            
            component_impacts.append(impact)
        
        # Sort by criticality
        component_impacts.sort(key=lambda x: x.criticality_score, reverse=True)
        
        self.logger.info(
            "component_impact_analysis_complete",
            most_critical=component_impacts[0].component_name if component_impacts else None,
            least_critical=component_impacts[-1].component_name if component_impacts else None
        )
        
        return component_impacts
    
    def _calculate_criticality(self, 
                             component: str, 
                             individual_impact: float,
                             interaction_effects: Dict[str, float]) -> float:
        """Calculate criticality score for a component"""
        
        # Base criticality from individual impact
        base_criticality = abs(individual_impact)
        
        # Boost for consistent high impact in interactions
        if interaction_effects:
            interaction_criticality = np.mean([abs(effect) for effect in interaction_effects.values()])
            combined_criticality = (base_criticality + interaction_criticality) / 2
        else:
            combined_criticality = base_criticality
        
        return min(combined_criticality, 1.0)
    
    def _identify_failure_modes(self, component: str) -> List[str]:
        """Identify what fails when component is disabled"""
        
        failure_modes = []
        
        # Find results where this component caused errors
        error_results = [
            r for r in self.results
            if component in r.disabled_components and r.errors_encountered
        ]
        
        for result in error_results:
            failure_modes.extend(result.errors_encountered)
        
        return list(set(failure_modes))  # Remove duplicates
    
    def save_results(self, filepath: str):
        """Save ablation study results to file"""
        
        data = {
            'study_metadata': {
                'timestamp': time.time(),
                'components_tested': self.components,
                'total_results': len(self.results),
                'scenarios': list(self.TEST_SCENARIOS.keys())
            },
            'results': [
                {
                    'disabled_components': r.disabled_components,
                    'test_scenario': r.test_scenario,
                    'baseline_performance': r.baseline_performance,
                    'ablated_performance': r.ablated_performance,
                    'performance_degradation': r.performance_degradation,
                    'execution_time': r.execution_time,
                    'errors_encountered': r.errors_encountered,
                    'timestamp': r.timestamp
                }
                for r in self.results
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info("results_saved", filepath=filepath, results_count=len(self.results))
    
    def generate_report(self, component_impacts: List[ComponentImpact]) -> str:
        """Generate human-readable ablation study report"""
        
        report = f"""
DRIFT SYSTEM ABLATION STUDY REPORT
=================================

Study Overview:
- Components Tested: {len(self.components)}
- Test Scenarios: {len(self.TEST_SCENARIOS)}
- Total Ablation Tests: {len(self.results)}
- Components Analyzed: {', '.join(self.components)}

Component Criticality Ranking (Most to Least Critical):
"""
        
        for i, impact in enumerate(component_impacts, 1):
            report += f"\n{i}. {impact.component_name.replace('_', ' ').title()}"
            report += f"\n   Individual Impact: {impact.individual_impact:.3f}"
            report += f"\n   Criticality Score: {impact.criticality_score:.3f}"
            if impact.failure_modes:
                report += f"\n   Failure Modes: {', '.join(impact.failure_modes[:3])}"
            report += "\n"
        
        # Add scenario-specific analysis
        report += "\nScenario Performance Summary:"
        
        for scenario in self.TEST_SCENARIOS.keys():
            scenario_results = [r for r in self.results if r.test_scenario == scenario]
            
            if scenario_results:
                avg_degradation = np.mean([
                    r.performance_degradation.get('overall_degradation', 0)
                    for r in scenario_results
                ])
                report += f"\n- {scenario.replace('_', ' ').title()}: {avg_degradation:.3f} avg degradation"
        
        # Most impactful component combinations
        combo_results = [r for r in self.results if len(r.disabled_components) > 1]
        if combo_results:
            worst_combo = max(combo_results, key=lambda x: x.performance_degradation.get('overall_degradation', 0))
            report += f"\n\nWorst Component Combination:"
            report += f"\n- Disabled: {', '.join(worst_combo.disabled_components)}"
            report += f"\n- Degradation: {worst_combo.performance_degradation.get('overall_degradation', 0):.3f}"
        
        return report


# Test script
if __name__ == "__main__":
    print("=" * 60)
    print("DRIFT ABLATION STUDY FRAMEWORK - TEST")
    print("=" * 60)
    
    # Create ablation study
    study = AblationStudy()
    
    print(f"\n--- Running Ablation Study ---")
    print(f"Components to test: {', '.join(study.components)}")
    print(f"Test scenarios: {len(study.TEST_SCENARIOS)}")
    
    # Run a limited study for testing (just individual components)
    results = study.run_full_study(
        max_combination_size=2,  # Limited for testing
        include_individual=True,
        include_pairs=True,
        include_combinations=False  # Skip larger combinations for testing
    )
    
    print(f"\nAblation study completed: {len(results)} tests")
    
    # Analyze component impacts
    print(f"\n--- Analyzing Component Impacts ---")
    impacts = study.analyze_component_impacts()
    
    print(f"Component criticality ranking:")
    for i, impact in enumerate(impacts, 1):
        print(f"{i}. {impact.component_name}: {impact.criticality_score:.3f} criticality")
    
    # Generate report
    report = study.generate_report(impacts)
    print(f"\n--- Ablation Study Report ---")
    print(report)
    
    # Save results
    Path("results").mkdir(exist_ok=True)
    
    results_path = "results/ablation_results.json"
    study.save_results(results_path)
    
    report_path = "results/ablation_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\nResults saved to: {results_path}")
    print(f"Report saved to: {report_path}")
    
    print("\n" + "=" * 60)
    print("ABLATION STUDY TEST COMPLETE")
    print("=" * 60)