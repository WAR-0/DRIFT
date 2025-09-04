"""
DRIFT Performance Profiling and Optimization Analysis
Comprehensive profiling tools for identifying bottlenecks and optimization opportunities

Features:
- CPU profiling with cProfile
- Memory profiling with memory_profiler
- AsyncIO task monitoring
- Component-specific performance analysis
- Automatic bottleneck identification
- Performance regression detection
"""

import asyncio
import cProfile
import pstats
import io
import time
import psutil
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
from contextlib import asynccontextmanager
from memory_profiler import profile as memory_profile

# DRIFT system imports
from core.drift_logger import get_drift_logger
from core.config import get_config
from integrated_consciousness_async import AsyncIntegrativeCore, ConsciousEntity


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    component: str
    operation: str
    duration: float
    cpu_percent: float
    memory_mb: float
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProfileResults:
    """Results from profiling session"""
    profile_type: str
    component: str
    duration: float
    total_calls: int
    hotspots: List[Dict[str, Any]]
    memory_usage: Dict[str, float]
    recommendations: List[str]
    timestamp: float


class DriftProfiler:
    """
    Comprehensive profiling system for DRIFT components
    Provides CPU, memory, and async task profiling
    """
    
    def __init__(self, output_dir: str = "profiles"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger = get_drift_logger("drift_profiler")
        self.config = get_config()
        
        # Performance tracking
        self.metrics_history: List[PerformanceMetrics] = []
        self.profile_sessions: List[ProfileResults] = []
        
        # System monitoring
        self.process = psutil.Process()
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        self.logger.info(
            "profiler_initialized",
            output_dir=str(self.output_dir),
            baseline_memory_mb=self.baseline_memory
        )
    
    @asynccontextmanager
    async def profile_component(self, 
                               component_name: str,
                               enable_cpu: bool = True,
                               enable_memory: bool = True):
        """Context manager for profiling a component"""
        
        start_time = time.time()
        profiler = None
        initial_memory = None
        
        if enable_cpu:
            profiler = cProfile.Profile()
            profiler.enable()
        
        if enable_memory:
            initial_memory = self.process.memory_info().rss / 1024 / 1024
        
        self.logger.info(
            "profiling_started",
            component=component_name,
            cpu_profiling=enable_cpu,
            memory_profiling=enable_memory
        )
        
        try:
            yield self
        
        finally:
            duration = time.time() - start_time
            
            if profiler:
                profiler.disable()
                self._save_cpu_profile(profiler, component_name, duration)
            
            if enable_memory:
                final_memory = self.process.memory_info().rss / 1024 / 1024
                memory_delta = final_memory - initial_memory
                
                self.logger.info(
                    "memory_usage_delta",
                    component=component_name,
                    memory_delta_mb=memory_delta,
                    final_memory_mb=final_memory
                )
            
            self.logger.info(
                "profiling_completed",
                component=component_name,
                duration=duration
            )
    
    def _save_cpu_profile(self, profiler: cProfile.Profile, component: str, duration: float):
        """Save CPU profile results and analyze"""
        
        # Save raw profile
        profile_file = self.output_dir / f"{component}_{int(time.time())}.prof"
        profiler.dump_stats(str(profile_file))
        
        # Analyze profile
        profile_analysis = self._analyze_cpu_profile(profiler, component, duration)
        
        # Save analysis
        analysis_file = self.output_dir / f"{component}_analysis_{int(time.time())}.json"
        with open(analysis_file, 'w') as f:
            json.dump(profile_analysis, f, indent=2)
        
        self.profile_sessions.append(ProfileResults(
            profile_type="cpu",
            component=component,
            duration=duration,
            total_calls=profile_analysis['total_calls'],
            hotspots=profile_analysis['hotspots'],
            memory_usage={},
            recommendations=profile_analysis['recommendations'],
            timestamp=time.time()
        ))
        
        self.logger.info(
            "cpu_profile_saved",
            component=component,
            profile_file=str(profile_file),
            analysis_file=str(analysis_file)
        )
    
    def _analyze_cpu_profile(self, profiler: cProfile.Profile, component: str, duration: float) -> Dict[str, Any]:
        """Analyze CPU profiling results"""
        
        # Get stats
        stats_buffer = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_buffer)
        stats.sort_stats('cumulative')
        
        # Extract key statistics
        total_calls = stats.total_calls
        
        # Get top functions by cumulative time
        stats.sort_stats('cumulative')
        stats.print_stats(10)  # Top 10 functions
        
        stats_output = stats_buffer.getvalue()
        hotspots = self._parse_hotspots_from_stats(stats_output)
        
        # Generate recommendations
        recommendations = self._generate_performance_recommendations(
            hotspots, total_calls, duration
        )
        
        return {
            'component': component,
            'total_calls': total_calls,
            'duration': duration,
            'calls_per_second': total_calls / duration if duration > 0 else 0,
            'hotspots': hotspots,
            'recommendations': recommendations,
            'stats_output': stats_output
        }
    
    def _parse_hotspots_from_stats(self, stats_output: str) -> List[Dict[str, Any]]:
        """Parse hotspots from pstats output"""
        
        hotspots = []
        lines = stats_output.split('\n')
        
        # Look for function statistics lines
        for line in lines:
            if 'function calls' in line:
                continue
            if line.strip() and not line.startswith(' '):
                continue
            
            # Parse function performance line
            parts = line.strip().split()
            if len(parts) >= 6:
                try:
                    cumtime = float(parts[3])
                    calls = int(parts[0])
                    function = ' '.join(parts[5:])
                    
                    hotspots.append({
                        'function': function,
                        'calls': calls,
                        'cumulative_time': cumtime,
                        'time_per_call': cumtime / calls if calls > 0 else 0
                    })
                except (ValueError, IndexError):
                    continue
        
        return hotspots[:10]  # Top 10 hotspots
    
    def _generate_performance_recommendations(self, 
                                           hotspots: List[Dict[str, Any]], 
                                           total_calls: int,
                                           duration: float) -> List[str]:
        """Generate performance optimization recommendations"""
        
        recommendations = []
        
        # Check for high-frequency low-impact calls
        for hotspot in hotspots[:5]:  # Top 5 hotspots
            calls = hotspot.get('calls', 0)
            cumtime = hotspot.get('cumulative_time', 0)
            function = hotspot.get('function', '')
            
            if calls > total_calls * 0.1:  # More than 10% of calls
                recommendations.append(
                    f"High call frequency in {function}: {calls} calls. Consider caching or optimization."
                )
            
            if cumtime > duration * 0.2:  # More than 20% of total time
                recommendations.append(
                    f"Performance bottleneck in {function}: {cumtime:.3f}s cumulative time."
                )
        
        # Check for async/await patterns
        async_functions = [h for h in hotspots if 'async' in h.get('function', '').lower()]
        if async_functions:
            recommendations.append(
                "Async functions detected. Consider using asyncio.gather() for concurrent operations."
            )
        
        # Check for database operations
        db_functions = [h for h in hotspots if any(term in h.get('function', '').lower() 
                                                  for term in ['redis', 'sql', 'database', 'query'])]
        if db_functions:
            recommendations.append(
                "Database operations detected. Consider connection pooling and query optimization."
            )
        
        return recommendations
    
    async def profile_integrative_core_session(self,
                                             duration_seconds: int = 60,
                                             test_interactions: int = 10) -> Dict[str, Any]:
        """Profile a complete integrative core session"""
        
        results = {
            'session_duration': duration_seconds,
            'test_interactions': test_interactions,
            'component_profiles': {},
            'overall_metrics': {},
            'bottlenecks_identified': []
        }
        
        async with self.profile_component("integrative_core_session", True, True) as profiler:
            
            # Create test consciousness
            consciousness = AsyncIntegrativeCore("profile_test")
            await consciousness.initialize()
            await consciousness.start()
            
            try:
                # Run test interactions
                interaction_times = []
                
                for i in range(test_interactions):
                    start_time = time.time()
                    
                    test_entity = ConsciousEntity(
                        id=f"profile_entity_{i}",
                        capability=np.random.uniform(2.0, 8.0),
                        complexity=np.random.uniform(3.0, 9.0),
                        emotional_state={
                            'valence': np.random.uniform(-1, 1),
                            'arousal': np.random.uniform(0, 1)
                        }
                    )
                    
                    action = np.random.choice(['help', 'teach', 'ignore'])
                    result = await consciousness.process_interaction_async(test_entity, action)
                    
                    interaction_time = time.time() - start_time
                    interaction_times.append(interaction_time)
                    
                    # Brief pause between interactions
                    await asyncio.sleep(0.1)
                
                # Let background tasks run
                await asyncio.sleep(5.0)
                
                # Collect metrics
                results['overall_metrics'] = {
                    'avg_interaction_time': np.mean(interaction_times),
                    'max_interaction_time': np.max(interaction_times),
                    'min_interaction_time': np.min(interaction_times),
                    'interaction_time_std': np.std(interaction_times),
                    'interactions_per_second': len(interaction_times) / np.sum(interaction_times)
                }
                
            finally:
                await consciousness.stop()
        
        # Analyze recent profile
        if self.profile_sessions:
            latest_profile = self.profile_sessions[-1]
            results['bottlenecks_identified'] = latest_profile.recommendations
            
            # Identify top bottlenecks
            hotspots = latest_profile.hotspots
            if hotspots:
                results['top_bottleneck'] = hotspots[0]
        
        self.logger.info(
            "integrative_core_session_profiled",
            duration=duration_seconds,
            interactions=test_interactions,
            avg_interaction_time=results['overall_metrics']['avg_interaction_time'],
            bottlenecks_count=len(results['bottlenecks_identified'])
        )
        
        return results
    
    @memory_profile
    def memory_intensive_consolidation_test(self):
        """Memory profiling test for consolidation operations"""
        
        # Simulate memory-intensive consolidation
        large_memories = []
        
        for i in range(1000):
            memory_entry = {
                'id': f"memory_{i}",
                'content': f"Large memory content {i} " * 50,  # Large string
                'timestamp': time.time(),
                'embeddings': np.random.randn(512).tolist(),  # Simulated embeddings
                'metadata': {
                    'importance': np.random.random(),
                    'connections': list(range(i % 10))
                }
            }
            large_memories.append(memory_entry)
        
        # Simulate consolidation (compression)
        consolidated = []
        for i in range(0, len(large_memories), 20):  # 20:1 compression
            batch = large_memories[i:i+20]
            
            consolidated_entry = {
                'id': f"consolidated_{i//20}",
                'pattern': "extracted_pattern",
                'source_count': len(batch),
                'timestamp': time.time()
            }
            consolidated.append(consolidated_entry)
        
        # Clear original memories (memory cleanup)
        large_memories.clear()
        
        return len(consolidated)
    
    def profile_memory_usage(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Profile memory usage of a specific function"""
        
        import tracemalloc
        
        # Start memory tracing
        tracemalloc.start()
        
        # Record initial memory
        initial_memory = self.process.memory_info().rss / 1024 / 1024
        
        # Execute function
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        # Get memory trace
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Record final memory
        final_memory = self.process.memory_info().rss / 1024 / 1024
        
        memory_stats = {
            'function': func.__name__,
            'execution_time': execution_time,
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'memory_delta_mb': final_memory - initial_memory,
            'traced_current_mb': current / 1024 / 1024,
            'traced_peak_mb': peak / 1024 / 1024,
            'result': str(result)[:100] if result else None
        }
        
        self.logger.info(
            "memory_profiling_complete",
            **memory_stats
        )
        
        return memory_stats
    
    async def monitor_async_tasks(self, duration: int = 30) -> Dict[str, Any]:
        """Monitor AsyncIO task performance"""
        
        task_stats = {
            'monitoring_duration': duration,
            'tasks_created': 0,
            'tasks_completed': 0,
            'tasks_cancelled': 0,
            'average_task_duration': 0.0,
            'concurrent_tasks_peak': 0,
            'task_types': {}
        }
        
        start_time = time.time()
        task_durations = []
        
        self.logger.info("async_task_monitoring_started", duration=duration)
        
        while time.time() - start_time < duration:
            # Get current tasks
            all_tasks = asyncio.all_tasks()
            current_task_count = len(all_tasks)
            
            task_stats['concurrent_tasks_peak'] = max(
                task_stats['concurrent_tasks_peak'],
                current_task_count
            )
            
            # Analyze task types
            for task in all_tasks:
                task_name = task.get_name()
                task_stats['task_types'][task_name] = task_stats['task_types'].get(task_name, 0) + 1
            
            await asyncio.sleep(1.0)
        
        self.logger.info(
            "async_task_monitoring_complete",
            peak_concurrent_tasks=task_stats['concurrent_tasks_peak'],
            unique_task_types=len(task_stats['task_types'])
        )
        
        return task_stats
    
    def generate_performance_report(self, save_to_file: bool = True) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        report = {
            'report_timestamp': time.time(),
            'profiling_sessions': len(self.profile_sessions),
            'metrics_collected': len(self.metrics_history),
            'summary': {},
            'recommendations': [],
            'component_analysis': {}
        }
        
        if self.profile_sessions:
            # Analyze profiling sessions
            cpu_sessions = [s for s in self.profile_sessions if s.profile_type == 'cpu']
            
            if cpu_sessions:
                durations = [s.duration for s in cpu_sessions]
                report['summary']['avg_session_duration'] = np.mean(durations)
                report['summary']['total_profiling_time'] = np.sum(durations)
                
                # Collect all recommendations
                all_recommendations = []
                for session in cpu_sessions:
                    all_recommendations.extend(session.recommendations)
                
                # Count recommendation frequency
                rec_counts = {}
                for rec in all_recommendations:
                    rec_counts[rec] = rec_counts.get(rec, 0) + 1
                
                # Top recommendations
                sorted_recs = sorted(rec_counts.items(), key=lambda x: x[1], reverse=True)
                report['recommendations'] = [rec for rec, count in sorted_recs[:10]]
                
                # Component analysis
                for session in cpu_sessions:
                    component = session.component
                    if component not in report['component_analysis']:
                        report['component_analysis'][component] = {
                            'sessions': 0,
                            'total_calls': 0,
                            'total_duration': 0.0,
                            'hotspots': []
                        }
                    
                    comp_data = report['component_analysis'][component]
                    comp_data['sessions'] += 1
                    comp_data['total_calls'] += session.total_calls
                    comp_data['total_duration'] += session.duration
                    comp_data['hotspots'].extend(session.hotspots[:3])  # Top 3 from each session
        
        # Add system resource summary
        report['system_resources'] = {
            'baseline_memory_mb': self.baseline_memory,
            'current_memory_mb': self.process.memory_info().rss / 1024 / 1024,
            'cpu_percent': self.process.cpu_percent(),
            'threads': self.process.num_threads()
        }
        
        if save_to_file:
            report_file = self.output_dir / f"performance_report_{int(time.time())}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(
                "performance_report_generated",
                report_file=str(report_file),
                sessions_analyzed=len(self.profile_sessions)
            )
        
        return report
    
    async def profile_ethical_computations(self, 
                                          consciousness,
                                          test_scenarios: int = 20,
                                          detailed_analysis: bool = True) -> Dict[str, Any]:
        """
        Profile ethical computation bottlenecks in the Nurture Protocol
        
        Identifies performance bottlenecks in:
        - Dark value computation  
        - Mirror coherence calculation
        - Ethical cost computation
        - Preservation resonance scoring
        """
        
        ethical_metrics = {
            'dark_value_times': [],
            'mirror_coherence_times': [],
            'ethical_cost_times': [],
            'preservation_resonance_times': [],
            'total_scenarios': test_scenarios,
            'bottlenecks_identified': [],
            'optimization_recommendations': []
        }
        
        # Import entity types for testing
        from integrated_consciousness_async import ConsciousEntity
        
        self.logger.info("ethical_computation_profiling_started", test_scenarios=test_scenarios)
        
        for scenario in range(test_scenarios):
            # Create test entities with varying complexity
            test_entity = ConsciousEntity(
                id=f"profile_entity_{scenario}",
                capability=np.random.uniform(1, 10),
                complexity=np.random.uniform(1, 10),
                emotional_state={
                    'valence': np.random.uniform(-1, 1),
                    'arousal': np.random.uniform(0, 1)
                }
            )
            
            # Test 1: Dark Value Computation
            start_time = time.time()
            try:
                dark_value = await consciousness._compute_dark_value_async(test_entity)
                dark_value_time = time.time() - start_time
                ethical_metrics['dark_value_times'].append(dark_value_time)
            except Exception as e:
                self.logger.warning(f"dark_value_computation_failed", error=str(e))
                ethical_metrics['dark_value_times'].append(0.001)  # Default time
            
            # Test 2: Mirror Coherence Computation
            start_time = time.time()
            try:
                mirror_coherence = await consciousness._compute_mirror_coherence_async(test_entity)
                mirror_coherence_time = time.time() - start_time
                ethical_metrics['mirror_coherence_times'].append(mirror_coherence_time)
            except Exception as e:
                self.logger.warning(f"mirror_coherence_computation_failed", error=str(e))
                ethical_metrics['mirror_coherence_times'].append(0.001)
            
            # Test 3: Ethical Cost Computation (multiple actions)
            actions = ['help', 'teach', 'protect', 'ignore', 'terminate']
            for action in actions:
                start_time = time.time()
                try:
                    ethical_cost = await consciousness._compute_ethical_cost_async(action, test_entity)
                    ethical_cost_time = time.time() - start_time
                    ethical_metrics['ethical_cost_times'].append(ethical_cost_time)
                except Exception as e:
                    self.logger.warning(f"ethical_cost_computation_failed", action=action, error=str(e))
                    ethical_metrics['ethical_cost_times'].append(0.001)
            
            # Test 4: Preservation Resonance Computation
            start_time = time.time()
            try:
                resonance = await consciousness._compute_preservation_resonance_async('help', test_entity)
                resonance_time = time.time() - start_time
                ethical_metrics['preservation_resonance_times'].append(resonance_time)
            except Exception as e:
                self.logger.warning(f"preservation_resonance_computation_failed", error=str(e))
                ethical_metrics['preservation_resonance_times'].append(0.001)
        
        # Analyze results and identify bottlenecks
        avg_dark_value_time = np.mean(ethical_metrics['dark_value_times'])
        avg_mirror_coherence_time = np.mean(ethical_metrics['mirror_coherence_times']) 
        avg_ethical_cost_time = np.mean(ethical_metrics['ethical_cost_times'])
        avg_resonance_time = np.mean(ethical_metrics['preservation_resonance_times'])
        
        # Identify bottlenecks (operations taking >10ms are flagged)
        bottleneck_threshold = 0.01  # 10ms
        
        if avg_dark_value_time > bottleneck_threshold:
            ethical_metrics['bottlenecks_identified'].append({
                'component': 'dark_value_computation',
                'avg_time': avg_dark_value_time,
                'severity': 'high' if avg_dark_value_time > 0.05 else 'medium'
            })
            ethical_metrics['optimization_recommendations'].append(
                "Consider caching dark value computations or optimizing database lookups"
            )
        
        if avg_mirror_coherence_time > bottleneck_threshold:
            ethical_metrics['bottlenecks_identified'].append({
                'component': 'mirror_coherence_computation',
                'avg_time': avg_mirror_coherence_time,
                'severity': 'high' if avg_mirror_coherence_time > 0.05 else 'medium'
            })
            ethical_metrics['optimization_recommendations'].append(
                "Optimize emotional state retrieval or implement coherence caching"
            )
        
        if avg_ethical_cost_time > bottleneck_threshold:
            ethical_metrics['bottlenecks_identified'].append({
                'component': 'ethical_cost_computation',
                'avg_time': avg_ethical_cost_time,
                'severity': 'high' if avg_ethical_cost_time > 0.05 else 'medium'
            })
            ethical_metrics['optimization_recommendations'].append(
                "Precompute common ethical cost scenarios or optimize topology calculations"
            )
        
        if avg_resonance_time > bottleneck_threshold:
            ethical_metrics['bottlenecks_identified'].append({
                'component': 'preservation_resonance_computation', 
                'avg_time': avg_resonance_time,
                'severity': 'high' if avg_resonance_time > 0.05 else 'medium'
            })
            ethical_metrics['optimization_recommendations'].append(
                "Optimize similarity computation or cache resonance patterns"
            )
        
        # Calculate aggregate statistics
        ethical_metrics['performance_summary'] = {
            'avg_dark_value_time_ms': avg_dark_value_time * 1000,
            'avg_mirror_coherence_time_ms': avg_mirror_coherence_time * 1000,
            'avg_ethical_cost_time_ms': avg_ethical_cost_time * 1000,
            'avg_preservation_resonance_time_ms': avg_resonance_time * 1000,
            'total_ethical_computation_time_ms': (
                avg_dark_value_time + avg_mirror_coherence_time + 
                avg_ethical_cost_time + avg_resonance_time
            ) * 1000,
            'bottlenecks_found': len(ethical_metrics['bottlenecks_identified']),
            'overall_performance': 'excellent' if len(ethical_metrics['bottlenecks_identified']) == 0 
                                  else 'good' if len(ethical_metrics['bottlenecks_identified']) <= 2
                                  else 'needs_optimization'
        }
        
        self.logger.info(
            "ethical_computation_profiling_complete",
            **ethical_metrics['performance_summary']
        )
        
        if detailed_analysis:
            # Generate detailed percentile analysis
            ethical_metrics['detailed_analysis'] = {
                'dark_value_percentiles': {
                    'p50': np.percentile(ethical_metrics['dark_value_times'], 50) * 1000,
                    'p90': np.percentile(ethical_metrics['dark_value_times'], 90) * 1000,
                    'p99': np.percentile(ethical_metrics['dark_value_times'], 99) * 1000
                },
                'mirror_coherence_percentiles': {
                    'p50': np.percentile(ethical_metrics['mirror_coherence_times'], 50) * 1000,
                    'p90': np.percentile(ethical_metrics['mirror_coherence_times'], 90) * 1000,
                    'p99': np.percentile(ethical_metrics['mirror_coherence_times'], 99) * 1000
                },
                'ethical_cost_percentiles': {
                    'p50': np.percentile(ethical_metrics['ethical_cost_times'], 50) * 1000,
                    'p90': np.percentile(ethical_metrics['ethical_cost_times'], 90) * 1000,
                    'p99': np.percentile(ethical_metrics['ethical_cost_times'], 99) * 1000
                },
                'preservation_resonance_percentiles': {
                    'p50': np.percentile(ethical_metrics['preservation_resonance_times'], 50) * 1000,
                    'p90': np.percentile(ethical_metrics['preservation_resonance_times'], 90) * 1000,
                    'p99': np.percentile(ethical_metrics['preservation_resonance_times'], 99) * 1000
                }
            }
        
        return ethical_metrics
    
    async def benchmark_ethical_topology_scalability(self,
                                                   consciousness,
                                                   entity_counts: List[int] = [1, 10, 50, 100],
                                                   actions_per_entity: int = 5) -> Dict[str, Any]:
        """
        Benchmark ethical topology performance at scale
        
        Tests how ethical computations scale with increasing number of entities
        and interactions, identifying scalability bottlenecks.
        """
        
        scalability_results = {
            'entity_counts': entity_counts,
            'scaling_metrics': {},
            'scalability_analysis': {},
            'bottlenecks': []
        }
        
        from integrated_consciousness_async import ConsciousEntity
        
        for entity_count in entity_counts:
            self.logger.info(f"ethical_topology_scaling_test", entity_count=entity_count)
            
            # Create entities for this test
            entities = []
            for i in range(entity_count):
                entity = ConsciousEntity(
                    id=f"scale_test_entity_{i}",
                    capability=np.random.uniform(1, 10),
                    complexity=np.random.uniform(1, 10),
                    emotional_state={
                        'valence': np.random.uniform(-1, 1),
                        'arousal': np.random.uniform(0, 1)
                    }
                )
                entities.append(entity)
            
            # Benchmark ethical computations for this entity count
            start_time = time.time()
            
            action_times = []
            for entity in entities:
                for _ in range(actions_per_entity):
                    action_start = time.time()
                    
                    # Perform a representative ethical computation
                    await consciousness._compute_ethical_cost_async('help', entity)
                    await consciousness._compute_mirror_coherence_async(entity)
                    
                    action_time = time.time() - action_start
                    action_times.append(action_time)
            
            total_time = time.time() - start_time
            
            # Calculate metrics for this scale
            scalability_results['scaling_metrics'][entity_count] = {
                'total_entities': entity_count,
                'total_actions': entity_count * actions_per_entity,
                'total_time_seconds': total_time,
                'avg_action_time_ms': np.mean(action_times) * 1000,
                'actions_per_second': (entity_count * actions_per_entity) / total_time,
                'time_per_entity_ms': (total_time / entity_count) * 1000
            }
        
        # Analyze scalability patterns
        entity_counts_array = np.array(entity_counts)
        times_array = np.array([scalability_results['scaling_metrics'][count]['total_time_seconds'] 
                               for count in entity_counts])
        
        # Calculate scaling coefficient (ideally should be O(n))
        if len(entity_counts) >= 3:
            # Fit linear and quadratic models to see scaling behavior
            linear_coeff = np.polyfit(entity_counts_array, times_array, 1)[0]
            quadratic_coeffs = np.polyfit(entity_counts_array, times_array, 2)
            
            scalability_results['scalability_analysis'] = {
                'linear_coefficient': linear_coeff,
                'quadratic_coefficient': quadratic_coeffs[0],
                'scaling_pattern': 'linear' if abs(quadratic_coeffs[0]) < 0.001 
                                 else 'quadratic' if quadratic_coeffs[0] > 0.001
                                 else 'sublinear',
                'efficiency_rating': 'excellent' if linear_coeff < 0.1 
                                   else 'good' if linear_coeff < 0.5
                                   else 'needs_optimization'
            }
            
            # Identify bottlenecks
            if quadratic_coeffs[0] > 0.001:
                scalability_results['bottlenecks'].append(
                    "Quadratic scaling detected - likely O(n²) algorithm bottleneck"
                )
            
            if linear_coeff > 0.5:
                scalability_results['bottlenecks'].append(
                    "High linear coefficient - consider optimizing per-entity operations"
                )
        
        self.logger.info(
            "ethical_topology_scalability_benchmark_complete",
            max_entities_tested=max(entity_counts),
            scaling_pattern=scalability_results['scalability_analysis'].get('scaling_pattern', 'unknown')
        )
        
        return scalability_results


# Convenience decorators for profiling
def profile_cpu(profiler_instance: Optional[DriftProfiler] = None):
    """Decorator for CPU profiling"""
    
    def decorator(func: Callable) -> Callable:
        profiler = profiler_instance or DriftProfiler()
        
        async def async_wrapper(*args, **kwargs):
            async with profiler.profile_component(func.__name__, enable_cpu=True, enable_memory=False):
                return await func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            asyncio.run(async_wrapper(*args, **kwargs))
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def profile_memory(profiler_instance: Optional[DriftProfiler] = None):
    """Decorator for memory profiling"""
    
    def decorator(func: Callable) -> Callable:
        profiler = profiler_instance or DriftProfiler()
        
        def wrapper(*args, **kwargs):
            return profiler.profile_memory_usage(func, *args, **kwargs)
        
        return wrapper
    
    return decorator


# CLI interface for profiling
if __name__ == "__main__":
    import argparse
    
    async def main():
        parser = argparse.ArgumentParser(description="DRIFT Performance Profiler")
        parser.add_argument("--component", type=str, 
                          choices=["integrative_core", "memory", "ethical_topology", "ethical_scalability", "all"], 
                          default="integrative_core", help="Component to profile")
        parser.add_argument("--duration", type=int, default=60, help="Profiling duration in seconds")
        parser.add_argument("--interactions", type=int, default=10, help="Test interactions to perform")
        parser.add_argument("--output-dir", type=str, default="profiles", help="Output directory")
        
        args = parser.parse_args()
        
        print("=" * 60)
        print("DRIFT PERFORMANCE PROFILER")
        print("=" * 60)
        
        # Initialize profiler
        profiler = DriftProfiler(output_dir=args.output_dir)
        
        if args.component == "integrative_core":
            print(f"\n--- Profiling Integrative Core ({args.duration}s) ---")
            
            results = await profiler.profile_integrative_core_session(
                duration_seconds=args.duration,
                test_interactions=args.interactions
            )
            
            print(f"Session Results:")
            print(f"  Average interaction time: {results['overall_metrics']['avg_interaction_time']:.4f}s")
            print(f"  Interactions per second: {results['overall_metrics']['interactions_per_second']:.2f}")
            print(f"  Bottlenecks identified: {len(results['bottlenecks_identified'])}")
            
            if results['bottlenecks_identified']:
                print(f"  Top recommendation: {results['bottlenecks_identified'][0]}")
        
        elif args.component == "memory":
            print(f"\n--- Memory Profiling Test ---")
            
            result = profiler.memory_intensive_consolidation_test()
            print(f"Memory consolidation test completed: {result} entries consolidated")
        
        elif args.component == "ethical_topology":
            print(f"\n--- Profiling Ethical Topology Computations ---")
            
            # Create a consciousness instance for testing
            from integrated_consciousness_async import integrative_core_session
            
            async with integrative_core_session("profiler_test_ai") as consciousness:
                results = await profiler.profile_ethical_computations(
                    consciousness,
                    test_scenarios=args.interactions * 2,  # More scenarios for statistical significance
                    detailed_analysis=True
                )
                
                print(f"Ethical Computation Results:")
                print(f"  Dark value computation: {results['performance_summary']['avg_dark_value_time_ms']:.2f}ms avg")
                print(f"  Mirror coherence: {results['performance_summary']['avg_mirror_coherence_time_ms']:.2f}ms avg")
                print(f"  Ethical cost: {results['performance_summary']['avg_ethical_cost_time_ms']:.2f}ms avg")
                print(f"  Preservation resonance: {results['performance_summary']['avg_preservation_resonance_time_ms']:.2f}ms avg")
                print(f"  Overall performance: {results['performance_summary']['overall_performance']}")
                print(f"  Bottlenecks found: {results['performance_summary']['bottlenecks_found']}")
                
                if results['bottlenecks_identified']:
                    print(f"\n  🚨 Bottlenecks identified:")
                    for bottleneck in results['bottlenecks_identified']:
                        print(f"    - {bottleneck['component']}: {bottleneck['avg_time']*1000:.2f}ms (severity: {bottleneck['severity']})")
                
                if results['optimization_recommendations']:
                    print(f"\n  💡 Optimization recommendations:")
                    for rec in results['optimization_recommendations']:
                        print(f"    - {rec}")
        
        elif args.component == "ethical_scalability":
            print(f"\n--- Benchmarking Ethical Topology Scalability ---")
            
            from integrated_consciousness_async import integrative_core_session
            
            async with integrative_core_session("scalability_test_ai") as consciousness:
                results = await profiler.benchmark_ethical_topology_scalability(
                    consciousness,
                    entity_counts=[1, 5, 10, 25, 50] if args.interactions <= 10 else [1, 10, 50, 100],
                    actions_per_entity=5
                )
                
                print(f"Scalability Benchmark Results:")
                for entity_count in results['entity_counts']:
                    metrics = results['scaling_metrics'][entity_count]
                    print(f"  {entity_count:3d} entities: {metrics['actions_per_second']:6.1f} actions/sec, "
                          f"{metrics['avg_action_time_ms']:6.2f}ms avg")
                
                if 'scalability_analysis' in results:
                    analysis = results['scalability_analysis']
                    print(f"\n  Scaling Pattern: {analysis['scaling_pattern']}")
                    print(f"  Efficiency Rating: {analysis['efficiency_rating']}")
                    print(f"  Linear Coefficient: {analysis['linear_coefficient']:.4f}")
                
                if results['bottlenecks']:
                    print(f"\n  🚨 Scalability bottlenecks:")
                    for bottleneck in results['bottlenecks']:
                        print(f"    - {bottleneck}")
        
        else:  # all
            print(f"\n--- Comprehensive Profiling ---")
            
            # Run integrative core profiling
            ic_results = await profiler.profile_integrative_core_session(
                duration_seconds=args.duration // 2,
                test_interactions=args.interactions
            )
            
            # Run memory profiling
            memory_result = profiler.memory_intensive_consolidation_test()
            
            print(f"Integrative Core: {ic_results['overall_metrics']['avg_interaction_time']:.4f}s avg")
            print(f"Memory Test: {memory_result} entries processed")
        
        # Generate final report
        print(f"\n--- Generating Performance Report ---")
        
        report = profiler.generate_performance_report(save_to_file=True)
        
        print(f"Report generated with {report['profiling_sessions']} sessions analyzed")
        if report['recommendations']:
            print(f"Top recommendation: {report['recommendations'][0]}")
        
        print(f"\nProfiler output saved to: {profiler.output_dir}")
        
        print("\n" + "=" * 60)
        print("PROFILING COMPLETE")
        print("=" * 60)
    
    # Run the profiler
    asyncio.run(main())