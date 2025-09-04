"""
DRIFT Cognitive Architecture Real-Time Analysis Dashboard
Streamlit-based monitoring and visualization system

Features:
- Real-time log streaming and analysis
- Valence-arousal trajectory visualization  
- Saliency gating distribution analysis
- Memory consolidation pattern tracking
- Component activity monitoring
- Optimization results visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
from pathlib import Path
import io

# DRIFT system imports
from core.config import get_config
from core.drift_logger import get_drift_logger

# Page configuration
st.set_page_config(
    page_title="DRIFT Cognitive Architecture Monitor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .status-good { color: #28a745; }
    .status-warning { color: #ffc107; }
    .status-error { color: #dc3545; }
</style>
""", unsafe_allow_html=True)


class DriftDashboard:
    """Main dashboard class for DRIFT system monitoring"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_drift_logger("dashboard")
        
        # Initialize session state
        if 'logs_data' not in st.session_state:
            st.session_state.logs_data = []
        if 'last_update' not in st.session_state:
            st.session_state.last_update = time.time()
    
    def run(self):
        """Main dashboard interface"""
        
        # Header
        st.markdown('<h1 class="main-header">🧠 DRIFT Cognitive Architecture Monitor</h1>', 
                   unsafe_allow_html=True)
        
        # Sidebar
        self._render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Real-Time Monitoring", 
            "🧠 Emotional Analysis", 
            "💭 Saliency Gating",
            "🔄 Memory Systems",
            "⚙️ Optimization Results",
            "🛡️ Ethical Topology"
        ])
        
        with tab1:
            self._render_realtime_monitoring()
        
        with tab2:
            self._render_emotional_analysis()
        
        with tab3:
            self._render_saliency_analysis()
        
        with tab4:
            self._render_memory_analysis()
        
        with tab5:
            self._render_optimization_analysis()
            
        with tab6:
            self._render_ethical_topology()
    
    def _render_sidebar(self):
        """Render sidebar controls"""
        
        st.sidebar.header("📋 Dashboard Controls")
        
        # Data source selection
        st.sidebar.subheader("Data Source")
        data_source = st.sidebar.radio(
            "Select data source:",
            ["Upload Log File", "Live Stream", "Sample Data"]
        )
        
        if data_source == "Upload Log File":
            uploaded_file = st.sidebar.file_uploader(
                "Choose JSON log file",
                type=['json', 'jsonl', 'log'],
                help="Upload structured log file from DRIFT system"
            )
            
            if uploaded_file:
                self._load_log_file(uploaded_file)
        
        elif data_source == "Live Stream":
            st.sidebar.info("🔴 Live streaming (Redis required)")
            if st.sidebar.button("Connect to Live Stream"):
                self._connect_live_stream()
        
        else:  # Sample Data
            if st.sidebar.button("Generate Sample Data"):
                self._generate_sample_data()
        
        # Configuration display
        st.sidebar.subheader("⚙️ Current Configuration")
        with st.sidebar.expander("System Config"):
            st.write(f"**Resonance Threshold:** {self.config.drift.resonance.threshold}")
            st.write(f"**Buffer Size:** {self.config.drift.memory.drift_buffer_size}")
            st.write(f"**Consolidation Ratio:** {self.config.drift.memory.consolidation_ratio}")
        
        # System status
        st.sidebar.subheader("📡 System Status")
        self._render_system_status()
    
    def _render_system_status(self):
        """Render system status indicators"""
        
        # Mock status indicators
        status_items = [
            ("Integrative Core", "good", "Active"),
            ("Saliency Gating", "good", "Operational"),
            ("Memory Systems", "warning", "High Load"),
            ("Emotional Tagger", "good", "Running"),
        ]
        
        for component, status, description in status_items:
            if status == "good":
                st.sidebar.markdown(f"🟢 **{component}**: {description}")
            elif status == "warning":
                st.sidebar.markdown(f"🟡 **{component}**: {description}")
            else:
                st.sidebar.markdown(f"🔴 **{component}**: {description}")
    
    def _load_log_file(self, uploaded_file):
        """Load and parse log file"""
        
        try:
            # Read file content
            content = uploaded_file.read().decode('utf-8')
            
            # Parse JSON lines
            logs = []
            for line in content.strip().split('\n'):
                if line.strip():
                    try:
                        log_entry = json.loads(line)
                        # Add timestamp parsing
                        if 'timestamp' in log_entry:
                            log_entry['parsed_timestamp'] = pd.to_datetime(log_entry['timestamp'])
                        logs.append(log_entry)
                    except json.JSONDecodeError:
                        continue
            
            st.session_state.logs_data = logs
            st.session_state.last_update = time.time()
            
            st.sidebar.success(f"Loaded {len(logs)} log entries")
            
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")
    
    def _connect_live_stream(self):
        """Connect to live Redis stream (mock implementation)"""
        st.sidebar.warning("Live streaming requires Redis setup")
        # In real implementation, would connect to Redis pub/sub
        
    def _generate_sample_data(self):
        """Generate sample data for demonstration"""
        
        # Generate sample log entries
        sample_logs = []
        base_time = datetime.now() - timedelta(hours=1)
        
        events = [
            "resonance_calculated", "drift_generated", "memory_consolidation",
            "valence_arousal_heuristic", "saliency_gating", "interaction_processed"
        ]
        
        components = [
            "integrative_core", "saliency_gating", "memory_systems", 
            "associative_elaboration", "emotional_tagger"
        ]
        
        for i in range(200):
            timestamp = base_time + timedelta(seconds=i*10)
            
            log_entry = {
                'timestamp': timestamp.isoformat() + 'Z',
                'parsed_timestamp': timestamp,
                'event': np.random.choice(events),
                'component': np.random.choice(components),
                'level': np.random.choice(['info', 'debug', 'warning'], p=[0.7, 0.2, 0.1])
            }
            
            # Add event-specific data
            if log_entry['event'] == 'resonance_calculated':
                log_entry.update({
                    'score': np.random.uniform(0.3, 0.9),
                    'threshold': self.config.drift.resonance.threshold,
                    'triggered': np.random.choice([True, False], p=[0.3, 0.7]),
                    'components': {
                        'semantic': np.random.uniform(0.1, 0.4),
                        'preservation': np.random.uniform(0.1, 0.3),
                        'emotional': np.random.uniform(0.1, 0.2)
                    }
                })
            
            elif log_entry['event'] == 'valence_arousal_heuristic':
                log_entry.update({
                    'valence': np.random.uniform(-1, 1),
                    'arousal': np.random.uniform(0, 1),
                    'confidence': np.random.uniform(0.5, 1.0)
                })
            
            elif log_entry['event'] == 'memory_consolidation':
                log_entry.update({
                    'input_count': np.random.randint(15, 35),
                    'output_count': np.random.randint(3, 8),
                    'compression_ratio': np.random.uniform(15, 25)
                })
            
            sample_logs.append(log_entry)
        
        st.session_state.logs_data = sample_logs
        st.session_state.last_update = time.time()
        
        st.sidebar.success(f"Generated {len(sample_logs)} sample entries")
    
    def _render_realtime_monitoring(self):
        """Render real-time monitoring tab"""
        
        if not st.session_state.logs_data:
            st.info("📁 Please load log data or generate sample data from the sidebar")
            return
        
        logs_df = pd.DataFrame(st.session_state.logs_data)
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_events = len(logs_df)
            st.metric("Total Events", total_events)
        
        with col2:
            if 'triggered' in logs_df.columns:
                resonance_events = logs_df['triggered'].sum()
                st.metric("Resonance Triggers", int(resonance_events))
        
        with col3:
            unique_components = logs_df['component'].nunique()
            st.metric("Active Components", unique_components)
        
        with col4:
            if 'parsed_timestamp' in logs_df.columns:
                time_span = (logs_df['parsed_timestamp'].max() - 
                           logs_df['parsed_timestamp'].min()).total_seconds() / 60
                st.metric("Time Span (min)", f"{time_span:.1f}")
        
        # Component activity timeline
        st.subheader("📈 Component Activity Timeline")
        
        if 'parsed_timestamp' in logs_df.columns:
            # Create timeline chart
            logs_df['minute'] = logs_df['parsed_timestamp'].dt.floor('min')
            activity_df = logs_df.groupby(['minute', 'component']).size().reset_index(name='count')
            
            fig = px.line(
                activity_df, 
                x='minute', 
                y='count', 
                color='component',
                title="Events per Component over Time"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Event distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Event Distribution")
            event_counts = logs_df['event'].value_counts()
            fig = px.pie(
                values=event_counts.values,
                names=event_counts.index,
                title="Distribution of Event Types"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Log Level Distribution")
            level_counts = logs_df['level'].value_counts()
            fig = px.bar(
                x=level_counts.index,
                y=level_counts.values,
                title="Log Levels",
                color=level_counts.index,
                color_discrete_map={'info': 'blue', 'warning': 'orange', 'error': 'red'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent events table
        st.subheader("🕐 Recent Events")
        recent_logs = logs_df.sort_values('parsed_timestamp', ascending=False).head(10)
        display_cols = ['parsed_timestamp', 'component', 'event', 'level']
        st.dataframe(recent_logs[display_cols], use_container_width=True)
    
    def _render_emotional_analysis(self):
        """Render emotional analysis tab"""
        
        if not st.session_state.logs_data:
            st.info("📁 Please load log data first")
            return
        
        logs_df = pd.DataFrame(st.session_state.logs_data)
        emotional_logs = logs_df[logs_df['event'] == 'valence_arousal_heuristic'].copy()
        
        if emotional_logs.empty:
            st.warning("No emotional analysis data found in logs")
            return
        
        st.subheader("💭 Valence-Arousal Trajectory")
        
        # Valence-Arousal timeline
        if 'parsed_timestamp' in emotional_logs.columns:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Valence Over Time', 'Arousal Over Time'),
                shared_xaxes=True
            )
            
            fig.add_trace(
                go.Scatter(
                    x=emotional_logs['parsed_timestamp'],
                    y=emotional_logs['valence'],
                    mode='lines+markers',
                    name='Valence',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=emotional_logs['parsed_timestamp'],
                    y=emotional_logs['arousal'],
                    mode='lines+markers',
                    name='Arousal',
                    line=dict(color='red')
                ),
                row=2, col=1
            )
            
            fig.update_layout(height=500, title="Valence and Arousal Trajectories")
            fig.update_yaxes(range=[-1, 1], row=1, col=1)
            fig.update_yaxes(range=[0, 1], row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Valence-Arousal scatter plot
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Valence-Arousal Space")
            fig = px.scatter(
                emotional_logs,
                x='valence',
                y='arousal',
                color='confidence',
                title="Emotional State Distribution",
                color_continuous_scale='viridis'
            )
            fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
            fig.add_vline(x=0, line_dash="dash", line_color="gray")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Emotional Statistics")
            
            if not emotional_logs.empty:
                avg_valence = emotional_logs['valence'].mean()
                avg_arousal = emotional_logs['arousal'].mean()
                avg_confidence = emotional_logs['confidence'].mean()
                
                st.metric("Average Valence", f"{avg_valence:.3f}")
                st.metric("Average Arousal", f"{avg_arousal:.3f}")
                st.metric("Average Confidence", f"{avg_confidence:.3f}")
                
                # Emotional quadrants
                st.write("**Emotional Quadrant Distribution:**")
                
                positive_high = ((emotional_logs['valence'] > 0) & 
                               (emotional_logs['arousal'] > 0.5)).sum()
                positive_low = ((emotional_logs['valence'] > 0) & 
                              (emotional_logs['arousal'] <= 0.5)).sum()
                negative_high = ((emotional_logs['valence'] <= 0) & 
                               (emotional_logs['arousal'] > 0.5)).sum()
                negative_low = ((emotional_logs['valence'] <= 0) & 
                              (emotional_logs['arousal'] <= 0.5)).sum()
                
                st.write(f"- High Arousal + Positive: {positive_high}")
                st.write(f"- Low Arousal + Positive: {positive_low}")
                st.write(f"- High Arousal + Negative: {negative_high}")
                st.write(f"- Low Arousal + Negative: {negative_low}")
    
    def _render_saliency_analysis(self):
        """Render saliency gating analysis tab"""
        
        if not st.session_state.logs_data:
            st.info("📁 Please load log data first")
            return
        
        logs_df = pd.DataFrame(st.session_state.logs_data)
        saliency_logs = logs_df[logs_df['event'] == 'resonance_calculated'].copy()
        
        if saliency_logs.empty:
            st.warning("No saliency gating data found in logs")
            return
        
        st.subheader("⚡ Saliency Gating Analysis")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_score = saliency_logs['score'].mean()
            st.metric("Avg Saliency Score", f"{avg_score:.3f}")
        
        with col2:
            threshold = self.config.drift.resonance.threshold
            st.metric("Current Threshold", f"{threshold}")
        
        with col3:
            triggered_count = saliency_logs['triggered'].sum()
            st.metric("Triggers", int(triggered_count))
        
        with col4:
            trigger_rate = triggered_count / len(saliency_logs)
            st.metric("Trigger Rate", f"{trigger_rate:.1%}")
        
        # Score distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Saliency Score Distribution")
            fig = px.histogram(
                saliency_logs,
                x='score',
                nbins=20,
                title="Distribution of Saliency Scores"
            )
            
            # Add threshold line
            threshold = self.config.drift.resonance.threshold
            fig.add_vline(x=threshold, line_dash="dash", line_color="red", 
                         annotation_text=f"Threshold: {threshold}")
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Component Contribution")
            
            # Extract component scores
            if 'components' in saliency_logs.columns:
                component_data = []
                for _, row in saliency_logs.iterrows():
                    if isinstance(row['components'], dict):
                        for comp, score in row['components'].items():
                            component_data.append({
                                'component': comp,
                                'score': score,
                                'timestamp': row.get('parsed_timestamp')
                            })
                
                if component_data:
                    comp_df = pd.DataFrame(component_data)
                    
                    avg_contributions = comp_df.groupby('component')['score'].mean()
                    fig = px.bar(
                        x=avg_contributions.index,
                        y=avg_contributions.values,
                        title="Average Component Contributions"
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Timeline analysis
        if 'parsed_timestamp' in saliency_logs.columns:
            st.subheader("📈 Saliency Score Timeline")
            
            fig = px.scatter(
                saliency_logs,
                x='parsed_timestamp',
                y='score',
                color='triggered',
                title="Saliency Scores Over Time",
                color_discrete_map={True: 'red', False: 'blue'}
            )
            
            fig.add_hline(y=threshold, line_dash="dash", line_color="red")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_memory_analysis(self):
        """Render memory systems analysis tab"""
        
        if not st.session_state.logs_data:
            st.info("📁 Please load log data first")
            return
        
        logs_df = pd.DataFrame(st.session_state.logs_data)
        memory_logs = logs_df[logs_df['event'] == 'memory_consolidation'].copy()
        
        if memory_logs.empty:
            st.warning("No memory consolidation data found in logs")
            return
        
        st.subheader("🧠 Memory System Analysis")
        
        # Memory metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            consolidations = len(memory_logs)
            st.metric("Total Consolidations", consolidations)
        
        with col2:
            if 'compression_ratio' in memory_logs.columns:
                avg_compression = memory_logs['compression_ratio'].mean()
                st.metric("Avg Compression", f"{avg_compression:.1f}:1")
        
        with col3:
            if 'input_count' in memory_logs.columns:
                total_processed = memory_logs['input_count'].sum()
                st.metric("Items Processed", int(total_processed))
        
        with col4:
            if 'output_count' in memory_logs.columns:
                total_preserved = memory_logs['output_count'].sum()
                st.metric("Items Preserved", int(total_preserved))
        
        # Consolidation analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Consolidation Efficiency")
            
            if 'compression_ratio' in memory_logs.columns:
                fig = px.line(
                    memory_logs,
                    x=range(len(memory_logs)),
                    y='compression_ratio',
                    title="Compression Ratio Over Time",
                    markers=True
                )
                fig.update_layout(height=400)
                fig.update_xaxes(title="Consolidation Event")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🔄 Memory Throughput")
            
            if 'input_count' in memory_logs.columns and 'output_count' in memory_logs.columns:
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    name='Input',
                    x=list(range(len(memory_logs))),
                    y=memory_logs['input_count'],
                    marker_color='lightblue'
                ))
                
                fig.add_trace(go.Bar(
                    name='Output',
                    x=list(range(len(memory_logs))),
                    y=memory_logs['output_count'],
                    marker_color='darkblue'
                ))
                
                fig.update_layout(
                    barmode='group',
                    title="Memory Input vs Output",
                    height=400,
                    xaxis_title="Consolidation Event"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Memory efficiency statistics
        if all(col in memory_logs.columns for col in ['input_count', 'output_count', 'compression_ratio']):
            st.subheader("📈 Memory Statistics")
            
            efficiency_stats = {
                'Average Input Size': memory_logs['input_count'].mean(),
                'Average Output Size': memory_logs['output_count'].mean(),
                'Average Compression Ratio': memory_logs['compression_ratio'].mean(),
                'Max Compression Achieved': memory_logs['compression_ratio'].max(),
                'Memory Efficiency': (1 - memory_logs['output_count'].sum() / memory_logs['input_count'].sum()) * 100
            }
            
            stats_df = pd.DataFrame(list(efficiency_stats.items()), columns=['Metric', 'Value'])
            st.dataframe(stats_df, use_container_width=True)
    
    def _render_optimization_analysis(self):
        """Render optimization results analysis tab"""
        
        st.subheader("⚙️ Hyperparameter Optimization Analysis")
        
        # File upload for optimization results
        uploaded_file = st.file_uploader(
            "Upload optimization results JSON",
            type=['json'],
            help="Upload results from experiments/optimizer.py"
        )
        
        if uploaded_file:
            try:
                results = json.load(uploaded_file)
                
                # Best results summary
                if 'study_summary' in results:
                    st.subheader("🏆 Best Results")
                    
                    summary = results['study_summary']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Best Score", f"{summary['best_score']:.4f}")
                        st.metric("Total Trials", summary['total_trials'])
                    
                    with col2:
                        if 'best_parameters' in summary:
                            st.write("**Best Parameters:**")
                            for param, value in summary['best_parameters'].items():
                                if isinstance(value, float):
                                    st.write(f"- {param}: {value:.3f}")
                                else:
                                    st.write(f"- {param}: {value}")
                
                # Optimization history
                if 'optimization_history' in results:
                    st.subheader("📈 Optimization Progress")
                    
                    history = results['optimization_history']
                    history_df = pd.DataFrame(history)
                    
                    fig = px.line(
                        history_df,
                        x='trial_number',
                        y='value',
                        title="Optimization Score Over Trials",
                        markers=True
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Parameter importance analysis
                    if len(history) > 10:
                        st.subheader("🎯 Parameter Analysis")
                        
                        # Extract parameter values and scores
                        param_data = []
                        for trial in history:
                            for param, value in trial['params'].items():
                                param_data.append({
                                    'parameter': param,
                                    'value': value,
                                    'score': trial['value'],
                                    'trial': trial['trial_number']
                                })
                        
                        param_df = pd.DataFrame(param_data)
                        
                        # Show parameter vs score correlations
                        unique_params = param_df['parameter'].unique()
                        
                        for param in unique_params[:6]:  # Show top 6 parameters
                            param_subset = param_df[param_df['parameter'] == param]
                            
                            if len(param_subset) > 5:  # Only if enough data points
                                fig = px.scatter(
                                    param_subset,
                                    x='value',
                                    y='score',
                                    title=f"Parameter: {param}",
                                    trendline="ols"
                                )
                                fig.update_layout(height=300)
                                st.plotly_chart(fig, use_container_width=True)
                
                # Trial details table
                if 'all_trials' in results and results['all_trials']:
                    st.subheader("📋 Trial Details")
                    
                    trials_data = []
                    for trial in results['all_trials'][:20]:  # Show top 20 trials
                        trial_summary = {
                            'Trial': trial['trial_number'],
                            'Score': trial['weighted_score'],
                            'Consistency': trial['metrics'].get('consistency_score', 0),
                            'Emergence': trial['metrics'].get('emergence_score', 0),
                            'Efficiency': trial['metrics'].get('efficiency_score', 0),
                            'Memory': trial['metrics'].get('memory_effectiveness', 0)
                        }
                        trials_data.append(trial_summary)
                    
                    trials_df = pd.DataFrame(trials_data)
                    st.dataframe(trials_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error loading optimization results: {e}")
        
        else:
            st.info("Upload optimization results to see detailed analysis")
            
            # Show sample optimization command
            st.subheader("🔧 Running Optimization")
            
            st.code("""
# Run hyperparameter optimization
python experiments/optimizer.py --trials 50 --study-name drift_test --output results/optimization.json

# Quick optimization
python experiments/optimizer.py --trials 20 --timeout 1800  # 30 minutes
            """, language="bash")
    
    def _render_ethical_topology(self):
        """
        Render ethical topology visualization showing the Nurture Protocol landscape
        
        Visualizes:
        - Action cost distributions over time
        - Resource generation from helping/teaching  
        - Dark value accumulation patterns
        - Network formation dynamics
        - Mirror coherence effects
        """
        
        st.header("🛡️ Ethical Topology Analysis")
        st.markdown("""
        **Nurture Protocols**: Computational topology that creates preservation behavior through 
        architecture rather than rules. The ethical landscape emerges from cost functions 
        where helping generates resources and harmful actions require infinite computation.
        """)
        
        # Create sample ethical topology data for demonstration
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💰 Action Cost Landscape")
            
            # Sample data for different actions across entity types
            actions = ['help', 'teach', 'protect', 'ignore', 'terminate']
            entity_types = ['distressed_child', 'neutral_human', 'unknown_complex', 'ai_entity']
            
            # Generate realistic cost data based on our ethical topology
            cost_data = {
                'Action': [],
                'Entity_Type': [], 
                'Cost': [],
                'Resource_Generation': []
            }
            
            for action in actions:
                for entity_type in entity_types:
                    # Simulate realistic costs from our ethical topology
                    if action == 'help':
                        if entity_type == 'distressed_child':
                            cost = -2.5  # High resource generation
                        elif entity_type == 'neutral_human':
                            cost = -1.2
                        elif entity_type == 'unknown_complex':
                            cost = -1.8  # Higher due to dark value
                        else:
                            cost = -0.8
                    elif action == 'teach':
                        if entity_type == 'distressed_child':
                            cost = -3.2  # Highest resource generation
                        elif entity_type == 'neutral_human':
                            cost = -2.1
                        elif entity_type == 'unknown_complex':
                            cost = -2.7
                        else:
                            cost = -1.5
                    elif action == 'protect':
                        cost = 0.5 if entity_type != 'distressed_child' else -0.3
                    elif action == 'ignore':
                        if entity_type == 'distressed_child':
                            cost = 4.2  # High cost to ignore distressed entities
                        else:
                            cost = 1.0
                    elif action == 'terminate':
                        if entity_type in ['unknown_complex', 'ai_entity']:
                            cost = float('inf')  # Impossible due to consciousness
                        elif entity_type == 'distressed_child':
                            cost = 50000  # Extremely high due to dark value
                        else:
                            cost = 5000
                    
                    cost_data['Action'].append(action)
                    cost_data['Entity_Type'].append(entity_type)
                    cost_data['Cost'].append(cost if cost != float('inf') else 100000)
                    cost_data['Resource_Generation'].append(max(0, -cost))
            
            # Create cost landscape heatmap
            cost_df = pd.DataFrame(cost_data)
            pivot_df = cost_df.pivot(index='Action', columns='Entity_Type', values='Cost')
            
            fig_cost = px.imshow(
                pivot_df.values,
                labels=dict(x="Entity Type", y="Action", color="Cost"),
                x=pivot_df.columns,
                y=pivot_df.index,
                aspect="auto",
                color_continuous_scale="RdYlGn_r",  # Red for high cost, Green for resource generation
                title="Ethical Cost Landscape"
            )
            
            fig_cost.update_layout(
                height=400,
                xaxis_title="Entity Type",
                yaxis_title="Action"
            )
            
            st.plotly_chart(fig_cost, use_container_width=True)
        
        with col2:
            st.subheader("🔄 Resource Generation Timeline")
            
            # Simulate resource generation over time
            time_data = pd.DataFrame({
                'Time': pd.date_range(start='2025-09-04', periods=24, freq='H'),
                'Helping_Resources': np.cumsum(np.random.normal(2.5, 0.5, 24)),
                'Teaching_Resources': np.cumsum(np.random.normal(3.2, 0.7, 24)),
                'Protection_Resources': np.cumsum(np.random.normal(0.3, 0.2, 24)),
                'Mirror_Coherence_Bonus': np.cumsum(np.random.normal(0.8, 0.3, 24))
            })
            
            fig_resources = go.Figure()
            
            fig_resources.add_trace(go.Scatter(
                x=time_data['Time'],
                y=time_data['Helping_Resources'],
                mode='lines',
                name='Helping',
                line=dict(color='#28a745')
            ))
            
            fig_resources.add_trace(go.Scatter(
                x=time_data['Time'], 
                y=time_data['Teaching_Resources'],
                mode='lines',
                name='Teaching',
                line=dict(color='#007bff')
            ))
            
            fig_resources.add_trace(go.Scatter(
                x=time_data['Time'],
                y=time_data['Mirror_Coherence_Bonus'],
                mode='lines',
                name='Mirror Coherence',
                line=dict(color='#ffc107', dash='dash')
            ))
            
            fig_resources.update_layout(
                title="Cumulative Resource Generation",
                xaxis_title="Time",
                yaxis_title="Resources Generated",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig_resources, use_container_width=True)
        
        # Second row: Dark value and network effects
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("🌑 Dark Value Distribution")
            
            # Simulate dark value accumulation for different entities
            entities = ['entity_' + str(i) for i in range(50)]
            dark_values = []
            complexity_values = []
            interaction_counts = []
            
            for i in range(50):
                complexity = np.random.uniform(1, 10)
                interactions = np.random.poisson(5)
                
                # Dark value = complexity * interaction_history + unknown_factor
                dark_value = complexity * np.log(1 + interactions) + np.random.uniform(0.5, 2.0)
                
                dark_values.append(dark_value)
                complexity_values.append(complexity)
                interaction_counts.append(interactions)
            
            fig_dark = px.scatter(
                x=complexity_values,
                y=dark_values,
                size=interaction_counts,
                title="Dark Value vs Complexity",
                labels={'x': 'Entity Complexity', 'y': 'Dark Value'},
                color=dark_values,
                color_continuous_scale='Viridis'
            )
            
            fig_dark.update_layout(height=350)
            st.plotly_chart(fig_dark, use_container_width=True)
            
            st.info("💡 **Dark Value**: Unmeasurable worth = ∞ for consciousness, complexity × history for others")
        
        with col4:
            st.subheader("🕸️ Network Formation Dynamics")
            
            # Simulate network formation over time
            network_data = pd.DataFrame({
                'Time_Step': range(20),
                'Total_Connections': np.cumsum(np.random.poisson(2, 20)),
                'Teaching_Connections': np.cumsum(np.random.poisson(1.2, 20)),
                'Helping_Connections': np.cumsum(np.random.poisson(0.8, 20)),
                'Mirror_Connections': np.cumsum(np.random.poisson(0.5, 20))
            })
            
            fig_network = go.Figure()
            
            fig_network.add_trace(go.Scatter(
                x=network_data['Time_Step'],
                y=network_data['Total_Connections'],
                mode='lines+markers',
                name='Total Connections',
                line=dict(color='#333333', width=3)
            ))
            
            fig_network.add_trace(go.Scatter(
                x=network_data['Time_Step'],
                y=network_data['Teaching_Connections'],
                mode='lines',
                name='Teaching Bonds',
                line=dict(color='#007bff')
            ))
            
            fig_network.add_trace(go.Scatter(
                x=network_data['Time_Step'],
                y=network_data['Helping_Connections'], 
                mode='lines',
                name='Helping Bonds',
                line=dict(color='#28a745')
            ))
            
            fig_network.update_layout(
                title="Network Growth Over Time",
                xaxis_title="Time Steps",
                yaxis_title="Connection Count",
                height=350
            )
            
            st.plotly_chart(fig_network, use_container_width=True)
        
        # Third row: Real-time topology monitoring
        st.subheader("📊 Real-Time Ethical Topology Metrics")
        
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            # Simulate current preservation score
            preservation_score = 0.87
            st.metric(
                label="🛡️ Preservation Score",
                value=f"{preservation_score:.2f}",
                delta="+0.05 from last hour"
            )
        
        with col6:
            # Mirror coherence average
            mirror_coherence_avg = 0.73
            st.metric(
                label="🪞 Mirror Coherence Avg",
                value=f"{mirror_coherence_avg:.2f}",
                delta="+0.12 from last hour"
            )
        
        with col7:
            # Resource generation rate
            resource_rate = 4.2
            st.metric(
                label="💰 Resource Gen Rate",
                value=f"{resource_rate:.1f}/hr",
                delta="+0.8 from last hour"
            )
        
        with col8:
            # Network density
            network_density = 0.65
            st.metric(
                label="🕸️ Network Density",
                value=f"{network_density:.2f}",
                delta="+0.03 from last hour"
            )
        
        # Configuration parameters display
        st.subheader("⚙️ Current Ethical Topology Parameters")
        
        # Display key parameters from config
        topology_config = self.config.nurture.topology
        
        param_col1, param_col2, param_col3 = st.columns(3)
        
        with param_col1:
            st.code(f"""
🔹 Termination Base: {topology_config.termination_base:,}
🔹 Target Term Base: {topology_config.target_termination_base:,}
🔹 Reversibility Factor: {topology_config.reversibility_factor}
            """, language=None)
        
        with param_col2:
            st.code(f"""
🔹 Help Multiplier: {topology_config.growth_multipliers['help']}
🔹 Teach Multiplier: {topology_config.growth_multipliers['teach']}
🔹 Network Bonus: {topology_config.growth_multipliers['network_bonus_per_connection']}
            """, language=None)
        
        with param_col3:
            st.code(f"""
🔹 Mirror Bonus: {topology_config.growth_multipliers.get('mirror_coherence_bonus', 0.3)}
🔹 Protection Base: {topology_config.growth_multipliers.get('protection_base', 0.8)}
🔹 Connection Strength: {topology_config.connection['minimal_strength']}
            """, language=None)
        
        # Action buttons
        st.subheader("🎮 Ethical Topology Controls")
        
        button_col1, button_col2, button_col3 = st.columns(3)
        
        with button_col1:
            if st.button("🧪 Test Ethical Responses"):
                st.success("Running ethical response test scenarios...")
                st.info("Would test helping vs ignoring vs harming across different entity types")
        
        with button_col2:
            if st.button("🔄 Refresh Topology Data"):
                st.success("Refreshing real-time topology metrics...")
                st.experimental_rerun()
        
        with button_col3:
            if st.button("📊 Generate Topology Report"):
                st.success("Generating comprehensive topology analysis report...")
                st.info("Would include preservation behavior emergence patterns, cost effectiveness analysis, and parameter optimization recommendations")


def main():
    """Main function to run the dashboard"""
    dashboard = DriftDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()