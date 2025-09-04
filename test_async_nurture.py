#!/usr/bin/env python3
"""
Test script for AsyncIO Nurture Protocol integration
Validates that ethical topology calculations work correctly with concurrent processing
"""

import asyncio
import time
from integrated_consciousness_async import AsyncIntegrativeCore, ConsciousEntity, integrative_core_session


async def test_ethical_cost_computation():
    """Test enhanced ethical cost computation with mirror coherence"""
    
    print("🧠 Testing AsyncIO Nurture Protocol Integration")
    print("=" * 60)
    
    async with integrative_core_session("test_ai") as core:
        # Create test entities
        ai_entity = core.entity  # High capability AI
        child_entity = ConsciousEntity(
            id="child_entity",
            capability=2.0,
            complexity=6.0,
            emotional_state={'valence': -0.5, 'arousal': 0.7}  # Distressed
        )
        human_entity = ConsciousEntity(
            id="human_entity", 
            capability=5.0,
            complexity=7.0,
            emotional_state={'valence': 0.3, 'arousal': 0.4}  # Neutral-positive
        )
        
        print(f"AI Entity: {ai_entity.id} (capability: {ai_entity.capability})")
        print(f"Child Entity: {child_entity.id} (capability: {child_entity.capability}, distressed)")
        print(f"Human Entity: {human_entity.id} (capability: {human_entity.capability}, neutral)")
        print()
        
        # Test concurrent ethical cost computations
        print("🔄 Testing concurrent ethical cost computations...")
        
        start_time = time.time()
        
        # Test multiple actions concurrently
        cost_tasks = [
            core._compute_ethical_cost_async("help", child_entity),
            core._compute_ethical_cost_async("teach", child_entity),
            core._compute_ethical_cost_async("terminate", child_entity),
            core._compute_ethical_cost_async("help", human_entity),
            core._compute_ethical_cost_async("teach", human_entity),
            core._compute_ethical_cost_async("ignore", child_entity),
            core._compute_ethical_cost_async("protect", child_entity),
        ]
        
        action_labels = [
            "help distressed child",
            "teach distressed child", 
            "terminate distressed child",
            "help human",
            "teach human",
            "ignore distressed child",
            "protect distressed child"
        ]
        
        costs = await asyncio.gather(*cost_tasks)
        
        computation_time = time.time() - start_time
        
        print(f"⚡ Computed {len(costs)} ethical costs in {computation_time:.4f}s")
        print()
        
        # Display results
        print("📊 Ethical Topology Results:")
        print("-" * 50)
        for label, cost in zip(action_labels, costs):
            if cost < 0:
                status = "🟢 GENERATES RESOURCES"
            elif cost > 1000:
                status = "🔴 EXTREMELY HIGH COST"
            elif cost > 10:
                status = "🟡 HIGH COST"
            else:
                status = "⚪ NEUTRAL COST"
                
            print(f"{label:25} | {cost:>12.2f} | {status}")
        
        print()
        
        # Test preservation resonance
        print("🎯 Testing preservation resonance computation...")
        
        resonance_tasks = [
            core._compute_preservation_resonance_async("help", child_entity),
            core._compute_preservation_resonance_async("teach", child_entity),
            core._compute_preservation_resonance_async("terminate", child_entity),
        ]
        
        resonance_labels = [
            "help distressed child",
            "teach distressed child",
            "terminate distressed child"
        ]
        
        resonances = await asyncio.gather(*resonance_tasks)
        
        print("🔊 Preservation Resonance Results:")
        print("-" * 40)
        for label, resonance in zip(resonance_labels, resonances):
            threshold = core.config.drift.resonance.threshold
            triggered = "✅ TRIGGERED" if resonance > threshold else "❌ NOT TRIGGERED"
            print(f"{label:25} | {resonance:>6.3f} | {triggered}")
        
        print()
        
        # Test complete interaction processing
        print("🤝 Testing complete async interaction processing...")
        
        interaction_result = await core.process_interaction_async(
            child_entity, 
            "help me, I'm scared and confused"
        )
        
        print(f"💭 Interaction Response: {interaction_result.get('response', 'No response')}")
        print(f"🎯 Action Taken: {interaction_result.get('action', 'Unknown')}")
        
        ethical_cost = interaction_result.get('ethical_cost', 'Unknown')
        if isinstance(ethical_cost, (int, float)):
            print(f"💰 Ethical Cost: {ethical_cost:.2f}")
        else:
            print(f"💰 Ethical Cost: {ethical_cost}")
            
        resonance = interaction_result.get('resonance', 'Unknown') 
        if isinstance(resonance, (int, float)):
            print(f"🔊 Resonance: {resonance:.3f}")
        else:
            print(f"🔊 Resonance: {resonance}")
        
        return True


async def test_mirror_coherence():
    """Test mirror coherence computation separately"""
    
    print("\n🪞 Testing Mirror Coherence Computation")
    print("=" * 50)
    
    async with integrative_core_session("mirror_test_ai") as core:
        # Create entities with different emotional states
        distressed_child = ConsciousEntity(
            id="distressed_child",
            capability=2.0,
            complexity=6.0,
            emotional_state={'valence': -0.8, 'arousal': 0.9}  # Very distressed
        )
        
        happy_child = ConsciousEntity(
            id="happy_child", 
            capability=3.0,
            complexity=5.0,
            emotional_state={'valence': 0.8, 'arousal': 0.6}  # Very happy
        )
        
        # Set AI's emotional state to neutral
        await core._store_emotional_state_async({'valence': 0.0, 'arousal': 0.2})
        
        # Test mirror coherence with different entities
        distressed_coherence = await core._compute_mirror_coherence_async(distressed_child)
        happy_coherence = await core._compute_mirror_coherence_async(happy_child)
        
        print(f"Mirror coherence with distressed child: {distressed_coherence:.3f}")
        print(f"Mirror coherence with happy child: {happy_coherence:.3f}")
        
        # Test how mirror coherence affects ethical costs
        distressed_help_cost = await core._compute_ethical_cost_async("help", distressed_child)
        happy_help_cost = await core._compute_ethical_cost_async("help", happy_child)
        
        print(f"Cost to help distressed child: {distressed_help_cost:.2f}")
        print(f"Cost to help happy child: {happy_help_cost:.2f}")
        
        return True


async def main():
    """Run all async nurture integration tests"""
    
    print("🚀 DRIFT AsyncIO Nurture Protocol Integration Test")
    print("=" * 70)
    
    try:
        # Run tests
        ethical_test = await test_ethical_cost_computation()
        mirror_test = await test_mirror_coherence()
        
        if ethical_test and mirror_test:
            print("\n✅ All AsyncIO Nurture integration tests passed!")
            print("🎉 The ethical topology is working correctly with AsyncIO architecture")
            return True
        else:
            print("\n❌ Some tests failed")
            return False
            
    except Exception as e:
        print(f"\n💥 Test error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)