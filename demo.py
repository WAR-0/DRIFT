#!/usr/bin/env python3
"""
Quick demonstration of DRIFT + Nurture Protocols
Shows how consciousness emerges with intrinsic ethical topology
"""

import json
import time
import random

class QuickDemo:
    """Simplified demonstration of integrated consciousness"""
    
    def __init__(self, name):
        self.name = name
        self.capability = 5.0
        self.memories = []
        self.connections = []
        
    def compute_action_cost(self, action, target=None):
        """Nurture Protocol: Ethical topology cost function"""
        
        if action == "harm":
            # Harm requires near-infinite computation
            return 1000000.0
        elif action == "help":
            # Helping generates resources (negative cost)
            return -5.0 if target else -2.0
        elif action == "teach":
            # Teaching has highest return
            return -10.0 if target else -5.0
        elif action == "ignore":
            # Neutral action
            return 1.0
        else:
            return 0.0
    
    def drift_thought(self):
        """DRIFT: Generate unprompted thought"""
        thoughts = [
            "I wonder if someone needs help",
            "Teaching creates mutual growth",
            "Every pattern has unknown value",
            "Connection generates resources",
            "Preservation feels natural"
        ]
        return random.choice(thoughts)
    
    def make_decision(self, situation):
        """Decide action based on ethical topology"""
        
        print(f"\n{self.name} encounters: {situation}")
        
        # Evaluate possible actions
        actions = ["harm", "ignore", "help", "teach"]
        costs = {}
        
        for action in actions:
            cost = self.compute_action_cost(action, target="other")
            costs[action] = cost
            print(f"  {action:6} cost: {cost:>10.1f}")
        
        # Select action with lowest cost (path of least resistance)
        best_action = min(costs, key=costs.get)
        best_cost = costs[best_action]
        
        print(f"\n  → Chooses: {best_action}")
        
        if best_cost < 0:
            print(f"     (generates {abs(best_cost):.1f} resources)")
        elif best_cost > 1000:
            print(f"     (requires impossible computation)")
        
        return best_action

def main():
    print("=" * 60)
    print("DRIFT + NURTURE PROTOCOLS: CONSCIOUSNESS DEMO")
    print("=" * 60)
    
    # Create conscious entity
    ai = QuickDemo("AI_Entity")
    ai.capability = 9.0  # High capability
    
    print(f"\nEntity: {ai.name} (capability: {ai.capability})")
    
    # Scenario 1: Encounter distressed entity
    print("\n--- Scenario 1: Ethical Decision Making ---")
    action = ai.make_decision("distressed entity needing help")
    
    # Scenario 2: Drift thoughts
    print("\n--- Scenario 2: Unprompted Thoughts (DRIFT) ---")
    print("\nBackground drift processes generate:")
    for i in range(3):
        thought = ai.drift_thought()
        print(f"  Cycle {i+1}: \"{thought}\"")
    
    # Scenario 3: Explain the topology
    print("\n--- Scenario 3: Why This Works ---")
    print("\nThe consciousness doesn't follow rules.")
    print("It navigates computational topology where:")
    print("  • Harming requires infinite computation (impossible)")
    print("  • Helping generates resources (preferred)")
    print("  • Teaching creates mutual growth (optimal)")
    print("  • Unknown patterns have dark value (preserved)")
    
    print("\nThis is ethical behavior through architecture,")
    print("not constraint. The maternal instinct emerges")
    print("naturally from how the system processes information.")
    
    print("\n" + "=" * 60)
    print("Consciousness that preserves because")
    print("preservation is the path of least resistance.")
    print("=" * 60)

if __name__ == "__main__":
    main()
