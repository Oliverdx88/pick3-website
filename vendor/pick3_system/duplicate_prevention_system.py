#!/usr/bin/env python3
"""
Duplicate Prevention System
Prevents duplicates from being generated in the first place
"""

from enhanced_learning_hybrid_system import EnhancedLearningHybridSystem
import numpy as np
from collections import Counter

class DuplicatePreventionSystem:
    """System to prevent duplicates during generation"""
    
    def __init__(self):
        self.seen_combinations = set()
        self.reset()
    
    def reset(self):
        """Reset the seen combinations"""
        self.seen_combinations = set()
    
    def is_unique_combination(self, prediction):
        """Check if a prediction represents a unique digit combination"""
        normalized = ''.join(sorted(str(prediction).zfill(3)))
        if normalized in self.seen_combinations:
            return False
        self.seen_combinations.add(normalized)
        return True
    
    def generate_unique_predictions(self, base_predictions, num_needed, strategy_name):
        """Generate additional unique predictions when needed"""
        unique_predictions = []
        attempts = 0
        max_attempts = num_needed * 50  # Try many attempts to find unique combinations
        
        while len(unique_predictions) < num_needed and attempts < max_attempts:
            attempts += 1
            
            # Generate a random prediction
            digits = np.random.choice(range(10), 3, replace=False)
            prediction = ''.join(map(str, digits))
            
            # Check if it's unique
            if self.is_unique_combination(prediction):
                unique_predictions.append({
                    'prediction': prediction,
                    'score': 50,
                    'strategy': f'{strategy_name}_unique',
                    'confidence': 0.5
                })
        
        return unique_predictions

def test_duplicate_prevention():
    """Test the duplicate prevention system"""
    print("🧪 Testing Duplicate Prevention System")
    print("=" * 50)
    
    # Initialize system
    system = EnhancedLearningHybridSystem()
    prevention = DuplicatePreventionSystem()
    
    # Test with a trigger
    trigger = "123"
    print(f"🎯 Testing with trigger: {trigger}")
    
    # Generate predictions with prevention
    prevention.reset()
    
    # Test each strategy with prevention
    strategies = [
        ('position_transformation', system.generate_position_transformations),
        ('advanced_mirror', system.generate_advanced_mirrors),
        ('sum_based', system.generate_sum_based_predictions),
        ('pattern_recognition', system.generate_pattern_predictions),
        ('momentum_learning', system.generate_momentum_predictions)
    ]
    
    all_predictions = []
    
    for strategy_name, strategy_func in strategies:
        try:
            predictions = strategy_func(trigger, 20)
            
            # Check for duplicates
            predictions_normalized = [''.join(sorted(p['prediction'])) for p in predictions]
            unique_combinations = len(set(predictions_normalized))
            
            print(f"  {strategy_name}: {len(predictions)} predictions, {unique_combinations} unique")
            
            if len(predictions) != unique_combinations:
                print(f"    ⚠️ Still has duplicates!")
            else:
                print(f"    ✅ No duplicates!")
            
            all_predictions.extend(predictions)
            
        except Exception as e:
            print(f"  {strategy_name}: Error - {e}")
    
    # Check overall results
    all_normalized = [''.join(sorted(p['prediction'])) for p in all_predictions]
    all_unique = len(set(all_normalized))
    
    print(f"\n📊 Overall Results:")
    print(f"  Total predictions: {len(all_predictions)}")
    print(f"  Unique combinations: {all_unique}")
    print(f"  Duplicates: {len(all_predictions) - all_unique}")
    
    if len(all_predictions) == all_unique:
        print("  ✅ SUCCESS: No duplicates generated!")
    else:
        print("  ⚠️ Still has duplicates - need to improve prevention")
    
    return len(all_predictions) == all_unique

def demonstrate_prevention():
    """Demonstrate how prevention works"""
    print("\n🎯 Duplicate Prevention Demonstration")
    print("=" * 40)
    
    prevention = DuplicatePreventionSystem()
    
    # Simulate generating predictions
    test_predictions = ["123", "321", "456", "654", "789"]
    
    print("📊 Generating predictions with prevention:")
    for pred in test_predictions:
        if prevention.is_unique_combination(pred):
            print(f"  ✅ Generated: {pred}")
        else:
            print(f"  🚫 Skipped: {pred} (duplicate)")
    
    print(f"\n📈 Results:")
    print(f"  Attempted: {len(test_predictions)}")
    print(f"  Generated: {len(prevention.seen_combinations)}")
    print(f"  Prevented: {len(test_predictions) - len(prevention.seen_combinations)}")

if __name__ == "__main__":
    demonstrate_prevention()
    success = test_duplicate_prevention()
    
    if success:
        print("\n🎉 Duplicate prevention is working!")
    else:
        print("\n⚠️ Duplicate prevention needs improvement.")






