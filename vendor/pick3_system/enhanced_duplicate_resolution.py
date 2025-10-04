#!/usr/bin/env python3
"""
Enhanced duplicate resolution system
Considers multiple factors when deciding which version of a duplicate to keep
"""

from enhanced_learning_hybrid_system import EnhancedLearningHybridSystem

def calculate_prediction_quality(prediction, last_draw, strategy_weights):
    """Calculate overall quality score for a prediction"""
    score = 0
    
    # Base score from prediction
    base_score = prediction.get('score', 50)
    score += base_score * 0.4  # 40% weight to base score
    
    # Strategy weight
    strategy = prediction.get('strategy', 'unknown')
    strategy_weight = strategy_weights.get(strategy, 0.5)
    score += strategy_weight * 100 * 0.3  # 30% weight to strategy
    
    # Historical frequency (if available)
    if hasattr(prediction, 'frequency'):
        score += prediction.frequency * 0.2  # 20% weight to frequency
    
    # Pattern strength (if available)
    if hasattr(prediction, 'pattern_strength'):
        score += prediction.pattern_strength * 0.1  # 10% weight to pattern
    
    return score

def smart_duplicate_resolution(predictions, last_draw, strategy_weights):
    """Smart duplicate resolution that keeps the best version"""
    seen = {}
    unique_predictions = []
    resolved_count = 0
    
    for pred in predictions:
        prediction = str(pred['prediction']).zfill(3)
        normalized = ''.join(sorted(prediction))
        
        if normalized not in seen:
            seen[normalized] = pred
            unique_predictions.append(pred)
        else:
            # We have a duplicate - calculate quality scores
            existing_pred = seen[normalized]
            
            # Calculate quality scores
            existing_quality = calculate_prediction_quality(existing_pred, last_draw, strategy_weights)
            new_quality = calculate_prediction_quality(pred, last_draw, strategy_weights)
            
            # Keep the one with higher quality
            if new_quality > existing_quality:
                # Replace existing with better one
                seen[normalized] = pred
                # Update the list
                for i, p in enumerate(unique_predictions):
                    if p == existing_pred:
                        unique_predictions[i] = pred
                        break
                print(f"🔄 Replaced {existing_pred['prediction']} (quality: {existing_quality:.1f}) with {prediction} (quality: {new_quality:.1f})")
            else:
                print(f"🔄 Kept {existing_pred['prediction']} (quality: {existing_quality:.1f}) over {prediction} (quality: {new_quality:.1f})")
            
            resolved_count += 1
    
    if resolved_count > 0:
        print(f"✅ Smart resolved {resolved_count} duplicates, kept {len(unique_predictions)} best predictions")
    
    return unique_predictions

def test_smart_resolution():
    """Test the smart duplicate resolution system"""
    print("🧪 Testing Smart Duplicate Resolution")
    print("=" * 50)
    
    # Initialize system
    system = EnhancedLearningHybridSystem()
    
    # Create test predictions with duplicates
    test_predictions = [
        {'prediction': '502', 'score': 85, 'strategy': 'pattern_recognition'},
        {'prediction': '205', 'score': 92, 'strategy': 'position_transformation'},  # Better score
        {'prediction': '123', 'score': 78, 'strategy': 'sum_based'},
        {'prediction': '321', 'score': 75, 'strategy': 'momentum_learning'},  # Lower score
        {'prediction': '456', 'score': 88, 'strategy': 'advanced_mirror'},
        {'prediction': '654', 'score': 90, 'strategy': 'pattern_recognition'},  # Better score
    ]
    
    print("📊 Original predictions:")
    for i, pred in enumerate(test_predictions):
        print(f"  {i+1}. {pred['prediction']} (Score: {pred['score']}, Strategy: {pred['strategy']})")
    
    # Test smart resolution
    strategy_weights = {
        'pattern_recognition': 0.9,
        'position_transformation': 0.8,
        'advanced_mirror': 0.7,
        'sum_based': 0.6,
        'momentum_learning': 0.5
    }
    
    resolved_predictions = smart_duplicate_resolution(test_predictions, "123", strategy_weights)
    
    print(f"\n📊 Resolved predictions:")
    for i, pred in enumerate(resolved_predictions):
        print(f"  {i+1}. {pred['prediction']} (Score: {pred['score']}, Strategy: {pred['strategy']})")
    
    # Show which duplicates were resolved
    original_normalized = [''.join(sorted(p['prediction'])) for p in test_predictions]
    resolved_normalized = [''.join(sorted(p['prediction'])) for p in resolved_predictions]
    
    print(f"\n🔍 Resolution Summary:")
    print(f"Original: {len(test_predictions)} predictions")
    print(f"Resolved: {len(resolved_predictions)} predictions")
    print(f"Duplicates resolved: {len(test_predictions) - len(resolved_predictions)}")
    
    return resolved_predictions

if __name__ == "__main__":
    test_smart_resolution()






