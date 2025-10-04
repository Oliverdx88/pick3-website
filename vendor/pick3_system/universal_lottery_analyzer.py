#!/usr/bin/env python3
"""
BRAIN: ENHANCED LEARNING HYBRID SYSTEM
==================================

Advanced prediction system with manual input learning:
- Manual entry of last 2-4 drawing results
- Real-time learning from recent patterns
- Adaptive strategy weighting
- 90-98% accuracy target
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import json
import os
import warnings
warnings.filterwarnings('ignore')

class EnhancedLearningHybridSystem:
    def __init__(self, data_file='data_files/nj_pick3_history.csv'):
        self.data_file = data_file
        self.df = None
        self.patterns = {}
        self.quality_threshold = 0.005
        
        # Enhanced learning parameters
        self.strategy_performance = defaultdict(list)
        self.learning_rate = 0.15  # Increased learning rate
        self.memory_window = 20    # Shorter memory for faster adaptation
        self.adaptive_weights = {}
        
        # Manual input tracking
        self.recent_drawings = []  # Last 2-4 manual entries
        self.manual_learning_enabled = True
        self.manual_confidence_boost = 0.3  # 30% boost for manual input
        
        # Strategy definitions
        self.strategies = {
            'position_transformation': {'weight': 0.25, 'success_rate': 0.0},
            'advanced_mirror': {'weight': 0.25, 'success_rate': 0.0},
            'sum_based': {'weight': 0.20, 'success_rate': 0.0},
            'pattern_recognition': {'weight': 0.15, 'success_rate': 0.0},
            'momentum_learning': {'weight': 0.15, 'success_rate': 0.0}
        }
        
        # Load and analyze data
        self.load_data()
        self.analyze_enhanced_patterns()
        self.initialize_learning()
        
        # Analyze transition patterns
        self.analyze_transition_patterns()
        
        # Train transition model
        if self.df is not None and len(self.df) > 1:
            history = self.df['numbers'].tolist()
            self.fit_transitions(history)
            # Also train stream-specific models
            self.fit_transitions_by_stream()
        
        # Update learning from data files
        self.update_learning_from_data_files()
        
        # Update NJ UPDATED DATES.txt file
        self.update_nj_dates_file()
        
    def update_nj_dates_file(self):
        """Update the NJ UPDATED DATES.txt file with latest learning information"""
        try:
            # Get the latest learning data
            latest_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            learning_summary = self.get_learning_summary()
            
            # Create/update the NJ UPDATED DATES.txt file
            dates_file_path = 'DATA FILES/NJ UPDATED DATES.txt'
            
            # Read existing content
            existing_content = ""
            try:
                with open(dates_file_path, 'r', encoding='utf-8') as f:
                    existing_content = f.read()
            except FileNotFoundError:
                pass
            
            # Create learning summary section
            learning_section = f"""
=== NJ LEARNING SYSTEM UPDATE ===
Last Updated: {latest_date}
Total Draws Analyzed: {len(self.df) if self.df is not None else 0}
Learning Status: ACTIVE

Strategy Performance Summary:
{learning_summary}

Latest Learning Updates:
- Position Transformation: {self.strategies.get('position_transformation', {}).get('weight', 0):.3f}
- Advanced Mirror: {self.strategies.get('advanced_mirror', {}).get('weight', 0):.3f}
- Sum-based: {self.strategies.get('sum_based', {}).get('weight', 0):.3f}
- Pattern Recognition: {self.strategies.get('pattern_recognition', {}).get('weight', 0):.3f}
- Momentum Learning: {self.strategies.get('momentum_learning', {}).get('weight', 0):.3f}

Manual Learning Entries: {len(self.recent_drawings)}
Recent Manual Drawings: {', '.join(self.recent_drawings[-5:]) if self.recent_drawings else 'None'}

=== END LEARNING UPDATE ===

"""
            
            # Combine existing content with new learning section
            updated_content = learning_section + existing_content
            
            # Write back to file
            with open(dates_file_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
                
            print(f"SUCCESS: Updated NJ UPDATED DATES.txt with learning information")
            
        except Exception as e:
            print(f"WARNING:  Warning: Could not update NJ UPDATED DATES.txt: {e}")
    
    def get_learning_summary(self):
        """Get a summary of the current learning state"""
        summary = []
        
        for strategy, data in self.strategies.items():
            weight = data.get('weight', 0)
            success_rate = data.get('success_rate', 0)
            summary.append(f"- {strategy.replace('_', ' ').title()}: Weight={weight:.3f}, Success={success_rate:.1%}")
        
        return '\n'.join(summary)
        
    def load_data(self):
        """Load and prepare historical data"""
        print("Loading historical data...")
        try:
            self.df = pd.read_csv(self.data_file)
            self.df['draw_date'] = pd.to_datetime(self.df['draw_date'])
            self.df['numbers'] = self.df['numbers'].astype(str).str.zfill(3)
            
            # Add time-based features
            self.df['time_segment'] = self.df['draw_time'].map({'midday': 'midday', 'evening': 'evening'})
            self.df['day_of_week'] = self.df['draw_date'].dt.dayofweek
            self.df['month'] = self.df['draw_date'].dt.month
            
            print(f"SUCCESS: Loaded {len(self.df)} draws")
        except FileNotFoundError:
            print(f"WARNING: Data file '{self.data_file}' not found. Creating sample data...")
            # Create sample data if file doesn't exist
            sample_data = {
                'draw_date': pd.date_range('2024-01-01', periods=50, freq='D'),
                'draw_time': ['evening'] * 50,
                'numbers': ['123', '456', '789', '234', '567', '890', '345', '678', '901', '234',
                           '567', '890', '123', '456', '789', '234', '567', '890', '345', '678',
                           '901', '234', '567', '890', '123', '456', '789', '234', '567', '890',
                           '345', '678', '901', '234', '567', '890', '123', '456', '789', '234',
                           '567', '890', '345', '678', '901', '234', '567', '890', '123', '456']
            }
            self.df = pd.DataFrame(sample_data)
            self.df['numbers'] = self.df['numbers'].astype(str).str.zfill(3)
            self.df['time_segment'] = self.df['draw_time'].map({'midday': 'midday', 'evening': 'evening'})
            self.df['day_of_week'] = self.df['draw_date'].dt.dayofweek
            self.df['month'] = self.df['draw_date'].dt.month
            print(f"SUCCESS: Created sample data with {len(self.df)} draws")
        except Exception as e:
            print(f"ERROR: Error loading data: {e}")
            print("Creating minimal sample data...")
            # Create minimal sample data
            sample_data = {
                'draw_date': pd.date_range('2024-01-01', periods=20, freq='D'),
                'draw_time': ['evening'] * 20,
                'numbers': ['123', '456', '789', '234', '567', '890', '345', '678', '901', '234',
                           '567', '890', '123', '456', '789', '234', '567', '890', '345', '678']
            }
            self.df = pd.DataFrame(sample_data)
            self.df['numbers'] = self.df['numbers'].astype(str).str.zfill(3)
            self.df['time_segment'] = self.df['draw_time'].map({'midday': 'midday', 'evening': 'evening'})
            self.df['day_of_week'] = self.df['draw_date'].dt.dayofweek
            self.df['month'] = self.df['draw_date'].dt.month
            print(f"SUCCESS: Created minimal sample data with {len(self.df)} draws")
        
    def analyze_enhanced_patterns(self):
        """Analyze enhanced patterns with manual input consideration"""
        print("SEARCH: Analyzing enhanced patterns...")
        
        # 1. Position transformation patterns
        self.position_transformations = defaultdict(list)
        for i in range(len(self.df) - 1):
            current = self.df.iloc[i]['numbers']
            next_draw = self.df.iloc[i + 1]['numbers']
            
            # Track position transformations
            for pos in range(3):
                current_digit = int(current[pos])
                next_digit = int(next_draw[pos])
                transformation = (current_digit, next_digit)
                self.position_transformations[pos].append(transformation)
        
        # 2. Advanced mirror patterns
        self.mirror_patterns = defaultdict(list)
        for i in range(len(self.df) - 1):
            current = self.df.iloc[i]['numbers']
            next_draw = self.df.iloc[i + 1]['numbers']
            
            # Track mirror relationships
            for pos in range(3):
                current_digit = int(current[pos])
                next_digit = int(next_draw[pos])
                mirror_digit = 9 - current_digit
                
                if next_digit == mirror_digit:
                    self.mirror_patterns[pos].append((current_digit, next_digit))
        
        # 3. Sum-based patterns
        self.sum_patterns = defaultdict(list)
        for i in range(len(self.df) - 1):
            current_sum = sum(int(d) for d in self.df.iloc[i]['numbers'])
            next_draw = self.df.iloc[i + 1]['numbers']
            self.sum_patterns[current_sum].append(next_draw)
        
        # 4. Pattern recognition
        self.pattern_sequences = defaultdict(list)
        for i in range(len(self.df) - 2):
            seq = (self.df.iloc[i]['numbers'], self.df.iloc[i + 1]['numbers'])
            next_draw = self.df.iloc[i + 2]['numbers']
            self.pattern_sequences[seq].append(next_draw)
        
        print("SUCCESS: Enhanced pattern analysis complete")
        
    def initialize_learning(self):
        """Initialize learning system with default weights"""
        print("BRAIN: Initializing enhanced learning system...")
        
        # Initialize strategy weights
        for strategy in self.strategies.keys():
            self.adaptive_weights[strategy] = self.strategies[strategy]['weight']
            self.strategy_performance[strategy] = []
        
        print("SUCCESS: Learning system initialized")
        
    def add_manual_drawing(self, drawing_result):
        """Add manual drawing result to recent drawings"""
        if len(drawing_result) == 3 and drawing_result.isdigit():
            self.recent_drawings.append(drawing_result)
            
            # Keep only last 4 drawings
            if len(self.recent_drawings) > 4:
                self.recent_drawings = self.recent_drawings[-4:]
            
            print(f"SUCCESS: Added manual drawing: {drawing_result}")
            print(f"DATA: Recent drawings: {self.recent_drawings}")
            
            # Update learning with new data
            if len(self.recent_drawings) >= 2:
                self.update_learning_from_manual()
        else:
            print("ERROR: Invalid drawing format. Must be 3 digits.")
            
    def update_learning_from_manual(self):
        """Update learning system based on manual input"""
        print(" Updating learning from manual input...")
        
        if len(self.recent_drawings) < 2:
            return
            
        # Analyze patterns in recent manual drawings
        for i in range(len(self.recent_drawings) - 1):
            trigger = self.recent_drawings[i]
            result = self.recent_drawings[i + 1]
            
            # Test each strategy
            for strategy in self.strategies.keys():
                predictions = self.get_strategy_predictions(trigger, strategy, 10)
                predicted_numbers = [p['prediction'] for p in predictions]
                
                # Record hit or miss with manual confidence boost
                hit = 1 if result in predicted_numbers else 0
                self.strategy_performance[strategy].append(hit)
                
                # Keep only recent performance
                if len(self.strategy_performance[strategy]) > self.memory_window:
                    self.strategy_performance[strategy] = self.strategy_performance[strategy][-self.memory_window:]
                
                # Update adaptive weight with manual boost
                recent_performance = self.strategy_performance[strategy][-self.memory_window:]
                if recent_performance:
                    new_performance = np.mean(recent_performance)
                    current_weight = self.adaptive_weights.get(strategy, 0.5)
                    
                    # Enhanced learning with manual confidence boost
                    manual_boost = self.manual_confidence_boost if len(self.recent_drawings) >= 3 else 0.1
                    updated_weight = current_weight + (self.learning_rate + manual_boost) * (new_performance - current_weight)
                    self.adaptive_weights[strategy] = max(0.01, min(0.99, updated_weight))
        
        print("DATA: Updated Strategy Performance (with manual input):")
        for strategy, weight in self.adaptive_weights.items():
            recent_perf = self.strategy_performance[strategy][-10:]
            recent_avg = np.mean(recent_perf) if recent_perf else 0
            print(f"  {strategy}: {recent_avg:.3f} (weight: {weight:.3f})")
        
        # Update NJ UPDATED DATES.txt file after manual learning
        self.update_nj_dates_file()
            
    def get_strategy_predictions(self, last_draw, strategy, num_predictions):
        """Get predictions for a specific strategy"""
        predictions = []
        
        if strategy == 'position_transformation':
            predictions = self.generate_position_transformations(last_draw, num_predictions)
        elif strategy == 'advanced_mirror':
            predictions = self.generate_advanced_mirrors(last_draw, num_predictions)
        elif strategy == 'sum_based':
            predictions = self.generate_sum_based_predictions(last_draw, num_predictions)
        elif strategy == 'pattern_recognition':
            predictions = self.generate_pattern_predictions(last_draw, num_predictions)
        elif strategy == 'momentum_learning':
            predictions = self.generate_momentum_predictions(last_draw, num_predictions)
            
        return predictions
        
    def generate_position_transformations(self, last_draw, num_predictions):
        """Generate predictions using position transformations"""
        predictions = []
        
        for pos in range(3):
            current_digit = int(last_draw[pos])
            transformations = self.position_transformations[pos]
            
            if transformations:
                # Find most common transformations
                transform_counts = Counter(transformations)
                most_common = transform_counts.most_common(5)
                
                for (from_digit, to_digit), count in most_common:
                    if from_digit == current_digit:
                        # Generate prediction with this transformation
                        pred_digits = list(last_draw)
                        pred_digits[pos] = str(to_digit)
                        prediction = ''.join(pred_digits)
                        
                        success_rate = count / len(transformations)
                        predictions.append({
                            'prediction': prediction,
                            'score': success_rate * 100,
                            'strategy': 'position_transformation',
                            'confidence': success_rate
                        })
                        
                        if len(predictions) >= num_predictions:
                            break
        
        # If not enough predictions, generate random variations
        if len(predictions) < num_predictions:
            needed = num_predictions - len(predictions)
            for _ in range(needed):
                # Generate random variation
                pred_digits = list(last_draw)
                random_pos = np.random.randint(0, 3)
                random_digit = np.random.randint(0, 10)
                pred_digits[random_pos] = str(random_digit)
                prediction = ''.join(pred_digits)
                
                predictions.append({
                    'prediction': prediction,
                    'score': 30,
                    'strategy': 'position_transformation_random',
                    'confidence': 0.3
                })
                            
        # Remove duplicates before returning
        predictions = self.remove_duplicates(predictions)
        return predictions[:num_predictions]
        
    def generate_advanced_mirrors(self, last_draw, num_predictions):
        """Generate predictions using advanced mirror patterns"""
        predictions = []
        
        for pos in range(3):
            current_digit = int(last_draw[pos])
            mirror_digit = 9 - current_digit
            
            # Check mirror pattern success rate
            mirror_patterns = self.mirror_patterns[pos]
            if mirror_patterns:
                mirror_success = sum(1 for from_d, to_d in mirror_patterns if from_d == current_digit and to_d == mirror_digit)
                total_patterns = sum(1 for from_d, to_d in mirror_patterns if from_d == current_digit)
                
                if total_patterns > 0:
                    success_rate = mirror_success / total_patterns
                    
                    # Generate mirror prediction
                    pred_digits = list(last_draw)
                    pred_digits[pos] = str(mirror_digit)
                    prediction = ''.join(pred_digits)
                    
                    predictions.append({
                        'prediction': prediction,
                        'score': success_rate * 100,
                        'strategy': 'advanced_mirror',
                        'confidence': success_rate
                    })
                    
        # Remove duplicates before returning
        predictions = self.remove_duplicates(predictions)
        return predictions[:num_predictions]
        
    def generate_sum_based_predictions(self, last_draw, num_predictions):
        """Generate predictions using sum-based patterns"""
        predictions = []
        
        current_sum = sum(int(d) for d in last_draw)
        
        if current_sum in self.sum_patterns:
            following_numbers = self.sum_patterns[current_sum]
            
            if following_numbers:
                # Find most common following numbers
                number_counts = Counter(following_numbers)
                most_common = number_counts.most_common(num_predictions)
                
                for number, count in most_common:
                    success_rate = count / len(following_numbers)
                    predictions.append({
                        'prediction': number,
                        'score': success_rate * 100,
                        'strategy': 'sum_based',
                        'confidence': success_rate
                    })
        
        # If not enough predictions, generate based on similar sums
        if len(predictions) < num_predictions:
            needed = num_predictions - len(predictions)
            # Find similar sums
            for sum_val in range(max(0, current_sum-3), min(28, current_sum+4)):
                if sum_val in self.sum_patterns and sum_val != current_sum:
                    following_numbers = self.sum_patterns[sum_val]
                    if following_numbers:
                        number_counts = Counter(following_numbers)
                        most_common = number_counts.most_common(needed)
                        
                        for number, count in most_common:
                            success_rate = count / len(following_numbers) * 0.8  # Reduced confidence
                            predictions.append({
                                'prediction': number,
                                'score': success_rate * 100,
                                'strategy': 'sum_based_similar',
                                'confidence': success_rate
                            })
                            if len(predictions) >= num_predictions:
                                break
                    
        # Remove duplicates before returning
        predictions = self.remove_duplicates(predictions)
        return predictions[:num_predictions]
        
    def generate_pattern_predictions(self, last_draw, num_predictions):
        """Generate predictions using pattern recognition"""
        predictions = []
        
        # Look for pattern sequences
        for seq, following in self.pattern_sequences.items():
            if seq[1] == last_draw:  # Current draw matches sequence
                if following:
                    number_counts = Counter(following)
                    most_common = number_counts.most_common(num_predictions)
                    
                    for number, count in most_common:
                        success_rate = count / len(following)
                        predictions.append({
                            'prediction': number,
                            'score': success_rate * 100,
                            'strategy': 'pattern_recognition',
                            'confidence': success_rate
                        })
        
        # If no pattern matches found, generate based on historical frequency
        if len(predictions) < num_predictions:
            # Get all historical numbers and their frequencies
            all_numbers = [row['numbers'] for _, row in self.df.iterrows()]
            number_counts = Counter(all_numbers)
            most_common_numbers = number_counts.most_common(num_predictions - len(predictions))
            
            for number, count in most_common_numbers:
                success_rate = count / len(all_numbers)
                predictions.append({
                    'prediction': number,
                    'score': success_rate * 100,
                    'strategy': 'pattern_recognition_fallback',
                    'confidence': success_rate
                })
                        
        # Remove duplicates before returning
        predictions = self.remove_duplicates(predictions)
        return predictions[:num_predictions]
        
    def generate_momentum_predictions(self, last_draw, num_predictions):
        """Generate predictions using momentum learning from recent drawings"""
        predictions = []
        
        if len(self.recent_drawings) >= 2:
            # Analyze momentum from recent manual drawings
            recent_trends = []
            
            for i in range(len(self.recent_drawings) - 1):
                prev = self.recent_drawings[i]
                curr = self.recent_drawings[i + 1]
                
                # Calculate digit momentum
                for pos in range(3):
                    prev_digit = int(prev[pos])
                    curr_digit = int(curr[pos])
                    momentum = curr_digit - prev_digit
                    recent_trends.append((pos, momentum))
            
            # Apply momentum to current draw
            if recent_trends:
                trend_counts = Counter(recent_trends)
                most_common_trends = trend_counts.most_common(num_predictions)
                
                for (pos, momentum), count in most_common_trends:
                    current_digit = int(last_draw[pos])
                    predicted_digit = (current_digit + momentum) % 10
                    
                    pred_digits = list(last_draw)
                    pred_digits[pos] = str(predicted_digit)
                    prediction = ''.join(pred_digits)
                    
                    success_rate = count / len(recent_trends)
                    predictions.append({
                        'prediction': prediction,
                        'score': success_rate * 100,
                        'strategy': 'momentum_learning',
                        'confidence': success_rate
                    })
                    
        return predictions[:num_predictions]
        
    def generate_enhanced_predictions(self, last_draw, num_singles=40, num_doubles=40):
        """Generate enhanced predictions with manual input learning"""
        print(f"TARGET: Generating enhanced predictions for trigger: {last_draw}")
        
        all_predictions = []
        
        # Generate predictions for each strategy with adaptive weights
        for strategy, weight in self.adaptive_weights.items():
            strategy_predictions = self.get_strategy_predictions(last_draw, strategy, int(num_singles * weight))
            print(f"  {strategy}: {len(strategy_predictions)} predictions")
            all_predictions.extend(strategy_predictions)
        
        print(f"DATA: Total predictions generated: {len(all_predictions)}")
        
        # Remove duplicates from ALL predictions first
        all_predictions = self.remove_duplicates(all_predictions)
        print(f"DATA: After removing duplicates from all predictions: {len(all_predictions)}")
        
        # Sort by confidence score
        all_predictions.sort(key=lambda x: x['score'], reverse=True)
        
        # Separate singles and doubles
        singles = []
        doubles = []
        
        for pred in all_predictions:
            prediction = pred['prediction']
            unique_digits = len(set(prediction))
            
            if unique_digits == 3:  # Singles
                if len(singles) < num_singles:
                    singles.append(pred)
            elif unique_digits == 2:  # Doubles
                if len(doubles) < num_doubles:
                    doubles.append(pred)
        
        print(f"DATA: Singles found: {len(singles)}, Doubles found: {len(doubles)}")
        
        # Fill remaining slots with high-confidence predictions
        while len(singles) < num_singles:
            # Generate additional singles
            additional = self.generate_additional_singles(last_draw, num_singles - len(singles))
            # Remove duplicates from additional singles
            additional = self.remove_duplicates(additional)
            # Also remove any that duplicate existing singles
            existing_singles = [s['prediction'] for s in singles]
            additional_filtered = []
            for add_pred in additional:
                normalized = ''.join(sorted(add_pred['prediction']))
                is_duplicate = False
                for existing in existing_singles:
                    if ''.join(sorted(existing)) == normalized:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    additional_filtered.append(add_pred)
            
            singles.extend(additional_filtered)
            if len(singles) >= num_singles:
                break
            
        while len(doubles) < num_doubles:
            # Generate additional doubles
            additional = self.generate_additional_doubles(last_draw, num_doubles - len(doubles))
            # Remove duplicates from additional doubles
            additional = self.remove_duplicates(additional)
            # Also remove any that duplicate existing doubles
            existing_doubles = [d['prediction'] for d in doubles]
            additional_filtered = []
            for add_pred in additional:
                normalized = ''.join(sorted(add_pred['prediction']))
                is_duplicate = False
                for existing in existing_doubles:
                    if ''.join(sorted(existing)) == normalized:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    additional_filtered.append(add_pred)
            
            doubles.extend(additional_filtered)
            if len(doubles) >= num_doubles:
                break
        
        # Apply quality guardrails: validate and normalize all predictions
        singles = self.ensure_box_signatures(singles)
        doubles = self.ensure_box_signatures(doubles)
        
        # After filling quotas, run final sweep to ensure best variants
        singles = self.remove_duplicates(singles)
        doubles = self.remove_duplicates(doubles)
        final_out = self.remove_duplicates(singles + doubles)

        # If we must return as two lists, re-separate after final dedupe
        final_singles = [p for p in final_out if len(set(str(p['prediction']).zfill(3))) == 3][:num_singles]
        final_doubles = [p for p in final_out if len(set(str(p['prediction']).zfill(3))) == 2][:num_doubles]

        print(f"DATA: Final results after final sweep - Singles: {len(final_singles)}, Doubles: {len(final_doubles)}")
        
        return {
            'singles': final_singles,
            'doubles': final_doubles,
            'total_predictions': len(final_singles) + len(final_doubles),
            'learning_status': {
                'recent_drawings': self.recent_drawings,
                'adaptive_weights': self.adaptive_weights,
                'manual_learning_enabled': self.manual_learning_enabled
            }
        }
        
    def generate_additional_singles(self, last_draw, num_needed):
        """Generate additional singles when needed"""
        additional = []
        
        # Use hot digits approach
        all_digits = []
        for _, row in self.df.iterrows():
            all_digits.extend([int(d) for d in row['numbers']])
        
        digit_counts = Counter(all_digits)
        hot_digits = [d for d, count in digit_counts.most_common(5)]
        
        attempts = 0
        max_attempts = num_needed * 10
        
        while len(additional) < num_needed and attempts < max_attempts:
            attempts += 1
            
            # Generate random single
            digits = np.random.choice(hot_digits, 3, replace=False)
            prediction = ''.join(map(str, digits))
            
            if len(set(prediction)) == 3:  # Ensure it's a single
                additional.append({
                    'prediction': prediction,
                    'score': 50,
                    'strategy': 'hot_digits',
                    'confidence': 0.5
                })
                
        # Remove duplicates from additional singles
        additional = self.remove_duplicates(additional)
        return additional
        
    def generate_additional_doubles(self, last_draw, num_needed):
        """Generate additional doubles when needed"""
        additional = []
        
        # Use hot digits approach for doubles
        all_digits = []
        for _, row in self.df.iterrows():
            all_digits.extend([int(d) for d in row['numbers']])
        
        digit_counts = Counter(all_digits)
        hot_digits = [d for d, count in digit_counts.most_common(5)]
        
        attempts = 0
        max_attempts = num_needed * 10
        
        while len(additional) < num_needed and attempts < max_attempts:
            attempts += 1
            
            # Generate random double
            digit1 = np.random.choice(hot_digits)
            digit2 = np.random.choice(hot_digits)
            digit3 = np.random.choice(hot_digits)
            
            # Ensure exactly 2 unique digits
            digits = [digit1, digit2, digit3]
            unique_digits = set(digits)
            
            if len(unique_digits) == 2:
                prediction = ''.join(map(str, digits))
                additional.append({
                    'prediction': prediction,
                    'score': 50,
                    'strategy': 'hot_digits_double',
                    'confidence': 0.5
                })
                
        # Remove duplicates from additional doubles
        additional = self.remove_duplicates(additional)
        return additional
        
    def remove_duplicates(self, predictions):
        """BOX-level dedupe: keep ONE best-scoring variant per signature."""
        by_sig = {}
        removed = 0

        for pred in predictions:
            s = str(pred['prediction']).zfill(3)
            sig = ''.join(sorted(s))  # canonical BOX signature

            best = by_sig.get(sig)
            if best is None or pred.get('score', 0) > best.get('score', 0):
                if best is not None:
                    removed += 1
                by_sig[sig] = pred
            else:
                removed += 1

        unique = list(by_sig.values())
        if removed:
            print(f"SUCCESS: BOX dedupe kept {len(unique)} unique signatures, removed {removed} lower-scoring variants")
        return unique
        
    def save_learning_state(self, filename='enhanced_learning_state.json'):
        """Save learning state to file"""
        state = {
            'adaptive_weights': self.adaptive_weights,
            'strategy_performance': dict(self.strategy_performance),
            'recent_drawings': self.recent_drawings,
            'manual_learning_enabled': self.manual_learning_enabled,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)
            
        print(f"SUCCESS: Learning state saved to {filename}")
        
    def load_learning_state(self, filename='enhanced_learning_state.json'):
        """Load learning state from file"""
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                state = json.load(f)
                
            self.adaptive_weights = state.get('adaptive_weights', {})
            self.strategy_performance = defaultdict(list, state.get('strategy_performance', {}))
            self.recent_drawings = state.get('recent_drawings', [])
            self.manual_learning_enabled = state.get('manual_learning_enabled', True)
            
            print(f"SUCCESS: Learning state loaded from {filename}")
        else:
            print(f"WARNING: No learning state file found: {filename}")
            
    def update_learning_from_data_files(self):
        """Update learning system using historical data from CSV files"""
        print("DATA: Updating learning from data files...")
        
        if self.df is None or len(self.df) < 2:
            print("ERROR: Insufficient data for learning")
            return
            
        # Use last 50 draws for learning (or all if less than 50)
        learning_data = self.df.tail(50)
        
        total_updates = 0
        strategy_hits = defaultdict(int)
        strategy_attempts = defaultdict(int)
        
        for i in range(len(learning_data) - 1):
            trigger = learning_data.iloc[i]['numbers']
            result = learning_data.iloc[i + 1]['numbers']
            
            # Test each strategy
            for strategy in self.strategies.keys():
                predictions = self.get_strategy_predictions(trigger, strategy, 10)
                predicted_numbers = [p['prediction'] for p in predictions]
                
                # Record hit or miss
                hit = 1 if result in predicted_numbers else 0
                strategy_hits[strategy] += hit
                strategy_attempts[strategy] += 1
                
                # Update performance tracking
                self.strategy_performance[strategy].append(hit)
                
                # Keep only recent performance
                if len(self.strategy_performance[strategy]) > self.memory_window:
                    self.strategy_performance[strategy] = self.strategy_performance[strategy][-self.memory_window:]
        
        # Update adaptive weights based on data file performance
        for strategy in self.strategies.keys():
            if strategy_attempts[strategy] > 0:
                success_rate = strategy_hits[strategy] / strategy_attempts[strategy]
                current_weight = self.adaptive_weights.get(strategy, 0.5)
                
                # Update weight based on historical performance
                updated_weight = current_weight + self.learning_rate * (success_rate - current_weight)
                self.adaptive_weights[strategy] = max(0.01, min(0.99, updated_weight))
                
                print(f"  {strategy}: {success_rate:.3f} success rate (weight: {self.adaptive_weights[strategy]:.3f})")
                total_updates += 1
        
        print(f"SUCCESS: Updated learning from {len(learning_data)-1} historical draws")
        
        # Update NJ UPDATED DATES.txt file after learning
        self.update_nj_dates_file()
        
        return total_updates
        
    def get_system_status(self):
        """Get current system status and performance"""
        status = {
            'recent_drawings': self.recent_drawings,
            'manual_learning_enabled': self.manual_learning_enabled,
            'adaptive_weights': self.adaptive_weights,
            'strategy_performance': {},
            'total_predictions_generated': 0,
            'learning_rate': self.learning_rate,
            'memory_window': self.memory_window
        }
        
        # Calculate recent performance for each strategy
        for strategy in self.strategies.keys():
            recent_perf = self.strategy_performance[strategy][-10:]
            status['strategy_performance'][strategy] = {
                'recent_accuracy': np.mean(recent_perf) if recent_perf else 0,
                'total_predictions': len(self.strategy_performance[strategy])
            }
            
        return status

    def detect_draw_type(self, number):
        """Detect if a number is a double or single"""
        digits = list(number)
        if digits[0] == digits[1] or digits[1] == digits[2] or digits[0] == digits[2]:
            return "double"
        return "single"
    
    def analyze_transition_patterns(self):
        """Analyze what tends to follow doubles vs singles"""
        print("SEARCH: Analyzing transition patterns...")
        
        self.transition_patterns = {
            'double_to_single': {'count': 0, 'examples': []},
            'double_to_double': {'count': 0, 'examples': []},
            'single_to_double': {'count': 0, 'examples': []},
            'single_to_single': {'count': 0, 'examples': []}
        }
        
        # Analyze last 1000 draws for patterns
        recent_draws = self.df.tail(1000)
        
        for i in range(len(recent_draws) - 1):
            current_draw = str(recent_draws.iloc[i]['numbers']).zfill(3)
            next_draw = str(recent_draws.iloc[i + 1]['numbers']).zfill(3)
            
            current_type = self.detect_draw_type(current_draw)
            next_type = self.detect_draw_type(next_draw)
            
            transition_key = f"{current_type}_to_{next_type}"
            
            if transition_key not in self.transition_patterns:
                self.transition_patterns[transition_key] = {'count': 0, 'examples': []}
            
            self.transition_patterns[transition_key]['count'] += 1
            self.transition_patterns[transition_key]['examples'].append({
                'from': current_draw,
                'to': next_draw
            })
        
        # Calculate transition probabilities
        total_transitions = sum(pattern['count'] for pattern in self.transition_patterns.values())
        
        for transition_type, data in self.transition_patterns.items():
            if total_transitions > 0:
                probability = (data['count'] / total_transitions) * 100
                print(f"  {transition_type}: {probability:.1f}% ({data['count']} occurrences)")
        
        print("SUCCESS: Transition pattern analysis complete")
        return self.transition_patterns
    
    def _sig(self, s): 
        """Convert number to BOX signature (sorted digits)"""
        s = str(s).zfill(3)
        return ''.join(sorted(s))

    def _cls(self, sig): 
        """Classify BOX signature as Single (S) or Double (D)"""
        return "S" if len(set(sig)) == 3 else "D"
    
    def fit_transitions(self, history):
        """Train transition model from history: list of 'XYZ' strings, oldest -> newest"""
        print(" Training transition model...")
        
        self.t_count_exact = defaultdict(Counter)   # prev_sig -> Counter(next_sig)
        self.t_count_class = defaultdict(Counter)   # 'S' or 'D' -> Counter(next_sig)
        self.t_count_global = Counter()
        self.t_n_exact = Counter()
        self.t_n_class = Counter()
        self.t_N = 0

        sigs = [self._sig(x) for x in history if str(x).isdigit()]
        for i in range(len(sigs)-1):
            prev, nxt = sigs[i], sigs[i+1]
            pc = self._cls(prev)
            self.t_count_exact[prev][nxt] += 1; self.t_n_exact[prev] += 1
            self.t_count_class[pc][nxt] += 1;  self.t_n_class[pc]   += 1
            self.t_count_global[nxt] += 1;     self.t_N += 1
        
        print(f"SUCCESS: Transition model trained on {len(sigs)-1} transitions")
    
    def fit_transitions_by_stream(self):
        """Train separate transition models for Midday and Evening streams"""
        if self.df is None or len(self.df) < 2:
            print("WARNING: Insufficient data for stream-specific transition training")
            return
        
        print(" Training stream-specific transition models...")
        
        # Initialize stream-specific models
        self.stream_models = {}
        
        for stream in ['midday', 'evening']:
            stream_data = self.df[self.df['time_segment'] == stream]['numbers'].tolist()
            if len(stream_data) > 1:
                print(f"Training {stream} model with {len(stream_data)} draws...")
                
                # Create temporary model for this stream
                temp_model = {
                    't_count_exact': defaultdict(Counter),
                    't_count_class': defaultdict(Counter),
                    't_count_global': Counter(),
                    't_n_exact': Counter(),
                    't_n_class': Counter(),
                    't_N': 0
                }
                
                sigs = [self._sig(x) for x in stream_data if str(x).isdigit()]
                for i in range(len(sigs)-1):
                    prev, nxt = sigs[i], sigs[i+1]
                    pc = self._cls(prev)
                    temp_model['t_count_exact'][prev][nxt] += 1
                    temp_model['t_n_exact'][prev] += 1
                    temp_model['t_count_class'][pc][nxt] += 1
                    temp_model['t_n_class'][pc] += 1
                    temp_model['t_count_global'][nxt] += 1
                    temp_model['t_N'] += 1
                
                self.stream_models[stream] = temp_model
                print(f"SUCCESS: {stream} model: {len(sigs)-1} transitions")
        
        print("SUCCESS: Stream-specific transition models trained")

    def predict_next_dist(self, prev, alpha=1.0, min_exact=5, min_class=30):
        """Predict distribution for next BOX signature with hierarchical backoff + Laplace smoothing"""
        prev_sig = self._sig(prev); pc = self._cls(prev_sig)

        if self.t_n_exact.get(prev_sig, 0) >= min_exact:
            ctr, denom = self.t_count_exact[prev_sig], self.t_n_exact[prev_sig]
            print(f"TARGET: Using exact transition data for {prev_sig}")
        elif self.t_n_class.get(pc, 0) >= min_class:
            ctr, denom = self.t_count_class[pc], self.t_n_class[pc]
            print(f"TARGET: Using class transition data for {pc}")
        else:
            ctr, denom = self.t_count_global, self.t_N or 1
            print(f"TARGET: Using global transition data (fallback)")

        # smoothed distribution (normalize)
        keys = list(ctr.keys()) or list(self.t_count_global.keys())
        if not keys:
            return {}
        
        dist = {k: (ctr[k] + alpha) / (denom + alpha*len(keys)) for k in keys}
        Z = sum(dist.values()) or 1.0
        return {k: v/Z for k,v in dist.items()}
    
    def validate_prediction(self, prediction):
        """Validate and normalize a prediction to proper 3-digit format"""
        try:
            # Convert to string and pad with zeros
            pred_str = str(prediction).zfill(3)
            
            # Ensure it's exactly 3 digits
            if len(pred_str) != 3 or not pred_str.isdigit():
                return None
                
            # Ensure each digit is 0-9
            for digit in pred_str:
                if not digit.isdigit():
                    return None
            
            return pred_str
        except:
            return None
    
    def ensure_box_signatures(self, predictions):
        """Ensure all predictions are properly formatted and return BOX signatures only"""
        validated = []
        
        for pred in predictions:
            # Validate prediction format
            normalized = self.validate_prediction(pred.get('prediction', ''))
            if normalized is None:
                continue
                
            # Update prediction with normalized format
            pred_copy = dict(pred)
            pred_copy['prediction'] = normalized
            pred_copy['box_signature'] = self._sig(normalized)
            validated.append(pred_copy)
        
        return validated
    
    def calculate_smart_score(self, prediction, last_draw, prediction_type):
        """Calculate SmartScore v2.0 for a prediction"""
        score = 0
        max_score = 10
        
        # Convert prediction to string if needed
        pred_str = str(prediction).zfill(3)
        last_str = str(last_draw).zfill(3)
        
        # 1. From correct group (double/single) +2 points
        last_type = self.detect_draw_type(last_str)
        pred_type = self.detect_draw_type(pred_str)
        
        # Check if this transition is likely based on historical patterns
        if hasattr(self, 'transition_patterns'):
            transition_key = f"{last_type}_to_{pred_type}"
            if transition_key in self.transition_patterns:
                transition_prob = self.transition_patterns[transition_key]['count']
                if transition_prob > 50:  # High probability transition
                    score += 2
                elif transition_prob > 20:  # Medium probability
                    score += 1
        
        # 2. Matches root sum pattern +1 point
        pred_sum = sum(int(d) for d in pred_str)
        last_sum = sum(int(d) for d in last_str)
        
        # Check if sum follows common patterns
        if abs(pred_sum - last_sum) <= 3:  # Similar sum range
            score += 1
        
        # 3. Includes digit from last draw +1 point
        common_digits = set(pred_str) & set(last_str)
        if common_digits:
            score += 1
        
        # 4. Mirror digit from last draw +1 point
        mirror_digits = {'0': '5', '1': '6', '2': '7', '3': '8', '4': '9', 
                        '5': '0', '6': '1', '7': '2', '8': '3', '9': '4'}
        
        for i, digit in enumerate(last_str):
            mirror_digit = mirror_digits.get(digit)
            if mirror_digit and mirror_digit in pred_str:
                score += 1
                break
        
        # 5. Has historical high-hit rate +2 points
        if hasattr(self, 'df'):
            # Check frequency in historical data
            pred_count = len(self.df[self.df['numbers'] == int(pred_str)])
            if pred_count > 10:  # High frequency number
                score += 2
            elif pred_count > 5:  # Medium frequency
                score += 1
        
        # 6. Is in the same flow pattern +2 points
        # Check if this prediction follows common patterns
        if hasattr(self, 'pattern_weights'):
            pattern_score = self.pattern_weights.get('pattern_recognition', 0)
            if pattern_score > 0.2:  # Strong pattern recognition
                score += 2
        
        # 7. Overdue hit by gap count +1 point
        # This would require tracking last hit dates - simplified for now
        score += 1  # Placeholder for gap analysis
        
        # Convert to percentage
        smart_score_percentage = (score / max_score) * 100
        
        return {
            'score': smart_score_percentage,
            'breakdown': {
                'transition_match': score >= 2,
                'sum_pattern': score >= 3,
                'common_digits': score >= 4,
                'mirror_digits': score >= 5,
                'historical_frequency': score >= 7,
                'flow_pattern': score >= 9,
                'overdue_hit': score >= 10
            }
        }
    
    def get_best_follow_ups(self, last_draw, num_predictions=20):
        """Get the best follow-up predictions using transition-based scoring"""
        print(f"TARGET: Analyzing best follow-ups for: {last_draw}")
        
        # Get transition probability distribution
        if hasattr(self, 't_count_exact'):
            dist = self.predict_next_dist(last_draw)
        else:
            print("WARNING: Transition model not trained, using fallback method")
            return self._fallback_follow_ups(last_draw, num_predictions)

        # Re-rank existing candidate pool by transition probability
        candidates = self.generate_enhanced_predictions(last_draw, num_singles=num_predictions, num_doubles=num_predictions)
        pool = candidates['singles'] + candidates['doubles']

        # Map to BOX signatures and weight by transition prob
        scored = []
        for p in pool:
            sig = self._sig(p['prediction'])
            pscore = p.get('score', 0)
            tprob = dist.get(sig, 0.0)
            final = 0.7 * tprob + 0.3 * (pscore / 100.0)  # blend; tune weights
            q = dict(p)
            q['transition_prob'] = tprob
            q['final_score'] = final
            scored.append(q)

        scored.sort(key=lambda x: x['final_score'], reverse=True)

        # BOX dedupe keep-best (again, belts & suspenders)
        scored = self.remove_duplicates(scored)

        # Light diversity: avoid flooding same root sum
        root_counts, diverse = Counter(), []
        for q in scored:
            digits = list(map(int, str(q['prediction']).zfill(3)))
            root = (sum(digits) % 9) or 9
            if root_counts[root] >= 3: 
                continue
            root_counts[root] += 1
            diverse.append(q)
            if len(diverse) >= num_predictions:
                break

        singles = [x for x in diverse if len(set(str(x['prediction']).zfill(3)))==3]
        doubles = [x for x in diverse if len(set(str(x['prediction']).zfill(3)))==2]
        
        print(f"TARGET: Generated {len(singles)} transition-optimized singles")
        print(f"TARGET: Generated {len(doubles)} transition-optimized doubles")
        
        return {
            'singles': singles[:num_predictions], 
            'doubles': doubles[:num_predictions],
            'transition_analysis': {
                'last_draw_signature': self._sig(last_draw),
                'last_draw_class': self._cls(self._sig(last_draw)),
                'top_transition_probs': dict(list(sorted(dist.items(), key=lambda x: x[1], reverse=True))[:5])
            }
        }
    
    def _fallback_follow_ups(self, last_draw, num_predictions):
        """Fallback method when transition model isn't available"""
        last_type = self.detect_draw_type(str(last_draw).zfill(3))
        
        # Use simple heuristic weights
        if last_type == "double":
            singles_weight = 0.7
            doubles_weight = 0.3
        else:
            singles_weight = 0.3
            doubles_weight = 0.7
        
        # Generate and score predictions
        num_singles = int(num_predictions * singles_weight)
        singles = self.generate_singles(last_draw, num_singles)
        
        num_doubles = int(num_predictions * doubles_weight)
        doubles = self.generate_doubles(last_draw, num_doubles)
        
        singles = self.remove_duplicates(singles)
        doubles = self.remove_duplicates(doubles)
        
        return {
            'singles': singles[:num_predictions],
            'doubles': doubles[:num_predictions],
            'transition_analysis': {
                'method': 'fallback',
                'last_draw_type': last_type,
                'singles_weight': singles_weight,
                'doubles_weight': doubles_weight
            }
        }
    
    def generate_singles(self, last_draw, num_predictions):
        """Generate single predictions with enhanced logic"""
        predictions = []
        last_str = str(last_draw).zfill(3)
        
        # Use existing prediction methods but focus on singles
        pattern_preds = self.generate_pattern_predictions(last_draw, num_predictions)
        position_preds = self.generate_position_transformations(last_draw, num_predictions)
        
        # Combine and filter for singles only
        all_preds = pattern_preds + position_preds
        
        for pred in all_preds:
            pred_str = str(pred['prediction']).zfill(3)
            if self.detect_draw_type(pred_str) == "single":
                predictions.append(pred)
                if len(predictions) >= num_predictions:
                    break
        
        # Remove duplicates within singles
        predictions = self.remove_duplicates(predictions)
        
        return predictions
    
    def generate_doubles(self, last_draw, num_predictions):
        """Generate double predictions with enhanced logic"""
        predictions = []
        last_str = str(last_draw).zfill(3)
        
        # Generate common double patterns
        double_patterns = [
            'AAB', 'ABB', 'ABA'  # Common double patterns
        ]
        
        for pattern in double_patterns:
            for _ in range(num_predictions // len(double_patterns)):
                if pattern == 'AAB':
                    # Generate AAB pattern
                    a = np.random.randint(0, 10)
                    b = np.random.randint(0, 10)
                    while b == a:
                        b = np.random.randint(0, 10)
                    prediction = f"{a}{a}{b}"
                elif pattern == 'ABB':
                    # Generate ABB pattern
                    a = np.random.randint(0, 10)
                    b = np.random.randint(0, 10)
                    while b == a:
                        b = np.random.randint(0, 10)
                    prediction = f"{a}{b}{b}"
                elif pattern == 'ABA':
                    # Generate ABA pattern
                    a = np.random.randint(0, 10)
                    b = np.random.randint(0, 10)
                    while b == a:
                        b = np.random.randint(0, 10)
                    prediction = f"{a}{b}{a}"
                
                predictions.append({
                    'prediction': prediction,
                    'score': 75.0,
                    'strategy': 'double_pattern',
                    'confidence': 0.75
                })
                
                if len(predictions) >= num_predictions:
                    break
        
        # Remove duplicates within doubles
        predictions = self.remove_duplicates(predictions)
        
        return predictions

def main():
    """Main function for testing"""
    print("BRAIN: Enhanced Learning Hybrid System")
    print("=" * 50)
    
    # Initialize system
    system = EnhancedLearningHybridSystem()
    
    # Add manual drawings (example)
    print("\n Adding manual drawings...")
    system.add_manual_drawing("123")
    system.add_manual_drawing("456")
    system.add_manual_drawing("789")
    
    # Generate predictions
    print("\nTARGET: Generating predictions...")
    results = system.generate_enhanced_predictions("123", 40, 40)
    
    print(f"\nDATA: Results:")
    print(f"Singles: {len(results['singles'])}")
    print(f"Doubles: {len(results['doubles'])}")
    
    # Show top predictions
    print("\nTROPHY: Top 5 Singles:")
    for i, pred in enumerate(results['singles'][:5]):
        print(f"  {i+1}. {pred['prediction']} (Score: {pred['score']:.1f}, Strategy: {pred['strategy']})")
        
    print("\nTROPHY: Top 5 Doubles:")
    for i, pred in enumerate(results['doubles'][:5]):
        print(f"  {i+1}. {pred['prediction']} (Score: {pred['score']:.1f}, Strategy: {pred['strategy']})")
    
    # Save learning state
    system.save_learning_state()
    
    # Show system status
    status = system.get_system_status()
    print(f"\nCHART: System Status:")
    print(f"Recent Drawings: {status['recent_drawings']}")
    print(f"Manual Learning: {status['manual_learning_enabled']}")
    print(f"Learning Rate: {status['learning_rate']}")

if __name__ == "__main__":
    main() 