import numpy as np
from collections import deque

class VolatilityAnalyzer:
    def _init_(self):
        self.ensemble = EmotionEnsemble()
    
    def calculate_user_volatility(self, posts_timeline):
        """Calculate emotional volatility for a user over time"""
        emotions = []
        timestamps = []
        
        for post in sorted(posts_timeline, key=lambda x: x['timestamp']):
            result = self.ensemble.ensemble_prediction(post['text'])
            emotions.append(result['ensemble_score'])
            timestamps.append(post['timestamp'])
        
        # Calculate volatility metrics
        volatility_metrics = {
            'standard_deviation': np.std(emotions),
            'mean_emotion': np.mean(emotions),
            'emotion_range': max(emotions) - min(emotions),
            'swing_count': self._count_swings(emotions, threshold=0.3),
            'stability_periods': self._identify_stable_periods(emotions),
            'crisis_indicators': self._detect_crisis_patterns(emotions)
        }
        
        return {
            'volatility_metrics': volatility_metrics,
            'emotion_timeline': list(zip(timestamps, emotions)),
            'overall_volatility_score': self._calculate_overall_volatility(volatility_metrics)
        }
    
    def _count_swings(self, emotions, threshold=0.3):
        """Count dramatic emotional changes"""
        swings = 0
        for i in range(1, len(emotions)):
            if abs(emotions[i] - emotions[i-1]) > threshold:
                swings += 1
        return swings
    
    def _identify_stable_periods(self, emotions, stability_threshold=0.2):
        """Find periods of emotional stability"""
        stable_periods = []
        current_period_start = 0
        
        for i in range(1, len(emotions)):
            if abs(emotions[i] - emotions[i-1]) > stability_threshold:
                if i - current_period_start >= 3:  # Minimum 3 posts
                    stable_periods.append({
                        'start': current_period_start,
                        'end': i-1,
                        'length': i - current_period_start,
                        'avg_emotion': np.mean(emotions[current_period_start:i])
                    })
                current_period_start = i
        
        return stable_periods
    
    def _detect_crisis_patterns(self, emotions):
        """Identify potential crisis indicators"""
        crisis_indicators = []
        
        # Pattern: Sustained negative emotion + high local volatility
        window_size = 5
        for i in range(len(emotions) - window_size):
            window = emotions[i:i+window_size]
            avg_emotion = np.mean(window)
            local_volatility = np.std(window)
            
            if avg_emotion < -0.4 and local_volatility > 0.6:
                crisis_indicators.append({
                    'type': 'sustained_negative_volatility',
                    'position': i,
                    'severity': abs(avg_emotion) * local_volatility,
                    'window_emotions': window
                })
        
        return crisis_indicators
    
    def _calculate_overall_volatility(self, metrics):
        """Combine different volatility measures into single score"""
        # Normalize and weight different components
        std_component = min(metrics['standard_deviation'] / 2.0, 1.0)  # Cap at 1.0
        swing_component = min(metrics['swing_count'] / 10.0, 1.0)  # Normalize by expected swings
        range_component = min(metrics['emotion_range'] / 2.0, 1.0)  # Cap at 1.0
        
        overall_score = (
            std_component * 0.4 +
            swing_component * 0.3 +
            range_component * 0.3
        )
