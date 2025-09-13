import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta

def calculate_comprehensive_metrics(df):
    """Calculate all emotional volatility metrics for the dashboard"""
    
    if df.empty or 'sentiment' not in df.columns:
        return {}
    
    # Ensure data is sorted by time
    df = df.sort_values('time')
    
    # Basic Statistics
    mean_sentiment = df['sentiment'].mean()
    sentiment_std = df['sentiment'].std()
    sentiment_range = df['sentiment'].max() - df['sentiment'].min()
    
    # Volatility Metrics
    volatility_score = sentiment_std  # Primary volatility measure
    emotional_swings = count_emotional_swings(df['sentiment'])
    swing_frequency = emotional_swings / len(df) if len(df) > 0 else 0
    
    # Stability Analysis
    stability_periods = identify_stability_periods(df)
    avg_stability_duration = np.mean([p['duration'] for p in stability_periods]) if stability_periods else 0
    stability_ratio = sum([p['duration'] for p in stability_periods]) / len(df) if stability_periods else 0
    
    # Trend Analysis
    if len(df) > 1:
        trend_slope, trend_p_value = calculate_trend(df)
        trend_direction = "Improving" if trend_slope > 0.01 else "Declining" if trend_slope < -0.01 else "Stable"
    else:
        trend_slope, trend_p_value, trend_direction = 0, 1, "Insufficient Data"
    
    # Risk Indicators
    crisis_risk = calculate_crisis_risk(df)
    negative_streaks = count_negative_streaks(df['sentiment'])
    extreme_events = count_extreme_events(df['sentiment'])
    
    # Time-based Patterns
    daily_pattern = analyze_daily_patterns(df) if 'time' in df.columns else {}
    
    return {
        # Core Volatility Metrics
        'volatility_score': round(volatility_score, 3),
        'emotional_swings': emotional_swings,
        'swing_frequency': round(swing_frequency, 3),
        'sentiment_range': round(sentiment_range, 3),
        
        # Stability Metrics
        'avg_stability_duration': round(avg_stability_duration, 1),
        'stability_ratio': round(stability_ratio, 3),
        'stability_periods_count': len(stability_periods),
        
        # Trend Analysis
        'trend_direction': trend_direction,
        'trend_slope': round(trend_slope, 4),
        'trend_significance': 'Significant' if trend_p_value < 0.05 else 'Not Significant',
        
        # Risk Assessment
        'crisis_risk': crisis_risk,
        'negative_streaks': negative_streaks,
        'extreme_events': extreme_events,
        
        # Basic Stats
        'mean_sentiment': round(mean_sentiment, 3),
        'posts_analyzed': len(df),
        'time_span_days': calculate_time_span(df) if 'time' in df.columns else 0,
        
        # Distribution
        'positive_ratio': round((df['sentiment'] > 0.1).mean(), 3),
        'negative_ratio': round((df['sentiment'] < -0.1).mean(), 3),
        'neutral_ratio': round((df['sentiment'].between(-0.1, 0.1)).mean(), 3),
    }

def count_emotional_swings(sentiment_series, threshold=0.3):
    """Count significant emotional changes between consecutive posts"""
    if len(sentiment_series) < 2:
        return 0
    
    swings = 0
    for i in range(1, len(sentiment_series)):
        if abs(sentiment_series.iloc[i] - sentiment_series.iloc[i-1]) > threshold:
            swings += 1
    return swings

def identify_stability_periods(df, stability_threshold=0.15):
    """Identify periods of emotional stability"""
    if len(df) < 3:
        return []
    
    stable_periods = []
    current_period_start = 0
    
    for i in range(1, len(df)):
        sentiment_diff = abs(df['sentiment'].iloc[i] - df['sentiment'].iloc[i-1])
        
        if sentiment_diff > stability_threshold:
            # End of stable period
            if i - current_period_start >= 3:  # Minimum 3 posts for stability
                stable_periods.append({
                    'start_index': current_period_start,
                    'end_index': i-1,
                    'duration': i - current_period_start,
                    'avg_sentiment': df['sentiment'].iloc[current_period_start:i].mean()
                })
            current_period_start = i
    
    # Check final period
    if len(df) - current_period_start >= 3:
        stable_periods.append({
            'start_index': current_period_start,
            'end_index': len(df)-1,
            'duration': len(df) - current_period_start,
            'avg_sentiment': df['sentiment'].iloc[current_period_start:].mean()
        })
    
    return stable_periods

def calculate_trend(df):
    """Calculate overall sentiment trend using linear regression"""
    if len(df) < 3:
        return 0, 1
    
    x = np.arange(len(df))
    y = df['sentiment'].values
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return slope, p_value

def calculate_crisis_risk(df):
    """Calculate crisis risk based on multiple factors"""
    if df.empty:
        return "Low"
    
    risk_score = 0
    
    # High volatility
    if df['sentiment'].std() > 0.5:
        risk_score += 2
    
    # Recent negative trend
    recent_data = df.tail(10)  # Last 10 posts
    if len(recent_data) > 3 and recent_data['sentiment'].mean() < -0.3:
        risk_score += 2
    
    # Extreme negative posts
    extreme_negative = (df['sentiment'] < -0.7).sum()
    if extreme_negative > len(df) * 0.2:  # More than 20% extremely negative
        risk_score += 1
    
    # Sustained negativity
    negative_streak = count_negative_streaks(df['sentiment'])
    if negative_streak > 5:
        risk_score += 1
    
    if risk_score >= 4:
        return "High"
    elif risk_score >= 2:
        return "Medium"
    else:
        return "Low"

def count_negative_streaks(sentiment_series):
    """Count maximum consecutive negative posts"""
    if sentiment_series.empty:
        return 0
    
    max_streak = 0
    current_streak = 0
    
    for sentiment in sentiment_series:
        if sentiment < -0.1:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    
    return max_streak

def count_extreme_events(sentiment_series):
    """Count extremely positive or negative posts"""
    if sentiment_series.empty:
        return 0
    
    extreme_positive = (sentiment_series > 0.8).sum()
    extreme_negative = (sentiment_series < -0.8).sum()
    
    return extreme_positive + extreme_negative

def calculate_time_span(df):
    """Calculate time span of data in days"""
    if df.empty or 'time' not in df.columns:
        return 0
    
    time_diff = df['time'].max() - df['time'].min()
    return time_diff.days

def analyze_daily_patterns(df):
    """Analyze sentiment patterns by time of day"""
    if df.empty or 'time' not in df.columns:
        return {}
    
    df['hour'] = df['time'].dt.hour
    
    # Group by time periods
    morning = df[df['hour'].between(6, 12)]['sentiment'].mean()
    afternoon = df[df['hour'].between(12, 18)]['sentiment'].mean()
    evening = df[df['hour'].between(18, 24)]['sentiment'].mean()
    night = df[df['hour'] < 6]['sentiment'].mean()
    
    return {
        'morning_avg': round(morning, 3) if not pd.isna(morning) else 0,
        'afternoon_avg': round(afternoon, 3) if not pd.isna(afternoon) else 0,
        'evening_avg': round(evening, 3) if not pd.isna(evening) else 0,
        'night_avg': round(night, 3) if not pd.isna(night) else 0,
    }

# Usage in your Streamlit app
def display_metrics(df):
    """Display metrics in Streamlit dashboard"""
    metrics = calculate_comprehensive_metrics(df)
    
    # Main metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Volatility Score", metrics.get('volatility_score', 0))
    col2.metric("Emotional Swings", metrics.get('emotional_swings', 0))
    col3.metric("Crisis Risk", metrics.get('crisis_risk', 'Low'))
    col4.metric("Trend", metrics.get('trend_direction', 'Stable'))
    
    # Secondary metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Stability Ratio", f"{metrics.get('stability_ratio', 0):.1%}")
    col2.metric("Posts Analyzed", metrics.get('posts_analyzed', 0))
    col3.metric("Time Span (Days)", metrics.get('time_span_days', 0))
    col4.metric("Negative Streaks", metrics.get('negative_streaks', 0))
    
    return metrics