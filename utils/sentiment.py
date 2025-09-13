from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk

def _init_(self):
        self.vader = SentimentIntensityAnalyzer()
        nltk.download('vader_lexicon', quiet=True)
        self.nltk_analyzer = NLTKAnalyzer()
        
    def analyze_text(self, text):
        """Combine multiple sentiment analysis methods"""
        
        # VADER Analysis
        vader_scores = self.vader.polarity_scores(text)
        vader_compound = vader_scores['compound']
        
        # TextBlob Analysis  
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        
        # NLTK VADER (alternative implementation)
        nltk_scores = self.nltk_analyzer.polarity_scores(text)
        nltk_compound = nltk_scores['compound']
        
        return {
            'vader': vader_compound,
            'textblob': textblob_polarity,
            'nltk': nltk_compound,
            'raw_scores': {
                'vader_detailed': vader_scores,
                'textblob_subjectivity': blob.sentiment.subjectivity,
                'nltk_detailed': nltk_scores
            }
        }
    
    def ensemble_prediction(self, text, method='weighted_average'):
        """Combine predictions using different ensemble methods"""
        scores = self.analyze_text(text)
        
        if method == 'weighted_average':
            # Weights based on research performance on mental health text
            weights = {
                'vader': 0.4,      # Generally performs well on social media
                'textblob': 0.3,   # Good for general sentiment
                'nltk': 0.3        # Backup/validation
            }
            
            ensemble_score = (
                scores['vader'] * weights['vader'] +
                scores['textblob'] * weights['textblob'] +
                scores['nltk'] * weights['nltk']
            )
            
        elif method == 'simple_average':
            ensemble_score = (scores['vader'] + scores['textblob'] + scores['nltk']) / 3
            
        elif method == 'confidence_weighted':
            # Weight by confidence (absolute value)
            confidences = [abs(scores['vader']), abs(scores['textblob']), abs(scores['nltk'])]
            total_confidence = sum(confidences)
            
            if total_confidence > 0:
                ensemble_score = (
                    scores['vader'] * (abs(scores['vader']) / total_confidence) +
                    scores['textblob'] * (abs(scores['textblob']) / total_confidence) +
                    scores['nltk'] * (abs(scores['nltk']) / total_confidence)
                )
            else:
                ensemble_score = 0
                
        return {
            'ensemble_score': ensemble_score,
            'individual_scores': scores,
            'confidence': self._calculate_confidence(scores),
            'agreement': self._calculate_agreement(scores)
        }
    
    def _calculate_confidence(self, scores):
        """Calculate confidence based on score magnitudes"""
        magnitudes = [abs(scores['vader']), abs(scores['textblob']), abs(scores['nltk'])]
        return np.mean(magnitudes)
    
    def _calculate_agreement(self, scores):
        """Calculate agreement between different methods"""
        values = [scores['vader'], scores['textblob'], scores['nltk']]
        # Check if all have same sign (positive/negative agreement)
        signs = [1 if v > 0 else -1 if v < 0 else 0 for v in values]
        agreement = len(set(signs)) == 1 and signs[0] != 0
        
        # Calculate variance as disagreement measure
        variance = np.var(values)
        
        return {
            'directional_agreement': agreement,
            'variance': variance,
            'agreement_score': 1 / (1 + variance)  # Higher = more agreement
        }def _init_(self):
        self.vader = SentimentIntensityAnalyzer()
        nltk.download('vader_lexicon', quiet=True)
        self.nltk_analyzer = NLTKAnalyzer()
        
    def analyze_text(self, text):
        """Combine multiple sentiment analysis methods"""
        
        # VADER Analysis
        vader_scores = self.vader.polarity_scores(text)
        vader_compound = vader_scores['compound']
        
        # TextBlob Analysis  
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        
        # NLTK VADER (alternative implementation)
        nltk_scores = self.nltk_analyzer.polarity_scores(text)
        nltk_compound = nltk_scores['compound']
        
        return {
            'vader': vader_compound,
            'textblob': textblob_polarity,
            'nltk': nltk_compound,
            'raw_scores': {
                'vader_detailed': vader_scores,
                'textblob_subjectivity': blob.sentiment.subjectivity,
                'nltk_detailed': nltk_scores
            }
        }
    
    def ensemble_prediction(self, text, method='weighted_average'):
        """Combine predictions using different ensemble methods"""
        scores = self.analyze_text(text)
        
        if method == 'weighted_average':
            # Weights based on research performance on mental health text
            weights = {
                'vader': 0.4,      # Generally performs well on social media
                'textblob': 0.3,   # Good for general sentiment
                'nltk': 0.3        # Backup/validation
            }
            
            ensemble_score = (
                scores['vader'] * weights['vader'] +
                scores['textblob'] * weights['textblob'] +
                scores['nltk'] * weights['nltk']
            )
            
        elif method == 'simple_average':
            ensemble_score = (scores['vader'] + scores['textblob'] + scores['nltk']) / 3
            
        elif method == 'confidence_weighted':
            # Weight by confidence (absolute value)
            confidences = [abs(scores['vader']), abs(scores['textblob']), abs(scores['nltk'])]
            total_confidence = sum(confidences)
            
            if total_confidence > 0:
                ensemble_score = (
                    scores['vader'] * (abs(scores['vader']) / total_confidence) +
                    scores['textblob'] * (abs(scores['textblob']) / total_confidence) +
                    scores['nltk'] * (abs(scores['nltk']) / total_confidence)
                )
            else:
                ensemble_score = 0
                
        return {
            'ensemble_score': ensemble_score,
            'individual_scores': scores,
            'confidence': self._calculate_confidence(scores),
            'agreement': self._calculate_agreement(scores)
        }
    
    def _calculate_confidence(self, scores):
        """Calculate confidence based on score magnitudes"""
        magnitudes = [abs(scores['vader']), abs(scores['textblob']), abs(scores['nltk'])]
        return np.mean(magnitudes)
    
    def _calculate_agreement(self, scores):
        """Calculate agreement between different methods"""
        values = [scores['vader'], scores['textblob'], scores['nltk']]
        # Check if all have same sign (positive/negative agreement)
        signs = [1 if v > 0 else -1 if v < 0 else 0 for v in values]
        agreement = len(set(signs)) == 1 and signs[0] != 0
        
        # Calculate variance as disagreement measure
        variance = np.var(values)
        
        return {
            'directional_agreement': agreement,
            'variance': variance,
            'agreement_score': 1 / (1 + variance)  # Higher = more agreement
        }