"""
Robust Emotional Tagger v2
Transformer-based emotional analysis with valence-arousal mapping

Replaces lexicon-based approach with j-hartmann/emotion-english-distilroberta-base
for better handling of negation, sarcasm, and context.
"""

# Handle missing dependencies gracefully
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

import numpy as np
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import warnings
import logging

# Suppress transformer warnings for cleaner output if available
if HAS_TRANSFORMERS:
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("transformers").setLevel(logging.ERROR)


@dataclass
class EmotionalAnalysis:
    """Result of emotional analysis"""
    valence: float  # -1 to 1 (negative to positive)
    arousal: float  # 0 to 1 (calm to excited)
    confidence: float  # 0 to 1 (certainty of analysis)
    dominant_emotion: str  # Primary detected emotion
    emotion_scores: Dict[str, float]  # All emotion probabilities


class RobustEmotionalTagger:
    """
    GPU-accelerated transformer-based emotional tagger
    Maps 7 emotions to valence-arousal space with confidence scoring
    """
    
    # Mapping from Ekman emotions to valence-arousal coordinates
    EMOTION_MAPPING = {
        'joy': {'valence': 0.8, 'arousal': 0.7},
        'anger': {'valence': -0.6, 'arousal': 0.8}, 
        'disgust': {'valence': -0.7, 'arousal': 0.5},
        'fear': {'valence': -0.5, 'arousal': 0.8},
        'sadness': {'valence': -0.8, 'arousal': 0.3},
        'surprise': {'valence': 0.1, 'arousal': 0.9},
        'neutral': {'valence': 0.0, 'arousal': 0.2}
    }
    
    # Contextual modifiers for complex expressions
    NEGATION_PATTERNS = [
        'not', 'no', 'never', 'nothing', 'nowhere', 'none', 'neither',
        'without', 'hardly', 'scarcely', 'barely', 'rarely', 'seldom',
        "n't", 'cannot', 'can\'t', 'won\'t', 'shouldn\'t', 'wouldn\'t'
    ]
    
    INTENSIFIERS = {
        'very': 1.3, 'extremely': 1.5, 'incredibly': 1.4, 'really': 1.2,
        'quite': 1.1, 'rather': 1.1, 'pretty': 1.1, 'fairly': 1.05,
        'somewhat': 0.8, 'slightly': 0.7, 'a bit': 0.7, 'a little': 0.7
    }
    
    def __init__(self, 
                 model_name: str = "j-hartmann/emotion-english-distilroberta-base",
                 device: Optional[int] = None,
                 batch_size: int = 8):
        """
        Initialize the robust emotional tagger
        
        Args:
            model_name: HuggingFace model identifier
            device: GPU device (0 for first GPU, -1 for CPU, None for auto)
            batch_size: Batch size for processing multiple texts
        """
        
        # Determine device
        if device is None:
            device = 0 if (HAS_TORCH and torch.cuda.is_available()) else -1
        self.device = device
        self.batch_size = batch_size
        
        # Initialize the emotion classification pipeline
        if HAS_TRANSFORMERS and HAS_TORCH:
            try:
                self.classifier = pipeline(
                    "text-classification",
                    model=model_name,
                    tokenizer=model_name,
                    return_all_scores=True,
                    device=device
                )
                
                # Test with simple input to ensure model works
                test_result = self.classifier("I am happy")
                print(f"✓ Emotional tagger initialized on {'GPU' if device >= 0 else 'CPU'}")
                
            except Exception as e:
                print(f"✗ Failed to initialize transformer model: {e}")
                print("Falling back to rule-based approach...")
                self.classifier = None
        else:
            print("✗ PyTorch/Transformers not available")
            print("Using rule-based emotional analysis...")
            self.classifier = None
            
        self.model_name = model_name
        
    def tag(self, text: Union[str, List[str]]) -> Union[EmotionalAnalysis, List[EmotionalAnalysis]]:
        """
        Analyze emotional content of text(s)
        
        Args:
            text: Single text string or list of texts
            
        Returns:
            EmotionalAnalysis object or list of objects
        """
        if isinstance(text, str):
            return self._tag_single(text)
        else:
            return self._tag_batch(text)
    
    def _tag_single(self, text: str) -> EmotionalAnalysis:
        """Tag single text with emotional analysis"""
        
        if not text or not text.strip():
            return EmotionalAnalysis(
                valence=0.0, arousal=0.0, confidence=0.0,
                dominant_emotion='neutral', emotion_scores={}
            )
        
        # Clean and preprocess text
        cleaned_text = self._preprocess_text(text)
        
        if self.classifier is None:
            # Fallback to simple rule-based analysis
            return self._rule_based_analysis(cleaned_text)
        
        try:
            # Get emotion predictions
            results = self.classifier(cleaned_text)
            
            # Parse results (format depends on model)
            if isinstance(results, list) and len(results) > 0:
                emotion_scores = {item['label'].lower(): item['score'] 
                                for item in results}
            else:
                emotion_scores = {}
            
            # Find dominant emotion
            if emotion_scores:
                dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
                max_confidence = emotion_scores[dominant_emotion]
            else:
                dominant_emotion = 'neutral'
                max_confidence = 0.5
                emotion_scores = {'neutral': 0.5}
            
            # Map to valence-arousal space
            valence, arousal = self._map_to_valence_arousal(
                emotion_scores, cleaned_text
            )
            
            return EmotionalAnalysis(
                valence=valence,
                arousal=arousal, 
                confidence=max_confidence,
                dominant_emotion=dominant_emotion,
                emotion_scores=emotion_scores
            )
            
        except Exception as e:
            print(f"Warning: Emotion classification failed: {e}")
            return self._rule_based_analysis(cleaned_text)
    
    def _tag_batch(self, texts: List[str]) -> List[EmotionalAnalysis]:
        """Tag multiple texts efficiently in batch"""
        
        if self.classifier is None:
            return [self._rule_based_analysis(text) for text in texts]
        
        results = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            cleaned_batch = [self._preprocess_text(text) for text in batch]
            
            try:
                # Batch classification
                batch_results = self.classifier(cleaned_batch)
                
                for text, result in zip(cleaned_batch, batch_results):
                    emotion_scores = {item['label'].lower(): item['score'] 
                                    for item in result}
                    
                    dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
                    max_confidence = emotion_scores[dominant_emotion]
                    
                    valence, arousal = self._map_to_valence_arousal(
                        emotion_scores, text
                    )
                    
                    results.append(EmotionalAnalysis(
                        valence=valence,
                        arousal=arousal,
                        confidence=max_confidence,
                        dominant_emotion=dominant_emotion,
                        emotion_scores=emotion_scores
                    ))
                    
            except Exception as e:
                print(f"Warning: Batch emotion classification failed: {e}")
                # Fallback to individual processing
                for text in cleaned_batch:
                    results.append(self._rule_based_analysis(text))
        
        return results
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for analysis"""
        # Basic cleaning
        text = text.strip()
        
        # Handle common contractions that affect sentiment
        contractions = {
            "isn't": "is not", "aren't": "are not", "wasn't": "was not",
            "weren't": "were not", "haven't": "have not", "hasn't": "has not",
            "hadn't": "had not", "won't": "will not", "wouldn't": "would not",
            "don't": "do not", "doesn't": "does not", "didn't": "did not",
            "can't": "cannot", "couldn't": "could not", "shouldn't": "should not",
            "mightn't": "might not", "mustn't": "must not"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
            text = text.replace(contraction.capitalize(), expansion.capitalize())
        
        return text
    
    def _map_to_valence_arousal(self, 
                               emotion_scores: Dict[str, float], 
                               text: str) -> tuple[float, float]:
        """Map emotion probabilities to valence-arousal coordinates"""
        
        valence = 0.0
        arousal = 0.0
        
        # Weight by emotion probabilities
        for emotion, score in emotion_scores.items():
            if emotion in self.EMOTION_MAPPING:
                coords = self.EMOTION_MAPPING[emotion]
                valence += coords['valence'] * score
                arousal += coords['arousal'] * score
        
        # Apply contextual modifiers
        valence, arousal = self._apply_contextual_modifiers(text, valence, arousal)
        
        # Clamp to valid ranges
        valence = np.clip(valence, -1.0, 1.0)
        arousal = np.clip(arousal, 0.0, 1.0)
        
        return float(valence), float(arousal)
    
    def _apply_contextual_modifiers(self, text: str, valence: float, arousal: float) -> tuple[float, float]:
        """Apply negation and intensifier modifications"""
        
        text_lower = text.lower()
        words = text_lower.split()
        
        # Check for negation (flips valence)
        has_negation = any(neg in text_lower for neg in self.NEGATION_PATTERNS)
        if has_negation:
            valence *= -0.8  # Partial negation, not complete reversal
            
        # Check for intensifiers
        intensity_multiplier = 1.0
        for word in words:
            if word in self.INTENSIFIERS:
                intensity_multiplier *= self.INTENSIFIERS[word]
                
        # Apply intensity (affects both valence magnitude and arousal)
        valence *= intensity_multiplier
        arousal *= min(intensity_multiplier, 1.2)  # Cap arousal intensification
        
        return valence, arousal
    
    def _rule_based_analysis(self, text: str) -> EmotionalAnalysis:
        """Simple rule-based fallback when transformer fails"""
        
        text_lower = text.lower()
        
        # Simple keyword-based emotion detection
        positive_words = ['happy', 'joy', 'love', 'good', 'great', 'amazing', 'wonderful']
        negative_words = ['sad', 'angry', 'hate', 'bad', 'terrible', 'awful', 'horrible']
        high_arousal_words = ['excited', 'thrilled', 'angry', 'furious', 'terrified', 'shocked']
        
        valence = 0.0
        arousal = 0.2  # Default low arousal
        
        # Count sentiment words
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            valence = 0.6
            dominant_emotion = 'joy'
        elif neg_count > pos_count:
            valence = -0.6
            dominant_emotion = 'sadness'
        else:
            valence = 0.0
            dominant_emotion = 'neutral'
            
        # Check for high arousal
        if any(word in text_lower for word in high_arousal_words):
            arousal = 0.8
            
        # Apply contextual modifiers
        valence, arousal = self._apply_contextual_modifiers(text, valence, arousal)
        
        return EmotionalAnalysis(
            valence=valence,
            arousal=arousal,
            confidence=0.4,  # Lower confidence for rule-based
            dominant_emotion=dominant_emotion,
            emotion_scores={dominant_emotion: 0.6, 'neutral': 0.4}
        )
    
    def analyze_emotional_trajectory(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze emotional changes over a sequence of texts"""
        
        analyses = self.tag(texts)
        
        if not analyses:
            return {'error': 'No texts to analyze'}
            
        valences = [a.valence for a in analyses]
        arousals = [a.arousal for a in analyses]
        confidences = [a.confidence for a in analyses]
        
        return {
            'trajectory': analyses,
            'summary': {
                'mean_valence': float(np.mean(valences)),
                'mean_arousal': float(np.mean(arousals)),
                'mean_confidence': float(np.mean(confidences)),
                'valence_std': float(np.std(valences)),
                'arousal_std': float(np.std(arousals)),
                'emotional_volatility': float(np.std(valences) + np.std(arousals)),
                'trend': 'positive' if valences[-1] > valences[0] else 'negative' if len(valences) > 1 else 'stable'
            }
        }
        
    def get_emotional_state_dict(self, text: str) -> Dict[str, float]:
        """Get emotional state in format compatible with existing DRIFT code"""
        analysis = self.tag(text)
        return {
            'valence': analysis.valence,
            'arousal': analysis.arousal,
            'confidence': analysis.confidence
        }


# Convenience functions for integration
def analyze_emotion(text: str, device: Optional[int] = None) -> EmotionalAnalysis:
    """Quick emotion analysis for single text"""
    tagger = RobustEmotionalTagger(device=device)
    return tagger.tag(text)


def get_emotional_state(text: str, device: Optional[int] = None) -> Dict[str, float]:
    """Get emotion state dict compatible with existing code"""
    tagger = RobustEmotionalTagger(device=device)
    return tagger.get_emotional_state_dict(text)


# Test the tagger
if __name__ == "__main__":
    print("=" * 60)
    print("ROBUST EMOTIONAL TAGGER V2 - TEST SUITE")
    print("=" * 60)
    
    # Initialize tagger
    tagger = RobustEmotionalTagger()
    
    # Test cases covering various emotional complexities
    test_cases = [
        "I am very happy today!",
        "I'm not happy at all",  # Negation
        "I am terrified but trying to stay calm",  # Mixed emotions
        "This is absolutely amazing!!!",  # Intensification
        "I don't think this is working",  # Subtle negativity
        "Whatever, I don't care anymore",  # Indifference/resignation
        "I'm so excited I can barely contain myself!",  # High arousal positive
        "I feel empty and numb inside",  # Low arousal negative
        "",  # Empty text
        "The weather is nice today"  # Neutral
    ]
    
    print("\n--- Single Text Analysis ---")
    for text in test_cases:
        if not text:
            continue
            
        result = tagger.tag(text)
        print(f"\nText: \"{text}\"")
        print(f"  Valence: {result.valence:+.3f} | Arousal: {result.arousal:.3f} | Confidence: {result.confidence:.3f}")
        print(f"  Emotion: {result.dominant_emotion}")
    
    print("\n--- Batch Processing ---")
    batch_results = tagger.tag(test_cases[:-2])  # Exclude empty text
    print(f"Processed {len(batch_results)} texts in batch")
    
    print("\n--- Emotional Trajectory ---")
    conversation = [
        "Hi, how are you?",
        "I'm feeling a bit down today",
        "Oh no, what happened?", 
        "Work has been really stressful",
        "That sounds tough, but you'll get through it",
        "Thanks, I'm feeling a bit better already"
    ]
    
    trajectory = tagger.analyze_emotional_trajectory(conversation)
    print(f"Conversation emotional trend: {trajectory['summary']['trend']}")
    print(f"Average valence: {trajectory['summary']['mean_valence']:+.3f}")
    print(f"Emotional volatility: {trajectory['summary']['emotional_volatility']:.3f}")
    
    print("\n" + "=" * 60)
    print("EMOTIONAL TAGGER V2 TEST COMPLETE")
    print("=" * 60)