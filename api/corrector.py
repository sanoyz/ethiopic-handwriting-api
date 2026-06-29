"""
Post-processing correction using deployment data
"""

import re
import json
from typing import Dict, List, Tuple, Any, Optional
from difflib import SequenceMatcher


class TextCorrector:
    """Post-processing correction for Ethiopic text using deployment data"""

    def __init__(self, deployment_data: Optional[Dict] = None):
        self.confusion_map = {}
        self.context_rules = []
        self.common_words = {}
        self.confidence_threshold = 0.6
        self.char_confidence = {}

        if deployment_data:
            self._load_from_deployment_data(deployment_data)
        else:
            self._initialize_default_rules()

    def _initialize_default_rules(self):
        """Initialize default correction rules"""
        self.confusion_map = {
            'ዚ': [('ማ', 0.85), ('መ', 0.10), ('ሞ', 0.05)],
            'ተተ': [('ተ', 0.90), ('ት', 0.05), ('ታ', 0.05)],
            'ዋ': [('ቹ', 0.65), ('ዎ', 0.20), ('ወ', 0.10)],
            'ስ': [('ዛ', 0.60), ('ሰ', 0.25), ('ሳ', 0.10)],
            'ነ': [('ከ', 0.55), ('ኔ', 0.20), ('ን', 0.15)],
            'ተ': [('በ', 0.50), ('ት', 0.30), ('ታ', 0.15)],
            'መ': [('ል', 0.50), ('ም', 0.30), ('ማ', 0.15)],
        }

        self.context_rules = [
            (r'ተ([ማሪ])', r'ተማ'),
            (r'አ([ዲስ])', r'አዲስ'),
            (r'ተዚ', r'ተማ'),
            (r'\s+', r' '),
            (r'\s([።፡፤፥?!])', r''),
            (r'([።፡፤፥?!])([^\s])', r' '),
        ]

        self.common_words = {
            'ተማሪ': True,
            'አዲስ': True,
            'አበባ': True,
            'የኢትዮጵያ': True,
            'ዋና': True,
            'ከተማ': True,
            'ናት': True,
            'ፍቅር': True,
            'ሰላም': True,
        }

    def _load_from_deployment_data(self, deployment_data: Dict):
        """Load correction rules from deployment data"""
        # Load top confusions
        if 'top_confusions' in deployment_data:
            for item in deployment_data['top_confusions']:
                target = item.get('target')
                predicted = item.get('predicted')
                rate = item.get('rate', 0)
                if target and predicted and rate > 0.05:
                    if target not in self.confusion_map:
                        self.confusion_map[target] = []
                    self.confusion_map[target].append((predicted, rate))

        # Load character confidence
        if 'char_confidence' in deployment_data:
            self.char_confidence = deployment_data['char_confidence']
            for char, stats in self.char_confidence.items():
                if stats.get('mean', 1.0) < 0.5:
                    self.confidence_threshold = min(
                        self.confidence_threshold, 
                        stats.get('mean', 0.6)
                    )

        # Load correction data
        if 'correction_data' in deployment_data:
            correction_data = deployment_data['correction_data']
            if 'characters' in correction_data:
                for char in correction_data['characters']:
                    self.common_words[char] = True

    def _get_context(self, text: str, pos: int, window: int = 3) -> str:
        """Get surrounding characters for context"""
        start = max(0, pos - window)
        end = min(len(text), pos + window + 1)
        return text[start:end]

    def _context_matches(self, context: str, suggestion: str) -> bool:
        """Check if suggestion fits context"""
        patterns = {
            'ማ': ['ተ', 'ሪ', 'ዎ', 'ሁ'],
            'ሪ': ['ተ', 'ማ', 'ዎ', 'ቹ'],
            'ቹ': ['ማ', 'ሪ', 'ዎ'],
        }

        for pattern_char, followers in patterns.items():
            if suggestion in followers and pattern_char in context:
                return True
        return True

    def _apply_character_corrections(self, text: str) -> Tuple[str, List[Dict]]:
        """Apply character-level corrections"""
        chars = list(text)
        corrections = []

        for i, char in enumerate(chars):
            if char in self.confusion_map:
                suggestions = self.confusion_map[char]
                if suggestions:
                    best_suggestion, conf = max(suggestions, key=lambda x: x[1])
                    if conf > self.confidence_threshold:
                        context = self._get_context(text, i)
                        if self._context_matches(context, best_suggestion):
                            chars[i] = best_suggestion
                            corrections.append({
                                'position': i,
                                'original': char,
                                'corrected': best_suggestion,
                                'confidence': conf
                            })

        return ''.join(chars), corrections

    def _apply_context_rules(self, text: str) -> Tuple[str, List[Dict]]:
        """Apply contextual rules"""
        corrections = []
        for pattern, replacement in self.context_rules:
            if re.search(pattern, text):
                text = re.sub(pattern, replacement, text)
                corrections.append({
                    'pattern': pattern,
                    'replacement': replacement
                })
        return text, corrections

    def _apply_dictionary_correction(self, text: str) -> Tuple[str, List[Dict]]:
        """Apply dictionary-based corrections"""
        words = text.split()
        corrected_words = []
        corrections = []

        for i, word in enumerate(words):
            if word in self.common_words:
                corrected_words.append(word)
                continue

            best_match = None
            best_score = 0

            for dict_word in self.common_words:
                score = SequenceMatcher(None, word, dict_word).ratio()
                if score > 0.7 and score > best_score:
                    best_match = dict_word
                    best_score = score

            if best_match and best_score > 0.7:
                corrected_words.append(best_match)
                corrections.append({
                    'position': i,
                    'original': word,
                    'corrected': best_match,
                    'confidence': best_score
                })
            else:
                corrected_words.append(word)

        return ' '.join(corrected_words), corrections

    def _fix_spacing(self, text: str) -> str:
        """Fix spacing issues"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s([።፡፤፥?!])', r'', text)
        text = re.sub(r'([።፡፤፥?!])([^\s])', r' ', text)
        return text.strip()

    def correct_text(self, text: str, confidence: float = 0.8) -> Dict[str, Any]:
        """Apply all correction strategies"""
        if not text or len(text) < 1:
            return {
                'original': text,
                'corrected': text,
                'corrections': [],
                'was_corrected': False,
                'confidence': confidence
            }

        original = text
        corrected = text
        all_corrections = []

        if confidence > 0.9:
            # High confidence - minimal correction only
            corrected, spacing_corrections = self._apply_context_rules(corrected)
            all_corrections.extend(spacing_corrections)
        else:
            # Apply all correction strategies
            corrected, char_corrections = self._apply_character_corrections(corrected)
            all_corrections.extend(char_corrections)

            corrected, context_corrections = self._apply_context_rules(corrected)
            all_corrections.extend(context_corrections)

            corrected, dict_corrections = self._apply_dictionary_correction(corrected)
            all_corrections.extend(dict_corrections)

        corrected = self._fix_spacing(corrected)
        was_corrected = original != corrected

        return {
            'original': original,
            'corrected': corrected,
            'corrections': all_corrections,
            'was_corrected': was_corrected,
            'correction_count': len(all_corrections),
            'confidence': confidence
        }
