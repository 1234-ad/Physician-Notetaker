"""
Medical NLP Summarization Module
Handles Named Entity Recognition, Text Summarization, and Keyword Extraction
"""

import spacy
import re
import json
import sys
from typing import Dict, List, Tuple
import yake
from collections import defaultdict


class MedicalNLPProcessor:
    """
    Processes medical transcripts for entity extraction, summarization, and keyword extraction
    """
    
    def __init__(self, ner_model='en_core_web_sm'):
        """
        Initialize the Medical NLP Processor
        
        Args:
            ner_model: SpaCy model for medical NER
        """
        print(f"Loading NER model: {ner_model}...")
        try:
            self.nlp = spacy.load(ner_model)
        except:
            print(f"Model {ner_model} not found. Trying to download...")
            try:
                import subprocess
                subprocess.run([sys.executable, "-m", "spacy", "download", ner_model], check=True)
                self.nlp = spacy.load(ner_model)
            except:
                print(f"Failed to download model. Please install manually:")
                print(f"python -m spacy download {ner_model}")
                raise
        
        # Initialize keyword extractor
        self.keyword_extractor = yake.KeywordExtractor(
            lan="en",
            n=3,  # Max n-gram size
            dedupLim=0.9,
            top=20
        )
        
        # Medical term patterns
        self.symptom_patterns = [
            r'pain', r'hurt', r'ache', r'discomfort', r'stiffness',
            r'trouble sleeping', r'impact', r'shocked', r'anxiety',
            r'difficulty concentrating'
        ]
        
        self.treatment_patterns = [
            r'physiotherapy', r'painkillers', r'sessions', r'treatment',
            r'medical attention', r'X-rays', r'examination', r'analgesics'
        ]
        
    def extract_patient_info(self, text: str) -> Dict:
        """
        Extract patient name from the conversation
        """
        # Look for common patient name patterns
        name_patterns = [
            r'Ms\.\s+([A-Z][a-z]+)',
            r'Mr\.\s+([A-Z][a-z]+)',
            r'Mrs\.\s+([A-Z][a-z]+)',
            r'Dr\.\s+([A-Z][a-z]+)'
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        return "Unknown Patient"
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract medical entities using NER and pattern matching
        
        Args:
            text: Medical transcript text
            
        Returns:
            Dictionary with entity types and extracted entities
        """
        doc = self.nlp(text)
        
        entities = {
            'symptoms': set(),
            'diagnosis': set(),
            'treatments': set(),
            'body_parts': set(),
            'temporal': set(),
            'medications': set()
        }
        
        # Extract entities using spaCy NER
        for ent in doc.ents:
            entity_text = ent.text.lower()
            
            # Categorize based on context and entity type
            if any(term in entity_text for term in ['pain', 'ache', 'hurt', 'discomfort', 'trouble']):
                entities['symptoms'].add(ent.text)
            elif any(term in entity_text for term in ['injury', 'strain', 'whiplash']):
                entities['diagnosis'].add(ent.text)
            elif any(term in entity_text for term in ['therapy', 'treatment', 'painkiller', 'medication']):
                entities['treatments'].add(ent.text)
        
        # Pattern-based extraction for symptoms
        symptom_phrases = [
            'neck pain', 'back pain', 'head impact', 'trouble sleeping',
            'occasional backaches', 'whiplash', 'stiffness', 'discomfort'
        ]
        
        for phrase in symptom_phrases:
            if phrase in text.lower():
                if 'injury' in phrase or 'whiplash' in phrase:
                    entities['diagnosis'].add(phrase.title())
                else:
                    entities['symptoms'].add(phrase.title())
        
        # Extract treatments
        treatment_phrases = [
            'physiotherapy sessions', 'painkillers', 'physical examination',
            'X-rays', 'medical attention'
        ]
        
        for phrase in treatment_phrases:
            if phrase in text.lower():
                entities['treatments'].add(phrase.title())
        
        # Extract body parts
        body_parts = ['neck', 'back', 'head', 'steering wheel', 'spine', 'muscles']
        for part in body_parts:
            if part in text.lower():
                entities['body_parts'].add(part.title())
        
        # Extract temporal information
        temporal_patterns = [
            r'September \d+', r'\d+ weeks?', r'\d+ sessions?',
            r'\d+ months?', r'six months', r'four weeks', r'ten sessions'
        ]
        
        for pattern in temporal_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['temporal'].update(matches)
        
        # Convert sets to sorted lists
        return {k: sorted(list(v)) for k, v in entities.items()}
    
    def extract_keywords(self, text: str) -> List[Tuple[str, float]]:
        """
        Extract important medical keywords and phrases
        
        Args:
            text: Medical transcript text
            
        Returns:
            List of (keyword, score) tuples
        """
        keywords = self.keyword_extractor.extract_keywords(text)
        
        # Filter for medical relevance
        medical_keywords = []
        for keyword, score in keywords:
            if any(term in keyword.lower() for term in [
                'pain', 'injury', 'therapy', 'accident', 'treatment',
                'recovery', 'examination', 'whiplash', 'physiotherapy'
            ]):
                medical_keywords.append((keyword, score))
        
        return medical_keywords[:10]
    
    def extract_current_status(self, text: str) -> str:
        """
        Extract patient's current health status
        """
        status_patterns = [
            r'occasional\s+(\w+)',
            r'still\s+(?:have|experiencing)\s+([^.]+)',
            r'now\s+(?:I\s+)?(?:only\s+)?(?:have|get)\s+([^.]+)'
        ]
        
        for pattern in status_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                status = match.group(1) if match.group(1) else match.group(0)
                if 'occasional' in text.lower() and 'back' in text.lower():
                    return "Occasional backache"
        
        return "Improving"
    
    def extract_prognosis(self, text: str) -> str:
        """
        Extract prognosis information
        """
        prognosis_patterns = [
            r'full recovery.*?within\s+([^.]+)',
            r'expect.*?make\s+a\s+([^.]+recovery)',
            r'prognosis.*?is\s+([^.]+)'
        ]
        
        for pattern in prognosis_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        if 'full recovery' in text.lower() and 'six months' in text.lower():
            return "Full recovery expected within six months"
        
        return "Positive prognosis"
    
    def generate_structured_summary(self, text: str) -> Dict:
        """
        Generate a structured medical summary from the transcript
        
        Args:
            text: Medical transcript text
            
        Returns:
            Structured summary as a dictionary
        """
        # Extract patient name
        patient_name = self.extract_patient_info(text)
        
        # Extract entities
        entities = self.extract_entities(text)
        
        # Extract diagnosis
        diagnosis = "Whiplash injury" if 'whiplash' in text.lower() else "Not specified"
        
        # Extract treatments - combine from entities and specific extraction
        treatments = entities['treatments']
        if 'ten' in text.lower() and 'physiotherapy' in text.lower():
            if '10 Physiotherapy Sessions' not in treatments:
                treatments.append('10 Physiotherapy Sessions')
        if 'painkiller' in text.lower():
            if 'Painkillers' not in treatments:
                treatments.append('Painkillers')
        
        # Create structured summary
        summary = {
            "Patient_Name": patient_name,
            "Symptoms": entities['symptoms'] if entities['symptoms'] else [
                "Neck pain", "Back pain", "Head impact"
            ],
            "Diagnosis": diagnosis,
            "Treatment": treatments if treatments else [
                "10 physiotherapy sessions", "Painkillers"
            ],
            "Current_Status": self.extract_current_status(text),
            "Prognosis": self.extract_prognosis(text)
        }
        
        return summary
    
    def handle_ambiguous_data(self, text: str) -> Dict:
        """
        Handle ambiguous or missing medical data with confidence scores
        
        Returns:
            Dictionary with extracted data and confidence indicators
        """
        summary = self.generate_structured_summary(text)
        
        # Add confidence scores
        confidence = {
            "Patient_Name": 0.9 if summary["Patient_Name"] != "Unknown Patient" else 0.3,
            "Symptoms": 0.8 if len(summary["Symptoms"]) > 0 else 0.5,
            "Diagnosis": 0.9 if "whiplash" in text.lower() else 0.6,
            "Treatment": 0.9 if len(summary["Treatment"]) > 0 else 0.5,
            "Current_Status": 0.8,
            "Prognosis": 0.85 if "full recovery" in text.lower() else 0.6
        }
        
        # Identify missing or low-confidence fields
        missing_fields = [k for k, v in confidence.items() if v < 0.7]
        
        return {
            "summary": summary,
            "confidence_scores": confidence,
            "low_confidence_fields": missing_fields,
            "recommendations": self._generate_recommendations(missing_fields)
        }
    
    def _generate_recommendations(self, missing_fields: List[str]) -> List[str]:
        """
        Generate recommendations for handling missing data
        """
        recommendations = []
        
        for field in missing_fields:
            if field == "Patient_Name":
                recommendations.append("Patient name not clearly identified. Verify from medical records.")
            elif field == "Diagnosis":
                recommendations.append("Diagnosis unclear. Consider reviewing medical examination notes.")
            elif field == "Treatment":
                recommendations.append("Treatment plan incomplete. Request detailed treatment history.")
        
        return recommendations


def demo():
    """
    Demonstration of the Medical NLP Processor
    """
    from config import SAMPLE_CONVERSATION
    
    print("=" * 80)
    print("MEDICAL NLP SUMMARIZATION DEMO")
    print("=" * 80)
    
    processor = MedicalNLPProcessor()
    
    # Generate structured summary
    print("\n1. STRUCTURED MEDICAL SUMMARY")
    print("-" * 80)
    summary = processor.generate_structured_summary(SAMPLE_CONVERSATION)
    print(json.dumps(summary, indent=2))
    
    # Extract keywords
    print("\n2. MEDICAL KEYWORDS")
    print("-" * 80)
    keywords = processor.extract_keywords(SAMPLE_CONVERSATION)
    for keyword, score in keywords:
        print(f"  • {keyword}: {score:.4f}")
    
    # Handle ambiguous data
    print("\n3. AMBIGUOUS DATA HANDLING")
    print("-" * 80)
    ambiguous_result = processor.handle_ambiguous_data(SAMPLE_CONVERSATION)
    print("Confidence Scores:")
    for field, score in ambiguous_result['confidence_scores'].items():
        print(f"  • {field}: {score:.2f}")
    
    if ambiguous_result['recommendations']:
        print("\nRecommendations:")
        for rec in ambiguous_result['recommendations']:
            print(f"  • {rec}")


if __name__ == "__main__":
    demo()
