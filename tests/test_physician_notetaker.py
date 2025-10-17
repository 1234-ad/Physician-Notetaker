"""
Unit Tests for Physician Notetaker
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.medical_nlp import MedicalNLPProcessor
from src.sentiment_analysis import SentimentIntentAnalyzer
from src.soap_generator import SOAPNoteGenerator


class TestMedicalNLP:
    """Test cases for Medical NLP Module"""
    
    @pytest.fixture
    def processor(self):
        return MedicalNLPProcessor()
    
    def test_extract_patient_info(self, processor):
        text = "Good morning, Ms. Jones. How are you feeling today?"
        patient_name = processor.extract_patient_info(text)
        assert patient_name == "Jones"
    
    def test_extract_entities(self, processor):
        text = "Patient has neck pain and back pain. Diagnosed with whiplash injury."
        entities = processor.extract_entities(text)
        
        assert 'symptoms' in entities
        assert 'diagnosis' in entities
        assert len(entities['symptoms']) > 0
    
    def test_generate_structured_summary(self, processor):
        text = "Patient Ms. Smith has neck pain. Diagnosed with whiplash. Prescribed painkillers."
        summary = processor.generate_structured_summary(text)
        
        assert 'Patient_Name' in summary
        assert 'Symptoms' in summary
        assert 'Diagnosis' in summary
        assert 'Treatment' in summary


class TestSentimentAnalysis:
    """Test cases for Sentiment Analysis Module"""
    
    @pytest.fixture
    def analyzer(self):
        return SentimentIntentAnalyzer()
    
    def test_analyze_sentiment_anxious(self, analyzer):
        text = "I'm worried about my back pain."
        result = analyzer.analyze_sentiment(text)
        
        assert 'Sentiment' in result
        assert result['Sentiment'] in ['Anxious', 'Neutral', 'Reassured']
    
    def test_analyze_sentiment_reassured(self, analyzer):
        text = "That's a relief! Thank you, doctor."
        result = analyzer.analyze_sentiment(text)
        
        assert result['Sentiment'] in ['Reassured', 'Neutral']
    
    def test_detect_intent(self, analyzer):
        text = "I'm worried about my back pain, will it get better?"
        result = analyzer.detect_intent(text)
        
        assert 'Intent' in result
        assert 'Confidence' in result
    
    def test_analyze_patient_dialogue(self, analyzer):
        text = "I hope my pain gets better soon."
        result = analyzer.analyze_patient_dialogue(text)
        
        assert 'Sentiment' in result
        assert 'Intent' in result
        assert 'Confidence' in result


class TestSOAPGenerator:
    """Test cases for SOAP Note Generator"""
    
    @pytest.fixture
    def generator(self):
        return SOAPNoteGenerator()
    
    def test_parse_conversation(self, generator):
        conversation = """
        Physician: How are you?
        Patient: I have pain.
        """
        parsed = generator.parse_conversation(conversation)
        
        assert 'physician' in parsed
        assert 'patient' in parsed
        assert len(parsed['patient']) > 0
    
    def test_generate_soap_note(self, generator):
        conversation = """
        Doctor: How are you feeling?
        Patient: I have back pain from a car accident.
        Doctor: I see whiplash injury. Let's do physiotherapy.
        """
        soap_note = generator.generate_soap_note(conversation)
        
        assert 'Subjective' in soap_note
        assert 'Objective' in soap_note
        assert 'Assessment' in soap_note
        assert 'Plan' in soap_note
    
    def test_extract_subjective(self, generator):
        conversation = "Patient reports neck and back pain for four weeks."
        parsed = {'patient': ["I have neck and back pain for four weeks"]}
        subjective = generator.extract_subjective(conversation, parsed)
        
        assert 'Chief_Complaint' in subjective
        assert 'History_of_Present_Illness' in subjective


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
