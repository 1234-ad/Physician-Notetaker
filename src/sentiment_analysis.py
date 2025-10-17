"""
Sentiment & Intent Analysis Module
Handles sentiment classification and intent detection for patient dialogue
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline,
    BertTokenizer,
    BertForSequenceClassification,
    DistilBertTokenizer,
    DistilBertForSequenceClassification
)
from typing import Dict, List, Tuple
import re
import json
import numpy as np


class SentimentIntentAnalyzer:
    """
    Analyzes patient sentiment and intent from medical conversations
    """
    
    def __init__(self, sentiment_model='distilbert-base-uncased-finetuned-sst-2-english'):
        """
        Initialize the Sentiment and Intent Analyzer
        
        Args:
            sentiment_model: Pre-trained model for sentiment analysis
        """
        print(f"Loading sentiment analysis model: {sentiment_model}...")
        
        try:
            # Load sentiment analysis pipeline
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=sentiment_model,
                tokenizer=sentiment_model
            )
            
            # Initialize tokenizer for custom processing
            self.tokenizer = AutoTokenizer.from_pretrained(sentiment_model)
        except Exception as e:
            print(f"Note: Using rule-based sentiment analysis (transformers model not loaded: {e})")
            self.sentiment_pipeline = None
            self.tokenizer = None
        
        # Intent keywords mapping
        self.intent_keywords = {
            'Seeking reassurance': [
                'worried', 'hope', 'concern', 'afraid', 'nervous',
                'anxious', 'will I', 'should I worry', 'is it serious'
            ],
            'Reporting symptoms': [
                'pain', 'hurt', 'feel', 'experiencing', 'have',
                'discomfort', 'ache', 'trouble', 'difficulty'
            ],
            'Expressing concern': [
                'worried', 'concerned', 'scared', 'frightened',
                'uncertain', 'unsure', 'what if', 'nervous'
            ],
            'Asking questions': [
                'what', 'when', 'why', 'how', 'will', 'can',
                'should', 'could', 'would', '?'
            ],
            'Confirming information': [
                'so', 'right', 'correct', 'that means', 'you mean',
                'I understand', 'okay', 'yes'
            ],
            'Providing history': [
                'was', 'were', 'had', 'happened', 'went to',
                'I did', 'I took', 'on', 'ago', 'last'
            ]
        }
        
        # Sentiment mapping
        self.sentiment_mapping = {
            'POSITIVE': 'Reassured',
            'NEGATIVE': 'Anxious',
            'NEUTRAL': 'Neutral'
        }
        
    def analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment of patient dialogue
        
        Args:
            text: Patient's dialogue text
            
        Returns:
            Dictionary with sentiment label and confidence score
        """
        # Get sentiment prediction
        if self.sentiment_pipeline:
            try:
                result = self.sentiment_pipeline(text)[0]
                label = result['label']
                score = result['score']
            except:
                # Fallback to rule-based
                label, score = self._rule_based_sentiment(text)
        else:
            # Use rule-based sentiment
            label, score = self._rule_based_sentiment(text)
        
        # Adjust for medical context
        medical_sentiment = self._adjust_for_medical_context(text, label, score)
        
        return {
            'Sentiment': medical_sentiment['label'],
            'Confidence': medical_sentiment['score'],
            'Raw_Sentiment': label
        }
    
    def _rule_based_sentiment(self, text: str) -> tuple:
        """
        Simple rule-based sentiment analysis fallback
        """
        text_lower = text.lower()
        
        # Count positive and negative words
        positive_words = ['good', 'better', 'relief', 'thank', 'great', 'appreciate', 'glad', 'happy']
        negative_words = ['worried', 'concern', 'afraid', 'nervous', 'bad', 'worse', 'pain', 'hurt']
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return 'POSITIVE', 0.7
        elif neg_count > pos_count:
            return 'NEGATIVE', 0.7
        else:
            return 'NEUTRAL', 0.6
    
    def _adjust_for_medical_context(self, text: str, label: str, score: float) -> Dict:
        """
        Adjust sentiment based on medical context
        """
        text_lower = text.lower()
        
        # Anxiety indicators
        anxiety_words = ['worried', 'concern', 'afraid', 'nervous', 'hope', 'but']
        anxiety_count = sum(1 for word in anxiety_words if word in text_lower)
        
        # Reassurance indicators
        reassurance_words = ['better', 'relief', 'good', 'thank', 'great', 'appreciate']
        reassurance_count = sum(1 for word in reassurance_words if word in text_lower)
        
        # Symptom reporting (usually neutral)
        symptom_words = ['pain', 'hurt', 'discomfort', 'ache']
        symptom_count = sum(1 for word in symptom_words if word in text_lower)
        
        # Determine adjusted sentiment
        if anxiety_count > reassurance_count and anxiety_count > 0:
            return {'label': 'Anxious', 'score': min(score + 0.1, 1.0)}
        elif reassurance_count > anxiety_count and reassurance_count > 0:
            return {'label': 'Reassured', 'score': score}
        elif symptom_count > 0 and anxiety_count == 0:
            return {'label': 'Neutral', 'score': score}
        else:
            # Use original sentiment
            if label == 'POSITIVE':
                return {'label': 'Reassured', 'score': score}
            elif label == 'NEGATIVE':
                return {'label': 'Anxious', 'score': score}
            else:
                return {'label': 'Neutral', 'score': score}
    
    def detect_intent(self, text: str) -> Dict:
        """
        Detect patient intent from dialogue
        
        Args:
            text: Patient's dialogue text
            
        Returns:
            Dictionary with detected intent and confidence
        """
        text_lower = text.lower()
        
        # Calculate intent scores
        intent_scores = {}
        for intent, keywords in self.intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            intent_scores[intent] = score
        
        # Get top intent
        if max(intent_scores.values()) > 0:
            top_intent = max(intent_scores, key=intent_scores.get)
            
            # Calculate confidence based on keyword matches
            total_matches = sum(intent_scores.values())
            confidence = intent_scores[top_intent] / total_matches if total_matches > 0 else 0.5
        else:
            top_intent = 'Providing information'
            confidence = 0.6
        
        # Get secondary intents
        secondary_intents = [
            intent for intent, score in intent_scores.items()
            if score > 0 and intent != top_intent
        ]
        
        return {
            'Intent': top_intent,
            'Confidence': min(confidence, 1.0),
            'Secondary_Intents': secondary_intents[:2]
        }
    
    def analyze_patient_dialogue(self, text: str) -> Dict:
        """
        Comprehensive analysis of patient dialogue
        
        Args:
            text: Patient's dialogue text
            
        Returns:
            Dictionary with sentiment and intent analysis
        """
        sentiment_result = self.analyze_sentiment(text)
        intent_result = self.detect_intent(text)
        
        return {
            'Text': text,
            'Sentiment': sentiment_result['Sentiment'],
            'Sentiment_Confidence': f"{sentiment_result['Confidence']:.2f}",
            'Intent': intent_result['Intent'],
            'Intent_Confidence': f"{intent_result['Confidence']:.2f}",
            'Secondary_Intents': intent_result['Secondary_Intents']
        }
    
    def analyze_conversation(self, conversation: str) -> List[Dict]:
        """
        Analyze an entire conversation, extracting patient dialogues
        
        Args:
            conversation: Full conversation text
            
        Returns:
            List of analysis results for each patient statement
        """
        # Split conversation into turns
        lines = conversation.strip().split('\n')
        
        patient_statements = []
        for line in lines:
            if line.strip().startswith('Patient:'):
                # Extract patient statement
                statement = line.replace('Patient:', '').strip()
                if statement:
                    analysis = self.analyze_patient_dialogue(statement)
                    patient_statements.append(analysis)
        
        return patient_statements
    
    def get_conversation_summary(self, conversation: str) -> Dict:
        """
        Get overall sentiment and intent summary for the conversation
        
        Args:
            conversation: Full conversation text
            
        Returns:
            Summary statistics
        """
        analyses = self.analyze_conversation(conversation)
        
        if not analyses:
            return {
                'Total_Patient_Statements': 0,
                'Overall_Sentiment': 'Unknown',
                'Most_Common_Intent': 'Unknown'
            }
        
        # Count sentiments
        sentiments = [a['Sentiment'] for a in analyses]
        sentiment_counts = {s: sentiments.count(s) for s in set(sentiments)}
        
        # Count intents
        intents = [a['Intent'] for a in analyses]
        intent_counts = {i: intents.count(i) for i in set(intents)}
        
        return {
            'Total_Patient_Statements': len(analyses),
            'Overall_Sentiment': max(sentiment_counts, key=sentiment_counts.get),
            'Sentiment_Distribution': sentiment_counts,
            'Most_Common_Intent': max(intent_counts, key=intent_counts.get),
            'Intent_Distribution': intent_counts,
            'Detailed_Analysis': analyses
        }
    
    @staticmethod
    def fine_tune_bert_for_medical_sentiment():
        """
        Instructions for fine-tuning BERT for medical sentiment detection
        
        Returns:
            Dictionary with fine-tuning guidelines
        """
        return {
            "Dataset_Requirements": [
                "Medical Patient-Doctor Conversations",
                "Labeled with sentiments: Anxious, Neutral, Reassured",
                "Minimum 5000-10000 labeled examples",
                "Balanced across sentiment classes"
            ],
            "Recommended_Datasets": [
                "MedDialog (Medical conversation dataset)",
                "MIMIC-III Clinical Notes (with annotation)",
                "Custom annotated patient conversation data",
                "Augmented data using paraphrasing"
            ],
            "Fine_Tuning_Process": {
                "1. Data Preparation": "Tokenize and format data for BERT input",
                "2. Model Selection": "Use BioClinicalBERT or BlueBERT for medical domain",
                "3. Training": "Fine-tune with learning rate 2e-5, batch size 16, 3-5 epochs",
                "4. Validation": "Use stratified k-fold cross-validation",
                "5. Evaluation": "Measure accuracy, F1-score, and confusion matrix"
            },
            "Code_Example": """
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments

# Load medical BERT model
model = BertForSequenceClassification.from_pretrained(
    'emilyalsentzer/Bio_ClinicalBERT',
    num_labels=3  # Anxious, Neutral, Reassured
)

# Training arguments
training_args = TrainingArguments(
    output_dir='./medical_sentiment_model',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    evaluation_strategy='epoch'
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)
trainer.train()
            """
        }


def demo():
    """
    Demonstration of Sentiment and Intent Analysis
    """
    from config import SAMPLE_CONVERSATION
    
    print("=" * 80)
    print("SENTIMENT & INTENT ANALYSIS DEMO")
    print("=" * 80)
    
    analyzer = SentimentIntentAnalyzer()
    
    # Test individual patient statements
    test_statements = [
        "I'm a bit worried about my back pain, but I hope it gets better soon.",
        "Good morning, doctor. I'm doing better, but I still have some discomfort now and then.",
        "The first four weeks were rough. My neck and back pain were really bad.",
        "That's a relief!",
        "Thank you, doctor. I appreciate it."
    ]
    
    print("\n1. INDIVIDUAL STATEMENT ANALYSIS")
    print("-" * 80)
    
    for i, statement in enumerate(test_statements, 1):
        print(f"\nStatement {i}: \"{statement}\"")
        result = analyzer.analyze_patient_dialogue(statement)
        print(f"  Sentiment: {result['Sentiment']} (Confidence: {result['Sentiment_Confidence']})")
        print(f"  Intent: {result['Intent']} (Confidence: {result['Intent_Confidence']})")
        if result['Secondary_Intents']:
            print(f"  Secondary Intents: {', '.join(result['Secondary_Intents'])}")
    
    # Analyze full conversation
    print("\n\n2. FULL CONVERSATION ANALYSIS")
    print("-" * 80)
    
    summary = analyzer.get_conversation_summary(SAMPLE_CONVERSATION)
    print(json.dumps({k: v for k, v in summary.items() if k != 'Detailed_Analysis'}, indent=2))
    
    # Fine-tuning instructions
    print("\n\n3. FINE-TUNING BERT FOR MEDICAL SENTIMENT")
    print("-" * 80)
    
    instructions = analyzer.fine_tune_bert_for_medical_sentiment()
    print("\nDataset Requirements:")
    for req in instructions['Dataset_Requirements']:
        print(f"  • {req}")
    
    print("\nRecommended Datasets:")
    for dataset in instructions['Recommended_Datasets']:
        print(f"  • {dataset}")


if __name__ == "__main__":
    demo()
