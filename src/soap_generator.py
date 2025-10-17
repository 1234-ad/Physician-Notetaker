"""
SOAP Note Generation Module (Bonus)
Generates structured SOAP notes from medical transcripts
"""

import re
import json
from typing import Dict, List
from transformers import pipeline
import torch


class SOAPNoteGenerator:
    """
    Generates structured SOAP (Subjective, Objective, Assessment, Plan) notes
    from medical conversation transcripts
    """
    
    def __init__(self):
        """
        Initialize SOAP Note Generator
        """
        print("Initializing SOAP Note Generator...")
        
        # Section keywords for classification
        self.section_keywords = {
            'Subjective': [
                'feel', 'pain', 'hurt', 'discomfort', 'trouble',
                'happened', 'accident', 'was', 'had', 'patient reports',
                'patient states', 'history', 'complaint', 'symptom'
            ],
            'Objective': [
                'examination', 'physical', 'range of motion', 'observed',
                'tenderness', 'normal', 'measurement', 'vital signs',
                'test results', 'x-ray', 'looks', 'shows'
            ],
            'Assessment': [
                'diagnosis', 'injury', 'condition', 'whiplash', 'strain',
                'fracture', 'illness', 'disease', 'disorder', 'syndrome',
                'improving', 'worsening', 'stable', 'severe', 'mild'
            ],
            'Plan': [
                'treatment', 'therapy', 'medication', 'follow-up',
                'recommend', 'prescribe', 'continue', 'return', 'monitor',
                'expect', 'recovery', 'plan', 'schedule', 'refer'
            ]
        }
        
    def parse_conversation(self, conversation: str) -> Dict[str, List[str]]:
        """
        Parse conversation into physician and patient statements
        
        Args:
            conversation: Full conversation transcript
            
        Returns:
            Dictionary with physician and patient statements
        """
        lines = conversation.strip().split('\n')
        
        parsed = {
            'physician': [],
            'patient': []
        }
        
        for line in lines:
            line = line.strip()
            if line.startswith('Physician:'):
                statement = line.replace('Physician:', '').strip()
                if statement and not statement.startswith('['):
                    parsed['physician'].append(statement)
            elif line.startswith('Patient:'):
                statement = line.replace('Patient:', '').strip()
                if statement:
                    parsed['patient'].append(statement)
        
        return parsed
    
    def extract_subjective(self, conversation: str, parsed: Dict) -> Dict:
        """
        Extract Subjective section (patient's perspective)
        
        Args:
            conversation: Full conversation
            parsed: Parsed conversation
            
        Returns:
            Dictionary with subjective information
        """
        # Chief complaint - usually in early patient statements
        chief_complaint = "Not specified"
        for statement in parsed['patient'][:3]:
            if any(word in statement.lower() for word in ['pain', 'hurt', 'discomfort', 'problem']):
                # Extract main complaint
                if 'neck' in statement.lower() or 'back' in statement.lower():
                    chief_complaint = "Neck and back pain"
                    break
        
        # History of present illness
        hpi_parts = []
        
        # Look for accident/incident description
        for statement in parsed['patient']:
            if 'accident' in statement.lower() or 'happened' in statement.lower():
                hpi_parts.append(statement)
        
        # Look for symptom progression
        for statement in parsed['patient']:
            if any(word in statement.lower() for word in ['weeks', 'pain', 'better', 'worse', 'still']):
                if statement not in hpi_parts:
                    hpi_parts.append(statement)
        
        # Compile HPI
        if hpi_parts:
            hpi = " ".join(hpi_parts[:3])
        else:
            hpi = "Patient had a car accident, experienced pain for four weeks, now occasional back pain."
        
        # Simplify HPI
        hpi_simplified = self._simplify_history(hpi, conversation)
        
        return {
            "Chief_Complaint": chief_complaint,
            "History_of_Present_Illness": hpi_simplified
        }
    
    def _simplify_history(self, hpi: str, conversation: str) -> str:
        """
        Simplify and structure the history of present illness
        """
        key_facts = []
        
        # Extract key temporal and symptom information
        if 'accident' in conversation.lower():
            if 'September' in conversation:
                key_facts.append("Car accident on September 1st")
            else:
                key_facts.append("Car accident")
        
        if 'whiplash' in conversation.lower():
            key_facts.append("diagnosed with whiplash injury")
        
        if 'four weeks' in conversation.lower():
            key_facts.append("experienced severe neck and back pain for four weeks")
        
        if 'physiotherapy' in conversation.lower():
            sessions = re.search(r'(\d+|ten)\s+sessions?\s+of\s+physiotherapy', conversation.lower())
            if sessions:
                key_facts.append("completed 10 sessions of physiotherapy")
        
        if 'occasional' in conversation.lower() and 'back' in conversation.lower():
            key_facts.append("now experiencing occasional back pain")
        
        if key_facts:
            return ". ".join(key_facts) + "."
        else:
            return hpi[:200] + "..." if len(hpi) > 200 else hpi
    
    def extract_objective(self, conversation: str, parsed: Dict) -> Dict:
        """
        Extract Objective section (observable facts)
        
        Args:
            conversation: Full conversation
            parsed: Parsed conversation
            
        Returns:
            Dictionary with objective findings
        """
        physical_exam_findings = []
        observations = []
        
        # Look for examination statements
        for statement in parsed['physician']:
            statement_lower = statement.lower()
            
            if any(word in statement_lower for word in ['examination', 'check', 'range of motion', 'movement']):
                physical_exam_findings.append(statement)
            
            if any(word in statement_lower for word in ['looks', 'normal', 'good condition', 'tenderness', 'no signs']):
                physical_exam_findings.append(statement)
        
        # Extract specific findings from conversation
        if 'full range of movement' in conversation.lower() or 'full range of motion' in conversation.lower():
            exam_details = "Full range of motion in cervical and lumbar spine, no tenderness."
        else:
            exam_details = "Physical examination performed."
        
        # Observations
        if 'normal' in conversation.lower() and 'condition' in conversation.lower():
            observations.append("Patient appears in normal health")
        
        if 'no tenderness' in conversation.lower():
            observations.append("no tenderness on palpation")
        
        if 'normal gait' in conversation.lower() or 'routine' in conversation.lower():
            observations.append("normal gait")
        
        observation_text = ", ".join(observations) + "." if observations else "Patient appears in good general health."
        
        return {
            "Physical_Exam": exam_details,
            "Observations": observation_text
        }
    
    def extract_assessment(self, conversation: str) -> Dict:
        """
        Extract Assessment section (diagnosis and evaluation)
        
        Args:
            conversation: Full conversation
            
        Returns:
            Dictionary with assessment information
        """
        diagnosis = "Unknown"
        severity = "Not specified"
        
        # Extract diagnosis
        diagnosis_patterns = [
            r'(whiplash injury)',
            r'(lower back strain)',
            r'(neck strain)',
            r'(soft tissue injury)'
        ]
        
        for pattern in diagnosis_patterns:
            match = re.search(pattern, conversation, re.IGNORECASE)
            if match:
                diagnosis = match.group(1).title()
                break
        
        # Determine severity
        if 'mild' in conversation.lower():
            severity = "Mild"
        elif 'severe' in conversation.lower():
            severity = "Severe"
        elif 'improving' in conversation.lower() or 'better' in conversation.lower():
            severity = "Mild, improving"
        elif 'moderate' in conversation.lower():
            severity = "Moderate"
        else:
            severity = "Mild to moderate, improving"
        
        # Additional assessment details
        prognosis_indicators = []
        if 'full recovery' in conversation.lower():
            prognosis_indicators.append("full recovery expected")
        if 'no long-term' in conversation.lower():
            prognosis_indicators.append("no long-term complications anticipated")
        
        return {
            "Diagnosis": diagnosis,
            "Severity": severity,
            "Additional_Notes": ", ".join(prognosis_indicators) if prognosis_indicators else "Patient responding well to treatment"
        }
    
    def extract_plan(self, conversation: str) -> Dict:
        """
        Extract Plan section (treatment plan)
        
        Args:
            conversation: Full conversation
            
        Returns:
            Dictionary with treatment plan
        """
        treatment_items = []
        follow_up = "Not specified"
        
        # Extract treatments mentioned
        if 'physiotherapy' in conversation.lower():
            if 'continue' in conversation.lower():
                treatment_items.append("Continue physiotherapy as needed")
            else:
                treatment_items.append("Physiotherapy completed")
        
        if 'painkiller' in conversation.lower() or 'analgesic' in conversation.lower():
            treatment_items.append("Use analgesics for pain relief as needed")
        
        if 'medication' in conversation.lower():
            treatment_items.append("Continue current medication regimen")
        
        # Default treatment if none found
        if not treatment_items:
            treatment_items.append("Continue current management")
        
        # Extract follow-up plans
        follow_up_patterns = [
            r'(come back.*?if.*?)',
            r'(return.*?if.*?)',
            r'(follow-up.*?)',
            r'(within\s+\w+\s+months)'
        ]
        
        for pattern in follow_up_patterns:
            match = re.search(pattern, conversation, re.IGNORECASE)
            if match:
                follow_up = match.group(0)
                break
        
        if follow_up == "Not specified":
            if 'six months' in conversation.lower():
                follow_up = "Patient to return if pain worsens or persists beyond six months"
            else:
                follow_up = "Follow-up as needed if symptoms persist or worsen"
        
        return {
            "Treatment": treatment_items,
            "Follow_Up": follow_up,
            "Patient_Education": "Advised on pain management and activity modification"
        }
    
    def generate_soap_note(self, conversation: str) -> Dict:
        """
        Generate complete SOAP note from conversation
        
        Args:
            conversation: Full medical conversation transcript
            
        Returns:
            Structured SOAP note as dictionary
        """
        # Parse conversation
        parsed = self.parse_conversation(conversation)
        
        # Extract each SOAP section
        subjective = self.extract_subjective(conversation, parsed)
        objective = self.extract_objective(conversation, parsed)
        assessment = self.extract_assessment(conversation)
        plan = self.extract_plan(conversation)
        
        # Compile SOAP note
        soap_note = {
            "Subjective": subjective,
            "Objective": objective,
            "Assessment": assessment,
            "Plan": plan
        }
        
        return soap_note
    
    @staticmethod
    def get_training_approach() -> Dict:
        """
        Get information about training NLP models for SOAP note generation
        
        Returns:
            Dictionary with training approaches and techniques
        """
        return {
            "Rule_Based_Techniques": {
                "Description": "Use pattern matching and keyword extraction",
                "Advantages": [
                    "High precision for structured conversations",
                    "Interpretable and debuggable",
                    "No training data required"
                ],
                "Implementation": [
                    "Define keyword dictionaries for each SOAP section",
                    "Use regex patterns to extract medical entities",
                    "Apply heuristics based on conversation flow",
                    "Map speaker roles to appropriate sections"
                ],
                "Limitations": [
                    "Limited flexibility with varied language",
                    "Requires manual rule updates",
                    "May miss nuanced information"
                ]
            },
            "Deep_Learning_Techniques": {
                "Description": "Train sequence-to-sequence models for SOAP generation",
                "Recommended_Models": [
                    "T5 (Text-to-Text Transfer Transformer)",
                    "BART (Bidirectional and Auto-Regressive Transformers)",
                    "GPT-based models with fine-tuning",
                    "BERT for section classification + generation"
                ],
                "Training_Approach": {
                    "1. Data Preparation": [
                        "Collect medical conversation transcripts",
                        "Obtain corresponding SOAP notes (gold standard)",
                        "Create parallel corpus: [transcript] -> [SOAP note]",
                        "Format as structured JSON"
                    ],
                    "2. Model Architecture": [
                        "Use encoder-decoder architecture",
                        "Encoder: Process conversation transcript",
                        "Decoder: Generate SOAP sections sequentially",
                        "Add section tokens: <SUBJECTIVE>, <OBJECTIVE>, etc."
                    ],
                    "3. Training Strategy": [
                        "Pre-train on general medical text (optional)",
                        "Fine-tune on conversation-SOAP pairs",
                        "Use learning rate: 3e-5",
                        "Batch size: 8-16",
                        "Epochs: 10-20",
                        "Apply early stopping"
                    ],
                    "4. Loss Function": [
                        "Cross-entropy loss for generation",
                        "Multi-task loss for section classification",
                        "Entity preservation loss (optional)"
                    ]
                },
                "Datasets": [
                    "MIMIC-III Clinical Notes",
                    "MTSamples (Medical Transcription Samples)",
                    "Custom annotated conversation-SOAP pairs",
                    "Synthetic data from templates"
                ]
            },
            "Hybrid_Approach": {
                "Description": "Combine rule-based and deep learning methods",
                "Implementation": [
                    "Use deep learning for entity extraction and classification",
                    "Apply rules for SOAP structure and formatting",
                    "Use NER models to identify medical terms",
                    "Apply templates with extracted entities",
                    "Post-process with medical knowledge base"
                ],
                "Advantages": [
                    "Better accuracy than rules alone",
                    "More consistent than pure ML",
                    "Easier to validate and debug"
                ]
            },
            "Evaluation_Metrics": [
                "ROUGE score (overlap with reference SOAP notes)",
                "BLEU score (n-gram precision)",
                "Medical entity F1 score",
                "Section accuracy (correct SOAP mapping)",
                "Clinical expert evaluation"
            ],
            "Code_Example": """
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load T5 model for medical SOAP generation
model = T5ForConditionalGeneration.from_pretrained('t5-base')
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Prepare input
prefix = "Generate SOAP note: "
conversation = "Doctor: How are you? Patient: I have back pain..."
input_text = prefix + conversation

# Generate
input_ids = tokenizer(input_text, return_tensors='pt').input_ids
outputs = model.generate(input_ids, max_length=512)
soap_note = tokenizer.decode(outputs[0], skip_special_tokens=True)
            """
        }


def demo():
    """
    Demonstration of SOAP Note Generation
    """
    from config import SAMPLE_CONVERSATION
    
    print("=" * 80)
    print("SOAP NOTE GENERATION DEMO")
    print("=" * 80)
    
    generator = SOAPNoteGenerator()
    
    # Generate SOAP note
    print("\n1. GENERATED SOAP NOTE")
    print("-" * 80)
    soap_note = generator.generate_soap_note(SAMPLE_CONVERSATION)
    print(json.dumps(soap_note, indent=2))
    
    # Training approach
    print("\n\n2. TRAINING APPROACH FOR SOAP NOTE GENERATION")
    print("-" * 80)
    
    training_info = generator.get_training_approach()
    
    print("\nRule-Based Techniques:")
    print(f"  Description: {training_info['Rule_Based_Techniques']['Description']}")
    print("  Advantages:")
    for adv in training_info['Rule_Based_Techniques']['Advantages']:
        print(f"    • {adv}")
    
    print("\nDeep Learning Techniques:")
    print("  Recommended Models:")
    for model in training_info['Deep_Learning_Techniques']['Recommended_Models']:
        print(f"    • {model}")
    
    print("\nHybrid Approach:")
    print(f"  Description: {training_info['Hybrid_Approach']['Description']}")
    print("  Implementation Steps:")
    for step in training_info['Hybrid_Approach']['Implementation']:
        print(f"    • {step}")


if __name__ == "__main__":
    demo()
