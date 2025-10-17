# 🩺 Physician Notetaker

An advanced AI system for medical transcription, NLP-based summarization, sentiment analysis, and SOAP note generation. This project processes physician-patient conversations to extract key medical information, analyze patient sentiment, and generate structured clinical documentation.

## 🌟 Features

### 1. Medical NLP Summarization
- **Named Entity Recognition (NER)**: Extract symptoms, diagnoses, treatments, medications, and body parts using SciSpacy
- **Structured Summarization**: Convert conversations into JSON medical reports
- **Keyword Extraction**: Identify important medical phrases using YAKE
- **Ambiguous Data Handling**: Confidence scoring and quality assessment

### 2. Sentiment & Intent Analysis
- **Sentiment Classification**: Detect patient emotions (Anxious, Neutral, Reassured) using DistilBERT
- **Intent Detection**: Identify patient communication goals (Seeking reassurance, Reporting symptoms, etc.)
- **Conversation Analysis**: Track sentiment progression throughout the consultation
- **Multi-statement Processing**: Analyze entire conversations with statistical summaries

### 3. SOAP Note Generation (Bonus)
- **Automated SOAP Notes**: Generate structured clinical documentation
- **Section Extraction**: Automatically categorize information into Subjective, Objective, Assessment, and Plan
- **Clinical Formatting**: Output in standard medical documentation format
- **Template-based + NLP**: Hybrid approach for accurate medical note generation

## 📁 Project Structure

```
Physician-Notetaker-main/
│
├── src/
│   ├── __init__.py
│   ├── medical_nlp.py          # NER, summarization, keyword extraction
│   ├── sentiment_analysis.py   # Sentiment and intent analysis
│   └── soap_generator.py        # SOAP note generation
│
├── main.py                      # Main application with CLI
├── config.py                    # Configuration and sample data
├── demo.ipynb                   # Interactive Jupyter notebook demo
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended

### Step 1: Clone the Repository

```bash
cd Physician-Notetaker-main
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download SpaCy Medical Model

```bash
# Install SciSpacy medical model
pip install https://s3-us-west-2.amazonaws.com/ai2-s3-scispacy/releases/v0.5.1/en_core_sci_md-0.5.1.tar.gz
```

### Step 5: Verify Installation

```bash
python -c "import spacy; import transformers; print('✅ Installation successful!')"
```

## 💻 Usage

### Command Line Interface

#### Run Demo with Sample Conversation

```bash
python main.py --demo
```

#### Process Custom Conversation File

```bash
python main.py --input conversation.txt --output results.json
```

#### Generate Text Report

```bash
python main.py --input conversation.txt --format text
```

### Python API

```python
from main import PhysicianNotetaker

# Initialize the system
app = PhysicianNotetaker()

# Process a conversation
conversation = """
Doctor: How are you feeling today?
Patient: I have severe back pain.
"""

results = app.process_conversation(conversation)
print(results)
```

### Jupyter Notebook Demo

```bash
jupyter notebook demo.ipynb
```

The interactive notebook provides:
- Step-by-step demonstrations of all features
- Visualizations of sentiment trends
- Formatted output displays
- Customizable examples

## 📊 Example Output

### Medical Summary (JSON)

```json
{
  "Patient_Name": "Jones",
  "Symptoms": ["Neck pain", "Back pain", "Head impact"],
  "Diagnosis": "Whiplash injury",
  "Treatment": ["10 Physiotherapy Sessions", "Painkillers"],
  "Current_Status": "Occasional backache",
  "Prognosis": "Full recovery expected within six months"
}
```

### Sentiment Analysis (JSON)

```json
{
  "Sentiment": "Anxious",
  "Intent": "Seeking reassurance",
  "Confidence": "0.87"
}
```

### SOAP Note (JSON)

```json
{
  "Subjective": {
    "Chief_Complaint": "Neck and back pain",
    "History_of_Present_Illness": "Car accident on September 1st. diagnosed with whiplash injury. experienced severe neck and back pain for four weeks. completed 10 sessions of physiotherapy. now experiencing occasional back pain."
  },
  "Objective": {
    "Physical_Exam": "Full range of motion in cervical and lumbar spine, no tenderness.",
    "Observations": "Patient appears in normal health, no tenderness on palpation, normal gait."
  },
  "Assessment": {
    "Diagnosis": "Whiplash Injury",
    "Severity": "Mild, improving"
  },
  "Plan": {
    "Treatment": ["Continue physiotherapy as needed", "Use analgesics for pain relief as needed"],
    "Follow_Up": "Patient to return if pain worsens or persists beyond six months"
  }
}
```

## 🧪 Running Individual Modules

### Test Medical NLP

```bash
python src/medical_nlp.py
```

### Test Sentiment Analysis

```bash
python src/sentiment_analysis.py
```

### Test SOAP Generation

```bash
python src/soap_generator.py
```

## 📚 Answers to Technical Questions

### Q1: How would you handle ambiguous or missing medical data in the transcript?

**Implementation in `medical_nlp.py`:**
- **Confidence Scoring**: Each extracted field has a confidence score (0-1)
- **Multi-source Validation**: Use pattern matching + NER + context analysis
- **Low Confidence Flagging**: Alert when data quality < 0.7 threshold
- **Recommendations**: Generate actionable suggestions for data verification
- **Fallback Strategies**: Use "Unknown" or "Not specified" with clear indicators

```python
# Example from code
confidence = {
    "Patient_Name": 0.9 if summary["Patient_Name"] != "Unknown Patient" else 0.3,
    "Diagnosis": 0.9 if "whiplash" in text.lower() else 0.6
}
```

### Q2: What pre-trained NLP models would you use for medical summarization?

**Models Used:**
1. **SciSpacy (en_core_sci_md)**: Medical NER with entity linking to medical ontologies
2. **DistilBERT**: Efficient sentiment classification
3. **BART/T5**: Abstractive text summarization (recommended for extension)
4. **BioClinicalBERT**: Medical domain BERT (recommended for fine-tuning)

**Why These Models:**
- Pre-trained on medical literature
- Understand clinical terminology
- Faster inference with good accuracy
- Easy to fine-tune on specific datasets

### Q3: How would you fine-tune BERT for medical sentiment detection?

**Implementation Strategy (in `sentiment_analysis.py`):**

```python
# Dataset Requirements:
- Medical patient-doctor conversations: 5K-10K examples
- Labeled with: Anxious, Neutral, Reassured
- Balanced across sentiment classes

# Recommended Approach:
1. Start with BioClinicalBERT (emilyalsentzer/Bio_ClinicalBERT)
2. Add classification head with 3 outputs
3. Fine-tune with learning rate: 2e-5
4. Batch size: 16, Epochs: 3-5
5. Use stratified k-fold validation
6. Evaluate: Accuracy, F1-score, Confusion Matrix

# Code example included in sentiment_analysis.py
```

### Q4: What datasets would you use for training a healthcare-specific sentiment model?

**Recommended Datasets:**
1. **MedDialog**: 260K+ medical conversations (Chinese + English)
2. **MIMIC-III Clinical Notes**: Real clinical documentation
3. **HealthCareMagic**: Patient-doctor conversations
4. **Reddit Medical Communities**: r/AskDocs, r/medical (with annotation)
5. **Custom Annotation**: Manually label hospital conversation transcripts

**Data Augmentation:**
- Paraphrasing with medical context
- Synthetic conversations using templates
- Back-translation while preserving medical terms

### Q5: How would you train an NLP model to map medical transcripts into SOAP format?

**Approach (implemented in `soap_generator.py`):**

**1. Rule-Based Approach (Current Implementation):**
- Keyword matching for section classification
- Pattern extraction for medical entities
- Conversation flow analysis
- Template-based structuring

**2. Deep Learning Approach (Recommended for Production):**

```python
# Model: T5 or BART for seq-to-seq generation
# Architecture:
1. Section Classification: BERT to classify each sentence
2. Entity Extraction: SciBERT NER
3. Generation: T5 conditioned on sections
4. Post-processing: Validation with medical ontologies

# Training:
- Dataset: Parallel corpus of [conversation] → [SOAP note]
- Format with section tokens: <SUBJECTIVE>, <OBJECTIVE>, etc.
- Multi-task loss: Classification + Generation
- Validation: ROUGE, BLEU, Clinical expert review
```

### Q6: What rule-based or deep-learning techniques would improve the accuracy of SOAP note generation?

**Implemented Techniques:**

1. **Rule-Based:**
   - Speaker role identification (Physician vs Patient)
   - Keyword dictionaries for section mapping
   - Temporal extraction (dates, durations)
   - Medical entity patterns

2. **Hybrid Approach (Recommended):**
   - NER for entity extraction (deep learning)
   - Template filling with extracted entities (rules)
   - Section classification with BERT
   - Validation with UMLS/SNOMED ontologies

3. **Accuracy Improvements:**
   - **Entity Preservation**: Ensure medical terms not lost in generation
   - **Consistency Checks**: Cross-validate extracted information
   - **Multi-task Learning**: Train on section classification + generation
   - **Expert Feedback Loop**: Incorporate clinical validation
   - **Attention Mechanisms**: Focus on relevant conversation parts

## 🔬 Technical Details

### Models Used

| Component | Model | Purpose |
|-----------|-------|---------|
| NER | SciSpacy (en_core_sci_md) | Medical entity extraction |
| Sentiment | DistilBERT | Patient emotion classification |
| Intent | Custom keyword-based | Patient communication goals |
| Keywords | YAKE | Important phrase extraction |

### Performance Considerations

- **Model Loading**: ~10-30 seconds initial load
- **Processing**: ~1-2 seconds per conversation
- **Memory**: ~2GB RAM for all models
- **GPU**: Optional, speeds up processing 5-10x

### Customization

You can customize behavior by editing `config.py`:

```python
# Add custom entity types
ENTITY_TYPES = ['SYMPTOM', 'DIAGNOSIS', 'TREATMENT', ...]

# Modify sentiment labels
SENTIMENT_LABELS = {...}

# Add intent categories
INTENT_CATEGORIES = [...]
```

## 🧪 Testing

### Run All Demos

```bash
# Test each module
python src/medical_nlp.py
python src/sentiment_analysis.py
python src/soap_generator.py

# Test main application
python main.py --demo
```

### Test with Custom Data

Create a file `test_conversation.txt`:

```
Doctor: What brings you in today?
Patient: I've been having chest pain for two days.
Doctor: Can you describe the pain?
Patient: It's a sharp pain that comes and goes.
```

Run analysis:

```bash
python main.py --input test_conversation.txt --output test_results.json
```

## 📈 Future Enhancements

- [ ] Fine-tuned medical BERT models
- [ ] Real-time audio transcription integration
- [ ] Multi-language support
- [ ] RESTful API with FastAPI
- [ ] Web interface for clinical use
- [ ] Database integration for patient records
- [ ] ICD-10 code suggestion
- [ ] Medication interaction checking
- [ ] Integration with EHR systems

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Fine-tuning models on medical datasets
- Adding more entity types
- Improving SOAP note accuracy
- Adding visualization features
- Creating unit tests

## 📝 License

This project is for educational purposes. Medical AI systems require clinical validation before real-world use.

## ⚠️ Disclaimer

This system is a prototype for educational and research purposes only. It is NOT intended for clinical use without proper validation, testing, and regulatory approval. Always consult qualified healthcare professionals for medical advice.

## 👨‍💻 Author

Created as a comprehensive solution for medical NLP tasks including transcription analysis, sentiment detection, and clinical documentation generation.

## 📞 Support

For issues or questions:
1. Check the demo notebook: `demo.ipynb`
2. Review technical documentation in source files
3. Test with sample conversation provided

---

**Built with ❤️ for healthcare AI innovation**
