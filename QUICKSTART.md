# 🚀 Quick Start Guide - Physician Notetaker

## Fast Setup (5 minutes)

### 1. Install Dependencies

```powershell
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# Run setup script (installs everything)
python setup.py
```

### 2. Run Demo

```powershell
# Quick demo with sample conversation
python main.py --demo
```

That's it! You should see:
- ✅ Medical summary in JSON format
- ✅ Sentiment analysis results
- ✅ SOAP note generation

## What Just Happened?

The system analyzed a physician-patient conversation and extracted:

1. **Medical Information**
   - Patient: Ms. Jones
   - Diagnosis: Whiplash injury
   - Symptoms: Neck pain, back pain
   - Treatment: Physiotherapy, painkillers
   - Prognosis: Full recovery in 6 months

2. **Patient Sentiment**
   - Overall: Moving from Anxious → Reassured
   - Intent: Seeking reassurance and providing history

3. **SOAP Note**
   - Structured clinical documentation ready for medical records

## Next Steps

### Try the Interactive Notebook
```powershell
jupyter notebook demo.ipynb
```

### Process Your Own Conversation
```powershell
# Create a text file with your conversation
echo "Doctor: How are you? Patient: I have pain." > my_conversation.txt

# Analyze it
python main.py --input my_conversation.txt --output my_results.json
```

### Use Python API
```python
from main import PhysicianNotetaker

app = PhysicianNotetaker()
results = app.process_conversation(your_conversation_text)
```

## Common Issues

### Issue: SpaCy model not found
**Solution:**
```powershell
pip install https://s3-us-west-2.amazonaws.com/ai2-s3-scispacy/releases/v0.5.1/en_core_sci_md-0.5.1.tar.gz
```

### Issue: Out of memory
**Solution:** Close other applications. System needs ~2GB RAM.

### Issue: Slow processing
**Solution:** First run loads models (30 sec). Subsequent runs are faster.

## Features Demonstration

### 1. Medical NLP (Task 1)
```powershell
python src/medical_nlp.py
```
Shows: Entity extraction, summarization, keywords

### 2. Sentiment Analysis (Task 2)
```powershell
python src/sentiment_analysis.py
```
Shows: Emotion detection, intent classification

### 3. SOAP Notes (Task 3 - Bonus)
```powershell
python src/soap_generator.py
```
Shows: Clinical note generation

## Example Usage

### Analyze a simple conversation
```powershell
python main.py --input examples/sample_conversation.txt --format text
```

### Save results as JSON
```powershell
python main.py --demo --output my_analysis.json
```

## Project Structure
```
Physician-Notetaker-main/
├── src/                    # Core modules
│   ├── medical_nlp.py     # Task 1: NLP & Summarization
│   ├── sentiment_analysis.py  # Task 2: Sentiment & Intent
│   └── soap_generator.py  # Task 3: SOAP Notes
├── main.py                # Main application
├── demo.ipynb             # Interactive demo
├── config.py              # Configuration
└── requirements.txt       # Dependencies
```

## Getting Help

1. **Check the demo notebook**: `demo.ipynb` has detailed examples
2. **Read README.md**: Comprehensive documentation
3. **Run tests**: `pytest tests/`
4. **Check module demos**: Each .py file in `src/` is runnable

## Performance Tips

- First run: ~30 seconds (model loading)
- Subsequent runs: ~2 seconds per conversation
- Use `--demo` for quick testing
- Process multiple conversations in batch for efficiency

## What's Included

✅ **Task 1: Medical NLP Summarization**
- Named Entity Recognition
- Text Summarization
- Keyword Extraction
- Ambiguous data handling

✅ **Task 2: Sentiment & Intent Analysis**
- Patient sentiment classification
- Intent detection
- Conversation analysis

✅ **Task 3: SOAP Note Generation (Bonus)**
- Automated SOAP notes
- Clinical documentation format

✅ **Extras:**
- Comprehensive README
- Interactive Jupyter notebook
- Unit tests
- Sample conversations
- Full documentation of technical approach

## Ready to Go!

You now have a complete medical NLP system. Start with:
```powershell
python main.py --demo
```

Enjoy! 🩺
