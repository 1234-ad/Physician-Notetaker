"""
Main Application Script for Physician Notetaker
Integrates all modules and provides a unified interface
"""

import sys
import json
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.medical_nlp import MedicalNLPProcessor
from src.sentiment_analysis import SentimentIntentAnalyzer
from src.soap_generator import SOAPNoteGenerator
from config import SAMPLE_CONVERSATION


class PhysicianNotetaker:
    """
    Main application class integrating all NLP components
    """
    
    def __init__(self):
        """
        Initialize all components
        """
        print("🩺 Initializing Physician Notetaker...")
        print("-" * 80)
        
        # Initialize modules
        self.nlp_processor = MedicalNLPProcessor()
        self.sentiment_analyzer = SentimentIntentAnalyzer()
        self.soap_generator = SOAPNoteGenerator()
        
        print("\n✅ All modules loaded successfully!\n")
    
    def process_conversation(self, conversation: str, output_file: str = None) -> Dict:
        """
        Process a complete medical conversation
        
        Args:
            conversation: Medical conversation transcript
            output_file: Optional file to save results
            
        Returns:
            Dictionary with all analysis results
        """
        print("=" * 80)
        print("PROCESSING MEDICAL CONVERSATION")
        print("=" * 80)
        
        results = {}
        
        # 1. Medical NLP Summarization
        print("\n📋 1. MEDICAL NLP SUMMARIZATION")
        print("-" * 80)
        
        summary = self.nlp_processor.generate_structured_summary(conversation)
        results['medical_summary'] = summary
        print(json.dumps(summary, indent=2))
        
        # Extract keywords
        keywords = self.nlp_processor.extract_keywords(conversation)
        results['keywords'] = [kw for kw, score in keywords]
        print(f"\n🔑 Key Medical Terms: {', '.join(results['keywords'][:5])}")
        
        # Handle ambiguous data
        ambiguous_check = self.nlp_processor.handle_ambiguous_data(conversation)
        results['data_quality'] = {
            'confidence_scores': ambiguous_check['confidence_scores'],
            'low_confidence_fields': ambiguous_check['low_confidence_fields']
        }
        
        if ambiguous_check['low_confidence_fields']:
            print(f"\n⚠️  Low confidence fields: {', '.join(ambiguous_check['low_confidence_fields'])}")
        
        # 2. Sentiment & Intent Analysis
        print("\n\n😊 2. SENTIMENT & INTENT ANALYSIS")
        print("-" * 80)
        
        sentiment_summary = self.sentiment_analyzer.get_conversation_summary(conversation)
        results['sentiment_analysis'] = sentiment_summary
        
        print(f"Total Patient Statements: {sentiment_summary['Total_Patient_Statements']}")
        print(f"Overall Sentiment: {sentiment_summary['Overall_Sentiment']}")
        print(f"Most Common Intent: {sentiment_summary['Most_Common_Intent']}")
        
        print("\nSentiment Distribution:")
        for sentiment, count in sentiment_summary['Sentiment_Distribution'].items():
            print(f"  • {sentiment}: {count}")
        
        # 3. SOAP Note Generation
        print("\n\n📝 3. SOAP NOTE GENERATION")
        print("-" * 80)
        
        soap_note = self.soap_generator.generate_soap_note(conversation)
        results['soap_note'] = soap_note
        print(json.dumps(soap_note, indent=2))
        
        # Save to file if requested
        if output_file:
            self.save_results(results, output_file)
            print(f"\n💾 Results saved to: {output_file}")
        
        return results
    
    def save_results(self, results: Dict, filename: str):
        """
        Save results to JSON file
        
        Args:
            results: Analysis results
            filename: Output filename
        """
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    def analyze_patient_statement(self, statement: str) -> Dict:
        """
        Quick analysis of a single patient statement
        
        Args:
            statement: Patient's statement
            
        Returns:
            Sentiment and intent analysis
        """
        return self.sentiment_analyzer.analyze_patient_dialogue(statement)
    
    def generate_report(self, conversation: str, format: str = 'json') -> str:
        """
        Generate a complete medical report
        
        Args:
            conversation: Medical conversation
            format: Output format ('json' or 'text')
            
        Returns:
            Formatted report
        """
        results = self.process_conversation(conversation)
        
        if format == 'text':
            return self._format_text_report(results)
        else:
            return json.dumps(results, indent=2)
    
    def _format_text_report(self, results: Dict) -> str:
        """
        Format results as human-readable text report
        """
        report = []
        report.append("=" * 80)
        report.append("MEDICAL CONVERSATION ANALYSIS REPORT")
        report.append("=" * 80)
        
        # Medical Summary
        report.append("\n📋 MEDICAL SUMMARY")
        report.append("-" * 80)
        summary = results['medical_summary']
        report.append(f"Patient: {summary['Patient_Name']}")
        report.append(f"Diagnosis: {summary['Diagnosis']}")
        report.append(f"Symptoms: {', '.join(summary['Symptoms'])}")
        report.append(f"Treatment: {', '.join(summary['Treatment'])}")
        report.append(f"Current Status: {summary['Current_Status']}")
        report.append(f"Prognosis: {summary['Prognosis']}")
        
        # SOAP Note
        report.append("\n\n📝 SOAP NOTE")
        report.append("-" * 80)
        soap = results['soap_note']
        
        report.append("\nSUBJECTIVE:")
        report.append(f"  Chief Complaint: {soap['Subjective']['Chief_Complaint']}")
        report.append(f"  History: {soap['Subjective']['History_of_Present_Illness']}")
        
        report.append("\nOBJECTIVE:")
        report.append(f"  Physical Exam: {soap['Objective']['Physical_Exam']}")
        report.append(f"  Observations: {soap['Objective']['Observations']}")
        
        report.append("\nASSESSMENT:")
        report.append(f"  Diagnosis: {soap['Assessment']['Diagnosis']}")
        report.append(f"  Severity: {soap['Assessment']['Severity']}")
        
        report.append("\nPLAN:")
        report.append(f"  Treatment: {', '.join(soap['Plan']['Treatment'])}")
        report.append(f"  Follow-Up: {soap['Plan']['Follow_Up']}")
        
        # Sentiment Analysis
        report.append("\n\n😊 SENTIMENT ANALYSIS")
        report.append("-" * 80)
        sentiment = results['sentiment_analysis']
        report.append(f"Overall Sentiment: {sentiment['Overall_Sentiment']}")
        report.append(f"Most Common Intent: {sentiment['Most_Common_Intent']}")
        
        return "\n".join(report)


def main():
    """
    Main CLI interface
    """
    parser = argparse.ArgumentParser(
        description='🩺 Physician Notetaker - Medical Conversation Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--input',
        type=str,
        help='Input file containing medical conversation'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for analysis results (JSON)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'text'],
        default='json',
        help='Output format (default: json)'
    )
    
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run demo with sample conversation'
    )
    
    args = parser.parse_args()
    
    # Initialize application
    app = PhysicianNotetaker()
    
    if args.demo or not args.input:
        # Run demo
        print("\n🎯 Running demo with sample conversation...\n")
        results = app.process_conversation(SAMPLE_CONVERSATION, args.output)
        
        if args.format == 'text':
            print("\n" + app._format_text_report(results))
    else:
        # Process input file
        try:
            with open(args.input, 'r', encoding='utf-8') as f:
                conversation = f.read()
            
            results = app.process_conversation(conversation, args.output)
            
            if args.format == 'text':
                print("\n" + app._format_text_report(results))
        
        except FileNotFoundError:
            print(f"❌ Error: Input file '{args.input}' not found")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error processing conversation: {str(e)}")
            sys.exit(1)
    
    print("\n\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
