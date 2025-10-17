"""
Visualization utilities for Physician Notetaker
"""

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import json


class MedicalVisualization:
    """
    Create visualizations for medical analysis results
    """
    
    def __init__(self):
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
    
    def plot_sentiment_timeline(self, sentiment_data: List[Dict]):
        """
        Plot patient sentiment over time
        
        Args:
            sentiment_data: List of sentiment analysis results
        """
        sentiment_map = {'Anxious': 1, 'Neutral': 2, 'Reassured': 3}
        sentiments = [sentiment_map.get(s['Sentiment'], 2) for s in sentiment_data]
        
        plt.figure(figsize=(14, 5))
        plt.plot(range(1, len(sentiments) + 1), sentiments, 
                marker='o', linewidth=2.5, markersize=10, color='#2ecc71')
        
        plt.yticks([1, 2, 3], ['😰 Anxious', '😐 Neutral', '😊 Reassured'])
        plt.xlabel('Patient Statement Number', fontsize=12, fontweight='bold')
        plt.ylabel('Sentiment', fontsize=12, fontweight='bold')
        plt.title('Patient Sentiment Throughout Conversation', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add annotations
        for i, sentiment in enumerate(sentiments):
            if i > 0 and sentiment != sentiments[i-1]:
                plt.annotate('Change', xy=(i+1, sentiment), 
                           xytext=(i+1, sentiment + 0.3),
                           arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                           fontsize=9, color='red')
        
        plt.tight_layout()
        plt.savefig('sentiment_timeline.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_sentiment_distribution(self, sentiment_counts: Dict[str, int]):
        """
        Plot distribution of sentiments
        
        Args:
            sentiment_counts: Dictionary with sentiment counts
        """
        sentiments = list(sentiment_counts.keys())
        counts = list(sentiment_counts.values())
        colors = {'Anxious': '#e74c3c', 'Neutral': '#95a5a6', 'Reassured': '#2ecc71'}
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(sentiments, counts, 
                      color=[colors.get(s, '#3498db') for s in sentiments],
                      alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.xlabel('Sentiment', fontsize=12, fontweight='bold')
        plt.ylabel('Count', fontsize=12, fontweight='bold')
        plt.title('Sentiment Distribution', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('sentiment_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_intent_distribution(self, intent_counts: Dict[str, int]):
        """
        Plot distribution of intents
        
        Args:
            intent_counts: Dictionary with intent counts
        """
        intents = list(intent_counts.keys())
        counts = list(intent_counts.values())
        
        plt.figure(figsize=(12, 6))
        bars = plt.barh(intents, counts, color='#3498db', alpha=0.7, edgecolor='black')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2.,
                    f' {int(width)}',
                    ha='left', va='center', fontsize=11, fontweight='bold')
        
        plt.xlabel('Count', fontsize=12, fontweight='bold')
        plt.ylabel('Intent', fontsize=12, fontweight='bold')
        plt.title('Intent Distribution', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig('intent_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confidence_scores(self, confidence_scores: Dict[str, float]):
        """
        Plot confidence scores for extracted fields
        
        Args:
            confidence_scores: Dictionary with field confidence scores
        """
        fields = list(confidence_scores.keys())
        scores = list(confidence_scores.values())
        colors = ['#2ecc71' if s >= 0.7 else '#e74c3c' for s in scores]
        
        plt.figure(figsize=(12, 6))
        bars = plt.barh(fields, scores, color=colors, alpha=0.7, edgecolor='black')
        
        # Add threshold line
        plt.axvline(x=0.7, color='orange', linestyle='--', linewidth=2, label='Threshold (0.7)')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2.,
                    f' {width:.2f}',
                    ha='left', va='center', fontsize=11, fontweight='bold')
        
        plt.xlabel('Confidence Score', fontsize=12, fontweight='bold')
        plt.ylabel('Field', fontsize=12, fontweight='bold')
        plt.title('Data Quality: Confidence Scores', fontsize=14, fontweight='bold')
        plt.xlim(0, 1.0)
        plt.legend(loc='lower right')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig('confidence_scores.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_entity_counts(self, entities: Dict[str, List[str]]):
        """
        Plot count of extracted entities by type
        
        Args:
            entities: Dictionary with entity types and lists
        """
        entity_types = [k.replace('_', ' ').title() for k in entities.keys()]
        counts = [len(v) for v in entities.values()]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(entity_types, counts, color='#9b59b6', alpha=0.7, edgecolor='black')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.xlabel('Entity Type', fontsize=12, fontweight='bold')
        plt.ylabel('Count', fontsize=12, fontweight='bold')
        plt.title('Extracted Medical Entities', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('entity_counts.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_comprehensive_dashboard(self, results: Dict):
        """
        Create a comprehensive dashboard with multiple plots
        
        Args:
            results: Complete analysis results
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🩺 Medical Conversation Analysis Dashboard', 
                     fontsize=16, fontweight='bold')
        
        # 1. Sentiment timeline
        if 'sentiment_analysis' in results and 'Detailed_Analysis' in results['sentiment_analysis']:
            sentiment_data = results['sentiment_analysis']['Detailed_Analysis']
            sentiment_map = {'Anxious': 1, 'Neutral': 2, 'Reassured': 3}
            sentiments = [sentiment_map.get(s['Sentiment'], 2) for s in sentiment_data]
            
            axes[0, 0].plot(range(1, len(sentiments) + 1), sentiments, 
                          marker='o', linewidth=2, markersize=8, color='#2ecc71')
            axes[0, 0].set_yticks([1, 2, 3], ['Anxious', 'Neutral', 'Reassured'])
            axes[0, 0].set_xlabel('Statement Number')
            axes[0, 0].set_ylabel('Sentiment')
            axes[0, 0].set_title('Sentiment Timeline')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Sentiment distribution
        if 'sentiment_analysis' in results and 'Sentiment_Distribution' in results['sentiment_analysis']:
            sentiment_counts = results['sentiment_analysis']['Sentiment_Distribution']
            sentiments = list(sentiment_counts.keys())
            counts = list(sentiment_counts.values())
            colors = {'Anxious': '#e74c3c', 'Neutral': '#95a5a6', 'Reassured': '#2ecc71'}
            
            axes[0, 1].bar(sentiments, counts, 
                         color=[colors.get(s, '#3498db') for s in sentiments], alpha=0.7)
            axes[0, 1].set_xlabel('Sentiment')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].set_title('Sentiment Distribution')
            axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. Intent distribution
        if 'sentiment_analysis' in results and 'Intent_Distribution' in results['sentiment_analysis']:
            intent_counts = results['sentiment_analysis']['Intent_Distribution']
            intents = list(intent_counts.keys())
            counts = list(intent_counts.values())
            
            axes[1, 0].barh(intents, counts, color='#3498db', alpha=0.7)
            axes[1, 0].set_xlabel('Count')
            axes[1, 0].set_ylabel('Intent')
            axes[1, 0].set_title('Intent Distribution')
            axes[1, 0].grid(axis='x', alpha=0.3)
        
        # 4. Confidence scores
        if 'data_quality' in results and 'confidence_scores' in results['data_quality']:
            confidence_scores = results['data_quality']['confidence_scores']
            fields = list(confidence_scores.keys())
            scores = list(confidence_scores.values())
            colors = ['#2ecc71' if s >= 0.7 else '#e74c3c' for s in scores]
            
            axes[1, 1].barh(fields, scores, color=colors, alpha=0.7)
            axes[1, 1].axvline(x=0.7, color='orange', linestyle='--', linewidth=2)
            axes[1, 1].set_xlabel('Confidence Score')
            axes[1, 1].set_ylabel('Field')
            axes[1, 1].set_title('Data Quality')
            axes[1, 1].set_xlim(0, 1.0)
            axes[1, 1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()


def demo():
    """
    Demonstration of visualization capabilities
    """
    from src.medical_nlp import MedicalNLPProcessor
    from src.sentiment_analysis import SentimentIntentAnalyzer
    from config import SAMPLE_CONVERSATION
    
    print("=" * 80)
    print("MEDICAL VISUALIZATION DEMO")
    print("=" * 80)
    
    # Initialize components
    nlp_processor = MedicalNLPProcessor()
    sentiment_analyzer = SentimentIntentAnalyzer()
    viz = MedicalVisualization()
    
    # Get analysis results
    sentiment_summary = sentiment_analyzer.get_conversation_summary(SAMPLE_CONVERSATION)
    entities = nlp_processor.extract_entities(SAMPLE_CONVERSATION)
    ambiguous_check = nlp_processor.handle_ambiguous_data(SAMPLE_CONVERSATION)
    
    # Create visualizations
    print("\n1. Generating sentiment timeline...")
    viz.plot_sentiment_timeline(sentiment_summary['Detailed_Analysis'])
    
    print("2. Generating sentiment distribution...")
    viz.plot_sentiment_distribution(sentiment_summary['Sentiment_Distribution'])
    
    print("3. Generating intent distribution...")
    viz.plot_intent_distribution(sentiment_summary['Intent_Distribution'])
    
    print("4. Generating confidence scores...")
    viz.plot_confidence_scores(ambiguous_check['confidence_scores'])
    
    print("5. Generating entity counts...")
    viz.plot_entity_counts(entities)
    
    print("\n✅ All visualizations generated successfully!")
    print("   Saved as PNG files in the current directory.")


if __name__ == "__main__":
    demo()
