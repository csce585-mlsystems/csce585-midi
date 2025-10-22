import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

def create_architecture_comparison():
    """Create visual comparison of current vs MuseNet architecture"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Color scheme
    colors = {
        'data': '#E8F4FD',
        'processing': '#B3D9FF', 
        'model': '#4D94FF',
        'output': '#0066CC',
        'discriminator': '#FF6B6B',
        'generator': '#4ECDC4',
        'attention': '#45B7D1'
    }
    
    # Current Architecture (Top)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 3)
    ax1.set_title('Current Architecture: Single LSTM Pipeline', fontsize=16, fontweight='bold', pad=20)
    
    # Current architecture components
    current_boxes = [
        {'pos': (0.5, 1), 'size': (1.5, 1), 'label': 'MIDI\nFiles', 'color': colors['data']},
        {'pos': (2.5, 1), 'size': (1.5, 1), 'label': 'Tokenization\n(Naive/MidiTok)', 'color': colors['processing']},
        {'pos': (4.5, 1), 'size': (1.5, 1), 'label': 'Single LSTM\nModel', 'color': colors['model']},
        {'pos': (6.5, 1), 'size': (1.5, 1), 'label': 'Output\nSequence', 'color': colors['processing']},
        {'pos': (8.5, 1), 'size': (1, 1), 'label': 'MIDI\nOut', 'color': colors['output']}
    ]
    
    # Draw current architecture boxes
    for box in current_boxes:
        rect = FancyBboxPatch(
            box['pos'], box['size'][0], box['size'][1],
            boxstyle="round,pad=0.1",
            facecolor=box['color'],
            edgecolor='black',
            linewidth=2
        )
        ax1.add_patch(rect)
        ax1.text(box['pos'][0] + box['size'][0]/2, box['pos'][1] + box['size'][1]/2,
                box['label'], ha='center', va='center', fontweight='bold', fontsize=11)
    
    # Arrows for current architecture
    arrow_props = dict(arrowstyle='->', lw=3, color='black')
    for i in range(len(current_boxes)-1):
        start_x = current_boxes[i]['pos'][0] + current_boxes[i]['size'][0]
        end_x = current_boxes[i+1]['pos'][0]
        y = current_boxes[i]['pos'][1] + current_boxes[i]['size'][1]/2
        ax1.annotate('', xy=(end_x, y), xytext=(start_x, y), arrowprops=arrow_props)
    
    # Add findings callout
    findings_box = FancyBboxPatch(
        (0.5, 0.1), 8.5, 0.6,
        boxstyle="round,pad=0.1",
        facecolor='#FFF3CD',
        edgecolor='#D6B656',
        linewidth=2
    )
    ax1.add_patch(findings_box)
    ax1.text(4.75, 0.4,
            'Key Finding: Naive tokenization outperforms MidiTok (392 vs 73 avg notes)\n' +
            'Insight: Simple architecture + complex tokenization = mismatch',
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    ax1.set_xticks([])
    ax1.set_yticks([])
    for spine in ax1.spines.values():
        spine.set_visible(False)
    
    # MuseNet Architecture (Bottom)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 4)
    ax2.set_title('Target Architecture: MuseNet-Inspired Two-Stage Pipeline', fontsize=16, fontweight='bold', pad=20)
    
    # Stage 1: Chord Prediction
    chord_boxes = [
        {'pos': (0.5, 2.5), 'size': (1.5, 0.8), 'label': 'Previous\nContext', 'color': colors['data']},
        {'pos': (2.5, 2.5), 'size': (2, 0.8), 'label': 'Chord Predictor\n(5-layer MLP)', 'color': colors['discriminator']},
        {'pos': (5, 2.5), 'size': (1.5, 0.8), 'label': 'Next Chord\nPrediction', 'color': colors['attention']}
    ]
    
    # Stage 2: Note Generation
    note_boxes = [
        {'pos': (2.5, 0.7), 'size': (2, 0.8), 'label': 'Note Generator\n(LSTM/Transformer)', 'color': colors['generator']},
        {'pos': (5, 0.7), 'size': (1.5, 0.8), 'label': 'Generated\nNotes', 'color': colors['processing']},
        {'pos': (7, 0.7), 'size': (1, 0.8), 'label': 'MIDI\nOut', 'color': colors['output']}
    ]
    
    all_boxes = chord_boxes + note_boxes
    
    # Draw all boxes
    for box in all_boxes:
        rect = FancyBboxPatch(
            box['pos'], box['size'][0], box['size'][1],
            boxstyle="round,pad=0.1",
            facecolor=box['color'],
            edgecolor='black',
            linewidth=2
        )
        ax2.add_patch(rect)
        ax2.text(box['pos'][0] + box['size'][0]/2, box['pos'][1] + box['size'][1]/2,
                box['label'], ha='center', va='center', fontweight='bold', fontsize=11)
    
    # Stage 1 arrows
    ax2.annotate('', xy=(2.5, 2.9), xytext=(2.0, 2.9), arrowprops=arrow_props)
    ax2.annotate('', xy=(5.0, 2.9), xytext=(4.5, 2.9), arrowprops=arrow_props)
    
    # Stage 2 arrows
    ax2.annotate('', xy=(5.0, 1.1), xytext=(4.5, 1.1), arrowprops=arrow_props)
    ax2.annotate('', xy=(7.0, 1.1), xytext=(6.5, 1.1), arrowprops=arrow_props)
    
    # Conditioning arrows
    # Chord to generator
    ax2.annotate('', xy=(3.5, 1.5), xytext=(5.75, 2.5),
                arrowprops=dict(arrowstyle='->', lw=3, color='red', connectionstyle="arc3,rad=0.3"))
    ax2.text(4.5, 1.9, 'Chord\nCondition', ha='center', va='center',
            fontsize=10, fontweight='bold', color='red')
    
    # Context to generator
    ax2.annotate('', xy=(2.5, 1.1), xytext=(1.25, 2.5),
                arrowprops=dict(arrowstyle='->', lw=3, color='blue', connectionstyle="arc3,rad=-0.3"))
    ax2.text(1.6, 1.7, 'Context', ha='center', va='center',
            fontsize=10, fontweight='bold', color='blue')
    
    # Stage labels
    ax2.text(0.2, 2.9, 'Stage 1', rotation=90, ha='center', va='center',
            fontsize=14, fontweight='bold', color='red')
    ax2.text(0.2, 1.1, 'Stage 2', rotation=90, ha='center', va='center',
            fontsize=14, fontweight='bold', color='blue')
    
    # Evolution benefits
    benefits_box = FancyBboxPatch(
        (0.5, 0.05), 7, 0.5,
        boxstyle="round,pad=0.1",
        facecolor='#D4EDDA',
        edgecolor='#28A745',
        linewidth=2
    )
    ax2.add_patch(benefits_box)
    ax2.text(4, 0.3,
            'Benefits: Separate harmonic structure from melodic generation\n' +
            'Result: Better long-term coherence and musical structure',
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    ax2.set_xticks([])
    ax2.set_yticks([])
    for spine in ax2.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    
    # Save the comparison
    output_path = Path('../outputs/architecture_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Architecture comparison saved to: {output_path}")
    
    return output_path

def create_evolution_timeline():
    """Create project evolution timeline"""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    # Timeline phases
    phases = [
        {
            'name': 'Phase 1\n(Current)',
            'status': 'Complete',
            'color': '#28A745',
            'y': 3,
            'tasks': ['LSTM Implementation', 'Naive vs MidiTok', 'Performance Analysis', 'Evaluation Metrics']
        },
        {
            'name': 'Phase 2\n(Next Week)', 
            'status': 'In Progress',
            'color': '#FFC107',
            'y': 2,
            'tasks': ['Fix MidiTok Training', 'Longer Sequences', 'Hyperparameter Tuning', 'Chord Analysis']
        },
        {
            'name': 'Phase 3\n(Next Month)',
            'status': 'Planned',
            'color': '#17A2B8', 
            'y': 1,
            'tasks': ['Chord Predictor MLP', 'Two-Stage Training', 'Bar Segmentation', 'Conditional Generation']
        },
        {
            'name': 'Phase 4\n(Future)',
            'status': 'Research Goal',
            'color': '#6F42C1',
            'y': 0,
            'tasks': ['Transformer Generator', 'Attention Mechanism', 'Large Dataset', 'Full MuseNet']
        }
    ]
    
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.7, 4)
    ax.set_title('Project Evolution: From LSTM to MuseNet Architecture', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Draw timeline
    for i, phase in enumerate(phases):
        # Main phase box
        main_box = FancyBboxPatch(
            (i, phase['y']), 0.8, 0.6,
            boxstyle="round,pad=0.05",
            facecolor=phase['color'],
            edgecolor='black',
            linewidth=2,
            alpha=0.9
        )
        ax.add_patch(main_box)
        
        # Phase labels
        ax.text(i + 0.4, phase['y'] + 0.45, phase['name'],
               ha='center', va='center', fontweight='bold', fontsize=11, color='white')
        ax.text(i + 0.4, phase['y'] + 0.15, phase['status'],
               ha='center', va='center', fontweight='bold', fontsize=9, color='white')
        
        # Task details
        task_box = FancyBboxPatch(
            (i - 0.3, phase['y'] - 0.5), 1.4, 0.4,
            boxstyle="round,pad=0.02",
            facecolor='white',
            edgecolor=phase['color'],
            linewidth=1.5,
            alpha=0.95
        )
        ax.add_patch(task_box)
        
        task_text = '\n'.join(f"• {task}" for task in phase['tasks'])
        ax.text(i + 0.4, phase['y'] - 0.3, task_text,
               ha='center', va='center', fontsize=9)
        
        # Progress arrows
        if i < len(phases) - 1:
            ax.annotate('', xy=(i + 0.9, phase['y'] + 0.3), xytext=(i + 0.8, phase['y'] + 0.3),
                       arrowprops=dict(arrowstyle='->', lw=3, color='gray'))
    
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Add key insight
    insight_box = FancyBboxPatch(
        (0.5, -0.6), 3, 0.4,
        boxstyle="round,pad=0.05",
        facecolor='#FFF3CD',
        edgecolor='#FFC107',
        linewidth=2
    )
    ax.add_patch(insight_box)
    ax.text(2, -0.4, 'Key Insight: Current results motivate architectural evolution\nSimple tokenization success → Need for structured generation approach',
           ha='center', va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    # Save timeline
    output_path = Path('../outputs/evolution_timeline.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Evolution timeline saved to: {output_path}")
    
    return output_path

if __name__ == "__main__":
    print("Creating architecture diagrams...")
    
    # Create architecture comparison
    arch_path = create_architecture_comparison()
    
    # Create evolution timeline  
    timeline_path = create_evolution_timeline()
    
    print(f"\n🎉 Architecture diagrams created!")
    print(f"🏗️  Architecture comparison: {arch_path}")
    print(f"📅 Evolution timeline: {timeline_path}")
    print(f"\n💡 Use these diagrams to show:")
    print(f"   • Current simple pipeline vs target two-stage architecture")
    print(f"   • Clear project roadmap from current state to MuseNet")
    print(f"   • How your findings motivate the architectural evolution")