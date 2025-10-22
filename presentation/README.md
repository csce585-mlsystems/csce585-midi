# Presentation Materials

This directory contains all scripts and materials for creating presentation visuals and analysis.

## Files

### Analysis Scripts
- `model_comparison.py` - Creates comparison plots between naive and MidiTok approaches
- `architecture_diagrams.py` - Generates architecture comparison and evolution timeline
- `create_presentation_plots.py` - Original automated plotting script
- `musenet_comparison.py` - Detailed MuseNet vs current approach analysis

### Generated Materials
- `presentation_summary.md` - Quick presentation guide with timing and key points
- `musenet_comparison.md` - Detailed technical comparison document

### Shell Scripts  
- `run_presentation_experiments.sh` - Automated experiment runner
- `generate_all_visuals.sh` - One-command script to create all presentation materials

## Usage

### Quick Setup
```bash
# Generate all presentation materials
./generate_all_visuals.sh
```

### Individual Scripts
```bash
# Model comparison plots
python model_comparison.py

# Architecture diagrams  
python architecture_diagrams.py

# Complete analysis
python musenet_comparison.py
```

## Outputs
All generated visuals are saved to `../outputs/` directory:
- Model comparison charts
- Architecture diagrams
- Evolution timeline
- Training loss plots