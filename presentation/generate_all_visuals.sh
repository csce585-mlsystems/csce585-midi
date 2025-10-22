#!/bin/bash

# Generate all presentation materials in one command
echo "🎨 Generating all presentation materials..."

# Change to presentation directory
cd "$(dirname "$0")"

echo "📊 Creating model comparison plots..."
python model_comparison.py

echo ""
echo "🏗️ Creating architecture diagrams..."
python architecture_diagrams.py

echo ""
echo "📋 Creating additional analysis plots..."
python create_presentation_plots.py

echo ""
echo "🎉 All presentation materials generated!"
echo ""
echo "📁 Generated files:"
echo "   • ../outputs/model_comparison_presentation.png"
echo "   • ../outputs/model_summary_table.png"
echo "   • ../outputs/architecture_comparison.png"
echo "   • ../outputs/evolution_timeline.png"
echo ""
echo "💡 Your presentation package is ready!"
echo "   Use these visuals to show your research progress and future vision."