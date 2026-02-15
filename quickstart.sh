#!/bin/bash

# Australian Property Investment Analyzer - Quick Start Script
# This script sets up everything you need to get started

echo "=================================================="
echo "Australian Property Investment Analyzer"
echo "Quick Start Setup"
echo "=================================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null
then
    echo "❌ Python 3 is not installed."
    echo "Please install Python 3.8+ from https://www.python.org/downloads/"
    echo "Then run this script again."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"
echo ""

# Check if pip is installed
if ! command -v pip3 &> /dev/null
then
    echo "❌ pip is not installed."
    echo "Installing pip..."
    python3 -m ensurepip --upgrade
fi

echo "✅ pip found"
echo ""

# Install required packages
echo "📦 Installing required Python packages..."
echo "This may take a few minutes..."
echo ""

echo "Attempting installation with --user flag (Mac-friendly)..."
pip3 install --user -q streamlit pandas plotly

if [ $? -eq 0 ]; then
    echo "✅ All packages installed successfully"
else
    echo "⚠️  First attempt failed, trying with --break-system-packages..."
    pip3 install --break-system-packages -q streamlit pandas plotly
    
    if [ $? -eq 0 ]; then
        echo "✅ All packages installed successfully"
    else
        echo "❌ Package installation failed"
        echo "Try manually: pip3 install --user streamlit pandas plotly"
        exit 1
    fi
fi

echo ""
echo "=================================================="
echo "✅ Setup Complete!"
echo "=================================================="
echo ""
echo "To start the dashboard:"
echo "  streamlit run property_analysis_dashboard.py"
echo ""
echo "The dashboard will open in your browser at:"
echo "  http://localhost:8501"
echo ""
echo "📖 For detailed instructions, see SETUP_GUIDE.md"
echo ""
echo "🎯 Next steps:"
echo "  1. Run the dashboard with the command above"
echo "  2. Go to 'Data Management' tab"
echo "  3. Import sample_economic_data.csv"
echo "  4. Import sample_property_data.csv"
echo "  5. Explore the dashboard!"
echo ""
echo "=================================================="
