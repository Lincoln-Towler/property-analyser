# 🏠 Australian Property Investment Analysis System

A comprehensive, data-driven tool for analyzing property market conditions and making informed investment decisions in the Australian property market.

![Dashboard Preview](https://img.shields.io/badge/Dashboard-Streamlit-red)
![Automation](https://img.shields.io/badge/Automation-n8n-orange)
![License](https://img.shields.io/badge/License-MIT-blue)

---

## 🎯 What This System Does

This system helps you make **objective, data-driven property investment decisions** by:

### ✅ Core Features

- **📊 Track 15+ Key Economic Indicators** - Interest rates, household debt, unemployment, rental vacancies, and more
- **📈 Visualize Historical Trends** - See how markets have moved over time with interactive charts
- **🎯 Get Buy/Hold/Wait Recommendations** - Based on multi-factor analysis, not gut feeling
- **📍 Compare Locations** - Objectively compare cities and regions for investment potential
- **🔔 Receive Automated Alerts** - Get notified when critical thresholds are crossed
- **📧 Weekly Market Summaries** - Automated reports delivered to your inbox
- **⏰ Track the Anderson Cycle** - See where we are in the 18.6-year property cycle
- **💾 Import/Export Data** - CSV support for bulk data management

### 🎨 Built With

- **Frontend:** Streamlit (Python-based interactive dashboards)
- **Automation:** n8n (workflow automation)
- **Database:** SQLite (lightweight, no server required)
- **Visualization:** Plotly (interactive charts)

---

## 🚀 Quick Start (5 Minutes)

### Option 1: Windows

1. **Download** all files to a folder
2. **Double-click** `quickstart.bat`
3. **Wait** for installation to complete
4. **Dashboard opens** automatically in your browser
5. **Import sample data** from Data Management tab

### Option 2: Mac/Linux

```bash
# Make script executable
chmod +x quickstart.sh

# Run setup
./quickstart.sh

# Start dashboard (if not auto-started)
streamlit run property_analysis_dashboard.py
```

### Option 3: Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run property_analysis_dashboard.py

# Open browser to http://localhost:8501
```

---

## 📸 Screenshots

### Main Dashboard
View overall market conditions, get Buy/Hold/Wait recommendations, and see key indicators at a glance.

### Economic Indicators
Track crash risk indicators (red) vs growth support indicators (green) with progress bars and historical trends.

### Location Comparison
Side-by-side comparison of investment locations with detailed metrics and price trend charts.

### Anderson Cycle Tracker
Visual representation of where we are in the 18.6-year property cycle with phase breakdowns and warnings.

---

## 📊 Tracked Indicators

### 🚨 Crash Risk Indicators (Bearish)

| Indicator | Current Target | What It Means |
|-----------|---------------|---------------|
| Household Debt to GDP | < 110% | Measures total household debt relative to economy size |
| Mortgage Stress Rate | < 30% | Percentage of borrowers struggling with repayments |
| Interest Rates | Trend | RBA cash rate - impacts borrowing costs |
| Unemployment Rate | < 5% | Job market health - affects ability to service debt |
| Auction Clearance Rate | > 65% | Measures buyer demand at auctions |
| Building Approvals | Trend | Leading indicator of future supply |

### 💪 Growth Support Indicators (Bullish)

| Indicator | Current Target | What It Means |
|-----------|---------------|---------------|
| Rental Vacancy Rate | < 2% | Tight = strong demand, high yields |
| Dwelling Supply Deficit | Count | How many homes we're short |
| Population Growth | Annual | More people = more housing demand |
| Mortgage Arrears Rate | < 2% | Low = healthy market, few forced sales |
| Credit Growth | Positive | Bank lending growth = market health |
| Wage Growth | Trend | Supports affordability over time |

---

## 🗺️ Supported Locations

The system comes with data templates for:
- **Sydney, NSW**
- **Melbourne, VIC**
- **Brisbane, QLD**
- **Perth, WA**
- **Adelaide, SA**
- **Albany, WA** (regional example)
- **+ Add your own locations**

---

## 🤖 Automation Features (Optional)

The included n8n workflow provides:

- **Daily Data Collection** (8 AM)
  - RBA interest rates
  - ABS unemployment data
  - Domain auction clearances
  - Automatic database updates

- **Real-Time Alerts**
  - Interest rate > 4.0%
  - Auction clearance < 60%
  - Custom threshold triggers

- **Weekly Summary Reports** (Monday 9 AM)
  - Market score calculation
  - Buy/Hold/Wait recommendation
  - Key indicator changes
  - Email delivery

---

## 📖 Documentation

- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Comprehensive setup instructions
- **Dashboard Help** - Built into each tab
- **Code Comments** - Detailed explanations throughout
- **Sample Data** - Included CSV files to get started

---

## 🎓 Understanding the System

### Market Score (0-100)

The system calculates an overall market score based on multiple factors:

- **0-30:** High Risk - Consider waiting
- **30-45:** Caution - Exercise care
- **45-55:** Neutral - Monitor closely
- **55-70:** Favorable - Consider buying
- **70-100:** Strong Buy - Conditions align

### Buy/Hold/Wait Recommendations

**Buy:**
- Multiple bullish indicators
- Supply shortage evident
- Low crash risk signals

**Hold:**
- Mixed signals
- Transitional period
- Await more data

**Wait:**
- Multiple bearish indicators
- High crash risk
- Potential cycle peak

### The Anderson 18.6-Year Cycle

Based on Phillip Anderson's theory:
- **Years 1-7:** Recovery phase
- **Years 7-9:** Mid-cycle slowdown
- **Years 9-14:** Boom phase
- **Years 14-16:** Peak/"Winner's Curse"
- **Years 16-18.6:** Crash/correction

Current position: **Year 15** (approaching peak)

---

## 🔧 Customization

### Adding New Indicators

Edit `property_analysis_dashboard.py`:

```python
indicators_config = {
    'your_new_indicator': {
        'threshold': 50,
        'type': 'bullish_above'  # or 'bearish_above'
    }
}
```

### Changing Alert Thresholds

Modify thresholds in the scoring logic:

```python
if value > YOUR_THRESHOLD:
    # Trigger alert
```

### Adding New Locations

Simply enter new locations when adding data - they're stored as text strings.

---

## 📅 Recommended Usage Schedule

### Daily (5 minutes)
- Check dashboard for any alerts
- Note major news/announcements

### Weekly (15 minutes)
- Update auction clearance rates
- Review market score
- Read weekly summary email (if n8n configured)

### Monthly (30 minutes)
- Update all economic indicators
- Update property data for tracked locations
- Review trends and adjust strategy
- Export data backup

---

## ⚠️ Important Disclaimers

### This Tool:
- ✅ Provides data and analysis
- ✅ Removes emotional bias
- ✅ Tracks multiple factors objectively
- ✅ Helps identify trends and patterns

### This Tool Does NOT:
- ❌ Replace professional financial advice
- ❌ Guarantee investment returns
- ❌ Predict future prices with certainty
- ❌ Account for your personal circumstances

**Always consult with:**
- Licensed financial advisor
- Qualified accountant
- Property investment specialist
- Mortgage broker

---

## 🆘 Troubleshooting

### Dashboard won't start
```bash
# Try:
pip install --upgrade streamlit pandas plotly
python -m streamlit run property_analysis_dashboard.py
```

### Database errors
- Ensure only one instance is running
- Check `property_data.db` is in the same folder
- Restart the dashboard

### No data showing
- Import sample CSV files from Data Management tab
- Check that dates are formatted correctly (YYYY-MM-DD)
- Verify indicator names match exactly

### n8n workflow not working
- Ensure workflow is set to "Active"
- Check database path in workflow nodes
- Verify email credentials are configured

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed troubleshooting.

---

## 🗺️ Roadmap / Future Enhancements

Potential additions:
- [ ] Mobile app version
- [ ] More data source integrations (CoreLogic API, etc.)
- [ ] Machine learning price predictions
- [ ] Portfolio tracking features
- [ ] Automated property listing alerts
- [ ] Integration with mortgage calculators
- [ ] Suburb-level granularity
- [ ] Property cashflow calculator

---

## 📄 File Structure

```
PropertyAnalyzer/
├── property_analysis_dashboard.py    # Main Streamlit app
├── n8n_workflow_property_analyzer.json  # n8n automation workflow
├── SETUP_GUIDE.md                    # Comprehensive setup guide
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
├── sample_economic_data.csv          # Example economic indicators
├── sample_property_data.csv          # Example property data
├── quickstart.sh                     # Mac/Linux quick setup
├── quickstart.bat                    # Windows quick setup
└── property_data.db                  # SQLite database (auto-created)
```

---

## 🤝 Contributing

This is a personal investment analysis tool, but if you:
- Find bugs
- Have feature suggestions
- Want to share improvements

Feel free to modify and enhance for your own use!

---

## 📜 License

MIT License - Free to use, modify, and distribute.

---

## 🙏 Acknowledgments

- **Phillip Anderson** - 18.6-year cycle theory
- **Streamlit** - Dashboard framework
- **n8n** - Workflow automation
- **Plotly** - Interactive charts
- **Australian Bureau of Statistics** - Public data
- **Reserve Bank of Australia** - Economic data

---

## 📞 Support

For setup help, see [SETUP_GUIDE.md](SETUP_GUIDE.md)

For Streamlit issues: https://docs.streamlit.io
For n8n issues: https://community.n8n.io

---

## ⭐ Final Notes

This system was created to help make **objective, data-driven** property investment decisions in the Australian market.

It synthesizes:
- Economic indicators
- Property market data
- Historical cycles
- Multiple data sources

...into actionable insights.

**Remember:** Markets are complex. This tool provides ONE perspective. Always:
- Do your own research
- Seek professional advice
- Consider your personal circumstances
- Invest within your means
- Plan for multiple scenarios

**Good luck with your property investment journey! 🏠**

---

*Last Updated: February 2026*
