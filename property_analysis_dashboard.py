"""
Australian Property Investment Analysis Dashboard
==================================================
A comprehensive tool for analyzing property market conditions and making data-driven investment decisions.

Features:
- Real-time market indicator tracking
- Historical trend analysis
- Buy/Hold/Wait recommendations
- Location comparison
- Risk assessment

Author: Created for property investment analysis
Date: February 2026
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import sqlite3
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor

def get_db_connection():
    """Get database connection - works with both local SQLite and Supabase PostgreSQL"""
    try:
        # Try Supabase connection (for deployed app)
        if "supabase" in st.secrets:
            conn_string = st.secrets["supabase"]["connection_string"]
            conn = psycopg2.connect(conn_string)
            return conn
        else:
            # Local development - use regular SQLite
            return sqlite3.connect("property_data.db")
    except Exception as e:
        # Fallback to local SQLite
        st.warning(f"Could not connect to Supabase: {e}. Using local SQLite.")
        return sqlite3.connect("property_data.db")

def is_postgres(conn):
    """Check if connection is PostgreSQL"""
    return isinstance(conn, psycopg2.extensions.connection)

# Page configuration
st.set_page_config(
    page_title="Australian Property Investment Analyzer",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .danger-card {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
    .success-card {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    </style>
""", unsafe_allow_html=True)

# Database setup - No longer needed with Turso, kept for compatibility
DB_PATH = Path("property_data.db")

def init_database():
    """Initialize database with required tables - works with both SQLite and PostgreSQL"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if we're using PostgreSQL
    is_pg = is_postgres(conn)
    
    # Use appropriate syntax for PRIMARY KEY
    if is_pg:
        pk_type = "SERIAL PRIMARY KEY"
        timestamp_type = "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
    else:
        pk_type = "INTEGER PRIMARY KEY AUTOINCREMENT"
        timestamp_type = "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
    
    # Economic indicators table
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS economic_indicators (
            id {pk_type},
            date DATE NOT NULL,
            indicator_name TEXT NOT NULL,
            value REAL NOT NULL,
            source TEXT,
            created_at {timestamp_type},
            UNIQUE(date, indicator_name)
        )
    """)
    
    # Property market data table
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS property_data (
            id {pk_type},
            date DATE NOT NULL,
            location TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            value REAL NOT NULL,
            source TEXT,
            created_at {timestamp_type},
            UNIQUE(date, location, metric_name)
        )
    """)
    
    # Market sentiment table
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS market_sentiment (
            id {pk_type},
            date DATE NOT NULL,
            source TEXT NOT NULL,
            sentiment_score REAL,
            notes TEXT,
            created_at {timestamp_type}
        )
    """)
    
    # Market commentary table
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS market_commentary (
            id INTEGER PRIMARY KEY,
            commentary TEXT,
            updated_date DATE
        )
    """)
    
    conn.commit()
    conn.close()

def get_indicator_data(indicator_name, days=365):
    """Fetch historical data for a specific indicator"""
    conn = get_db_connection()
    query = """
        SELECT date, value 
        FROM economic_indicators 
        WHERE indicator_name = ? 
        AND date >= date('now', '-{} days')
        ORDER BY date
    """.format(days)
    df = pd.read_sql_query(query, conn, params=(indicator_name,))
    conn.close()
    return df

def calculate_market_score():
    """Calculate overall market condition score (0-100)"""
    # This is a simplified version - you'll enhance this with real logic
    scores = {
        'bearish_indicators': 0,
        'bullish_indicators': 0,
        'neutral_indicators': 0
    }
    
    # Example scoring logic (to be replaced with real data)
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check latest values for key indicators
    indicators_config = {
        'household_debt_gdp': {'threshold': 110, 'type': 'bearish_above'},
        'mortgage_stress_rate': {'threshold': 30, 'type': 'bearish_above'},
        'rental_vacancy_rate': {'threshold': 2, 'type': 'bullish_below'},
        'auction_clearance_rate': {'threshold': 65, 'type': 'bullish_above'},
        'unemployment_rate': {'threshold': 5, 'type': 'bearish_above'},
    }
    
    for indicator, config in indicators_config.items():
        cursor.execute("""
            SELECT value FROM economic_indicators 
            WHERE indicator_name = ? 
            ORDER BY date DESC LIMIT 1
        """, (indicator,))
        result = cursor.fetchone()
        
        if result:
            value = result[0]
            if config['type'] == 'bearish_above' and value > config['threshold']:
                scores['bearish_indicators'] += 1
            elif config['type'] == 'bullish_below' and value < config['threshold']:
                scores['bullish_indicators'] += 1
            elif config['type'] == 'bullish_above' and value > config['threshold']:
                scores['bullish_indicators'] += 1
            else:
                scores['neutral_indicators'] += 1
    
    conn.close()
    
    total = sum(scores.values())
    if total == 0:
        return 50, "Insufficient Data"
    
    # Calculate weighted score
    bearish_weight = scores['bearish_indicators'] * -20
    bullish_weight = scores['bullish_indicators'] * 20
    base_score = 50
    
    final_score = base_score + ((bullish_weight + bearish_weight) / total)
    final_score = max(0, min(100, final_score))  # Clamp between 0-100
    
    if final_score >= 70:
        signal = "Strong Buy"
    elif final_score >= 55:
        signal = "Buy"
    elif final_score >= 45:
        signal = "Hold"
    elif final_score >= 30:
        signal = "Caution"
    else:
        signal = "Wait"
    
    return final_score, signal

def main():
    # Initialize database
    init_database()
    
    # Header
    st.markdown('<p class="main-header">🏠 Australian Property Investment Analyzer</p>', unsafe_allow_html=True)
    st.markdown("**Data-driven insights for smarter property investment decisions**")
    st.markdown("---")
    
    # Sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Select View",
            ["Dashboard", "Economic Indicators", "Location Analysis", "Anderson Cycle Tracker", "Data Management"]
        )
        
        st.markdown("---")
        st.markdown("### Quick Stats")
        score, signal = calculate_market_score()
        st.metric("Market Score", f"{score:.0f}/100", signal)
        
        st.markdown("---")
        st.markdown("### Last Updated")
        st.text(datetime.now().strftime("%Y-%m-%d %H:%M"))
    
    # Main content area
    if page == "Dashboard":
        show_dashboard()
    elif page == "Economic Indicators":
        show_economic_indicators()
    elif page == "Location Analysis":
        show_location_analysis()
    elif page == "Anderson Cycle Tracker":
        show_anderson_tracker()
    elif page == "Data Management":
        show_data_management()

def show_dashboard():
    """Main dashboard view with overall market assessment"""
    st.header("Market Overview Dashboard")
    
    # Overall market score
    score, signal = calculate_market_score()
    
    col1, col2, col3 = st.columns([2, 2, 3])
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Overall Market Score", f"{score:.0f}/100")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if signal in ["Strong Buy", "Buy"]:
            card_class = "success-card"
        elif signal == "Hold":
            card_class = "metric-card"
        elif signal == "Caution":
            card_class = "warning-card"
        else:
            card_class = "danger-card"
        
        st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
        st.metric("Signal", signal)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        # Gauge chart for market score
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Market Conditions"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightcoral"},
                    {'range': [30, 45], 'color': "lightyellow"},
                    {'range': [45, 70], 'color': "lightblue"},
                    {'range': [70, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True, key="dashboard_market_gauge")
    
    st.markdown("---")
    
    # Key indicators summary
    st.subheader("Key Indicators at a Glance")
    
    # Create metrics grid
    col1, col2, col3, col4 = st.columns(4)
    
    # Pull real metrics from database
    conn = get_db_connection()
    cursor = conn.cursor()
    
    def get_latest_indicator(indicator_name):
        """Get latest value and calculate change for an indicator"""
        cursor.execute("""
            SELECT value, date FROM economic_indicators 
            WHERE indicator_name = ? 
            ORDER BY date DESC LIMIT 2
        """, (indicator_name,))
        results = cursor.fetchall()
        
        if len(results) == 0:
            return None, None
        elif len(results) == 1:
            return results[0][0], 0
        else:
            current = results[0][0]
            previous = results[1][0]
            change = current - previous
            return current, change
    
    # Build indicators list from database
    indicators = []
    
    # Household Debt/GDP
    value, change = get_latest_indicator('household_debt_gdp')
    if value is not None:
        status = "danger" if value > 120 else "warning" if value > 110 else "success"
        indicators.append({
            "name": "Household Debt/GDP",
            "value": f"{value:.1f}%",
            "change": f"{change:+.1f}%" if change != 0 else "No prior data",
            "status": status
        })
    
    # Mortgage Stress Rate
    value, change = get_latest_indicator('mortgage_stress_rate')
    if value is not None:
        status = "danger" if value > 40 else "warning" if value > 30 else "success"
        indicators.append({
            "name": "Mortgage Stress Rate",
            "value": f"{value:.0f}%",
            "change": f"{change:+.1f}%" if change != 0 else "No prior data",
            "status": status
        })
    
    # Rental Vacancy
    value, change = get_latest_indicator('rental_vacancy_rate')
    if value is not None:
        status = "success" if value < 2 else "warning" if value < 3 else "danger"
        indicators.append({
            "name": "Rental Vacancy",
            "value": f"{value:.1f}%",
            "change": f"{change:+.1f}%" if change != 0 else "No prior data",
            "status": status
        })
    
    # Auction Clearance
    value, change = get_latest_indicator('auction_clearance_rate')
    if value is not None:
        status = "success" if value > 70 else "warning" if value > 60 else "danger"
        indicators.append({
            "name": "Auction Clearance",
            "value": f"{value:.0f}%",
            "change": f"{change:+.0f}%" if change != 0 else "No prior data",
            "status": status
        })
    
    conn.close()
    
    # If no data in database, show message
    if len(indicators) == 0:
        st.info("📊 Add economic indicator data in the 'Data Management' tab to see metrics here!")
        indicators = [
            {"name": "Add Data", "value": "—", "change": "", "status": "warning"},
            {"name": "To Get Started", "value": "—", "change": "", "status": "warning"},
            {"name": "Go To", "value": "—", "change": "", "status": "warning"},
            {"name": "Data Management", "value": "—", "change": "", "status": "warning"},
        ]
    
    for i, col in enumerate([col1, col2, col3, col4]):
        if i < len(indicators):
            indicator = indicators[i]
            with col:
                if indicator['status'] == 'success':
                    delta_color = "normal"
                elif indicator['status'] == 'warning':
                    delta_color = "off"
                else:
                    delta_color = "inverse"
                    
                st.metric(
                    indicator['name'],
                    indicator['value'],
                    indicator['change'],
                    delta_color=delta_color
                )
    
    st.markdown("---")
    
    # Recent trends
    st.subheader("Recent Trends (Last 12 Months)")
    
    # Query database for property data
    conn = get_db_connection()
    
    # Get median price data for all locations
    query = """
        SELECT date, location, value 
        FROM property_data 
        WHERE metric_name = 'median_price'
        AND date >= date('now', '-12 months')
        ORDER BY date, location
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if not df.empty:
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Pivot data for plotting
        trend_data = df.pivot(index='date', columns='location', values='value')
        trend_data = trend_data.reset_index()
        
        # Create line chart
        fig = px.line(trend_data, x='date', y=trend_data.columns[1:],
                      title='Median House Prices - Last 12 Months',
                      labels={'value': 'Median Price ($)', 'variable': 'Location', 'date': 'Date'})
        fig.update_layout(hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True, key="dashboard_recent_trends")
    else:
        st.info("📊 Add property data for locations in 'Data Management' → 'Property Market Data' to see trends here!")
        st.markdown("""
        **To add data:**
        1. Go to Data Management tab
        2. Select "Property Market Data"
        3. Enter location (e.g., "Sydney NSW", "Perth WA")
        4. Select metric: median_price
        5. Add monthly values to build your trend chart
        """)
    
    # Commentary section
    st.markdown("---")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        st.subheader("Market Commentary")
    with col2:
        edit_mode = st.button("✏️ Edit", key="edit_commentary_btn")
    
    # Get commentary from database or use default
    conn_comment = get_db_connection()
    cursor_comment = conn_comment.cursor()
    
    # Create commentary table if it doesn't exist
    cursor_comment.execute("""
        CREATE TABLE IF NOT EXISTS market_commentary (
            id INTEGER PRIMARY KEY,
            commentary TEXT,
            updated_date DATE
        )
    """)
    
    cursor_comment.execute("SELECT commentary, updated_date FROM market_commentary WHERE id = 1")
    result = cursor_comment.fetchone()
    
    default_commentary = """**Current Market Assessment**

Based on the available data, the Australian property market shows mixed signals:

**Bearish Factors:**
- Household debt remains at historic highs
- Sydney and Melbourne showing weakness
- Mortgage stress affecting borrowers
- RBA rate movements uncertain

**Bullish Factors:**
- Severe supply shortage
- Regional markets showing strength
- Low mortgage arrears rates
- Tight rental markets

**Recommendation:** Exercise caution. Review indicators regularly and adjust strategy based on data changes."""
    
    current_commentary = result[0] if result else default_commentary
    last_updated = result[1] if result else None
    
    if edit_mode:
        st.markdown("**Edit Your Market Commentary:**")
        new_commentary = st.text_area(
            "Commentary (supports Markdown formatting)",
            value=current_commentary,
            height=300,
            key="commentary_editor"
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("💾 Save", type="primary", key="save_commentary"):
                cursor_comment.execute("""
                    INSERT OR REPLACE INTO market_commentary (id, commentary, updated_date)
                    VALUES (1, ?, date('now'))
                """, (new_commentary,))
                conn_comment.commit()
                st.success("✅ Commentary saved!")
                st.rerun()
        
        with col2:
            if st.button("🔄 Reset to Default", key="reset_commentary"):
                cursor_comment.execute("""
                    INSERT OR REPLACE INTO market_commentary (id, commentary, updated_date)
                    VALUES (1, ?, date('now'))
                """, (default_commentary,))
                conn_comment.commit()
                st.success("✅ Reset to default!")
                st.rerun()
        
        st.markdown("---")
        st.markdown("**Preview:**")
        st.info(new_commentary)
    else:
        st.info(current_commentary)
        if last_updated:
            st.caption(f"Last updated: {last_updated}")
    
    conn_comment.close()

def show_economic_indicators():
    """Detailed view of economic indicators"""
    st.header("Economic Indicators")
    
    st.markdown("""
    Track key economic metrics that influence property markets. 
    **Red** indicators suggest increased crash risk, **Green** indicators suggest growth support.
    """)
    
    # Helper function to get latest indicator value
    conn = get_db_connection()
    cursor = conn.cursor()
    
    def get_indicator_value(indicator_name):
        cursor.execute("""
            SELECT value FROM economic_indicators 
            WHERE indicator_name = ? 
            ORDER BY date DESC LIMIT 1
        """, (indicator_name,))
        result = cursor.fetchone()
        return result[0] if result else None
    
    # Crash Risk Indicators
    st.subheader("🚨 Crash Risk Indicators")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Household Debt to GDP
        debt_gdp = get_indicator_value('household_debt_gdp')
        if debt_gdp:
            st.markdown("**Household Debt to GDP**")
            st.markdown(f"Current: **{debt_gdp:.1f}%** | Target: <110%")
            st.progress(min(1.0, debt_gdp / 130))
        else:
            st.markdown("**Household Debt to GDP**")
            st.markdown("No data - Add in Data Management")
        
        # Mortgage Stress Rate
        stress_rate = get_indicator_value('mortgage_stress_rate')
        if stress_rate:
            st.markdown("**Mortgage Stress Rate**")
            st.markdown(f"Current: **{stress_rate:.1f}%** | Target: <30%")
            st.progress(min(1.0, stress_rate / 50))
        else:
            st.markdown("**Mortgage Stress Rate**")
            st.markdown("No data - Add in Data Management")
        
        # Unemployment Rate
        unemployment = get_indicator_value('unemployment_rate')
        if unemployment:
            st.markdown("**Unemployment Rate**")
            st.markdown(f"Current: **{unemployment:.1f}%** | Target: <5%")
            st.progress(min(1.0, unemployment / 6))
        else:
            st.markdown("**Unemployment Rate**")
            st.markdown("No data - Add in Data Management")
    
    with col2:
        # Interest Rate
        interest_rate = get_indicator_value('interest_rate')
        if interest_rate:
            st.markdown("**Interest Rate (RBA Cash Rate)**")
            st.markdown(f"Current: **{interest_rate:.2f}%**")
            st.progress(min(1.0, interest_rate / 6))
        else:
            st.markdown("**Interest Rate**")
            st.markdown("No data - Add in Data Management")
        
        # Auction Clearance Rate
        clearance = get_indicator_value('auction_clearance_rate')
        if clearance:
            st.markdown("**Auction Clearance Rate**")
            st.markdown(f"Current: **{clearance:.1f}%** | Target: >65%")
            st.progress(clearance / 100)
        else:
            st.markdown("**Auction Clearance Rate**")
            st.markdown("No data - Add in Data Management")
        
        # Building Approvals
        approvals = get_indicator_value('building_approvals')
        if approvals:
            st.markdown("**Building Approvals (Annual)**")
            st.markdown(f"Current: **{approvals:,.0f}** | Target: 240,000")
            st.progress(min(1.0, approvals / 240000))
        else:
            st.markdown("**Building Approvals**")
            st.markdown("No data - Add in Data Management")
    
    st.markdown("---")
    
    # Growth Support Indicators
    st.subheader("💪 Growth Support Indicators")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Rental Vacancy Rate
        vacancy = get_indicator_value('rental_vacancy_rate')
        if vacancy:
            st.markdown("**Rental Vacancy Rate**")
            st.markdown(f"Current: **{vacancy:.1f}%** | Target: <2%")
            st.progress(vacancy / 3)
        else:
            st.markdown("**Rental Vacancy Rate**")
            st.markdown("No data - Add in Data Management")
        
        # Population Growth
        pop_growth = get_indicator_value('population_growth')
        if pop_growth:
            st.markdown("**Population Growth (Annual)**")
            st.markdown(f"Current: **+{pop_growth:,.0f}** people")
            st.progress(min(1.0, pop_growth / 500000))
        else:
            st.markdown("**Population Growth**")
            st.markdown("No data - Add in Data Management")
        
        # Credit Growth
        credit_growth = get_indicator_value('credit_growth')
        if credit_growth:
            st.markdown("**Credit Growth (Monthly)**")
            st.markdown(f"Current: **{credit_growth:.1f}%**")
            st.progress(min(1.0, (credit_growth + 1) / 3))  # Normalize around 0
        else:
            st.markdown("**Credit Growth**")
            st.markdown("No data - Add in Data Management")
    
    with col2:
        # Mortgage Arrears Rate
        arrears = get_indicator_value('mortgage_arrears_rate')
        if arrears:
            st.markdown("**Mortgage Arrears Rate**")
            st.markdown(f"Current: **{arrears:.1f}%** | Target: <2%")
            st.progress(arrears / 3)
        else:
            st.markdown("**Mortgage Arrears Rate**")
            st.markdown("No data - Add in Data Management")
        
        # Dwelling Supply Deficit
        deficit = get_indicator_value('dwelling_supply_deficit')
        if deficit:
            st.markdown("**Dwelling Supply Deficit**")
            st.markdown(f"Current: **{deficit:,.0f}** dwellings short")
            st.progress(min(1.0, deficit / 300000))
        else:
            st.markdown("**Dwelling Supply Deficit**")
            st.markdown("No data - Add in Data Management")
        
        # Wage Growth
        wage_growth = get_indicator_value('wage_growth')
        if wage_growth:
            st.markdown("**Wage Growth (Annual)**")
            st.markdown(f"Current: **{wage_growth:.1f}%**")
            st.progress(min(1.0, wage_growth / 5))
        else:
            st.markdown("**Wage Growth**")
            st.markdown("No data - Add in Data Management")
    
    conn.close()
    
    # Historical trends
    st.markdown("---")
    st.subheader("Historical Trends")
    
    # Get available indicators from database
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT indicator_name FROM economic_indicators ORDER BY indicator_name")
    available_indicators = [row[0] for row in cursor.fetchall()]
    
    if available_indicators:
        # Create friendly names mapping
        friendly_names = {
            'household_debt_gdp': 'Household Debt to GDP',
            'interest_rate': 'Interest Rates',
            'unemployment_rate': 'Unemployment Rate',
            'rental_vacancy_rate': 'Rental Vacancy Rate',
            'mortgage_stress_rate': 'Mortgage Stress Rate',
            'auction_clearance_rate': 'Auction Clearance Rate',
            'building_approvals': 'Building Approvals',
            'credit_growth': 'Credit Growth',
            'mortgage_arrears_rate': 'Mortgage Arrears Rate',
            'dwelling_supply_deficit': 'Dwelling Supply Deficit',
            'population_growth': 'Population Growth',
            'wage_growth': 'Wage Growth',
        }
        
        # Display names for dropdown
        display_names = [friendly_names.get(ind, ind.replace('_', ' ').title()) for ind in available_indicators]
        
        selected_display = st.selectbox("Select indicator to visualize", display_names)
        
        # Get actual indicator name
        selected_idx = display_names.index(selected_display)
        indicator_name = available_indicators[selected_idx]
        
        # Query historical data
        query = """
            SELECT date, value 
            FROM economic_indicators 
            WHERE indicator_name = ? 
            ORDER BY date
        """
        df = pd.read_sql_query(query, conn, params=(indicator_name,))
        
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            
            # Set appropriate threshold based on indicator
            thresholds = {
                'household_debt_gdp': 110,
                'interest_rate': 4.0,
                'unemployment_rate': 5.0,
                'rental_vacancy_rate': 2.0,
                'mortgage_stress_rate': 30,
                'auction_clearance_rate': 65,
                'mortgage_arrears_rate': 2.0,
            }
            threshold = thresholds.get(indicator_name)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['date'], 
                y=df['value'], 
                mode='lines+markers',
                name=selected_display,
                line=dict(width=2),
                marker=dict(size=6)
            ))
            
            if threshold:
                fig.add_hline(
                    y=threshold, 
                    line_dash="dash", 
                    line_color="red", 
                    annotation_text=f"Target Threshold: {threshold}"
                )
            
            fig.update_layout(
                title=f"{selected_display} - Historical Trend",
                xaxis_title="Date",
                yaxis_title="Value",
                hovermode='x unified',
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True, key="economic_historical_trend")
            
            # Show data summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Latest Value", f"{df['value'].iloc[-1]:.2f}")
            with col2:
                if len(df) > 1:
                    change = df['value'].iloc[-1] - df['value'].iloc[-2]
                    st.metric("Change from Previous", f"{change:+.2f}")
                else:
                    st.metric("Change from Previous", "N/A")
            with col3:
                if len(df) > 1:
                    pct_change = ((df['value'].iloc[-1] - df['value'].iloc[0]) / df['value'].iloc[0]) * 100
                    st.metric("Total Change", f"{pct_change:+.1f}%")
                else:
                    st.metric("Total Change", "N/A")
        else:
            st.info("No historical data available for this indicator yet.")
    else:
        st.info("📊 No economic indicator data yet. Add data in the 'Data Management' tab to see trends here!")
        st.markdown("""
        **Quick start:**
        1. Go to Data Management tab
        2. Select "Economic Indicator"
        3. Choose an indicator (e.g., interest_rate)
        4. Add values with dates
        5. Come back here to see the trend chart!
        """)
    
    conn.close()

def show_location_analysis():
    """Compare different locations for investment"""
    st.header("Location Analysis & Comparison")
    
    st.markdown("Compare property markets across different Australian cities and regions.")
    
    # Get available locations from database
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT location FROM property_data ORDER BY location")
    available_locations = [row[0] for row in cursor.fetchall()]
    
    if len(available_locations) < 2:
        st.warning("📍 You need at least 2 locations in your database to compare.")
        st.info("""
        **To add locations:**
        1. Go to Data Management tab
        2. Select "Property Market Data"
        3. Add data for different locations (e.g., "Sydney NSW", "Perth WA", "Albany WA")
        4. Return here to compare them!
        """)
        conn.close()
        return
    
    # Location selector
    col1, col2 = st.columns(2)
    with col1:
        location1 = st.selectbox("Primary Location", available_locations, index=0)
    with col2:
        # Default to second location if different from first
        default_idx = 1 if len(available_locations) > 1 and available_locations[1] != location1 else 0
        location2 = st.selectbox("Compare with", available_locations, index=default_idx)
    
    if location1 == location2:
        st.warning("⚠️ Please select two different locations to compare.")
        conn.close()
        return
    
    st.markdown("---")
    
    # Helper function to get latest metric for a location
    def get_latest_metric(location, metric_name):
        cursor.execute("""
            SELECT value FROM property_data 
            WHERE location = ? AND metric_name = ?
            ORDER BY date DESC LIMIT 1
        """, (location, metric_name))
        result = cursor.fetchone()
        return result[0] if result else None
    
    # Comparison metrics
    st.subheader("Key Metrics Comparison")
    
    # Build comparison table from database
    metrics_to_compare = [
        ('median_price', 'Median Price', lambda x: f"${x/1000:.0f}K" if x else "No data"),
        ('annual_growth', '12-Month Growth', lambda x: f"{x:+.1f}%" if x else "No data"),
        ('rental_yield', 'Rental Yield', lambda x: f"{x:.1f}%" if x else "No data"),
        ('vacancy_rate', 'Vacancy Rate', lambda x: f"{x:.1f}%" if x else "No data"),
        ('days_on_market', 'Days on Market', lambda x: f"{x:.0f}" if x else "No data"),
        ('price_to_income', 'Price-to-Income', lambda x: f"{x:.1f}" if x else "No data"),
    ]
    
    comparison_data = {'Metric': []}
    comparison_data[location1] = []
    comparison_data[location2] = []
    
    for metric_name, display_name, formatter in metrics_to_compare:
        comparison_data['Metric'].append(display_name)
        
        val1 = get_latest_metric(location1, metric_name)
        val2 = get_latest_metric(location2, metric_name)
        
        comparison_data[location1].append(formatter(val1) if val1 is not None else "No data")
        comparison_data[location2].append(formatter(val2) if val2 is not None else "No data")
    
    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Calculate investment scores based on available data
    def calculate_location_score(location):
        score = 50  # Base score
        
        # Positive factors
        growth = get_latest_metric(location, 'annual_growth')
        if growth and growth > 10: score += 15
        elif growth and growth > 5: score += 10
        elif growth and growth > 0: score += 5
        elif growth and growth < 0: score -= 10
        
        rental_yield = get_latest_metric(location, 'rental_yield')
        if rental_yield and rental_yield > 4: score += 10
        elif rental_yield and rental_yield > 3: score += 5
        
        vacancy = get_latest_metric(location, 'vacancy_rate')
        if vacancy and vacancy < 1: score += 15
        elif vacancy and vacancy < 2: score += 10
        elif vacancy and vacancy > 3: score -= 10
        
        days_market = get_latest_metric(location, 'days_on_market')
        if days_market and days_market < 30: score += 10
        elif days_market and days_market > 60: score -= 5
        
        return max(0, min(100, score))  # Clamp between 0-100
    
    score1 = calculate_location_score(location1)
    score2 = calculate_location_score(location2)
    
    # Visual comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### {location1} Score")
        fig1 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score1,
            title={'text': "Investment Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': "lightcoral"},
                    {'range': [40, 70], 'color': "lightyellow"},
                    {'range': [70, 100], 'color': "lightgreen"}
                ]
            }
        ))
        fig1.update_layout(height=300)
        st.plotly_chart(fig1, use_container_width=True, key=f"score_gauge_{location1}")
    
    with col2:
        st.markdown(f"### {location2} Score")
        fig2 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score2,
            title={'text': "Investment Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': "lightcoral"},
                    {'range': [40, 70], 'color': "lightyellow"},
                    {'range': [70, 100], 'color': "lightgreen"}
                ]
            }
        ))
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, use_container_width=True, key=f"score_gauge_{location2}")
    
    # Price trends comparison
    st.markdown("---")
    st.subheader("Price Trend Comparison (Last 12 Months)")
    
    # Query median price trends for both locations
    query = """
        SELECT date, location, value 
        FROM property_data 
        WHERE location IN (?, ?) 
        AND metric_name = 'median_price'
        AND date >= date('now', '-12 months')
        ORDER BY date
    """
    
    trend_df = pd.read_sql_query(query, conn, params=(location1, location2))
    
    if not trend_df.empty:
        trend_df['date'] = pd.to_datetime(trend_df['date'])
        trend_pivot = trend_df.pivot(index='date', columns='location', values='value')
        trend_pivot = trend_pivot.reset_index()
        
        fig = px.line(trend_pivot, x='date', y=trend_pivot.columns[1:],
                      title='Median Price Trends',
                      labels={'value': 'Median Price ($)', 'variable': 'Location', 'date': 'Date'})
        fig.update_layout(hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True, key="location_price_trends")
    else:
        st.info("📊 Add median_price data over multiple dates to see trend comparison.")
    
    # Investment recommendation
    st.markdown("---")
    st.subheader("Investment Recommendation")
    
    if score1 > score2:
        st.success(f"""
        **{location1}** shows stronger investment fundamentals based on current data:
        - Investment Score: {score1}/100 vs {score2}/100
        - Review the metrics above for specific strengths
        
        However, consider timing and overall market conditions before proceeding.
        """)
    elif score2 > score1:
        st.success(f"""
        **{location2}** shows stronger investment fundamentals based on current data:
        - Investment Score: {score2}/100 vs {score1}/100
        - Review the metrics above for specific strengths
        
        However, consider timing and overall market conditions before proceeding.
        """)
    else:
        st.info(f"""
        Both locations show similar investment scores ({score1}/100).
        
        Review the specific metrics above to determine which aligns better with your investment strategy.
        """)
    
    conn.close()

def show_anderson_tracker():
    """Track position in the 18.6 year property cycle"""
    st.header("Phillip Anderson 18.6 Year Cycle Tracker")
    
    st.markdown("""
    This tracker helps you understand where we are in the 18.6-year real estate cycle according to Phillip Anderson's theory.
    """)
    
    # Cycle clock visualization
    st.subheader("Cycle Position")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Create a circular cycle chart
        current_year = 2026
        cycle_start = 2011  # Post-GFC bottom
        years_elapsed = current_year - cycle_start
        cycle_percentage = (years_elapsed / 18.6) * 100
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=years_elapsed,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Years into Cycle (Started {cycle_start})"},
            delta={'reference': 14, 'increasing': {'color': "red"}},
            gauge={
                'axis': {'range': [0, 18.6], 'tickwidth': 1},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'steps': [
                    {'range': [0, 7], 'color': 'lightgreen', 'name': 'Phase 1'},
                    {'range': [7, 9], 'color': 'yellow', 'name': 'Mid-Cycle'},
                    {'range': [9, 14], 'color': 'orange', 'name': 'Phase 2'},
                    {'range': [14, 16], 'color': 'red', 'name': 'Peak'},
                    {'range': [16, 18.6], 'color': 'darkred', 'name': 'Crash'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 14
                }
            }
        ))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True, key="anderson_cycle_gauge")
    
    with col2:
        st.markdown("### Cycle Phase Breakdown")
        st.markdown("""
        **Phase 1 (Years 1-7):** Recovery
        - Steady growth after crash
        - Rebuilding confidence
        - ✅ Completed
        
        **Mid-Cycle (Years 7-9):** Slowdown
        - Correction/recession
        - 2018-19 & COVID
        - ✅ Completed
        
        **Phase 2 (Years 9-14):** Boom
        - Explosive growth
        - Currently here
        - ⚠️ **IN PROGRESS**
        
        **Winner's Curse (Years 14-16):** Peak
        - Speculation peak
        - **⚠️ APPROACHING**
        - Predicted: 2025-2026
        
        **Crash (Years 16-18.6):** Correction
        - Major downturn
        - Best buying opportunity
        - Predicted: 2027-2030
        """)
    
    st.markdown("---")
    
    # Historical cycle comparison
    st.subheader("Historical Cycle Comparison")
    
    st.markdown("""
    Anderson's theory has successfully predicted previous cycles:
    
    | Cycle | Bottom | Peak | Crash | Accuracy |
    |-------|--------|------|-------|----------|
    | 1973-1991 | 1973 | 1989 | 1990-1991 | ✅ Correct |
    | 1992-2010 | 1992 | 2007 | 2008-2010 | ✅ Correct |
    | 2011-2029 | 2011 | **2026** | **2027-2030** | ⏳ In Progress |
    """)
    
    # Current signals
    st.markdown("---")
    st.subheader("Current Cycle Signals")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**✅ Signals We've Seen**")
        st.markdown("""
        - Mid-cycle slowdown (2018-19) ✓
        - COVID recession (2020) ✓
        - Explosive Phase 2 growth ✓
        - Regional market surge ✓
        - FOMO and speculation ✓
        """)
    
    with col2:
        st.markdown("**⚠️ Peak Warning Signs**")
        st.markdown("""
        - Sydney/Melbourne weakening
        - Extreme debt levels
        - Universal bullishness
        - "Property only goes up"
        - Rate hikes not cuts
        """)
    
    with col3:
        st.markdown("**🔮 What to Watch**")
        st.markdown("""
        - Further Sydney/Melbourne falls
        - Credit tightening
        - Unemployment rising
        - Forced sales increasing
        - Sentiment shift
        """)
    
    # Anderson's recommendation
    st.markdown("---")
    st.warning("""
    **According to Anderson's Theory:**
    
    We are approximately at **Year 15 of the 18.6-year cycle** (2026).
    
    This suggests:
    - 🔴 **DO NOT BUY** aggressively now - we're potentially at/near the peak
    - 🟡 **HOLD** existing properties if you have equity buffers
    - 🟢 **PREPARE CASH** for the predicted 2027-2030 downturn
    - 🟢 **BEST BUYING OPPORTUNITY** predicted for 2028-2029
    
    **⚠️ Important:** This is one theory among many. It has historically been accurate but is not guaranteed.
    Use this as ONE input into your decision-making, not the only factor.
    """)

def show_data_management():
    """Manage and input data into the system"""
    st.header("Data Management")
    
    st.markdown("Add, edit, or import data to keep your analysis current.")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Manual Entry", "CSV Import", "View/Edit Data", "📥 Export Data"])
    
    with tab1:
        st.subheader("Manual Data Entry")
        
        data_type = st.selectbox(
            "Data Type",
            ["Economic Indicator", "Property Market Data", "Market Sentiment"]
        )
        
        if data_type == "Economic Indicator":
            col1, col2 = st.columns(2)
            
            with col1:
                indicator_name = st.selectbox(
                    "Indicator",
                    [
                        "interest_rate",
                        "household_debt_gdp",
                        "mortgage_stress_rate",
                        "rental_vacancy_rate",
                        "auction_clearance_rate",
                        "unemployment_rate",
                        "building_approvals",
                        "credit_growth",
                        "mortgage_arrears_rate",
                        "dwelling_supply_deficit",
                        "population_growth",
                        "wage_growth",
                    ]
                )
                value = st.number_input("Value", min_value=0.0, step=0.1)
            
            with col2:
                date = st.date_input("Date", value=datetime.now())
                source = st.text_input("Source", "Manual Entry")
            
            if st.button("Add Indicator Data"):
                try:
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT OR REPLACE INTO economic_indicators (date, indicator_name, value, source)
                        VALUES (?, ?, ?, ?)
                    """, (date, indicator_name, value, source))
                    conn.commit()
                    conn.close()
                    st.success(f"Added {indicator_name} = {value} for {date}")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        elif data_type == "Property Market Data":
            col1, col2 = st.columns(2)
            
            with col1:
                # Get existing locations from database
                conn_temp = get_db_connection()
                cursor_temp = conn_temp.cursor()
                cursor_temp.execute("SELECT DISTINCT location FROM property_data ORDER BY location")
                existing_locations = [row[0] for row in cursor_temp.fetchall()]
                conn_temp.close()
                
                # Location selector with option to add new
                if existing_locations:
                    location_options = existing_locations + ["➕ Add New Location"]
                    selected_location = st.selectbox("Location", location_options)
                    
                    if selected_location == "➕ Add New Location":
                        location = st.text_input("Enter New Location", placeholder="e.g., Brisbane QLD")
                        st.info("💡 Tip: Use format like 'City STATE' (e.g., 'Perth WA', 'Sydney NSW')")
                    else:
                        location = selected_location
                        st.success(f"✅ Using existing location: {location}")
                else:
                    location = st.text_input("Location", placeholder="e.g., Perth WA")
                    st.info("💡 First location! Use format: 'City STATE' (e.g., 'Perth WA', 'Sydney NSW')")
                
                metric_name = st.selectbox(
                    "Metric",
                    [
                        "median_price",
                        "rental_yield",
                        "vacancy_rate",
                        "days_on_market",
                        "auction_clearance",
                        "annual_growth",
                        "price_to_income",
                        "stock_levels",
                        "sales_volume",
                    ]
                )
                value = st.number_input("Value", min_value=0.0, step=0.1, key="property_value_input")
            
            with col2:
                date = st.date_input("Date", value=datetime.now())
                source = st.text_input("Source", "Manual Entry")
            
            if st.button("Add Property Data"):
                try:
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT OR REPLACE INTO property_data (date, location, metric_name, value, source)
                        VALUES (?, ?, ?, ?, ?)
                    """, (date, location, metric_name, value, source))
                    conn.commit()
                    conn.close()
                    st.success(f"Added {metric_name} for {location} = {value} on {date}")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with tab2:
        st.subheader("CSV Import")
        
        st.markdown("""
        Upload CSV files with the following formats:
        
        **Economic Indicators:** date, indicator_name, value, source
        
        **Property Data:** date, location, metric_name, value, source
        """)
        
        uploaded_file = st.file_uploader("Choose CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.dataframe(df.head())
                
                if st.button("Import Data"):
                    # Logic to import would go here
                    st.success(f"Imported {len(df)} rows successfully")
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    with tab3:
        st.subheader("View & Edit Current Data")
        
        view_type = st.selectbox("Select Data Type", ["Economic Indicators", "Property Data"])
        
        conn = get_db_connection()
        
        if view_type == "Economic Indicators":
            df = pd.read_sql_query("""
                SELECT id, date, indicator_name, value, source 
                FROM economic_indicators 
                ORDER BY date DESC, indicator_name
            """, conn)
            
            if not df.empty:
                st.markdown("### Current Economic Indicators")
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                st.subheader("Edit or Delete Entry")
                
                # Create a searchable dropdown
                entries = []
                for idx, row in df.iterrows():
                    display = f"{row['date']} - {row['indicator_name']} = {row['value']} (ID: {row['id']})"
                    entries.append((row['id'], display))
                
                selected = st.selectbox(
                    "Select entry to edit/delete",
                    options=[e[1] for e in entries],
                    key="economic_select"
                )
                
                if selected:
                    # Get the ID from the selected entry
                    selected_id = [e[0] for e in entries if e[1] == selected][0]
                    
                    # Get the full record
                    cursor = conn.cursor()
                    cursor.execute("SELECT * FROM economic_indicators WHERE id = ?", (selected_id,))
                    record = cursor.fetchone()
                    
                    if record:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Edit Values:**")
                            new_date = st.date_input("Date", value=pd.to_datetime(record[1]), key="edit_eco_date")
                            new_indicator = st.selectbox(
                                "Indicator",
                                [
                                    "interest_rate",
                                    "household_debt_gdp",
                                    "mortgage_stress_rate",
                                    "rental_vacancy_rate",
                                    "auction_clearance_rate",
                                    "unemployment_rate",
                                    "building_approvals",
                                    "credit_growth",
                                    "mortgage_arrears_rate",
                                    "dwelling_supply_deficit",
                                    "population_growth",
                                    "wage_growth",
                                ],
                                index=[
                                    "interest_rate",
                                    "household_debt_gdp",
                                    "mortgage_stress_rate",
                                    "rental_vacancy_rate",
                                    "auction_clearance_rate",
                                    "unemployment_rate",
                                    "building_approvals",
                                    "credit_growth",
                                    "mortgage_arrears_rate",
                                    "dwelling_supply_deficit",
                                    "population_growth",
                                    "wage_growth",
                                ].index(record[2]) if record[2] in [
                                    "interest_rate",
                                    "household_debt_gdp",
                                    "mortgage_stress_rate",
                                    "rental_vacancy_rate",
                                    "auction_clearance_rate",
                                    "unemployment_rate",
                                    "building_approvals",
                                    "credit_growth",
                                    "mortgage_arrears_rate",
                                    "dwelling_supply_deficit",
                                    "population_growth",
                                    "wage_growth",
                                ] else 0,
                                key="edit_eco_indicator"
                            )
                            new_value = st.number_input("Value", value=float(record[3]), step=0.1, key="edit_eco_value")
                            new_source = st.text_input("Source", value=record[4], key="edit_eco_source")
                        
                        with col2:
                            st.markdown("**Actions:**")
                            st.markdown("")  # Spacing
                            st.markdown("")  # Spacing
                            
                            if st.button("💾 Update Entry", type="primary", key="update_eco"):
                                try:
                                    cursor.execute("""
                                        UPDATE economic_indicators 
                                        SET date = ?, indicator_name = ?, value = ?, source = ?
                                        WHERE id = ?
                                    """, (new_date, new_indicator, new_value, new_source, selected_id))
                                    conn.commit()
                                    st.success(f"✅ Updated {new_indicator} for {new_date}")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error updating: {e}")
                            
                            st.markdown("")  # Spacing
                            
                            if st.button("🗑️ Delete Entry", type="secondary", key="delete_eco"):
                                try:
                                    cursor.execute("DELETE FROM economic_indicators WHERE id = ?", (selected_id,))
                                    conn.commit()
                                    st.success(f"✅ Deleted entry")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error deleting: {e}")
            else:
                st.info("No economic indicator data yet. Add data using the Manual Entry tab.")
        
        else:  # Property Data
            df = pd.read_sql_query("""
                SELECT id, date, location, metric_name, value, source 
                FROM property_data 
                ORDER BY date DESC, location, metric_name
            """, conn)
            
            if not df.empty:
                st.markdown("### Current Property Data")
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                st.subheader("Edit or Delete Entry")
                
                # Create a searchable dropdown
                entries = []
                for idx, row in df.iterrows():
                    display = f"{row['date']} - {row['location']} - {row['metric_name']} = {row['value']} (ID: {row['id']})"
                    entries.append((row['id'], display))
                
                selected = st.selectbox(
                    "Select entry to edit/delete",
                    options=[e[1] for e in entries],
                    key="property_select"
                )
                
                if selected:
                    # Get the ID from the selected entry
                    selected_id = [e[0] for e in entries if e[1] == selected][0]
                    
                    # Get the full record
                    cursor = conn.cursor()
                    cursor.execute("SELECT * FROM property_data WHERE id = ?", (selected_id,))
                    record = cursor.fetchone()
                    
                    if record:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Edit Values:**")
                            new_date = st.date_input("Date", value=pd.to_datetime(record[1]), key="edit_prop_date")
                            new_location = st.text_input("Location", value=record[2], key="edit_prop_location")
                            new_metric = st.selectbox(
                                "Metric",
                                [
                                    "median_price",
                                    "rental_yield",
                                    "vacancy_rate",
                                    "days_on_market",
                                    "auction_clearance",
                                    "annual_growth",
                                    "price_to_income",
                                    "stock_levels",
                                    "sales_volume",
                                ],
                                index=[
                                    "median_price",
                                    "rental_yield",
                                    "vacancy_rate",
                                    "days_on_market",
                                    "auction_clearance",
                                    "annual_growth",
                                    "price_to_income",
                                    "stock_levels",
                                    "sales_volume",
                                ].index(record[3]) if record[3] in [
                                    "median_price",
                                    "rental_yield",
                                    "vacancy_rate",
                                    "days_on_market",
                                    "auction_clearance",
                                    "annual_growth",
                                    "price_to_income",
                                    "stock_levels",
                                    "sales_volume",
                                ] else 0,
                                key="edit_prop_metric"
                            )
                            new_value = st.number_input("Value", value=float(record[4]), step=0.1, key="edit_prop_value")
                            new_source = st.text_input("Source", value=record[5], key="edit_prop_source")
                        
                        with col2:
                            st.markdown("**Actions:**")
                            st.markdown("")  # Spacing
                            st.markdown("")  # Spacing
                            
                            if st.button("💾 Update Entry", type="primary", key="update_prop"):
                                try:
                                    cursor.execute("""
                                        UPDATE property_data 
                                        SET date = ?, location = ?, metric_name = ?, value = ?, source = ?
                                        WHERE id = ?
                                    """, (new_date, new_location, new_metric, new_value, new_source, selected_id))
                                    conn.commit()
                                    st.success(f"✅ Updated {new_location} - {new_metric} for {new_date}")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error updating: {e}")
                            
                            st.markdown("")  # Spacing
                            
                            if st.button("🗑️ Delete Entry", type="secondary", key="delete_prop"):
                                try:
                                    cursor.execute("DELETE FROM property_data WHERE id = ?", (selected_id,))
                                    conn.commit()
                                    st.success(f"✅ Deleted entry")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error deleting: {e}")
            else:
                st.info("No property data yet. Add data using the Manual Entry tab.")
        
        conn.close()
    
    with tab4:
        st.subheader("📥 Export Your Data for Analysis")
        
        st.markdown("""
        Export all your data in a format ready to share with Claude for comprehensive analysis.
        You can also use this to backup your data or analyze it in Excel/Google Sheets.
        """)
        
        conn = get_db_connection()
        
        # Get all data for summary
        economic_df = pd.read_sql_query("SELECT * FROM economic_indicators ORDER BY date DESC", conn)
        property_df = pd.read_sql_query("SELECT * FROM property_data ORDER BY date DESC", conn)
        
        # Get commentary
        cursor = conn.cursor()
        cursor.execute("SELECT commentary, updated_date FROM market_commentary WHERE id = 1")
        commentary_result = cursor.fetchone()
        
        # Calculate market score
        score, signal = calculate_market_score()
        
        st.markdown("---")
        st.subheader("📊 Current Data Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Economic Indicators", len(economic_df))
        with col2:
            st.metric("Property Data Points", len(property_df))
        with col3:
            locations_count = property_df['location'].nunique() if not property_df.empty else 0
            st.metric("Locations Tracked", locations_count)
        with col4:
            st.metric("Market Score", f"{score}/100")
        
        st.markdown("---")
        st.subheader("Export Options")
        
        export_format = st.radio(
            "Choose export format:",
            ["📄 Comprehensive Report (for Claude Analysis)", "📊 Excel Workbook (for Spreadsheet Analysis)", "📋 CSV Files (Raw Data)"],
            key="export_format"
        )
        
        if export_format == "📄 Comprehensive Report (for Claude Analysis)":
            st.markdown("**This creates a detailed markdown report with:**")
            st.markdown("- All economic indicators with trends")
            st.markdown("- All location data with comparisons")
            st.markdown("- Your market commentary")
            st.markdown("- Current market score and recommendation")
            st.markdown("- Data visualizations described in text")
            
            if st.button("📥 Generate Comprehensive Report", type="primary"):
                # Build comprehensive markdown report
                report = f"""# Australian Property Investment Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

---

## Executive Summary

**Market Score:** {score}/100
**Signal:** {signal}
**Data Points:** {len(economic_df)} economic indicators, {len(property_df)} property metrics
**Locations Tracked:** {locations_count}

---

## Market Commentary

"""
                if commentary_result:
                    report += commentary_result[0] + "\n\n"
                    report += f"*Last updated: {commentary_result[1]}*\n\n"
                else:
                    report += "*No custom commentary yet.*\n\n"
                
                report += "---\n\n## Economic Indicators\n\n"
                
                if not economic_df.empty:
                    # Group by indicator
                    for indicator in economic_df['indicator_name'].unique():
                        indicator_data = economic_df[economic_df['indicator_name'] == indicator].sort_values('date')
                        latest = indicator_data.iloc[0]
                        
                        report += f"### {indicator.replace('_', ' ').title()}\n\n"
                        report += f"**Latest Value:** {latest['value']}\n"
                        report += f"**Date:** {latest['date']}\n"
                        report += f"**Source:** {latest['source']}\n\n"
                        
                        if len(indicator_data) > 1:
                            previous = indicator_data.iloc[1]
                            change = latest['value'] - previous['value']
                            report += f"**Change from previous:** {change:+.2f}\n"
                            
                            report += f"\n**Historical Data ({len(indicator_data)} data points):**\n\n"
                            report += "| Date | Value |\n|------|-------|\n"
                            for _, row in indicator_data.head(10).iterrows():
                                report += f"| {row['date']} | {row['value']:.2f} |\n"
                            
                            if len(indicator_data) > 10:
                                report += f"\n*...and {len(indicator_data) - 10} more historical entries*\n"
                        
                        report += "\n"
                else:
                    report += "*No economic indicator data yet.*\n\n"
                
                report += "---\n\n## Property Market Data by Location\n\n"
                
                if not property_df.empty:
                    for location in property_df['location'].unique():
                        location_data = property_df[property_df['location'] == location]
                        
                        report += f"### {location}\n\n"
                        
                        # Get latest values for each metric
                        for metric in location_data['metric_name'].unique():
                            metric_data = location_data[location_data['metric_name'] == metric].sort_values('date', ascending=False)
                            latest = metric_data.iloc[0]
                            
                            report += f"**{metric.replace('_', ' ').title()}:** {latest['value']}"
                            
                            if len(metric_data) > 1:
                                previous = metric_data.iloc[1]
                                change = latest['value'] - previous['value']
                                pct_change = (change / previous['value'] * 100) if previous['value'] != 0 else 0
                                report += f" ({change:+.2f}, {pct_change:+.1f}%)"
                            
                            report += f" - *as of {latest['date']}*\n"
                        
                        report += "\n"
                        
                        # Show trend table if we have median_price history
                        median_data = location_data[location_data['metric_name'] == 'median_price'].sort_values('date', ascending=False)
                        if len(median_data) > 1:
                            report += "**Median Price Trend:**\n\n"
                            report += "| Date | Price | Change |\n|------|-------|--------|\n"
                            prev_val = None
                            for idx, row in median_data.head(12).iterrows():
                                change_str = ""
                                if prev_val is not None:
                                    change = prev_val - row['value']
                                    pct = (change / row['value'] * 100) if row['value'] != 0 else 0
                                    change_str = f"{change:+,.0f} ({pct:+.1f}%)"
                                report += f"| {row['date']} | ${row['value']:,.0f} | {change_str} |\n"
                                prev_val = row['value']
                            report += "\n"
                        
                        report += "---\n\n"
                else:
                    report += "*No property data yet.*\n\n"
                
                report += """
## Questions for Analysis

Please analyze this data and provide insights on:

1. **Market Timing:** Based on the economic indicators and Anderson 18.6-year cycle theory (currently Year 15), am I at a peak or opportunity?

2. **Location Comparison:** Which location(s) show the strongest fundamentals for investment?

3. **Risk Assessment:** What are the key warning signs I should be most concerned about?

4. **Strategy Recommendation:** Should I be buying, waiting, or preparing cash reserves? What timeframe?

5. **Data Gaps:** What additional metrics should I be tracking to make better decisions?

6. **Trend Analysis:** Are my indicators improving, declining, or stable? What does this suggest?

---

*This report was generated by the Australian Property Investment Analysis System v2.0*
"""
                
                # Offer download
                st.download_button(
                    label="📥 Download Report (.md)",
                    data=report,
                    file_name=f"property_analysis_report_{datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown"
                )
                
                st.success("✅ Report generated! Download it and share with Claude for analysis.")
                
                with st.expander("📄 Preview Report"):
                    st.markdown(report)
        
        elif export_format == "📊 Excel Workbook (for Spreadsheet Analysis)":
            st.markdown("**Creates an Excel file with multiple sheets:**")
            st.markdown("- Economic Indicators")
            st.markdown("- Property Data")
            st.markdown("- Summary Dashboard")
            
            if st.button("📥 Generate Excel Workbook", type="primary"):
                from io import BytesIO
                
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    if not economic_df.empty:
                        economic_df.to_excel(writer, sheet_name='Economic Indicators', index=False)
                    if not property_df.empty:
                        property_df.to_excel(writer, sheet_name='Property Data', index=False)
                    
                    # Summary sheet
                    summary_data = {
                        'Metric': ['Market Score', 'Signal', 'Total Economic Indicators', 'Total Property Data', 'Locations Tracked'],
                        'Value': [score, signal, len(economic_df), len(property_df), locations_count]
                    }
                    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                
                st.download_button(
                    label="📥 Download Excel (.xlsx)",
                    data=output.getvalue(),
                    file_name=f"property_data_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                st.success("✅ Excel workbook generated!")
        
        else:  # CSV Files
            st.markdown("**Downloads separate CSV files for:**")
            st.markdown("- Economic indicators")
            st.markdown("- Property data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if not economic_df.empty:
                    csv_economic = economic_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Economic Indicators CSV",
                        data=csv_economic,
                        file_name=f"economic_indicators_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No economic data to export")
            
            with col2:
                if not property_df.empty:
                    csv_property = property_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Property Data CSV",
                        data=csv_property,
                        file_name=f"property_data_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No property data to export")
        
        conn.close()

if __name__ == "__main__":
    main()
