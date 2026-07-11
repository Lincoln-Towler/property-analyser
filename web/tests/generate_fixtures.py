"""Golden-master fixture generator.

Runs the REAL scoring functions from property_analysis_dashboard.py against
in-memory SQLite databases seeded with synthetic scenarios, and dumps the
exact outputs as JSON. The TypeScript port must reproduce these byte-for-byte
(modulo float formatting).

Dates are generated relative to today because the Python engine internally
uses date('now', ...) windows and datetime.now().year; the fixtures record
`generated_on` so the TS tests evaluate with the same clock.
"""
import json
import pathlib
import sqlite3
import sys
from datetime import date, datetime, timedelta

# repo root (this file lives at web/tests/)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import property_analysis_dashboard as dash

TODAY = date.today()


def d(days_ago):
    return (TODAY - timedelta(days=days_ago)).isoformat()


def make_db(indicator_rows, property_rows=()):
    """indicator_rows: list of (indicator_name, days_ago, value)"""
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE economic_indicators (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE NOT NULL,
            indicator_name TEXT NOT NULL,
            value REAL NOT NULL,
            source TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(date, indicator_name)
        )
    """)
    cur.execute("""
        CREATE TABLE property_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE NOT NULL,
            location TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            value REAL NOT NULL,
            source TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(date, location, metric_name)
        )
    """)
    for name, days_ago, value in indicator_rows:
        cur.execute(
            "INSERT INTO economic_indicators (date, indicator_name, value) VALUES (?,?,?)",
            (d(days_ago), name, value),
        )
    for location, metric, days_ago, value in property_rows:
        cur.execute(
            "INSERT INTO property_data (date, location, metric_name, value) VALUES (?,?,?,?)",
            (d(days_ago), location, metric, value),
        )
    conn.commit()
    # Reset the module's view cache so each scenario re-detects (no history table here)
    dash._ei_view_cache = {"name": None, "has_history": None}
    return conn


def series(name, spec):
    """spec: list of (days_ago, value)"""
    return [(name, days_ago, value) for days_ago, value in spec]


SCENARIOS = {}

# --- baseline: realistic full dataset, mixed conditions ---
SCENARIOS["baseline"] = dict(
    indicators=(
        series("interest_rate", [(150, 4.35), (90, 4.10), (30, 3.85)])
        + series("household_debt_gdp", [(200, 112.0), (100, 111.0), (20, 109.5)])
        + series("rental_vacancy_rate", [(120, 1.6), (60, 1.7), (10, 1.8)])
        + series("building_approvals", [(160, 165000), (80, 170000), (15, 172000)])
        + series("mortgage_stress_rate", [(140, 28.0), (70, 27.5), (25, 26.0)])
        + series("unemployment_rate", [(130, 4.2), (65, 4.3), (12, 4.4)])
        + series("auction_clearance_rate", [(100, 68.0), (50, 71.0), (8, 74.0)])
        + series("credit_growth", [(110, 0.5), (55, 0.6), (18, 0.65)])
        + series("wage_growth", [(180, 3.4), (90, 3.5), (22, 3.6)])
    ),
    properties=(),
)

# --- perfect storm: composite override must cap base at 25 ---
SCENARIOS["perfect_storm"] = dict(
    indicators=(
        series("interest_rate", [(60, 6.0), (30, 6.2), (5, 6.4)])
        + series("household_debt_gdp", [(60, 125.0), (30, 126.0), (5, 127.0)])
        + series("mortgage_stress_rate", [(60, 37.0), (30, 38.0), (5, 39.0)])
        + series("unemployment_rate", [(60, 5.2), (30, 5.4), (5, 5.6)])
        + series("rental_vacancy_rate", [(60, 2.8), (30, 2.9), (5, 3.0)])
    ),
    properties=(),
)

# --- crisis: stress>40 + unemployment>6 caps at 30; also volatile data ---
SCENARIOS["crisis"] = dict(
    indicators=(
        series("mortgage_stress_rate", [(90, 35.0), (60, 41.0), (30, 44.0), (5, 47.0)])
        + series("unemployment_rate", [(90, 5.5), (60, 6.2), (30, 6.8), (5, 7.4)])
        + series("interest_rate", [(90, 5.0), (60, 5.6), (30, 6.1), (5, 6.6)])
    ),
    properties=(),
)

# --- supply squeeze: vacancy<1.5 + approvals<180k + clearance>70 → +15 ---
SCENARIOS["supply_squeeze"] = dict(
    indicators=(
        series("rental_vacancy_rate", [(80, 1.2), (40, 1.1), (10, 1.0)])
        + series("building_approvals", [(80, 155000), (40, 152000), (10, 150000)])
        + series("auction_clearance_rate", [(80, 72.0), (40, 74.0), (10, 76.0)])
        + series("interest_rate", [(80, 3.5), (40, 3.5), (10, 3.5)])
        + series("wage_growth", [(80, 3.5), (40, 3.5), (10, 3.5)])
    ),
    properties=(),
)

# --- oversupply: vacancy>3.5 + approvals>250k → -15 ---
SCENARIOS["oversupply"] = dict(
    indicators=(
        series("rental_vacancy_rate", [(80, 3.8), (40, 4.0), (10, 4.2)])
        + series("building_approvals", [(80, 255000), (40, 260000), (10, 265000)])
        + series("auction_clearance_rate", [(80, 52.0), (40, 50.0), (10, 48.0)])
    ),
    properties=(),
)

# --- sparse: mirrors live DB — single points, stale data, insufficient volatility ---
SCENARIOS["sparse_like_production"] = dict(
    indicators=(
        [("household_debt_gdp", 380, 112.0)]           # one stale point
        + [("mortgage_stress_rate", 100, 27.0)]        # one point
        + series("interest_rate", [(120, 4.1), (45, 3.85)])   # two points → trend, no volatility
        + series("auction_clearance_rate", [(70, 66.0), (35, 69.0), (7, 73.0)])
        + series("rental_vacancy_rate", [(140, 1.7), (75, 1.8), (20, 1.9)])
        + series("building_approvals", [(150, 168000), (60, 171000)])
        + series("unemployment_rate", [(90, 4.3), (30, 4.4)])
    ),
    properties=(),
)

# --- with_regions: exercises regional divergence ---
SCENARIOS["with_regions"] = dict(
    indicators=SCENARIOS["baseline"]["indicators"],
    properties=[
        ("Perth WA", "median_price", 30, 780000),
        ("Perth WA", "rental_yield", 30, 5.1),
        ("Perth WA", "vacancy_rate", 30, 1.4),
        ("Perth WA", "annual_growth", 30, 8.0),
        ("Sydney NSW", "median_price", 30, 1450000),
        ("Sydney NSW", "rental_yield", 30, 3.1),
        ("Sydney NSW", "vacancy_rate", 30, 2.4),
        ("Sydney NSW", "annual_growth", 30, 2.0),
        ("Albany WA", "median_price", 30, 560000),
        ("Albany WA", "rental_yield", 30, 4.8),
        ("Albany WA", "vacancy_rate", 30, 1.1),
        ("Albany WA", "annual_growth", 30, 6.0),
    ],
)

# --- empty: no data at all ---
SCENARIOS["empty"] = dict(indicators=[], properties=())


def run():
    out = {
        "generated_on": TODAY.isoformat(),
        "current_year": datetime.now().year,
        "scenarios": {},
    }
    for name, spec in SCENARIOS.items():
        conn = make_db(spec["indicators"], spec.get("properties", ()))
        v2_score, v2_signal, v2_breakdown = dash.calculate_market_score_v2(conn)
        dash._ei_view_cache = {"name": None, "has_history": None}
        v3_score, v3_signal, v3_breakdown = dash.calculate_market_score_v3(conn)
        commentary = dash.generate_auto_commentary(conn)
        conn.close()
        out["scenarios"][name] = {
            "inputs": {
                "indicators": [
                    {"name": n, "date": d(a), "value": v}
                    for n, a, v in spec["indicators"]
                ],
                "property_data": [
                    {"location": l, "metric": m, "date": d(a), "value": v}
                    for l, m, a, v in spec.get("properties", ())
                ],
            },
            "v2": {"score": v2_score, "signal": v2_signal, "breakdown": v2_breakdown},
            "v3": {"score": v3_score, "signal": v3_signal, "breakdown": v3_breakdown},
            "commentary": commentary,
        }
    return out


if __name__ == "__main__":
    result = run()
    path = sys.argv[1] if len(sys.argv) > 1 else "fixtures.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"wrote {path}")
    for name, s in result["scenarios"].items():
        print(f"  {name}: v3={s['v3']['score']:.2f} {s['v3']['signal']}")
