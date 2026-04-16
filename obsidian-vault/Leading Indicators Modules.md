# Leading Indicators Modules

Five tracking modules accessible from the **Leading Indicators** navigation page. Each has its own database table, scoring function, and CSV import support.

## Module Overview

| Module | Table | Scoring Function | Score Range |
|--------|-------|-----------------|-------------|
| Infrastructure Tracker | `infrastructure_projects` | `calculate_infrastructure_score()` | 0-10 |
| Migration Monitor | `migration_data` | `calculate_population_score()` | 0-10 |
| Jobs Growth Tracker | `employment_data` + `employer_events` | `calculate_employment_score()` | 0-10 |
| Supply/Demand Analyzer | `property_data` | `calculate_supply_demand_score()` | 0-10 |
| Suburb Scorer | `suburb_scores` | Composite of all 4 above | 0-50 |

## 1. Infrastructure Tracker

**Function**: `show_infrastructure_tracker()`

Tracks major infrastructure projects (transport, hospitals, schools, commercial) that drive property values.

**Data fields**: project name, type, location, state, dates (announcement, construction start, expected/actual completion), budget (millions), status, lat/long, impact radius, source, notes.

**Scoring** (`calculate_infrastructure_score`):
- Counts projects within impact radius of a suburb
- Weights by status: under construction > approved > announced
- Weights by budget size and project type
- Score: 0-10

**CSV template**: `infrastructure_projects_template.csv`

## 2. Migration Monitor

**Function**: `show_migration_monitor()`

Tracks population movements that drive housing demand.

**Data fields**: date, state, interstate migration, overseas migration, international students, total population, source.

**Scoring** (`calculate_population_score`):
- Evaluates net migration (interstate + overseas)
- Considers international student numbers
- Compares growth rates over time
- Score: 0-10

**CSV template**: `migration_data_template.csv`

## 3. Jobs Growth Tracker

**Function**: `show_jobs_tracker()`

Tracks employment conditions and major employer events.

**Employment data fields**: date, region, total employed, unemployment rate, job ads count, employment growth rate, source.

**Employer events fields**: date, employer name, event type (expansion/closure/relocation/restructure), location, jobs impact, industry, source, notes.

**Scoring** (`calculate_employment_score`):
- Evaluates unemployment rate trend
- Considers job ads growth
- Factors in major employer events (expansions vs closures)
- Score: 0-10

**CSV templates**: `employment_data_template.csv`, `employer_events_template.csv`

## 4. Supply/Demand Analyzer

**Function**: `show_supply_demand_analyzer()`

Analyzes housing supply vs demand dynamics using property data.

**Uses existing `property_data` table** - metrics like median price, days on market, rental yield, vacancy rate, building approvals.

**Scoring** (`calculate_supply_demand_score`):
- Low vacancy + low approvals + high clearance = supply squeeze (high score)
- High vacancy + high approvals = oversupply (low score)
- Score: 0-10

## 5. Multi-Indicator Suburb Scorer

**Function**: `show_suburb_scorer()`

Combines all four module scores into a composite suburb rating.

**Calculation**:
```
Total Score = Infrastructure (0-10)
            + Population (0-10)
            + Employment (0-10)
            + Supply/Demand (0-10)
            + Credit/Gentrification (0-10, placeholder)
            = 0 to 50
```

**Features**:
- Score a specific suburb by entering suburb + state
- View historical scores table
- Scores saved to `suburb_scores` table with breakdown

## Data Management Integration

All leading indicator data types are accessible from the **Data Management** tab:

- **CSV Import** tab: 6 data types (Economic Indicators, Property Data, Infrastructure Projects, Migration Data, Employment Data, Employer Events)
- **View/Edit Data** tab: 10 data types with delete functionality (includes Infrastructure Projects, Migration Data, Employment Data, Employer Events, Suburb Scores)

## Related Notes

- [[Property Analyser - Architecture]]
- [[Score Calculation Pipeline]]
