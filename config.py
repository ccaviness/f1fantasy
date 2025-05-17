# config.py
"""Configuration settings for the F1 Fantasy application."""

# --- Main Configuration ---
INITIAL_BUDGET: float = 100.0  # Standard initial budget in F1 Fantasy (in millions)
"""The standard initial budget for a fantasy team, in millions."""

# --- Data Source URLs ---
ASSET_DATA_URL: str = (
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vRN0cf_B_U3KoYRAdCbiWrxxEplWZxiy0WQ6KIImEJ4E7mh147bDD5kSUsnDbGYFNChs6FNFQyfQThl/pub?gid=1838985806&single=true&output=csv"
)
"""URL for the CSV file containing F1 asset data (drivers and constructors)."""

MY_TEAM_URL: str = (
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vRN0cf_B_U3KoYRAdCbiWrxxEplWZxiy0WQ6KIImEJ4E7mh147bDD5kSUsnDbGYFNChs6FNFQyfQThl/pub?gid=2095757091&single=true&output=csv"
)
"""URL for the CSV file representing the user's current F1 fantasy team."""

MANUAL_ADJUSTMENTS_URL: str = (
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vRN0cf_B_U3KoYRAdCbiWrxxEplWZxiy0WQ6KIImEJ4E7mh147bDD5kSUsnDbGYFNChs6FNFQyfQThl/pub?gid=1943142711&single=true&output=csv"
)
"""URL for the CSV file containing manual adjustments to asset scores or data."""

# --- Data Structure Definitions ---
METADATA_COLUMNS: list[str] = ["ID", "Name", "Type", "Constructor", "Price", "Active"]
"""List of column names considered as metadata for assets."""

# --- Gameplay Parameters ---
DEFAULT_FREE_TRANSFERS: int = 3
"""The default number of free transfers allowed per game period."""

# --- Scoring Heuristic Configuration ---
# Define keys for weights consistently (used in WEIGHT_PROFILES and _calculate_derived_scores)
KEY_RECENT_FORM: str = "recent_form"
"""Dictionary key for the 'recent form' scoring component."""

KEY_LAST_RACE: str = "last_race"
"""Dictionary key for the 'last race performance' scoring component."""

KEY_PPM: str = "ppm"
"""Dictionary key for the 'points per million' (value) scoring component."""

KEY_TOTAL_POINTS: str = "total_points"
"""Dictionary key for the 'total points accumulated' scoring component."""

KEY_TREND: str = "trend"
"""Dictionary key for the 'performance trend' scoring component."""

WEIGHT_PROFILES: dict[str, dict[str, float]] = {
    "balanced": {
        KEY_RECENT_FORM: 0.30,
        KEY_LAST_RACE: 0.20,
        KEY_PPM: 0.25,
        KEY_TOTAL_POINTS: 0.15,
        KEY_TREND: 0.10,
    },
    "aggressive_form": {
        KEY_RECENT_FORM: 0.40,
        KEY_LAST_RACE: 0.25,
        KEY_PPM: 0.10,
        KEY_TOTAL_POINTS: 0.10,
        KEY_TREND: 0.15,
    },
    "value_focused": {
        KEY_RECENT_FORM: 0.15,
        KEY_LAST_RACE: 0.10,
        KEY_PPM: 0.40,
        KEY_TOTAL_POINTS: 0.25,
        KEY_TREND: 0.10,
    },
}
"""
A dictionary defining different weighting profiles for calculating combined scores.
Each profile is a dictionary mapping scoring component keys to their respective weights.
Profiles include:
    - "balanced": A general-purpose weighting.
    - "aggressive_form": Emphasizes recent performance and momentum.
    - "value_focused": Emphasizes points per million (PPM) and overall consistency.
"""

# --- Display Configuration ---
COLUMN_NAME_ABBREVIATIONS: dict[str, str] = {
    "ID": "ID",
    "Name": "Name",
    "Type": "Type",
    "Constructor": "Team",
    "Price": "Price",
    "Active": "Active",
    "Purchase_Price": "PurchPr",
    "Total_Points_So_Far": "TotPts",
    "Avg_Points_Last_3_Races": "AvgL3",
    "User_Adjusted_Avg_Points_Last_3_Races": "AdjAvgL3",
    "Point_Adjustment_Avg3Races": "AdjVal",
    "Points_Last_Race": "LastR",
    "Trend_Score": "Trend",
    "PPM_Current": "PPM_Cur",
    "PPM_on_Purchase": "PPM_Pur",
    "Combined_Score": "Score",
    "Norm_User_Adjusted_Avg_Points_Last_3": "N:AdjAvgL3",  # Corrected from your script's Norm_Avg_Points_Last_3
    "Norm_Points_Last_Race": "N:LastR",
    "Norm_PPM": "N:PPM",
    "Norm_Total_Points_So_Far": "N:TotPts",
    "Norm_Trend_Score": "N:Trend",
}
"""
A dictionary mapping full column names to their desired abbreviations for display purposes.
This helps in creating more compact and readable tables or outputs.
"""
