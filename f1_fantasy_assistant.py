# This script is a Python module for managing and analyzing F1 Fantasy teams.

import itertools
import os
import pulp
import pandas as pd
import numpy as np

# --- Configuration ---
INITIAL_BUDGET = 100.0  # Standard initial budget in F1 Fantasy (in millions)
ASSET_DATA_FILE = "asset_data.csv"
ASSET_DATA_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRN0cf_B_U3KoYRAdCbiWrxxEplWZxiy0WQ6KIImEJ4E7mh147bDD5kSUsnDbGYFNChs6FNFQyfQThl/pub?gid=1838985806&single=true&output=csv"
MY_TEAM_FILE = "my_team.csv"
MY_TEAM_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRN0cf_B_U3KoYRAdCbiWrxxEplWZxiy0WQ6KIImEJ4E7mh147bDD5kSUsnDbGYFNChs6FNFQyfQThl/pub?gid=2095757091&single=true&output=csv"
MANUAL_ADJUSTMENTS_FILE = "manual_adjustments.csv"
MANUAL_ADJUSTMENTS_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRN0cf_B_U3KoYRAdCbiWrxxEplWZxiy0WQ6KIImEJ4E7mh147bDD5kSUsnDbGYFNChs6FNFQyfQThl/pub?gid=1943142711&single=true&output=csv"

# Define the expected metadata columns in asset_data.csv
# All other columns will be treated as GP points columns
METADATA_COLUMNS = ["ID", "Name", "Type", "Constructor", "Price", "Active"]
DEFAULT_FREE_TRANSFERS = 2  # Standard free transfers per week, can be adjusted

# Define keys for weights consistently
KEY_RECENT_FORM = "recent_form"
KEY_LAST_RACE = "last_race"
KEY_PPM = "ppm"
KEY_TOTAL_POINTS = "total_points"
KEY_TREND = "trend"

WEIGHT_PROFILES = {
    "balanced": {
        KEY_RECENT_FORM: 0.30,
        KEY_LAST_RACE: 0.20,
        KEY_PPM: 0.25,
        KEY_TOTAL_POINTS: 0.15,
        KEY_TREND: 0.10,
    },
    "aggressive_form": {  # Emphasizes recent performance and momentum
        KEY_RECENT_FORM: 0.40,
        KEY_LAST_RACE: 0.25,
        KEY_PPM: 0.10,
        KEY_TOTAL_POINTS: 0.10,
        KEY_TREND: 0.15,
    },
    "value_focused": {  # Emphasizes PPM and overall consistency
        KEY_RECENT_FORM: 0.15,
        KEY_LAST_RACE: 0.10,
        KEY_PPM: 0.40,
        KEY_TOTAL_POINTS: 0.25,
        KEY_TREND: 0.10,
    },
}


def _load_raw_asset_df(asset_data_url):  # Parameter changed
    """Loads and performs initial validation on asset_data from a URL."""
    warnings = ""
    try:
        print(f"Attempting to load asset data from URL: {asset_data_url}")
        if (
            not asset_data_url
            or asset_data_url == "YOUR_GOOGLE_SHEET_URL_FOR_ASSET_DATA_CSV"
        ):  # Check if placeholder
            raise ValueError(
                "Asset data URL is a placeholder or empty. Please update it in the script's configuration."
            )
        df = pd.read_csv(asset_data_url)
        df.columns = df.columns.str.strip()
        for col in METADATA_COLUMNS:  # METADATA_COLUMNS is global
            if col not in df.columns:
                raise ValueError(
                    f"Essential metadata column '{col}' not found in data from {asset_data_url}. Required: {METADATA_COLUMNS}"
                )
        return df, warnings
    except (
        ValueError
    ) as e:  # Catch our specific ValueError for placeholder or missing columns
        return None, f"\nConfiguration or Data Error: {e}"
    except Exception as e:  # More general exception for URL/network issues
        return None, f"\nError loading asset data from URL {asset_data_url}: {e}"


def _calculate_points_metrics(df, metadata_cols_list):
    """Calculates Total_Points_So_Far, Avg_Points_Last_3_Races, and Points_Last_Race."""
    warnings = ""
    potential_gp_cols = [col for col in df.columns if col not in metadata_cols_list]
    completed_gp_cols = []

    if not potential_gp_cols:
        warnings += (
            "\nWarning: No potential GP points columns found. Points metrics set to 0."
        )
        df["Total_Points_So_Far"] = 0.0
        df["Avg_Points_Last_3_Races"] = 0.0
        df["Points_Last_Race"] = 0.0
    else:
        for col_name in potential_gp_cols:
            numeric_series = pd.to_numeric(df[col_name], errors="coerce")
            if numeric_series.notna().any():
                completed_gp_cols.append(col_name)
                df[col_name] = numeric_series
            else:
                df[col_name] = np.nan

        if not completed_gp_cols:
            warnings += (
                "\nWarning: No completed GP races identified. Points metrics set to 0."
            )
            df["Total_Points_So_Far"] = 0.0
            df["Avg_Points_Last_3_Races"] = 0.0
            df["Points_Last_Race"] = 0.0
        else:
            print(f"Identified completed GP columns: {completed_gp_cols}")
            df["Total_Points_So_Far"] = df[completed_gp_cols].sum(axis=1, skipna=True)

            num_races_to_avg = min(len(completed_gp_cols), 3)
            if num_races_to_avg > 0:
                last_n_cols = completed_gp_cols[-num_races_to_avg:]
                print(f"Calculating Avg_Points_Last_3_Races based on: {last_n_cols}")
                df["Avg_Points_Last_3_Races"] = (
                    df[last_n_cols].mean(axis=1, skipna=True).fillna(0)
                )
            else:
                df["Avg_Points_Last_3_Races"] = 0.0

            print(f"Calculating Points_Last_Race based on: {completed_gp_cols[-1]}")
            df["Points_Last_Race"] = df[completed_gp_cols[-1]].fillna(0)

    return df, warnings


def _preprocess_asset_attributes(df):
    """Parses Price and Active columns."""
    warnings = ""
    if "Price" in df.columns:
        if df["Price"].dtype == "object":
            df["Price"] = (
                df["Price"].replace({r"\$": "", "M": ""}, regex=True).astype(float)
            )
        else:
            df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0)
    else:
        df["Price"] = 0.0
        warnings += "\nCritical Warning: 'Price' column missing, defaulted to 0."

    if "Active" in df.columns:
        df["Active"] = df["Active"].astype(bool)
    else:
        df["Active"] = False  # Default to False if missing
        warnings += "\nCritical Warning: 'Active' column missing, defaulted to False."
    return df, warnings


def _apply_manual_adjustments(df, adjustments_url):
    warnings = ""
    # Initialize the target column in the main df first
    # This ensures it exists, even if no adjustments are loaded or merged.
    df["Point_Adjustment_Avg3Races"] = 0.0

    if (
        not adjustments_url
        or adjustments_url == "YOUR_GOOGLE_SHEET_URL_FOR_MANUAL_ADJUSTMENTS_CSV"
    ):
        print(
            "Info: Manual adjustments URL is a placeholder or empty. No adjustments will be applied."
        )
        # df['User_Adjusted_Avg_Points_Last_3_Races'] will be calculated later using the initialized 0.0
        if (
            "Avg_Points_Last_3_Races" in df.columns
            and "User_Adjusted_Avg_Points_Last_3_Races" not in df.columns
        ):
            df["User_Adjusted_Avg_Points_Last_3_Races"] = df["Avg_Points_Last_3_Races"]
        return df, warnings

    try:
        print(f"Attempting to load manual adjustments from URL: {adjustments_url}")
        adj_df = pd.read_csv(adjustments_url)
        adj_df.columns = adj_df.columns.str.strip()

        if "ID" in adj_df.columns and "Point_Adjustment_Avg3Races" in adj_df.columns:
            # Ensure the adjustment column from the file is numeric
            adj_df["Point_Adjustment_Avg3Races_from_file"] = pd.to_numeric(
                adj_df["Point_Adjustment_Avg3Races"], errors="coerce"
            ).fillna(0)

            # Merge. Suffixes will apply if 'Point_Adjustment_Avg3Races' was already in df (which it is, as 0.0)
            # However, since we want to overwrite, a map or update approach is cleaner.
            # Let's use map for simplicity if IDs are unique in adj_df, or merge and then select.

            # Create a mapping from ID to adjustment value
            adjustment_map = adj_df.set_index("ID")[
                "Point_Adjustment_Avg3Races_from_file"
            ]

            # Apply this map to the 'ID' column of the main df to create/update Point_Adjustment_Avg3Races
            # This will align adjustments by ID and fill with NaN for IDs in df not in adjustment_map
            df["Point_Adjustment_Avg3Races"] = df["ID"].map(adjustment_map).fillna(0)

            print("Successfully loaded and applied manual adjustments.")
        else:
            warnings += f"\nWarning: Data from {adjustments_url} is missing 'ID' or 'Point_Adjustment_Avg3Races' column. Adjustments not applied (using 0).."
    except Exception as e:
        warnings += f"\nError loading or processing adjustments from URL {adjustments_url}: {e}. Adjustments not applied (using 0)."

    # Ensure the 'User_Adjusted_Avg_Points_Last_3_Races' column is created
    if "Avg_Points_Last_3_Races" in df.columns:
        df["User_Adjusted_Avg_Points_Last_3_Races"] = (
            df["Avg_Points_Last_3_Races"] + df["Point_Adjustment_Avg3Races"]
        )
    else:
        warnings += "\nWarning: 'Avg_Points_Last_3_Races' missing for applying manual adjustments."
        # If Avg_Points is missing, User_Adjusted will just be the adjustment itself, or 0 if Point_Adjustment is also missing.
        df["User_Adjusted_Avg_Points_Last_3_Races"] = df.get(
            "Point_Adjustment_Avg3Races", 0.0
        )

    return df, warnings


def _calculate_derived_scores(df, selected_weights):  # Added selected_weights parameter
    """Calculates PPM_Current and an enhanced Combined_Score using provided weights."""
    warnings = ""

    # Calculate PPM_Current (assuming 'Price' and 'Total_Points_So_Far' exist and are numeric)
    # ... (PPM_Current calculation logic remains the same as your current version) ...
    df["PPM_Current"] = 0.0
    if "Price" in df.columns and "Total_Points_So_Far" in df.columns:
        non_zero_price_mask = (df["Price"].notna()) & (df["Price"] != 0)
        df.loc[non_zero_price_mask, "PPM_Current"] = (
            df.loc[non_zero_price_mask, "Total_Points_So_Far"]
            / df.loc[non_zero_price_mask, "Price"]
        )
        df["PPM_Current"] = df["PPM_Current"].replace([np.inf, -np.inf], 0).fillna(0)
    else:
        warnings += "\nWarning: 'Price' or 'Total_Points_So_Far' missing for PPM_Current. PPM_Current set to 0."

    # Calculate Trend_Score
    if (
        "Points_Last_Race" in df.columns
        and "User_Adjusted_Avg_Points_Last_3_Races" in df.columns
    ):
        df["Trend_Score"] = (
            df["Points_Last_Race"] - df["User_Adjusted_Avg_Points_Last_3_Races"]
        )
    else:
        warnings += "\nWarning: Columns for Trend_Score calculation missing. Trend_Score set to 0."
        df["Trend_Score"] = 0.0
    df["Trend_Score"] = df["Trend_Score"].fillna(0)

    # Initialize score columns
    df["Combined_Score"] = 0.0
    df["Norm_User_Adjusted_Avg_Points_Last_3"] = 0.5
    df["Norm_Points_Last_Race"] = 0.5
    df["Norm_PPM"] = 0.5
    df["Norm_Total_Points_So_Far"] = 0.5
    df["Norm_Trend_Score"] = 0.5

    for asset_type in ["Driver", "Constructor"]:
        type_mask = df["Type"] == asset_type
        if type_mask.sum() > 0:
            # Ensure source columns are filled before normalization
            df.loc[type_mask, "User_Adjusted_Avg_Points_Last_3_Races"] = df.loc[
                type_mask, "User_Adjusted_Avg_Points_Last_3_Races"
            ].fillna(0)
            df.loc[type_mask, "Points_Last_Race"] = df.loc[
                type_mask, "Points_Last_Race"
            ].fillna(0)
            df.loc[type_mask, "PPM_Current"] = df.loc[type_mask, "PPM_Current"].fillna(
                0
            )
            df.loc[type_mask, "Total_Points_So_Far"] = df.loc[
                type_mask, "Total_Points_So_Far"
            ].fillna(0)
            df.loc[type_mask, "Trend_Score"] = df.loc[type_mask, "Trend_Score"].fillna(
                0
            )

            df.loc[type_mask, "Norm_User_Adjusted_Avg_Points_Last_3"] = (
                normalize_series(
                    df.loc[type_mask, "User_Adjusted_Avg_Points_Last_3_Races"]
                )
            )
            df.loc[type_mask, "Norm_Points_Last_Race"] = normalize_series(
                df.loc[type_mask, "Points_Last_Race"]
            )
            df.loc[type_mask, "Norm_PPM"] = normalize_series(
                df.loc[type_mask, "PPM_Current"]
            )
            df.loc[type_mask, "Norm_Total_Points_So_Far"] = normalize_series(
                df.loc[type_mask, "Total_Points_So_Far"]
            )
            df.loc[type_mask, "Norm_Trend_Score"] = normalize_series(
                df.loc[type_mask, "Trend_Score"]
            )

            norm_avg3 = df.loc[
                type_mask, "Norm_User_Adjusted_Avg_Points_Last_3"
            ].fillna(0.5)
            norm_last_race = df.loc[type_mask, "Norm_Points_Last_Race"].fillna(0.5)
            norm_ppm = df.loc[type_mask, "Norm_PPM"].fillna(0.5)
            norm_total = df.loc[type_mask, "Norm_Total_Points_So_Far"].fillna(0.5)
            norm_trend = df.loc[type_mask, "Norm_Trend_Score"].fillna(0.5)

            # Use weights from the selected_weights dictionary
            df.loc[type_mask, "Combined_Score"] = (
                selected_weights[KEY_RECENT_FORM] * norm_avg3
                + selected_weights[KEY_LAST_RACE] * norm_last_race
                + selected_weights[KEY_PPM] * norm_ppm
                + selected_weights[KEY_TOTAL_POINTS] * norm_total
                + selected_weights[KEY_TREND] * norm_trend
            )

    df["Combined_Score"] = df["Combined_Score"].fillna(0)
    if warnings:
        # This print might be too verbose here, consider returning warnings
        # print(f"Warnings from _calculate_derived_scores: {warnings}")
        pass
    return df, warnings


def _load_and_process_team_df(team_url, all_assets_df_processed):  # Parameter changed
    """Loads team data from a URL, merges with asset data, and handles Purchase_Price."""
    warnings = ""
    my_team_df = None
    purchase_price_was_missing_in_file = False  # Renamed for clarity

    if (
        not team_url or team_url == "YOUR_GOOGLE_SHEET_URL_FOR_MY_TEAM_CSV"
    ):  # Check for placeholder
        warnings += "\nInfo: Team data URL is a placeholder or empty. Proceeding without team data."
        return None, warnings

    if all_assets_df_processed is None:
        warnings += "\nError: Asset data is not available, cannot process team data."
        return None, warnings

    try:
        print(f"Attempting to load team data from URL: {team_url}")
        my_team_df_raw = pd.read_csv(team_url)
        my_team_df_raw.columns = my_team_df_raw.columns.str.strip()

        cols_to_select_from_raw = ["ID"]
        if "Purchase_Price" not in my_team_df_raw.columns:
            purchase_price_was_missing_in_file = True
            warnings += f"\nWarning: 'Purchase_Price' column not found in team data from {team_url}."
        else:
            # ... (your existing Purchase_Price parsing logic for my_team_df_raw) ...
            cols_to_select_from_raw.append("Purchase_Price")
            if my_team_df_raw["Purchase_Price"].dtype == "object":
                my_team_df_raw["Purchase_Price"] = (
                    my_team_df_raw["Purchase_Price"]
                    .replace({r"\$": "", "M": ""}, regex=True)
                    .astype(float)
                )
            else:
                my_team_df_raw["Purchase_Price"] = pd.to_numeric(
                    my_team_df_raw["Purchase_Price"], errors="coerce"
                ).fillna(0)

        # ... (rest of the merge and processing logic for my_team_df, this part is largely the same) ...
        # Ensure cols_to_merge_from_assets and actual_cols_to_merge are correctly defined
        cols_to_merge_from_assets = [
            "ID",
            "Name",
            "Type",
            "Constructor",
            "Price",
            "Active",
            "Total_Points_So_Far",
            "Avg_Points_Last_3_Races",
            "User_Adjusted_Avg_Points_Last_3_Races",
            "Point_Adjustment_Avg3Races",
            "Points_Last_Race",
            "PPM_Current",
            "Combined_Score",
            "Norm_User_Adjusted_Avg_Points_Last_3",
            "Norm_Points_Last_Race",
            "Norm_PPM",
            "Norm_Total_Points_So_Far",
            "Norm_Trend_Score",
        ]  # This list should be comprehensive
        actual_cols_to_merge = [
            col
            for col in cols_to_merge_from_assets
            if col in all_assets_df_processed.columns
        ]

        my_team_df = pd.merge(
            my_team_df_raw[cols_to_select_from_raw],
            all_assets_df_processed[actual_cols_to_merge],
            on="ID",
            how="left",
        )

        # ... (the rest of your existing logic for defaulting Purchase_Price if it was missing, calculating PPM_on_Purchase, and filling NaNs) ...
        if my_team_df is not None and not my_team_df.empty:
            if purchase_price_was_missing_in_file:
                if "Price" in my_team_df.columns:
                    my_team_df["Purchase_Price"] = my_team_df["Price"]
                else:
                    my_team_df["Purchase_Price"] = 0.0
                warnings += f"\n 'Purchase_Price' defaulted to Current Price from asset data. Effective budget cap: ~${INITIAL_BUDGET:.2f}M."
            # Ensure Purchase_Price is numeric
            if "Purchase_Price" in my_team_df.columns:
                my_team_df["Purchase_Price"] = pd.to_numeric(
                    my_team_df["Purchase_Price"], errors="coerce"
                ).fillna(0)
            else:  # Should have been created if missing
                my_team_df["Purchase_Price"] = 0.0

            my_team_df["PPM_on_Purchase"] = 0.0
            if (
                "Total_Points_So_Far" in my_team_df.columns
                and "Purchase_Price" in my_team_df.columns
            ):
                non_zero_pp_mask = (my_team_df["Purchase_Price"].notna()) & (
                    my_team_df["Purchase_Price"] != 0
                )
                my_team_df.loc[non_zero_pp_mask, "PPM_on_Purchase"] = (
                    my_team_df.loc[non_zero_pp_mask, "Total_Points_So_Far"]
                    / my_team_df.loc[non_zero_pp_mask, "Purchase_Price"]
                )
            my_team_df["PPM_on_Purchase"] = (
                my_team_df["PPM_on_Purchase"].replace([np.inf, -np.inf], 0).fillna(0)
            )

            # ... (your NaNs filling logic for other columns)
            if my_team_df.isnull().values.any():
                warnings += "\nWarning: Some assets in your team file may have missing details after merge."
                # ... (your existing fillna loop)

        elif my_team_df is not None and my_team_df.empty:
            warnings += f"\nWarning: Team data is empty after merge (check IDs in team data from {team_url})."

        if (
            my_team_df is None
        ):  # Should be caught by earlier checks on all_assets_df_processed
            warnings += "\nError: Could not create final team dataframe, possibly due to asset data issues."

    except (
        Exception
    ) as e:  # More general exception for URL/network issues for team data
        warnings += f"\nError loading or processing team data from URL {team_url}: {e}\n{traceback.format_exc()}"
        my_team_df = None  # Ensure my_team_df is None on error

    return my_team_df, warnings


def load_and_process_data(
    asset_data_url, team_url, adjustments_url, selected_weights
):  # Parameters changed
    overall_warnings = []

    # 1. Load and validate raw asset data
    asset_data_df, warn = _load_raw_asset_df(asset_data_url)  # Use URL
    if warn:
        overall_warnings.append(warn)
    if asset_data_df is None:
        # ... (handle critical failure as before) ...
        final_warning_msg = "\n".join(filter(None, overall_warnings))
        if final_warning_msg.strip():
            print(
                f"\n--- Data Loading Log ---\n{final_warning_msg.strip()}\n------------------------"
            )
        return None, None, final_warning_msg

    # 2. Preprocess metadata
    asset_data_df, warn = _preprocess_asset_attributes(asset_data_df.copy())
    if warn:
        overall_warnings.append(warn)

    # 3. Calculate base points metrics
    asset_data_df, warn = _calculate_points_metrics(
        asset_data_df.copy(), METADATA_COLUMNS
    )
    if warn:
        overall_warnings.append(warn)

    # 4. Apply manual adjustments
    asset_data_df, warn = _apply_manual_adjustments(
        asset_data_df.copy(), adjustments_url
    )  # Use URL
    if warn:
        overall_warnings.append(warn)

    # 5. Calculate derived scores
    all_assets_df, warn = _calculate_derived_scores(
        asset_data_df.copy(), selected_weights
    )
    if warn:
        overall_warnings.append(warn)
    if all_assets_df is None:
        # ... (handle critical failure as before) ...
        final_warning_msg = "\n".join(filter(None, overall_warnings))
        if final_warning_msg.strip():
            print(
                f"\n--- Data Loading Log ---\n{final_warning_msg.strip()}\n------------------------"
            )
        return None, None, final_warning_msg

    # 6. Load and process team data
    my_team_df, warn = _load_and_process_team_df(team_url, all_assets_df)  # Use URL
    if warn:
        overall_warnings.append(warn)

    final_warning_msg = "\n".join(filter(None, overall_warnings))
    if final_warning_msg.strip():
        print(
            f"\n--- Data Loading Log ---\n{final_warning_msg.strip()}\n------------------------"
        )

    return all_assets_df, my_team_df, final_warning_msg


def display_team_and_budget_info(team_df, initial_budget, budget_warning_message):
    """Displays current team information and budget."""
    if team_df is None:
        return 0.0  # Return 0 for dynamic budget if team_df is None

    print("\n--- Your Current Team ---")
    if team_df.empty:
        print("Your team is currently empty.")
        team_current_value = 0.0
        team_purchase_cost = 0.0
    else:
        cols_to_display = [
            "ID",
            "Name",
            "Type",
            "Constructor",
            "Price",
            "Purchase_Price",
            "Total_Points_So_Far",
            "Avg_Points_Last_3_Races",
            "User_Adjusted_Avg_Points_Last_3_Races",  # New
            "Point_Adjustment_Avg3Races",  # New
            "Points_Last_Race",
            "PPM_Current",
            "PPM_on_Purchase",
            "Active",
            "Combined_Score",
        ]
        # Ensure all columns in cols_to_display actually exist in team_df
        displayable_cols = [col for col in cols_to_display if col in team_df.columns]
        print(team_df[displayable_cols].to_string(index=False, na_rep="NaN"))
        team_current_value = team_df["Price"].sum()
        team_purchase_cost = team_df["Purchase_Price"].sum()

    print(f"\nTotal Team Current Market Value: ${team_current_value:,.2f}M")
    print(f"Total Team Assumed Purchase Cost: ${team_purchase_cost:,.2f}M")

    value_gain_loss = team_current_value - team_purchase_cost
    dynamic_budget = initial_budget + value_gain_loss

    print(
        f"Team Value Gain/(Loss) since initial purchase (defaulted): ${value_gain_loss:,.2f}M"
    )
    print(f"Current Dynamic Budget: ${dynamic_budget:,.2f}M")

    if budget_warning_message:
        print(f"\n{budget_warning_message}")

    if not team_df.empty:
        num_drivers = len(team_df[team_df["Type"] == "Driver"])
        num_constructors = len(team_df[team_df["Type"] == "Constructor"])
        print(
            f"\nTeam Composition: {num_drivers} Drivers, {num_constructors} Constructors."
        )
        if num_drivers != 5 or num_constructors != 2:
            print(
                "WARNING: Team composition might be invalid (expected 5 Drivers, 2 Constructors)."
            )

    return dynamic_budget, team_current_value


def identify_mandatory_transfers(team_df):
    """Identifies inactive assets on the current team."""
    if team_df is None or team_df.empty:
        return pd.DataFrame()  # Return empty DataFrame if no team

    mandatory_transfers_df = team_df[
        ~team_df["Active"]
    ].copy()  # Select rows where Active is False

    if not mandatory_transfers_df.empty:
        print("\n--- Mandatory Transfers Required ---")
        print("The following assets on your team are INACTIVE and must be replaced:")
        print(
            mandatory_transfers_df[["ID", "Name", "Type", "Price"]].to_string(
                index=False
            )
        )
    else:
        print("\n--- Mandatory Transfers Required ---")
        print("No mandatory transfers required. All team members are active.")

    return mandatory_transfers_df


def _get_available_for_purchase(all_assets_df, owned_ids_list):
    """Filters all_assets_df for active assets not currently owned."""
    return all_assets_df[
        (all_assets_df["Active"]) & (~all_assets_df["ID"].isin(owned_ids_list))
    ].copy()


def _suggest_mandatory_replacements(
    mandatory_assets_to_sell_df,
    available_for_purchase_df,
    dynamic_budget,
    current_team_value,
    all_assets_df_ref,
):
    """
    Suggests replacements for mandatory transfers.
    Note: This version suggests replacements for each mandatory sale independently
          considering the initial dynamic_budget and current_team_value.
          If multiple mandatory sales occur, the user must manage the cumulative budget impact.
    """
    mandatory_replacement_suggestions = []
    budget_headroom = dynamic_budget - current_team_value

    for _, asset_to_sell in mandatory_assets_to_sell_df.iterrows():
        sell_name = asset_to_sell["Name"]
        sell_price = asset_to_sell["Price"]
        sell_type = asset_to_sell["Type"]

        max_buy_price = sell_price + budget_headroom

        potential_buys = available_for_purchase_df[
            (available_for_purchase_df["Type"] == sell_type)
            & (available_for_purchase_df["Price"] <= max_buy_price)
        ].sort_values(
            by="Combined_Score", ascending=False
        )  # Rank by Combined_Score

        suggestion_detail = {"sell": asset_to_sell, "options": pd.DataFrame()}
        if not potential_buys.empty:
            suggestion_detail["options"] = potential_buys.head(5)
        else:
            suggestion_detail["message"] = (
                f"No suitable replacements found for {sell_name} (max price: ${max_buy_price:.2f}M)."
            )
        mandatory_replacement_suggestions.append(suggestion_detail)

    return mandatory_replacement_suggestions


def _suggest_sequential_single_discretionary_swaps(
    initial_my_team_df,
    initial_available_for_purchase_df,
    all_assets_df_complete,
    dynamic_budget,
    initial_current_team_value,
    num_swaps_to_suggest,
):
    """
    Suggests a sequence of up to num_swaps_to_suggest single discretionary swaps.
    """
    final_discretionary_sequence = []

    hypothetical_team_df = initial_my_team_df.copy()
    hypothetical_current_team_value = initial_current_team_value

    # Initial set of owned IDs for filtering available_for_purchase
    hypothetical_owned_ids = list(hypothetical_team_df["ID"])

    for transfer_num in range(num_swaps_to_suggest):
        current_budget_headroom = dynamic_budget - hypothetical_current_team_value

        if "Active" not in hypothetical_team_df.columns:
            hypothetical_team_df = hypothetical_team_df.merge(
                all_assets_df_complete[["ID", "Active"]], on="ID", how="left"
            )
        active_team_members_to_sell = hypothetical_team_df[
            hypothetical_team_df["Active"]
        ].copy()

        best_swap_this_iteration = None
        highest_improvement_score_this_iteration = 0  # Only positive improvements

        # Dynamically update available_for_purchase based on hypothetical_owned_ids
        current_available_for_purchase_df = all_assets_df_complete[
            (all_assets_df_complete["Active"])
            & (~all_assets_df_complete["ID"].isin(hypothetical_owned_ids))
        ].copy()

        for _, asset_to_sell_row in active_team_members_to_sell.iterrows():
            sell_id = asset_to_sell_row["ID"]
            sell_price = asset_to_sell_row["Price"]
            sell_type = asset_to_sell_row["Type"]
            sell_combined_score = asset_to_sell_row["Combined_Score"]

            max_buy_price = sell_price + current_budget_headroom

            potential_buys = current_available_for_purchase_df[
                (current_available_for_purchase_df["Type"] == sell_type)
                & (current_available_for_purchase_df["Price"] <= max_buy_price)
                # ID != sell_id is implicitly handled by available_for_purchase not containing owned IDs
            ].copy()

            if not potential_buys.empty:
                potential_buys["Improvement_Score_Combined"] = (
                    potential_buys["Combined_Score"] - sell_combined_score
                )
                improved_options = potential_buys[
                    potential_buys["Improvement_Score_Combined"] > 0
                ].sort_values(by="Improvement_Score_Combined", ascending=False)

                if not improved_options.empty:
                    current_best_buy_for_this_sell = improved_options.iloc[0]
                    if (
                        current_best_buy_for_this_sell["Improvement_Score_Combined"]
                        > highest_improvement_score_this_iteration
                    ):
                        highest_improvement_score_this_iteration = (
                            current_best_buy_for_this_sell["Improvement_Score_Combined"]
                        )
                        best_swap_this_iteration = {
                            "sell_id": sell_id,
                            "sell_name": asset_to_sell_row["Name"],
                            "sell_price": sell_price,
                            "sell_type": sell_type,
                            "sell_score": asset_to_sell_row["Combined_Score"],
                            "sell_avg_points_raw": asset_to_sell_row[
                                "Avg_Points_Last_3_Races"
                            ],
                            "sell_last_race_raw": asset_to_sell_row["Points_Last_Race"],
                            # "sell_total_points_raw": asset_to_sell_row[
                            #     "Total_Points_So_Far"
                            # ],  # Ensure this is from asset_to_sell_row
                            # "sell_trend_raw": asset_to_sell_row[
                            #     "Trend_Score"
                            # ],  # Ensure this is from asset_to_sell_row
                            "sell_total_points_raw": asset_to_sell_row.get(
                                "Total_Points_So_Far", 0.0
                            ),
                            "sell_trend_raw": asset_to_sell_row.get("Trend_Score", 0.0),
                            "buy_id": current_best_buy_for_this_sell["ID"],
                            "buy_name": current_best_buy_for_this_sell["Name"],
                            "buy_price": current_best_buy_for_this_sell["Price"],
                            "buy_score": current_best_buy_for_this_sell[
                                "Combined_Score"
                            ],
                            "buy_avg_points_raw": current_best_buy_for_this_sell[
                                "Avg_Points_Last_3_Races"
                            ],
                            "buy_last_race_raw": current_best_buy_for_this_sell[
                                "Points_Last_Race"
                            ],
                            # "buy_total_points_raw": current_best_buy_for_this_sell[
                            #     "Total_Points_So_Far"
                            # ],  # From current_best_buy_for_this_sell
                            # "buy_trend_raw": current_best_buy_for_this_sell[
                            #     "Trend_Score"
                            # ],  # From current_best_buy_for_this_sell
                            "buy_total_points_raw": current_best_buy_for_this_sell.get(
                                "Total_Points_So_Far", 0.0
                            ),
                            "buy_trend_raw": current_best_buy_for_this_sell.get(
                                "Trend_Score", 0.0
                            ),
                            "improvement_score": highest_improvement_score_this_iteration,
                            # new_team_value, money_left_under_cap are calculated later in this function
                        }

        if best_swap_this_iteration:
            swap_sell_price = best_swap_this_iteration["sell_price"]
            swap_buy_price = best_swap_this_iteration["buy_price"]
            new_team_val_after_this_one_swap = (
                hypothetical_current_team_value - swap_sell_price + swap_buy_price
            )
            best_swap_this_iteration["new_team_value"] = (
                new_team_val_after_this_one_swap
            )
            best_swap_this_iteration["money_left_under_cap"] = (
                dynamic_budget - new_team_val_after_this_one_swap
            )
            final_discretionary_sequence.append(best_swap_this_iteration)

            hypothetical_current_team_value = new_team_val_after_this_one_swap
            hypothetical_team_df = hypothetical_team_df[
                hypothetical_team_df["ID"] != best_swap_this_iteration["sell_id"]
            ].copy()

            asset_to_add_details = all_assets_df_complete[
                all_assets_df_complete["ID"] == best_swap_this_iteration["buy_id"]
            ].copy()
            if (
                "Purchase_Price" not in asset_to_add_details.columns
            ):  # Should exist in all_assets_df_complete if processed by load_data
                asset_to_add_details["Purchase_Price"] = asset_to_add_details[
                    "Price"
                ]  # Default for hypothetical
            if "PPM_on_Purchase" not in asset_to_add_details.columns:
                asset_to_add_details["PPM_on_Purchase"] = (
                    0.0  # Default for hypothetical
                )
                if asset_to_add_details["Purchase_Price"].iloc[0] != 0:
                    asset_to_add_details["PPM_on_Purchase"] = (
                        asset_to_add_details["Total_Points_So_Far"]
                        / asset_to_add_details["Purchase_Price"]
                    )

            # Align columns before concat, using the structure of the initial my_team_df
            cols_for_hypothetical_team = list(initial_my_team_df.columns)
            for col in cols_for_hypothetical_team:
                if col not in asset_to_add_details.columns:
                    # This asset from all_assets_df might be missing some calculated team-specific columns
                    # like PPM_on_Purchase. Add with default if critical for structure.
                    if col == "PPM_on_Purchase":
                        asset_to_add_details[col] = 0.0
                    # Other columns should mostly come from all_assets_df_complete

            hypothetical_team_df = pd.concat(
                [
                    hypothetical_team_df,
                    asset_to_add_details[cols_for_hypothetical_team],
                ],
                ignore_index=True,
            )
            # Update the list of owned IDs for the next iteration
            hypothetical_owned_ids = list(hypothetical_team_df["ID"])
        else:
            if transfer_num == 0:
                print(
                    "No beneficial discretionary single swaps found based on Combined Score and current budget."
                )
            break

    return final_discretionary_sequence


def _suggest_true_double_discretionary_swaps(
    current_my_team_df,
    initial_available_for_purchase_df,
    all_assets_df_complete,
    dynamic_budget,
    initial_current_team_value,
):
    """
    Suggests true double swaps (2 out, 2 in) if beneficial.
    """
    true_double_swap_suggestions = []

    # Consider only active members of the current team for selling
    active_team_members_df = current_my_team_df[current_my_team_df["Active"]].copy()

    if len(active_team_members_df) < 2:
        return []  # Not enough players to perform a double swap

    budget_headroom = dynamic_budget - initial_current_team_value

    # Iterate through all unique pairs of assets to sell from the active team
    for s1_details, s2_details in itertools.combinations(
        active_team_members_df.to_dict("records"), 2
    ):
        sell_pair_ids = {s1_details["ID"], s2_details["ID"]}
        sell_pair_price = s1_details["Price"] + s2_details["Price"]
        sell_pair_combined_score = (
            s1_details["Combined_Score"] + s2_details["Combined_Score"]
        )

        type_needed1 = s1_details["Type"]
        type_needed2 = s2_details["Type"]

        max_combined_buy_price = sell_pair_price + budget_headroom

        # Filter available assets for the first buy candidate (B1)
        # Must not be one of the players being sold (already handled by initial_available_for_purchase_df)
        candidates_b1_df = initial_available_for_purchase_df[
            initial_available_for_purchase_df["Type"] == type_needed1
        ].copy()

        for _, b1_details_row in candidates_b1_df.iterrows():
            b1_details = b1_details_row.to_dict()
            price_b1 = b1_details["Price"]

            # Max price for the second buy candidate (B2)
            max_price_for_b2 = max_combined_buy_price - price_b1
            if max_price_for_b2 < 0:  # Not enough budget left for any B2
                continue

            # Filter available assets for the second buy candidate (B2)
            # Must not be B1 and must match type_needed2
            candidates_b2_df = initial_available_for_purchase_df[
                (initial_available_for_purchase_df["Type"] == type_needed2)
                & (
                    initial_available_for_purchase_df["ID"] != b1_details["ID"]
                )  # Cannot be the same as B1
                & (initial_available_for_purchase_df["Price"] <= max_price_for_b2)
            ].copy()

            for _, b2_details_row in candidates_b2_df.iterrows():
                b2_details = b2_details_row.to_dict()

                buy_pair_price = price_b1 + b2_details["Price"]
                buy_pair_combined_score = (
                    b1_details["Combined_Score"] + b2_details["Combined_Score"]
                )

                improvement_score = buy_pair_combined_score - sell_pair_combined_score

                if improvement_score > 0:  # Only consider beneficial swaps
                    new_team_total_value = (
                        initial_current_team_value - sell_pair_price + buy_pair_price
                    )
                    money_left_under_cap = dynamic_budget - new_team_total_value

                    true_double_swap_suggestions.append(
                        {
                            "sell1": s1_details,
                            "sell2": s2_details,
                            "buy1": b1_details,
                            "buy2": b2_details,
                            "improvement_score": improvement_score,
                            "new_team_value": new_team_total_value,
                            "money_left_under_cap": money_left_under_cap,
                            "sell_pair_price": sell_pair_price,
                            "buy_pair_price": buy_pair_price,
                        }
                    )

    # Sort all found double swaps by their improvement score
    if true_double_swap_suggestions:
        sorted_double_swaps = sorted(
            true_double_swap_suggestions,
            key=lambda x: x["improvement_score"],
            reverse=True,
        )
        return sorted_double_swaps[:5]  # Return top 5 (or configurable number)

    return []


def suggest_swaps(
    all_assets_df,
    my_team_df,
    mandatory_transfers_df,
    dynamic_budget,
    current_team_value,
    num_total_transfers_allowed,
    num_mandatory_transfers,
):
    """
    Orchestrates swap suggestions by calling helper functions.
    """
    suggestions = {
        "mandatory": [],
        "discretionary_sequence": [],
        "true_double_swaps": [],
    }

    if all_assets_df is None or my_team_df is None or my_team_df.empty:
        print("Cannot generate suggestions: missing asset data or team data.")
        return suggestions

    owned_ids = list(my_team_df["ID"])
    initial_available_for_purchase_df = _get_available_for_purchase(
        all_assets_df, owned_ids
    )

    # --- 1. Handle Mandatory Transfers ---
    if num_mandatory_transfers > 0:
        # ... (your existing mandatory transfer logic call) ...
        print("\n--- Suggestions for Mandatory Replacements ---")
        mandatory_replacement_suggestions = _suggest_mandatory_replacements(
            mandatory_transfers_df,
            initial_available_for_purchase_df,
            dynamic_budget,
            current_team_value,
            all_assets_df,
        )
        suggestions["mandatory"] = mandatory_replacement_suggestions
        if suggestions["mandatory"]:
            display_suggestions(
                suggestions["mandatory"], "Mandatory Replacements", dynamic_budget
            )

    # --- 2. Determine available discretionary transfers ---
    num_discretionary_transfers_available = (
        num_total_transfers_allowed - num_mandatory_transfers
    )

    # --- 3. Handle Sequential Discretionary Single Swaps ---
    # We can still offer this, especially if only 1 discretionary transfer is available
    if num_discretionary_transfers_available > 0:
        print(
            f"\n--- Sequential Suggestions for up to {num_discretionary_transfers_available} Discretionary Single Swap(s) (Using Combined Score) ---"
        )
        discretionary_sequence = _suggest_sequential_single_discretionary_swaps(
            my_team_df,
            initial_available_for_purchase_df,
            all_assets_df,
            dynamic_budget,
            current_team_value,
            num_discretionary_transfers_available,
        )
        suggestions["discretionary_sequence"] = discretionary_sequence
        if suggestions["discretionary_sequence"]:
            display_suggestions(
                suggestions["discretionary_sequence"],
                "Discretionary Single Swaps",
                dynamic_budget,
            )
        # Message for no beneficial swaps is now inside _suggest_sequential_single_discretionary_swaps

    # --- 4. Handle True Double Discretionary Swaps ---
    # Only suggest double swaps if 2 discretionary transfers are actually available
    if num_discretionary_transfers_available == 2:
        print(f"\n--- Suggestions for True Double Swap (2 out, 2 in) ---")
        double_swap_suggestions = _suggest_true_double_discretionary_swaps(
            my_team_df,  # Pass the original current team
            initial_available_for_purchase_df,  # Pass the initial available pool
            all_assets_df,  # Pass the complete asset data for lookups
            dynamic_budget,
            current_team_value,
        )
        suggestions["true_double_swaps"] = double_swap_suggestions
        if suggestions["true_double_swaps"]:
            display_suggestions(
                suggestions["true_double_swaps"], "True Double Swaps", dynamic_budget
            )
        else:
            print(
                "No beneficial true double swaps found based on Combined Score and current budget."
            )

    return suggestions


def suggest_target_based_double_swap(
    fixed_sell_id,
    fixed_buy_id,
    all_assets_df,
    current_my_team_df,
    dynamic_budget,
    initial_current_team_value,
):
    """
    User wants to swap fixed_sell_id for fixed_buy_id.
    This function finds the best accompanying second swap (if one is needed/beneficial)
    to make the overall 2-transfer operation budget-compliant and score-optimal.
    Assumes 2 transfers are available for this operation.
    """
    print(f"\n--- Target-Based Double Swap Assistant ---")
    print(f"Attempting to swap OUT: {fixed_sell_id} for IN: {fixed_buy_id}")

    # --- 1. Validate fixed part of the swap ---
    if fixed_sell_id not in current_my_team_df["ID"].values:
        print(f"Error: Asset to sell '{fixed_sell_id}' is not in your current team.")
        return

    asset_to_sell_1 = (
        current_my_team_df[current_my_team_df["ID"] == fixed_sell_id].iloc[0].to_dict()
    )

    if not all_assets_df[all_assets_df["ID"] == fixed_buy_id]["Active"].any():
        print(
            f"Error: Target asset to buy '{fixed_buy_id}' is not active or does not exist."
        )
        return

    asset_to_buy_1 = (
        all_assets_df[all_assets_df["ID"] == fixed_buy_id].iloc[0].to_dict()
    )

    if asset_to_sell_1["Type"] != asset_to_buy_1["Type"]:
        print(
            f"Error: Asset types do not match for the fixed swap ({asset_to_sell_1['Type']} vs {asset_to_buy_1['Type']})."
        )
        return

    if fixed_buy_id in current_my_team_df["ID"].values:
        print(
            f"Error: Target asset to buy '{fixed_buy_id}' is already on your team (cannot swap for itself)."
        )
        return

    print(
        f"Fixed Part 1: Sell {asset_to_sell_1['Name']} (Price ${asset_to_sell_1['Price']:.1f}M, Score {asset_to_sell_1['Combined_Score']:.2f})"
    )
    print(
        f"           Buy  {asset_to_buy_1['Name']} (Price ${asset_to_buy_1['Price']:.1f}M, Score {asset_to_buy_1['Combined_Score']:.2f})"
    )

    cost_of_first_swap = asset_to_buy_1["Price"] - asset_to_sell_1["Price"]
    score_change_from_first_swap = (
        asset_to_buy_1["Combined_Score"] - asset_to_sell_1["Combined_Score"]
    )

    print(
        f"Impact of this first swap: Cost Change = ${cost_of_first_swap:+.1f}M, Score Change = {score_change_from_first_swap:+.2f}"
    )

    # --- 2. Determine budget for the second part of the swap ---
    initial_budget_headroom = dynamic_budget - initial_current_team_value
    # This is how much the *second swap's net cost* can be (Price_Z - Price_Y)
    budget_allowance_for_second_swap_net_spend = (
        initial_budget_headroom - cost_of_first_swap
    )

    print(f"Initial budget headroom: ${initial_budget_headroom:.2f}M")
    print(
        f"Net spend allowance for second swap (Price_Z - Price_Y): ${budget_allowance_for_second_swap_net_spend:.2f}M"
    )

    # --- 3. Find optimal second swap (Sell Y, Buy Z) ---
    best_second_swap_details = None
    highest_net_score_improvement_for_total_operation = -float(
        "inf"
    )  # Initialize with a very low number

    # Potential assets to sell (Y) from the team, EXCLUDING the one already sold (asset_to_sell_1)
    # and ensuring they are active.
    potential_sell_y_df = current_my_team_df[
        (current_my_team_df["ID"] != fixed_sell_id) & (current_my_team_df["Active"])
    ].copy()

    # Assets available to purchase (Z) - active and not asset_to_buy_1, and not any other current team member
    # (except the one being sold as Y in this iteration)
    owned_ids_after_first_hypothetical_swap = list(potential_sell_y_df["ID"]) + [
        fixed_buy_id
    ]

    available_for_purchase_z_df = all_assets_df[
        (all_assets_df["Active"])
        & (~all_assets_df["ID"].isin(owned_ids_after_first_hypothetical_swap))
    ].copy()

    for _, asset_y_row in potential_sell_y_df.iterrows():
        asset_y = asset_y_row.to_dict()
        max_price_for_z = asset_y["Price"] + budget_allowance_for_second_swap_net_spend

        potential_buy_z_options = available_for_purchase_z_df[
            (available_for_purchase_z_df["Type"] == asset_y["Type"])
            & (available_for_purchase_z_df["Price"] <= max_price_for_z)
            & (
                available_for_purchase_z_df["ID"] != asset_y["ID"]
            )  # Should be covered by available_for_purchase
        ].copy()

        if not potential_buy_z_options.empty:
            # Score for this second swap part only
            potential_buy_z_options["Second_Swap_Score_Improvement"] = (
                potential_buy_z_options["Combined_Score"] - asset_y["Combined_Score"]
            )

            # We want the Z that maximizes (Score_Z - Score_Y)
            best_z_for_this_y = (
                potential_buy_z_options.sort_values(
                    by="Second_Swap_Score_Improvement", ascending=False
                )
                .iloc[0]
                .to_dict()
            )

            current_total_operation_score_improvement = (
                score_change_from_first_swap
                + best_z_for_this_y["Second_Swap_Score_Improvement"]
            )

            if (
                current_total_operation_score_improvement
                > highest_net_score_improvement_for_total_operation
            ):
                highest_net_score_improvement_for_total_operation = (
                    current_total_operation_score_improvement
                )

                final_team_value = (
                    initial_current_team_value
                    - asset_to_sell_1["Price"]
                    + asset_to_buy_1["Price"]
                    - asset_y["Price"]
                    + best_z_for_this_y["Price"]
                )

                best_second_swap_details = {
                    "sell_y": asset_y,
                    "buy_z": best_z_for_this_y,
                    "second_swap_score_improvement": best_z_for_this_y[
                        "Second_Swap_Score_Improvement"
                    ],
                    "total_operation_score_improvement": current_total_operation_score_improvement,
                    "final_team_value": final_team_value,
                    "money_left_under_cap": dynamic_budget - final_team_value,
                }

    # --- 4. Present the result ---
    if best_second_swap_details:
        print("\n--- Optimal Accompanying Second Swap Found ---")
        sy = best_second_swap_details["sell_y"]
        bz = best_second_swap_details["buy_z"]
        print(
            f"To make the primary swap (Out: {asset_to_sell_1['Name']}, In: {asset_to_buy_1['Name']}) work best:"
        )
        print(
            f"  Also Sell (Y): {sy['Name']} (ID: {sy['ID']}, Price ${sy['Price']:.1f}M, Score {sy['Combined_Score']:.2f})"
        )
        print(
            f"  And Buy   (Z): {bz['Name']} (ID: {bz['ID']}, Price ${bz['Price']:.1f}M, Score {bz['Combined_Score']:.2f})"
        )
        print(
            f"Score improvement from second swap (Z - Y): {best_second_swap_details['second_swap_score_improvement']:+.2f}"
        )
        print(
            f"Total Combined Score improvement for both transfers: {best_second_swap_details['total_operation_score_improvement']:+.2f}"
        )
        print(
            f"Final Team Value after both swaps: ${best_second_swap_details['final_team_value']:.2f}M"
        )
        print(
            f"Money Left Under Cap ({dynamic_budget:.2f}M): ${best_second_swap_details['money_left_under_cap']:.2f}M"
        )

        # Check if the first swap alone was already over budget before this second enabling swap
        value_after_just_first_swap = (
            initial_current_team_value
            - asset_to_sell_1["Price"]
            + asset_to_buy_1["Price"]
        )
        if value_after_just_first_swap > dynamic_budget:
            print(
                f"Note: The first swap alone (Out: {asset_to_sell_1['Name']}, In: {asset_to_buy_1['Name']}) would have resulted in a team value of ${value_after_just_first_swap:.2f}M."
            )
            print("The suggested second swap helps to make this financially viable.")
        elif best_second_swap_details["second_swap_score_improvement"] < 0:
            print(
                "Note: The suggested second swap results in a score decrease for that part, likely to enable the primary transfer financially."
            )

    else:  # No Y/Z swap found that works or improves things
        # Check if the first swap is viable on its own (as a single transfer)
        value_after_first_swap = (
            initial_current_team_value
            - asset_to_sell_1["Price"]
            + asset_to_buy_1["Price"]
        )
        if value_after_first_swap <= dynamic_budget:
            print("\nNo beneficial accompanying second swap found.")
            print(
                f"However, the primary swap (Out: {asset_to_sell_1['Name']}, In: {asset_to_buy_1['Name']}) is possible as a single transfer:"
            )
            print(
                f"  Cost Change: ${cost_of_first_swap:+.1f}M, Score Change: {score_change_from_first_swap:+.2f}"
            )
            print(
                f"  Resulting Team Value: ${value_after_first_swap:.2f}M / Money Left Under Cap: ${dynamic_budget - value_after_first_swap:.2f}M"
            )
            print("You would need to use 1 of your free transfers for this.")
        else:
            print(
                "\nNo accompanying second swap found that makes the primary transfer financially viable or beneficial."
            )
            print(
                f"The primary swap (Out: {asset_to_sell_1['Name']}, In: {asset_to_buy_1['Name']}) alone would result in a team value of ${value_after_first_swap:.2f}M, exceeding the budget cap of ${dynamic_budget:.2f}M."
            )


def display_suggestions(suggestion_list, suggestion_type_name, dynamic_budget=None):
    """
    Displays suggestions in a readable format.
    """
    if not suggestion_list:
        return

    if suggestion_type_name == "Mandatory Replacements":
        # ... (your existing mandatory replacements display logic) ...
        for i, suggestion_group in enumerate(suggestion_list):
            asset_to_sell = suggestion_group["sell"]
            options = suggestion_group["options"]
            message = suggestion_group.get("message")

            print(
                f"\n{i+1}. For mandatory sale of: {asset_to_sell.get('Name', 'N/A')} (ID: {asset_to_sell.get('ID', 'N/A')}, Price: ${asset_to_sell.get('Price', 0):.1f}M)"
            )
            if message:
                print(f"   {message}")
            elif not options.empty:
                print(
                    "   Potential replacements (ranked by Combined Score or Avg Points Last 3 Races):"
                )
                opt_display_cols = [
                    "ID",
                    "Name",
                    "Price",
                    "Combined_Score",
                    "Avg_Points_Last_3_Races",
                    "Total_Points_So_Far",
                ]
                actual_opt_display_cols = [
                    col for col in opt_display_cols if col in options.columns
                ]
                print(options[actual_opt_display_cols].to_string(index=False))
            else:
                print("   No replacement options found.")

    elif suggestion_type_name == "Discretionary Single Swaps":
        # ... (your existing discretionary single swaps display logic - this should already be up-to-date) ...
        title_cap_info = (
            f" (Team Budget Cap: ${dynamic_budget:.2f}M)"
            if dynamic_budget is not None
            else ""
        )
        print(
            f"\nTop {len(suggestion_list)} Discretionary Single Swap Option(s) (Using Combined Score){title_cap_info}:"
        )

        for i, swap in enumerate(suggestion_list):

            def format_value(val_key, precision=2, default_val="N/A"):
                val = swap.get(val_key)
                if isinstance(val, (int, float)) and not np.isnan(
                    val
                ):  # Check for NaN too
                    return f"{val:.{precision}f}"
                return default_val

            sell_score_display = format_value("sell_score", 2)
            buy_score_display = format_value("buy_score", 2)
            improvement_score_display = format_value("improvement_score", 2)

            sell_avg_raw_display = format_value("sell_avg_points_raw", 2)
            buy_avg_raw_display = format_value("buy_avg_points_raw", 2)

            sell_last_race_raw_display = format_value("sell_last_race_raw", 1)
            buy_last_race_raw_display = format_value("buy_last_race_raw", 1)

            sell_total_pts_raw_display = format_value("sell_total_points_raw", 1)
            buy_total_pts_raw_display = format_value("buy_total_points_raw", 1)

            sell_trend_raw_display = format_value("sell_trend_raw", 1)
            buy_trend_raw_display = format_value("buy_trend_raw", 1)

            print(
                f"\n{i+1}. Swap Out: {swap.get('sell_name','N/A')} (ID: {swap.get('sell_id','N/A')}, Price: ${swap.get('sell_price',0):.1f}M, Score: {sell_score_display}, AvgL3: {sell_avg_raw_display}, LastR: {sell_last_race_raw_display}, TotPts: {sell_total_pts_raw_display}, Trend: {sell_trend_raw_display})"
            )
            print(
                f"   Swap In:  {swap.get('buy_name','N/A')} (ID: {swap.get('buy_id','N/A')}, Price: ${swap.get('buy_price',0):.1f}M, Score: {buy_score_display}, AvgL3: {buy_avg_raw_display}, LastR: {buy_last_race_raw_display}, TotPts: {buy_total_pts_raw_display}, Trend: {buy_trend_raw_display})"
            )
            print(f"   Combined Score Improvement: +{improvement_score_display}")
            print(
                f"   Resulting Team Value: ${swap.get('new_team_value',0):.2f}M / Money Left Under Cap: ${swap.get('money_left_under_cap',0):.2f}M"
            )

    elif suggestion_type_name == "True Double Swaps":
        title_cap_info = (
            f" (Team Budget Cap: ${dynamic_budget:.2f}M)"
            if dynamic_budget is not None
            else ""
        )
        print(
            f"\nTop {len(suggestion_list)} True Double Swap Option(s){title_cap_info}:"
        )

        for i, swap_pair in enumerate(suggestion_list):
            s1 = swap_pair.get(
                "sell1", {}
            )  # Use .get with default empty dict for safety
            s2 = swap_pair.get("sell2", {})
            b1 = swap_pair.get("buy1", {})
            b2 = swap_pair.get("buy2", {})

            # Enhanced helper for formatting asset details for double swaps
            def format_asset_details_for_double_swap(asset_dict):
                # Inner helper for individual numeric values
                def format_num(val_key, precision=1, default_str="N/A"):
                    val = asset_dict.get(val_key)
                    if isinstance(val, (int, float)) and not np.isnan(val):
                        return f"{val:.{precision}f}"
                    return default_str

                name_id = f"{asset_dict.get('Name', 'N/A')} (ID: {asset_dict.get('ID', 'N/A')})"
                price = f"Price: ${asset_dict.get('Price', 0):.1f}M"
                score = f"Score: {asset_dict.get('Combined_Score', 0):.2f}"

                avg_l3_raw = format_num(
                    "Avg_Points_Last_3_Races", 2
                )  # Using 2 decimal for average
                last_r_raw = format_num("Points_Last_Race", 1)
                total_pts_raw = format_num("Total_Points_So_Far", 1)
                trend_raw = format_num("Trend_Score", 1)

                return (
                    f"{name_id}, {price}, {score}, "
                    f"AvgL3: {avg_l3_raw}, LastR: {last_r_raw}, "
                    f"TotPts: {total_pts_raw}, Trend: {trend_raw}"
                )

            print(f"\n{i+1}. Sell Pair:")
            print(f"   Out: {format_asset_details_for_double_swap(s1)}")
            print(f"   Out: {format_asset_details_for_double_swap(s2)}")
            print(
                f"   Combined Sell Price: ${swap_pair.get('sell_pair_price', 0):.1f}M"
            )
            print(f"   Buy Pair:")
            print(f"   In:  {format_asset_details_for_double_swap(b1)}")
            print(f"   In:  {format_asset_details_for_double_swap(b2)}")
            print(f"   Combined Buy Price: ${swap_pair.get('buy_pair_price', 0):.1f}M")

            improvement_score_val = swap_pair.get("improvement_score", 0)
            improvement_score_str = (
                f"{improvement_score_val:.2f}"
                if isinstance(improvement_score_val, (int, float))
                else "N/A"
            )
            print(f"   Combined Score Improvement: +{improvement_score_str}")

            new_team_val_str = f"{swap_pair.get('new_team_value', 0):.2f}"
            money_left_str = f"{swap_pair.get('money_left_under_cap', 0):.2f}"
            print(
                f"   Resulting Team Value: ${new_team_val_str}M / Money Left Under Cap: ${money_left_str}M"
            )


def normalize_series(series):
    """Normalizes a pandas Series to a 0-1 scale."""
    min_val = series.min()
    max_val = series.max()
    if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
        # Return a series of 0.5 if min/max are NaN (empty series) or all values are the same
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - min_val) / (max_val - min_val)


def optimize_wildcard_team(
    all_assets_df, budget, num_drivers_req=5, num_constructors_req=2
):
    """
    Optimizes a new team from scratch (Wildcard) using PuLP to maximize
    total Combined_Score within budget and composition constraints.
    """
    print(f"\n--- Wildcard Team Optimizer ---")
    print(
        f"Optimizing for a budget of ${budget:.2f}M: {num_drivers_req} Drivers, {num_constructors_req} Constructors."
    )

    # Filter for active assets only and ensure essential columns are numeric and not NaN
    active_assets = all_assets_df[all_assets_df["Active"]].copy()
    active_assets["Price"] = pd.to_numeric(
        active_assets["Price"], errors="coerce"
    ).fillna(
        0
    )  # Should be clean already
    active_assets["Combined_Score"] = pd.to_numeric(
        active_assets["Combined_Score"], errors="coerce"
    ).fillna(
        0
    )  # Should be clean

    # Remove assets with price > budget (cannot be part of any solution)
    # or assets that might have non-positive prices (though unlikely)
    active_assets = active_assets[active_assets["Price"] > 0]
    # active_assets = active_assets[active_assets['Price'] <= budget] # Further prune (solver handles budget constraint)

    if active_assets.empty:
        print("No active assets available for optimization.")
        return

    # Create the LP problem
    prob = pulp.LpProblem("WildcardTeamOptimization", pulp.LpMaximize)

    # Decision Variables: One for each asset, representing whether it's chosen (1) or not (0)
    # Using asset ID as key for variables for easier lookup
    asset_ids = active_assets["ID"].tolist()
    asset_vars = pulp.LpVariable.dicts("AssetSelected", asset_ids, cat="Binary")

    # Create lookup dictionaries for price, score, type for convenience
    prices = dict(zip(active_assets["ID"], active_assets["Price"]))
    scores = dict(zip(active_assets["ID"], active_assets["Combined_Score"]))
    types = dict(zip(active_assets["ID"], active_assets["Type"]))

    # Objective Function: Maximize total Combined_Score
    prob += (
        pulp.lpSum([scores[i] * asset_vars[i] for i in asset_ids]),
        "TotalCombinedScore",
    )

    # Constraints:
    # 1. Budget Constraint
    prob += (
        pulp.lpSum([prices[i] * asset_vars[i] for i in asset_ids]) <= budget,
        "BudgetConstraint",
    )

    # 2. Driver Count Constraint
    driver_ids = [i for i in asset_ids if types[i] == "Driver"]
    prob += (
        pulp.lpSum([asset_vars[i] for i in driver_ids]) == num_drivers_req,
        "DriverCountConstraint",
    )

    # 3. Constructor Count Constraint
    constructor_ids = [i for i in asset_ids if types[i] == "Constructor"]
    prob += (
        pulp.lpSum([asset_vars[i] for i in constructor_ids]) == num_constructors_req,
        "ConstructorCountConstraint",
    )

    # Solve the problem
    print("Solving optimization problem...")
    prob.solve(pulp.PULP_CBC_CMD(msg=0))  # msg=0 suppresses solver messages

    # Display the results
    print(f"\n--- Optimal Wildcard Team ---")
    status = pulp.LpStatus[prob.status]
    print(f"Optimization Status: {status}")

    if status == "Optimal":
        selected_asset_ids = [i for i in asset_vars if pulp.value(asset_vars[i]) == 1]

        optimal_team_df = all_assets_df[
            all_assets_df["ID"].isin(selected_asset_ids)
        ].copy()

        # Ensure correct columns for display
        display_cols = [
            "ID",
            "Name",
            "Type",
            "Constructor",
            "Price",
            "Combined_Score",
            "Avg_Points_Last_3_Races",
            "User_Adjusted_Avg_Points_Last_3_Races",
            "Points_Last_Race",
            "Total_Points_So_Far",
            "PPM_Current",
        ]

        # Filter for columns that actually exist in optimal_team_df
        actual_display_cols = [
            col for col in display_cols if col in optimal_team_df.columns
        ]

        print(optimal_team_df[actual_display_cols].to_string(index=False))

        total_cost = optimal_team_df["Price"].sum()
        total_score = optimal_team_df["Combined_Score"].sum()

        print(f"\nTotal Optimal Team Cost: ${total_cost:.2f}M (Budget: ${budget:.2f}M)")
        print(f"Total Optimal Team Combined Score: {total_score:.2f}")
    elif status == "Infeasible":
        print(
            "The problem is infeasible. It's not possible to select a team meeting all constraints (e.g., budget too low, not enough players of a certain type)."
        )
    else:
        print(
            "Could not find an optimal solution. The problem might be unbounded or have other issues."
        )


def optimize_limitless_team(all_assets_df, num_drivers_req=5, num_constructors_req=2):
    """
    Selects the optimal team (5 Drivers, 2 Constructors) based purely on
    maximizing Combined_Score, ignoring budget (for a "Limitless" chip scenario).
    """
    print(f"\n--- Limitless Chip Team Optimizer ---")
    print(
        f"Selecting {num_drivers_req} Drivers and {num_constructors_req} Constructors with the highest Combined_Score."
    )

    if all_assets_df is None or all_assets_df.empty:
        print("Asset data is not available. Cannot optimize limitless team.")
        return

    active_assets = all_assets_df[all_assets_df["Active"]].copy()
    if "Combined_Score" not in active_assets.columns:
        print(
            "Error: 'Combined_Score' not found in asset data. Please ensure data is processed correctly."
        )
        return

    # Select best drivers
    active_drivers = active_assets[active_assets["Type"] == "Driver"].sort_values(
        by="Combined_Score", ascending=False
    )
    limitless_drivers = active_drivers.head(num_drivers_req)

    # Select best constructors
    active_constructors = active_assets[
        active_assets["Type"] == "Constructor"
    ].sort_values(by="Combined_Score", ascending=False)
    limitless_constructors = active_constructors.head(num_constructors_req)

    if (
        len(limitless_drivers) < num_drivers_req
        or len(limitless_constructors) < num_constructors_req
    ):
        print(
            "\nWarning: Not enough active drivers or constructors available to form a full limitless team."
        )
        print(
            f"Found {len(limitless_drivers)} drivers and {len(limitless_constructors)} constructors."
        )

    limitless_team_df = pd.concat(
        [limitless_drivers, limitless_constructors], ignore_index=True
    )

    print(f"\n--- Optimal Limitless Team (Ignoring Budget) ---")
    if limitless_team_df.empty:
        print(
            "Could not form a limitless team (no active assets found or criteria not met)."
        )
        return

    display_cols = [
        "ID",
        "Name",
        "Type",
        "Constructor",
        "Price",
        "Combined_Score",
        "User_Adjusted_Avg_Points_Last_3_Races",
        "Points_Last_Race",
        "Total_Points_So_Far",
        "PPM_Current",
    ]

    actual_display_cols = [
        col for col in display_cols if col in limitless_team_df.columns
    ]

    print(limitless_team_df[actual_display_cols].to_string(index=False))

    total_hypothetical_cost = limitless_team_df["Price"].sum()
    total_score = limitless_team_df["Combined_Score"].sum()

    print(
        f"\nTotal Hypothetical Team Cost (if budget applied): ${total_hypothetical_cost:.2f}M"
    )
    print(f"Total Team Combined Score: {total_score:.2f}")


def main():
    # Ensure selected_weights is defined properly from profile selection
    # For example, if using profile_options and profile_choice_idx:
    profile_options = list(WEIGHT_PROFILES.keys())  # Defined globally
    default_profile_name = "balanced"
    selected_profile_name = default_profile_name  # Default

    print("\n--- Select Weighting Profile for Combined_Score ---")
    for i, name in enumerate(profile_options):
        print(f"{i+1}. {name.replace('_', ' ').title()}")

    try:
        profile_choice_input = input(
            f"Enter choice (1-{len(profile_options)}, default: {default_profile_name.title()}): "
        )
        if profile_choice_input:
            profile_choice_idx = int(profile_choice_input) - 1
            if 0 <= profile_choice_idx < len(profile_options):
                selected_profile_name = profile_options[profile_choice_idx]
            else:
                print(
                    f"Invalid choice. Using default '{default_profile_name.title()}' profile."
                )
        else:
            print(f"No input. Using default '{default_profile_name.title()}' profile.")
    except ValueError:
        print(f"Invalid input. Using default '{default_profile_name.title()}' profile.")

    selected_weights = WEIGHT_PROFILES.get(
        selected_profile_name, WEIGHT_PROFILES[default_profile_name]
    )  # Use .get for safety

    print(f"Using '{selected_profile_name.replace('_', ' ').title()}' profile.")

    all_assets_df, my_team_df, warning_msg = load_and_process_data(
        ASSET_DATA_URL, MY_TEAM_URL, MANUAL_ADJUSTMENTS_URL, selected_weights
    )

    if all_assets_df is None:
        print("Exiting due to critical error in data loading.")
        return

    dynamic_budget = INITIAL_BUDGET
    current_team_value = 0.0
    if my_team_df is not None and not my_team_df.empty:
        dynamic_budget, current_team_value = display_team_and_budget_info(
            my_team_df, INITIAL_BUDGET, warning_msg
        )
    elif MY_TEAM_FILE:
        # This part of the log is now handled inside load_and_process_data if team file fails
        # print(f"\nWarning: Could not load team data from {MY_TEAM_FILE}. Some modes will be unavailable.")
        pass

    print("\nChoose mode:")
    print("1. Weekly Transfer Suggestions (Automated)")
    print("2. Wildcard Team Optimizer (Budgeted)")
    print("3. Target-Based Double Transfer Assistant")
    print("4. Limitless Chip Optimizer (No Budget)")  # New Option

    mode_choice = input("Enter choice (1-4, default: 1): ") or "1"  # Default to 1

    if mode_choice == "1":
        if my_team_df is None or my_team_df.empty:
            print(
                "Cannot proceed with weekly transfers: Team data is missing or empty."
            )
            return
        # ... (existing weekly transfer logic calls display_team_and_budget_info if not already called)
        # We already called display_team_and_budget_info if my_team_df was loaded.
        # If it wasn't (e.g. my_team.csv missing), then this mode can't run.

        mandatory_transfers_df = identify_mandatory_transfers(my_team_df)
        num_mandatory_transfers = len(mandatory_transfers_df)
        # ... (rest of your existing mode '1' logic for calling suggest_swaps) ...
        print(
            f"\nYou have {num_mandatory_transfers} mandatory transfer(s)."
        )  # Duplicated from suggest_swaps context, can be streamlined
        print(
            f"The system will consider up to {DEFAULT_FREE_TRANSFERS} total transfers based on a team budget cap of ${dynamic_budget:.2f}M."
        )
        suggest_swaps(
            all_assets_df,
            my_team_df,
            mandatory_transfers_df,
            dynamic_budget,
            current_team_value,
            DEFAULT_FREE_TRANSFERS,
            num_mandatory_transfers,
        )

    elif mode_choice == "2":
        optimize_wildcard_team(all_assets_df, INITIAL_BUDGET)  # Budgeted wildcard

    elif mode_choice == "3":
        if my_team_df is None or my_team_df.empty:
            print(
                "Cannot proceed with target-based swap: Team data is missing or empty."
            )
            return
        # ... (your existing target-based swap logic, ensure dynamic_budget and current_team_value are used) ...
        print("\n--- Target-Based Double Transfer Input ---")
        fixed_sell_id = (
            input(
                f"Enter ID of asset to SELL from your team (e.g., {my_team_df['ID'].iloc[0] if not my_team_df.empty else 'ALP'}): "
            )
            .strip()
            .upper()
        )
        fixed_buy_id = (
            input("Enter ID of asset to BUY from all available assets (e.g., WIL): ")
            .strip()
            .upper()
        )

        if fixed_sell_id and fixed_buy_id:
            suggest_target_based_double_swap(
                fixed_sell_id,
                fixed_buy_id,
                all_assets_df,
                my_team_df,
                dynamic_budget,
                current_team_value,
            )
        else:
            print(
                "Both asset IDs (sell and buy) must be provided for target-based swap."
            )

    elif mode_choice == "4":  # New mode
        optimize_limitless_team(all_assets_df)
    else:
        print("Invalid mode choice. Exiting.")


if __name__ == "__main__":
    main()
