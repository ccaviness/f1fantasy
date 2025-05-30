#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is a Python module for managing and analyzing F1 Fantasy teams.

It provides functionalities for:
- Loading and processing asset data (drivers and constructors) from a CSV file.
- Calculating performance metrics such as total points, recent form, and points per million (PPM).
- Applying manual adjustments to asset values.
- Suggesting optimal team compositions based on various weighting profiles and budget constraints.
- Optimizing team transfers, including mandatory replacements and discretionary swaps.
- Simulating "Wildcard" and "Limitless" chip scenarios for team optimization.
- Displaying team information, asset statistics, and transfer suggestions in a user-friendly format.

The script relies on external libraries such as pandas for data manipulation,
numpy for numerical operations, and pulp for linear programming optimization.

Configuration parameters such as initial budget, data URLs, and weighting profiles
are defined at the beginning of the script and can be adjusted to customize the analysis.
"""

import argparse
import itertools
import logging
import sys
import pulp
import pandas as pd
import numpy as np
import config

logger = logging.getLogger("F1FantasyAssistant")

# --- Centralized CLI Argument Mappings and Choices ---

# For --profile argument
CLI_PROFILE_CHOICES_MAP: dict[str, str] = {
    # User-typed CLI choice : Internal key for config.WEIGHT_PROFILES
    "balanced": "balanced",
    "b": "balanced",
    "aggressive": "aggressive_form",
    "a": "aggressive_form",
    "value": "value_focused",
    "v": "value_focused",
}
# For more readable help text
CLI_PROFILE_HELP_OPTIONS: list[str] = [
    "balanced (b)",
    "aggressive (a - emphasizes recent form)",
    "value (v - emphasizes PPM and consistency)",
]
DEFAULT_CLI_PROFILE_CHOICE: str = "b"  # User-typed default

# For --mode argument
CLI_MODE_CHOICES_MAP: dict[str, str] = {
    # User-typed CLI choice : Internal mode string used in script logic
    "normal": "weekly",
    "n": "weekly",
    "wildcard": "wildcard",
    "w": "wildcard",
    "target": "target_swap",  # Using "target" as a user-friendly CLI option
    "ts": "target_swap",
    "limitless": "limitless",
    "l": "limitless",
    "all": "all_stats",  # Using "all" as a user-friendly CLI option
    "as": "all_stats",
}
# For more readable help text
CLI_MODE_HELP_OPTIONS: list[str] = [
    "normal (n): Regular weekly transfer suggestions.",
    "wildcard (w): Optimize a new team with a budget.",
    "target (ts): Assistant for a specific 2-part transfer.",
    "limitless (l): Optimize a new team with no budget (Limitless Chip).",
    "all (as): Display all asset statistics.",
]
DEFAULT_CLI_MODE_CHOICE: str = "n"  # User-typed default


def _load_raw_asset_df(asset_data_url: str) -> pd.DataFrame | None:
    """Loads and performs initial validation on asset_data from a URL."""
    try:
        logger.info("Attempting to load asset data from URL: %s", asset_data_url)
        # Check for placeholder URL from config (assuming config.ASSET_DATA_URL holds the placeholder if not changed by user)
        if not asset_data_url or (
            hasattr(config, "ASSET_DATA_URL")
            and asset_data_url == config.ASSET_DATA_URL
            and "YOUR_GOOGLE_SHEET_URL" in config.ASSET_DATA_URL
        ):
            err_msg = "Asset data URL is a placeholder or empty. Please update it in config.py."
            logger.error(err_msg)
            # No need to raise ValueError here if we return None, orchestrator will handle.
            return None

        df = pd.read_csv(asset_data_url)
        df.columns = df.columns.str.strip()
        for col in config.METADATA_COLUMNS:
            if col not in df.columns:
                err_msg = (
                    f"Essential metadata column '{col}' not found in data from {asset_data_url}. "
                    f"Required: {config.METADATA_COLUMNS}"
                )
                logger.error(err_msg)
                return None  # Indicate critical failure
        logger.debug("Raw asset data loaded and validated successfully from URL.")
        return df
    except pd.errors.EmptyDataError:  # Specific error for empty CSV from URL
        logger.error("No data found at asset URL (empty CSV): %s", asset_data_url)
        return None
    except Exception as e:
        logger.error(
            "Error loading asset data from URL %s: %s", asset_data_url, e, exc_info=True
        )
        return None


def _calculate_points_metrics(
    df: pd.DataFrame, metadata_cols_list: list
) -> pd.DataFrame | None:
    """Calculates Total_Points_So_Far, Avg_Points_Last_3_Races, and Points_Last_Race."""
    if df is None:
        return None  # Propagate failure

    potential_gp_cols = [col for col in df.columns if col not in metadata_cols_list]
    completed_gp_cols = []

    if not potential_gp_cols:
        logger.warning(
            "No potential GP points columns found in asset data. Points metrics set to 0."
        )
        df["Total_Points_So_Far"] = 0.0
        df["Avg_Points_Last_3_Races"] = 0.0
        df["Points_Last_Race"] = 0.0
    else:
        logger.debug("Found potential GP columns: %s", potential_gp_cols)
        for col_name in potential_gp_cols:
            numeric_series = pd.to_numeric(df[col_name], errors="coerce")
            if numeric_series.notna().any():
                completed_gp_cols.append(col_name)
                df[col_name] = numeric_series  # Store numeric version (with NaNs)
            else:
                df[col_name] = np.nan  # Ensure non-completed are NaN for sum/mean

        if not completed_gp_cols:
            logger.warning(
                "No completed GP races identified from data. Points metrics set to 0."
            )
            df["Total_Points_So_Far"] = 0.0
            df["Avg_Points_Last_3_Races"] = 0.0
            df["Points_Last_Race"] = 0.0
        else:
            logger.info("Identified completed GP columns: %s", completed_gp_cols)
            df["Total_Points_So_Far"] = df[completed_gp_cols].sum(axis=1, skipna=True)

            num_races_to_avg = min(len(completed_gp_cols), 3)
            if num_races_to_avg > 0:
                last_n_cols = completed_gp_cols[-num_races_to_avg:]
                logger.info(
                    "Calculating Avg_Points_Last_3_Races based on: %s", last_n_cols
                )
                df["Avg_Points_Last_3_Races"] = (
                    df[last_n_cols].mean(axis=1, skipna=True).fillna(0)
                )
            else:
                df["Avg_Points_Last_3_Races"] = 0.0

            last_race_col_name = completed_gp_cols[-1]
            logger.info("Calculating Points_Last_Race based on: %s", last_race_col_name)
            df["Points_Last_Race"] = df[last_race_col_name].fillna(0)
            logger.debug("Base points metrics calculated.")
    return df


def _preprocess_asset_attributes(df: pd.DataFrame) -> pd.DataFrame | None:
    """Parses Price and Active columns."""
    if df is None:
        return None

    try:
        if "Price" in df.columns:
            if df["Price"].dtype == "object":
                df["Price"] = (
                    df["Price"].replace({r"\$": "", "M": ""}, regex=True).astype(float)
                )
            else:  # If already numeric or other type, ensure it's float
                df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0)
        else:
            logger.error("'Price' column missing from asset data. Defaulting to 0.")
            df["Price"] = 0.0

        if "Active" in df.columns:
            df["Active"] = df["Active"].astype(bool)
        else:
            logger.error(
                "'Active' column missing from asset data. Defaulting to False."
            )
            df["Active"] = False
        logger.debug("Asset attributes (Price, Active) preprocessed.")
        return df
    except Exception as e:
        logger.error("Error during asset attribute preprocessing: %s", e, exc_info=True)
        return None


def _apply_manual_adjustments(
    df: pd.DataFrame, adjustments_url: str
) -> pd.DataFrame | None:
    """Applies manual point adjustments from a CSV URL."""
    if df is None:
        return None

    df["Point_Adjustment_Avg3Races"] = 0.0  # Initialize

    # Check for placeholder URL from config
    is_placeholder_url = not adjustments_url or (
        hasattr(config, "MANUAL_ADJUSTMENTS_URL")
        and adjustments_url == config.MANUAL_ADJUSTMENTS_URL
        and "YOUR_GOOGLE_SHEET_URL" in config.MANUAL_ADJUSTMENTS_URL
    )

    if is_placeholder_url:
        logger.info(
            "Manual adjustments URL is a placeholder or empty. No adjustments will be applied."
        )
    else:
        try:
            logger.info(
                "Attempting to load manual adjustments from URL: %s", adjustments_url
            )
            adj_df = pd.read_csv(adjustments_url)
            adj_df.columns = adj_df.columns.str.strip()

            if (
                "ID" in adj_df.columns
                and "Point_Adjustment_Avg3Races" in adj_df.columns
            ):
                adj_df["Point_Adjustment_Avg3Races_from_file"] = pd.to_numeric(
                    adj_df["Point_Adjustment_Avg3Races"], errors="coerce"
                ).fillna(0)

                adjustment_map = adj_df.set_index("ID")[
                    "Point_Adjustment_Avg3Races_from_file"
                ]
                df["Point_Adjustment_Avg3Races"] = (
                    df["ID"].map(adjustment_map).fillna(0)
                )
                logger.info("Successfully loaded and applied manual adjustments.")
            else:
                logger.warning(
                    "Data from %s is missing 'ID' or 'Point_Adjustment_Avg3Races' column. Adjustments not applied (using 0).",
                    adjustments_url,
                )
        except Exception as e:
            logger.error(
                "Error loading or processing adjustments from URL %s: %s. Adjustments not applied (using 0).",
                adjustments_url,
                e,
                exc_info=True,
            )

    if "Avg_Points_Last_3_Races" in df.columns:
        df["User_Adjusted_Avg_Points_Last_3_Races"] = (
            df["Avg_Points_Last_3_Races"] + df["Point_Adjustment_Avg3Races"]
        )
    else:
        logger.warning(
            "'Avg_Points_Last_3_Races' missing for applying manual adjustments. User_Adjusted will reflect only adjustments."
        )
        df["User_Adjusted_Avg_Points_Last_3_Races"] = df.get(
            "Point_Adjustment_Avg3Races", 0.0
        )

    logger.debug("Manual adjustments applied.")
    return df


def _calculate_ppm(df):
    """Calculates Points Per Million (PPM)."""
    df["PPM_Current"] = 0.0
    price_present = "Price" in df.columns
    total_points_present = "Total_Points_So_Far" in df.columns

    if price_present and total_points_present:
        non_zero_price_mask = (df["Price"].notna()) & (df["Price"] != 0)
        df.loc[non_zero_price_mask, "PPM_Current"] = (
            df.loc[non_zero_price_mask, "Total_Points_So_Far"]
            / df.loc[non_zero_price_mask, "Price"]
        )
        df["PPM_Current"] = df["PPM_Current"].replace([np.inf, -np.inf], 0).fillna(0)
    else:
        missing_cols = []
        if not price_present:
            missing_cols.append("Price")
        if not total_points_present:
            missing_cols.append("Total_Points_So_Far")
        logger.warning(
            "Missing columns %s for PPM_Current calculation. Setting PPM_Current to 0.",
            missing_cols,
        )
    return df


def _calculate_trend_score(df):
    """Calculates the Trend Score."""
    df["Trend_Score"] = 0.0
    last_race_present = "Points_Last_Race" in df.columns
    adjusted_avg_present = "User_Adjusted_Avg_Points_Last_3_Races" in df.columns

    if last_race_present and adjusted_avg_present:
        df["Trend_Score"] = (
            df["Points_Last_Race"] - df["User_Adjusted_Avg_Points_Last_3_Races"]
        )
    else:
        missing_cols = []
        if not last_race_present:
            missing_cols.append("Points_Last_Race")
        if not adjusted_avg_present:
            missing_cols.append("User_Adjusted_Avg_Points_Last_3_Races")
        logger.warning(
            "Missing columns %s for Trend_Score calculation. Setting Trend_Score to 0.",
            missing_cols,
        )
    df["Trend_Score"] = df["Trend_Score"].fillna(0)
    return df


def _normalize_series(series):
    """Normalizes a pandas Series to a 0-1 scale, handling edge cases."""
    min_val = series.min()
    max_val = series.max()

    if pd.isna(min_val) or pd.isna(max_val):
        logger.warning("Series contains NaN values. Returning series filled with 0.5.")
        return pd.Series([0.5] * len(series), index=series.index)

    if max_val == min_val:
        logger.warning(
            "Series has the same min and max value. Returning series filled with 0.5."
        )
        return pd.Series([0.5] * len(series), index=series.index)

    return (series - min_val) / (max_val - min_val)


def _calculate_combined_score(df, selected_weights):
    """Calculates the combined score based on normalized components and weights."""
    # Define the columns to be normalized
    columns_to_normalize = {
        "User_Adjusted_Avg_Points_Last_3_Races": "Norm_User_Adjusted_Avg_Points_Last_3",
        "Points_Last_Race": "Norm_Points_Last_Race",
        "PPM_Current": "Norm_PPM",
        "Total_Points_So_Far": "Norm_Total_Points_So_Far",
        "Trend_Score": "Norm_Trend_Score",
    }

    # Initialize score columns
    for norm_col in columns_to_normalize.values():
        df[norm_col] = 0.5  # Default value

    for asset_type in ["Driver", "Constructor"]:
        type_mask = df["Type"] == asset_type
        if type_mask.sum() > 0:
            # Normalize each column and assign to the corresponding Norm column
            for source_col, norm_col in columns_to_normalize.items():
                # Ensure source columns are filled before normalization
                df.loc[type_mask, source_col] = df.loc[type_mask, source_col].fillna(0)
                df.loc[type_mask, norm_col] = _normalize_series(
                    df.loc[type_mask, source_col]
                )

    # Calculate the combined score using the normalized columns and weights
    df["Combined_Score"] = (
        selected_weights["recent_form"] * df["Norm_User_Adjusted_Avg_Points_Last_3"]
        + selected_weights["last_race"] * df["Norm_Points_Last_Race"]
        + selected_weights["ppm"] * df["Norm_PPM"]
        + selected_weights["total_points"] * df["Norm_Total_Points_So_Far"]
        + selected_weights["trend"] * df["Norm_Trend_Score"]
    )

    df["Combined_Score"] = df["Combined_Score"].fillna(0)
    return df


def _calculate_derived_scores(
    df: pd.DataFrame, selected_weights: dict
) -> pd.DataFrame | None:
    """Orchestrates calculation of PPM, Trend Score, and Combined Score."""
    if df is None:
        return None
    try:
        df = _calculate_ppm(
            df.copy()
        )  # Pass copy to avoid modifying original in chain if an error occurs mid-way
        df = _calculate_trend_score(df.copy())
        df = _calculate_combined_score(df.copy(), selected_weights)
        logger.debug("Derived scores (PPM, Trend, Combined) calculated.")
        return df
    except Exception as e:
        logger.error("Error calculating derived scores: %s", e, exc_info=True)
        return None


def _load_and_process_team_df(
    team_url: str | None, all_assets_df_processed: pd.DataFrame | None
) -> pd.DataFrame | None:
    """Loads team data from a URL, merges with asset data, and handles Purchase_Price."""
    if not team_url:  # If MY_TEAM_URL in config is empty or None
        logger.info("No team URL provided. Skipping team data processing.")
        return None

    is_placeholder_url = (
        hasattr(config, "MY_TEAM_URL")
        and team_url == config.MY_TEAM_URL
        and "YOUR_GOOGLE_SHEET_URL" in config.MY_TEAM_URL
    )

    if is_placeholder_url:
        logger.info("Team data URL is a placeholder. Skipping team data processing.")
        return None

    if all_assets_df_processed is None:
        logger.error("Asset data is not available, cannot process team data.")
        return None

    my_team_df = None  # Initialize
    purchase_price_was_missing_in_file = False

    try:
        logger.info("Attempting to load team data from URL: %s", team_url)
        my_team_df_raw = pd.read_csv(team_url)
        my_team_df_raw.columns = my_team_df_raw.columns.str.strip()

        cols_to_select_from_raw = ["ID"]
        if "Purchase_Price" not in my_team_df_raw.columns:
            purchase_price_was_missing_in_file = True
            logger.warning(
                "'Purchase_Price' column not found in team data from %s.", team_url
            )
        else:
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

        # Define columns to bring in from all_assets_df_processed
        cols_to_merge_from_assets = [  # Ensure this list is comprehensive
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
        ]
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

        if my_team_df is not None and not my_team_df.empty:
            if purchase_price_was_missing_in_file:
                if "Price" in my_team_df.columns:
                    my_team_df["Purchase_Price"] = my_team_df["Price"]
                else:
                    my_team_df["Purchase_Price"] = 0.0
                logger.warning(
                    "'Purchase_Price' defaulted to Current Price from asset data. Effective budget cap will be approx $%.2fM.",
                    config.INITIAL_BUDGET,
                )

            if "Purchase_Price" in my_team_df.columns:
                my_team_df["Purchase_Price"] = pd.to_numeric(
                    my_team_df["Purchase_Price"], errors="coerce"
                ).fillna(0)
            else:  # Should have been created if missing and Price column was present
                my_team_df["Purchase_Price"] = 0.0
                logger.warning(
                    "'Purchase_Price' column could not be established, defaulted to 0."
                )

            my_team_df["PPM_on_Purchase"] = 0.0
            if (
                "Total_Points_So_Far" in my_team_df.columns
                and "Purchase_Price" in my_team_df.columns
                and my_team_df["Purchase_Price"].notna().all()
            ):  # Check if column exists and is not all NaN

                non_zero_pp_mask = my_team_df["Purchase_Price"] != 0
                my_team_df.loc[non_zero_pp_mask, "PPM_on_Purchase"] = (
                    my_team_df.loc[non_zero_pp_mask, "Total_Points_So_Far"]
                    / my_team_df.loc[non_zero_pp_mask, "Purchase_Price"]
                )
            my_team_df["PPM_on_Purchase"] = (
                my_team_df["PPM_on_Purchase"].replace([np.inf, -np.inf], 0).fillna(0)
            )

            # Check for assets in team file not found in asset_data (rows with NaNs in merged columns)
            # A more robust check for merge failure on an ID:
            check_cols_after_merge = [
                "Name",
                "Type",
                "Price",
            ]  # Core cols from asset_data
            if my_team_df[check_cols_after_merge].isnull().any().any():
                missing_ids_details = my_team_df[
                    my_team_df[check_cols_after_merge[0]].isnull()
                ]["ID"].tolist()
                if missing_ids_details:
                    logger.warning(
                        "Some asset IDs in team file were not found in asset data or have missing details after merge: %s. These rows may have incomplete data.",
                        missing_ids_details,
                    )
                # Fill NaNs for key display/logic columns that should have come from asset_data_df
                fill_cols = [
                    "Price",
                    "Total_Points_So_Far",
                    "Avg_Points_Last_3_Races",
                    "User_Adjusted_Avg_Points_Last_3_Races",
                    "Point_Adjustment_Avg3Races",
                    "Points_Last_Race",
                    "PPM_Current",
                    "Combined_Score",
                    "Active",
                    "Type",
                    "Name",
                    "Constructor",
                ]
                for col in fill_cols:
                    if col in my_team_df.columns:
                        my_team_df[col] = my_team_df[col].fillna(
                            False
                            if col == "Active"
                            else (
                                "Unknown"
                                if col in ["Name", "Type", "Constructor"]
                                else 0.0
                            )
                        )
                    else:  # If column itself is missing after merge (shouldn't happen if in actual_cols_to_merge)
                        my_team_df[col] = (
                            False
                            if col == "Active"
                            else (
                                "Unknown"
                                if col in ["Name", "Type", "Constructor"]
                                else 0.0
                            )
                        )
                        logger.warning(
                            "Column '%s' was missing in merged team data, defaulted.",
                            col,
                        )

        elif (
            my_team_df is not None and my_team_df.empty
        ):  # Merge happened but no matches
            logger.warning(
                "Team data is empty after merge (no matching IDs found between team data and asset data)."
            )

        if my_team_df is None:  # Should be caught by earlier checks
            logger.error(
                "Could not create final team dataframe, possibly due to asset data issues or empty team file."
            )

    except pd.errors.EmptyDataError:
        logger.warning(
            "No data found at team URL (empty CSV): %s. Proceeding without team data.",
            team_url,
        )
        my_team_df = None
    except Exception as e:
        logger.error(
            "Error loading or processing team data from URL %s: %s",
            team_url,
            e,
            exc_info=True,
        )
        my_team_df = None

    if my_team_df is not None:
        logger.debug("Team data loaded and processed.")
    return my_team_df


def load_and_process_data(
    asset_data_url: str,
    team_url: str | None,
    adjustments_url: str | None,
    selected_weights: dict,
) -> tuple[
    pd.DataFrame | None, pd.DataFrame | None, str
]:  # str is for legacy warning_msg, can be removed
    """
    Orchestrates loading and processing of all data by calling helper functions.
    """
    logger.info("Starting data loading and processing pipeline...")

    asset_data_df = _load_raw_asset_df(asset_data_url)
    if asset_data_df is None:
        logger.critical("Failed to load raw asset data. Cannot proceed.")
        return (
            None,
            None,
            "Critical: Raw asset data loading failed.",
        )  # Return legacy warning for now

    asset_data_df = _preprocess_asset_attributes(asset_data_df.copy())
    if asset_data_df is None:
        logger.critical("Failed to preprocess asset attributes. Cannot proceed.")
        return None, None, "Critical: Asset attribute preprocessing failed."

    asset_data_df = _calculate_points_metrics(
        asset_data_df.copy(), config.METADATA_COLUMNS
    )
    if asset_data_df is None:
        logger.critical("Failed to calculate base points metrics. Cannot proceed.")
        return None, None, "Critical: Base points metrics calculation failed."

    asset_data_df = _apply_manual_adjustments(asset_data_df.copy(), adjustments_url)
    if (
        asset_data_df is None
    ):  # Should not return None, just df with/without adjustments
        logger.warning(
            "Manual adjustments step had an issue but proceeding with potentially unadjusted data."
        )
        # Re-fetch from a prior step if needed, or ensure _apply_manual_adjustments always returns a df
        # For now, assume it returns asset_data_df even if adjustments file is missing/bad

    all_assets_df = _calculate_derived_scores(asset_data_df.copy(), selected_weights)
    if all_assets_df is None:
        logger.critical("Failed to calculate derived scores. Cannot proceed.")
        return None, None, "Critical: Derived score calculation failed."

    my_team_df = _load_and_process_team_df(team_url, all_assets_df)
    # my_team_df can be None if team_url is not provided or fails, which is acceptable for some modes.

    logger.info("Data loading and processing pipeline completed.")
    # The legacy warning_msg is less critical now as issues are logged.
    # We can return an empty string or a summary status.
    return all_assets_df, my_team_df, ""  # Return empty legacy warning message


def display_team_and_budget_info(team_df, initial_budget, budget_warning_message):
    """Displays current team information and budget with abbreviated column headers."""
    if team_df is None:
        # Added a print message for this case, and ensured it returns expected tuple
        print("\n--- Your Current Team ---")
        print("Team data is not available or empty.")
        if budget_warning_message:
            print(f"\n{budget_warning_message}")
        return initial_budget, 0.0  # Default dynamic_budget and current_team_value

    print("\n--- Your Current Team ---")
    if team_df.empty:
        print("Your team is currently empty.")
        team_current_value = 0.0
        team_purchase_cost = 0.0
    else:
        cols_to_display_original_names = [
            "ID",
            "Name",
            "Type",
            "Constructor",
            "Price",
            "Purchase_Price",
            "Total_Points_So_Far",
            "Avg_Points_Last_3_Races",
            "User_Adjusted_Avg_Points_Last_3_Races",
            "Point_Adjustment_Avg3Races",
            "Points_Last_Race",
            "Trend_Score",  # Added Trend_Score
            "PPM_Current",
            "PPM_on_Purchase",
            "Active",
            "Combined_Score",
            "Norm_User_Adjusted_Avg_Points_Last_3",
            "Norm_Points_Last_Race",
            "Norm_PPM",
            "Norm_Total_Points_So_Far",
            "Norm_Trend_Score",
        ]
        actual_display_cols = [
            col for col in cols_to_display_original_names if col in team_df.columns
        ]

        df_to_print = team_df[actual_display_cols].copy()
        # Rename only the columns that are present in df_to_print and in the abbreviation map
        rename_map = {
            k: v
            for k, v in config.COLUMN_NAME_ABBREVIATIONS.items()
            if k in df_to_print.columns
        }
        df_to_print.rename(columns=rename_map, inplace=True)

        # Temporarily set display options for this print
        with pd.option_context(
            "display.max_rows",
            None,
            "display.width",
            200,  # Adjust as needed
            "display.max_colwidth",
            None,
            "display.float_format",
            "{:.2f}".format,
        ):
            print(df_to_print.to_string(index=False, na_rep="NaN"))

        team_current_value = team_df["Price"].sum()  # Use original team_df for sums
        team_purchase_cost = team_df["Purchase_Price"].sum()

    # ... (rest of the budget printing logic remains the same) ...
    print(f"\nTotal Team Current Market Value: ${team_current_value:,.2f}M")
    print(f"Total Team Assumed Purchase Cost: ${team_purchase_cost:,.2f}M")
    value_gain_loss = team_current_value - team_purchase_cost
    dynamic_budget = initial_budget + value_gain_loss
    print(
        f"Team Value Gain/(Loss) since initial purchase (defaulted): ${value_gain_loss:,.2f}M"
    )
    print(f"Current Dynamic Budget: ${dynamic_budget:,.2f}M")
    if budget_warning_message:
        print(f"\n{budget_warning_message.strip()}")  # Ensure stripping newlines if any
    if not team_df.empty:
        num_drivers = len(team_df[team_df["Type"] == "Driver"])
        num_constructors = len(team_df[team_df["Type"] == "Constructor"])
        print(
            f"\nTeam Composition: {num_drivers} Drivers, {num_constructors} Constructors."
        )
        if num_drivers != 5 or num_constructors != 2:
            logger.warning(
                "Team composition might be invalid (expected 5 Drivers, 2 Constructors)."
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


# Ensure itertools is imported at the top of your script
# import itertools


def _suggest_true_triple_discretionary_swaps(
    current_my_team_df: pd.DataFrame,
    initial_available_for_purchase_df: pd.DataFrame,
    all_assets_df_complete: pd.DataFrame,
    dynamic_budget: float,
    initial_current_team_value: float,
) -> list:
    """
    Suggests true triple swaps (3 out, 3 in) if beneficial.
    """
    logger.info("Calculating true triple discretionary swaps...")
    true_triple_swap_suggestions = []

    active_team_members_df = current_my_team_df[current_my_team_df["Active"]].copy()

    if len(active_team_members_df) < 3:
        logger.info("Not enough active players on team to perform a triple swap.")
        return []

    budget_headroom = dynamic_budget - initial_current_team_value

    # Iterate through all unique combinations of three assets to sell
    for s1_details, s2_details, s3_details in itertools.combinations(
        active_team_members_df.to_dict("records"), 3
    ):
        sell_triple_ids = {s1_details["ID"], s2_details["ID"], s3_details["ID"]}
        sell_triple_price = (
            s1_details["Price"] + s2_details["Price"] + s3_details["Price"]
        )
        sell_triple_combined_score = (
            s1_details["Combined_Score"]
            + s2_details["Combined_Score"]
            + s3_details["Combined_Score"]
        )

        # Determine types needed for replacements
        types_to_sell = sorted(
            [s1_details["Type"], s2_details["Type"], s3_details["Type"]]
        )  # Sort for consistent handling if types are same

        max_combined_buy_price = sell_triple_price + budget_headroom

        # Filter available assets for candidates for B1, B2, B3
        # Ensure candidates are not the ones being sold (already handled by initial_available_for_purchase_df)

        # Get candidates for each slot based on type.
        # This nested loop approach can be computationally intensive.
        # We need to ensure B1, B2, B3 are distinct.

        # For simplicity, let's assume we are buying assets matching the types sold.
        # A more advanced version could try different type combinations if allowed by team structure (e.g. sell D,D,C -> buy D,D,C)

        # This part needs careful construction of loops to get unique B1, B2, B3 of correct types
        # For now, let's outline the structure for finding B1, B2, B3 matching types of S1, S2, S3 respectively.

        # Potential B1 candidates (matching type of S1)
        cand_b1_df = initial_available_for_purchase_df[
            initial_available_for_purchase_df["Type"] == s1_details["Type"]
        ].copy()

        for _, b1_row in cand_b1_df.iterrows():
            b1 = b1_row.to_dict()
            if b1["Price"] > max_combined_buy_price:
                continue  # Early exit if B1 alone is too expensive

            # Potential B2 candidates (matching type of S2, not B1)
            cand_b2_df = initial_available_for_purchase_df[
                (initial_available_for_purchase_df["Type"] == s2_details["Type"])
                & (initial_available_for_purchase_df["ID"] != b1["ID"])
            ].copy()

            for _, b2_row in cand_b2_df.iterrows():
                b2 = b2_row.to_dict()
                price_b1_b2 = b1["Price"] + b2["Price"]
                if price_b1_b2 > max_combined_buy_price:
                    continue

                # Potential B3 candidates (matching type of S3, not B1 or B2)
                max_price_for_b3 = max_combined_buy_price - price_b1_b2
                if max_price_for_b3 < 0:
                    continue

                cand_b3_df = initial_available_for_purchase_df[
                    (initial_available_for_purchase_df["Type"] == s3_details["Type"])
                    & (initial_available_for_purchase_df["ID"] != b1["ID"])
                    & (initial_available_for_purchase_df["ID"] != b2["ID"])
                    & (initial_available_for_purchase_df["Price"] <= max_price_for_b3)
                ].copy()

                for _, b3_row in cand_b3_df.iterrows():
                    b3 = b3_row.to_dict()

                    buy_triple_price = (
                        b1["Price"] + b2["Price"] + b3["Price"]
                    )  # Should be <= max_combined_buy_price
                    buy_triple_combined_score = (
                        b1["Combined_Score"]
                        + b2["Combined_Score"]
                        + b3["Combined_Score"]
                    )

                    improvement_score = (
                        buy_triple_combined_score - sell_triple_combined_score
                    )

                    if improvement_score > 0:
                        new_team_total_value = (
                            initial_current_team_value
                            - sell_triple_price
                            + buy_triple_price
                        )
                        money_left_under_cap = dynamic_budget - new_team_total_value

                        true_triple_swap_suggestions.append(
                            {
                                "sell1": s1_details,
                                "sell2": s2_details,
                                "sell3": s3_details,
                                "buy1": b1,
                                "buy2": b2,
                                "buy3": b3,
                                "improvement_score": improvement_score,
                                "new_team_value": new_team_total_value,
                                "money_left_under_cap": money_left_under_cap,
                                "sell_triple_price": sell_triple_price,
                                "buy_triple_price": buy_triple_price,
                            }
                        )

    if true_triple_swap_suggestions:
        sorted_triple_swaps = sorted(
            true_triple_swap_suggestions,
            key=lambda x: x["improvement_score"],
            reverse=True,
        )
        logger.info("Found %d potential true triple swaps.", len(sorted_triple_swaps))
        return sorted_triple_swaps[:5]  # Return top 5

    logger.info("No beneficial true triple swaps found.")
    return []


def suggest_swaps(
    all_assets_df: pd.DataFrame,
    my_team_df: pd.DataFrame,
    mandatory_transfers_df: pd.DataFrame,
    dynamic_budget: float,
    current_team_value: float,
    num_available_weekly_transfers: int,  # This is the raw number like 1, 2, or 3 from args.transfers
    num_mandatory_transfers: int,
) -> dict:
    """
    Orchestrates swap suggestions based on the number of available discretionary transfers.
    """
    suggestions = {
        "mandatory": [],
        "best_single_discretionary": [],  # For 1 transfer
        "discretionary_sequence": [],  # For N sequential single (can be 1, 2, or 3)
        "true_double_swaps": [],  # For 2 transfers
        "true_triple_swaps": [],  # For 3 transfers
    }

    if all_assets_df is None or my_team_df is None or my_team_df.empty:
        logger.error("Cannot generate suggestions: missing asset data or team data.")
        return suggestions

    owned_ids = list(my_team_df["ID"])
    initial_available_for_purchase_df = _get_available_for_purchase(
        all_assets_df, owned_ids
    )

    # --- 1. Handle Mandatory Transfers ---
    if num_mandatory_transfers > 0:
        logger.info("--- Suggestions for Mandatory Replacements ---")
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
        else:
            logger.info("No suitable replacements found for mandatory transfers.")

    # --- 2. Determine available discretionary transfers ---
    # This uses the user-provided number of transfers they intend to make this week
    num_discretionary_to_suggest = (
        num_available_weekly_transfers - num_mandatory_transfers
    )

    if num_discretionary_to_suggest <= 0:
        logger.info(
            "No discretionary transfers available or to suggest after mandatory ones."
        )
        return suggestions

    # --- 3. Generate Suggestions Based on Number of Discretionary Transfers ---
    logger.info(
        "Calculating suggestions for %d discretionary transfer(s).",
        num_discretionary_to_suggest,
    )

    # Option A: Always show best single if possible
    if num_discretionary_to_suggest >= 1:
        logger.info("Calculating best single discretionary swap...")
        best_single_swap = _suggest_sequential_single_discretionary_swaps(
            my_team_df,
            initial_available_for_purchase_df,
            all_assets_df,
            dynamic_budget,
            current_team_value,
            1,  # Request exactly one
        )
        if best_single_swap:  # This function returns a list, check if it's non-empty
            suggestions["best_single_discretionary"] = best_single_swap
            display_suggestions(
                suggestions["best_single_discretionary"],
                "Best Single Discretionary Swap",
                dynamic_budget,
            )
        else:
            logger.info("No beneficial single discretionary swap found.")

    # Option B: Show "true" optimal for the exact number of transfers
    if num_discretionary_to_suggest == 1:
        # The 'best_single_discretionary' already covers this.
        # Or, if you want the sequential to show its sequence for 1:
        # suggestions['discretionary_sequence'] = best_single_swap (already done)
        pass

    if num_discretionary_to_suggest == 2:
        logger.info("Calculating true double discretionary swaps...")
        double_swaps = _suggest_true_double_discretionary_swaps(
            my_team_df,
            initial_available_for_purchase_df,
            all_assets_df,
            dynamic_budget,
            current_team_value,
        )
        if double_swaps:
            suggestions["true_double_swaps"] = double_swaps
            display_suggestions(
                suggestions["true_double_swaps"], "True Double Swaps", dynamic_budget
            )
        else:
            logger.info("No beneficial true double swaps found.")

    if num_discretionary_to_suggest == 3:
        logger.info("Calculating true triple discretionary swaps...")
        triple_swaps = _suggest_true_triple_discretionary_swaps(
            my_team_df,
            initial_available_for_purchase_df,
            all_assets_df,
            dynamic_budget,
            current_team_value,
        )
        if triple_swaps:
            suggestions["true_triple_swaps"] = triple_swaps
            display_suggestions(
                suggestions["true_triple_swaps"], "True Triple Swaps", dynamic_budget
            )
        else:
            logger.info("No beneficial true triple swaps found.")

        # Optionally, also show the best sequential 3-single-swaps as an alternative for 3 transfers
        # logger.info("Calculating best sequential triple (3x single) discretionary swaps...")
        # seq_triple = _suggest_sequential_single_discretionary_swaps(
        #     my_team_df, initial_available_for_purchase_df, all_assets_df,
        #     dynamic_budget, current_team_value, 3
        # )
        # if seq_triple:
        #     suggestions['discretionary_sequence_for_3'] = seq_triple # Different key
        #     display_suggestions(suggestions['discretionary_sequence_for_3'], "Sequential Triple (3x Single) Discretionary Swaps", dynamic_budget)

    # Fallback: If specific N-swap logic didn't run or yield results, show sequential for N transfers
    # This part might become redundant if the above specific calls cover all cases.
    # For now, if a specific N-swap was requested and produced results, that's the primary output.
    # If only single was found when N > 1 was allowed, that's also fine.

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
    print("\n--- Target-Based Double Swap Assistant ---")
    print(f"Attempting to swap OUT: {fixed_sell_id} for IN: {fixed_buy_id}")

    # --- 1. Validate fixed part of the swap ---
    if fixed_sell_id not in current_my_team_df["ID"].values:
        logger.error("Asset to sell '%s' is not in your current team.", fixed_sell_id)
        return

    asset_to_sell_1 = (
        current_my_team_df[current_my_team_df["ID"] == fixed_sell_id].iloc[0].to_dict()
    )

    if not all_assets_df[all_assets_df["ID"] == fixed_buy_id]["Active"].any():
        logger.error(
            "Target asset to buy '%s' is not active or does not exist.", fixed_buy_id
        )
        return 1

    asset_to_buy_1 = (
        all_assets_df[all_assets_df["ID"] == fixed_buy_id].iloc[0].to_dict()
    )

    if asset_to_sell_1["Type"] != asset_to_buy_1["Type"]:
        logger.error(
            "Asset types do not match for the fixed swap (%s vs %s).",
            asset_to_sell_1["Type"],
            asset_to_buy_1["Type"],
        )
        return 1

    if fixed_buy_id in current_my_team_df["ID"].values:
        logger.error(
            "Target asset to buy '%s' is already on your team (cannot swap for itself).",
            fixed_buy_id,
        )
        return 1

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
        title_cap_info = (
            f" (Team Budget Cap: ${dynamic_budget:.2f}M)"
            if dynamic_budget is not None
            else ""
        )
        print(
            f"\nTop {len(suggestion_list)} Discretionary Single Swap Option(s) (Using Combined Score){title_cap_info}:"
        )

        for i, swap in enumerate(suggestion_list):

            def format_value(data_dict, val_key, precision=2, default_val="N/A"):
                val = data_dict.get(val_key)
                if isinstance(val, (int, float)) and not np.isnan(
                    val
                ):  # Check for NaN too
                    return f"{val:.{precision}f}"
                return default_val

            sell_score_display = format_value(swap, "sell_score", 2)
            buy_score_display = format_value(swap, "buy_score", 2)
            improvement_score_display = format_value(swap, "improvement_score", 2)

            sell_avg_raw_display = format_value(swap, "sell_avg_points_raw", 2)
            buy_avg_raw_display = format_value(swap, "buy_avg_points_raw", 2)

            sell_last_race_raw_display = format_value(swap, "sell_last_race_raw", 1)
            buy_last_race_raw_display = format_value(swap, "buy_last_race_raw", 1)

            sell_total_pts_raw_display = format_value(swap, "sell_total_points_raw", 1)
            buy_total_pts_raw_display = format_value(swap, "buy_total_points_raw", 1)

            sell_trend_raw_display = format_value(swap, "sell_trend_raw", 1)
            buy_trend_raw_display = format_value(swap, "buy_trend_raw", 1)

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
            print("   Buy Pair:")
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
    elif suggestion_type_name == "True Triple Swaps":
        title_cap_info = (
            f" (Team Budget Cap: ${dynamic_budget:.2f}M)"
            if dynamic_budget is not None
            else ""
        )
        print(
            f"\nTop {len(suggestion_list)} True Triple Swap Option(s){title_cap_info}:"
        )

        for i, swap_group in enumerate(suggestion_list):
            s1 = swap_group.get("sell1", {})
            s2 = swap_group.get("sell2", {})
            s3 = swap_group.get("sell3", {})
            b1 = swap_group.get("buy1", {})
            b2 = swap_group.get("buy2", {})
            b3 = swap_group.get("buy3", {})

            # Re-use or adapt your asset formatting helper
            def format_asset_details_for_triple_swap(asset_dict):
                # (Similar to format_asset_details_for_double_swap, showing relevant raw scores)
                def format_num(val_key, precision=1, default_str="N/A"):
                    val = asset_dict.get(val_key)
                    if isinstance(val, (int, float)) and not np.isnan(val):
                        return f"{val:.{precision}f}"
                    return default_str

                name_id = f"{asset_dict.get('Name', 'N/A')} (ID: {asset_dict.get('ID', 'N/A')})"
                price = f"Price: ${asset_dict.get('Price', 0):.1f}M"
                score = f"Score: {asset_dict.get('Combined_Score', 0):.2f}"
                avg_l3 = format_num("Avg_Points_Last_3_Races", 2)
                last_r = format_num("Points_Last_Race", 1)
                tot_pts = format_num("Total_Points_So_Far", 1)
                trend = format_num("Trend_Score", 1)
                return f"{name_id}, {price}, {score}, AvgL3: {avg_l3}, LastR: {last_r}, TotPts: {tot_pts}, Trend: {trend}"

            print(f"\n{i+1}. Sell Triple:")
            print(f"   Out: {format_asset_details_for_triple_swap(s1)}")
            print(f"   Out: {format_asset_details_for_triple_swap(s2)}")
            print(f"   Out: {format_asset_details_for_triple_swap(s3)}")
            print(
                f"   Combined Sell Price: ${swap_group.get('sell_triple_price', 0):.1f}M"
            )
            print(f"   Buy Triple:")
            print(f"   In:  {format_asset_details_for_triple_swap(b1)}")
            print(f"   In:  {format_asset_details_for_triple_swap(b2)}")
            print(f"   In:  {format_asset_details_for_triple_swap(b3)}")
            print(
                f"   Combined Buy Price: ${swap_group.get('buy_triple_price', 0):.1f}M"
            )

            improvement_score_val = swap_group.get("improvement_score", 0)
            improvement_score_str = (
                f"{improvement_score_val:.2f}"
                if isinstance(improvement_score_val, (int, float))
                else "N/A"
            )
            print(f"   Combined Score Improvement: +{improvement_score_str}")

            new_team_val_str = f"{swap_group.get('new_team_value', 0):.2f}"
            money_left_str = f"{swap_group.get('money_left_under_cap', 0):.2f}"
            print(
                f"   Resulting Team Value: ${new_team_val_str}M / Money Left Under Cap: ${money_left_str}M"
            )


def display_all_asset_stats(all_assets_df):
    """
    Displays a table of calculated statistics for all assets,
    separating Drivers and Constructors, using abbreviated column headers.
    Type column is omitted from display. Inactive assets are listed last.
    """
    if all_assets_df is None or all_assets_df.empty:
        logger.error("No asset data available to display.")
        return

    print("\n--- All Asset Statistics ---")

    # Define the original column names you want to see in this overview
    cols_to_display_original_names = [
        "ID",
        "Name",
        "Constructor",
        "Price",
        "Active",  # 'Type' is intentionally omitted for this display
        "Total_Points_So_Far",
        "Avg_Points_Last_3_Races",
        "User_Adjusted_Avg_Points_Last_3_Races",
        "Point_Adjustment_Avg3Races",
        "Points_Last_Race",
        "Trend_Score",
        "PPM_Current",
        "Combined_Score",
        "Norm_User_Adjusted_Avg_Points_Last_3",
        "Norm_Points_Last_Race",
        "Norm_PPM",
        "Norm_Total_Points_So_Far",
        "Norm_Trend_Score",
    ]

    # Filter for columns that actually exist in all_assets_df
    actual_display_cols = [
        col for col in cols_to_display_original_names if col in all_assets_df.columns
    ]

    # Set pandas display options
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 200)  # Adjust as needed, or None for terminal width
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.float_format", "{:.2f}".format)

    for asset_type_to_display in ["Driver", "Constructor"]:
        type_df = all_assets_df[all_assets_df["Type"] == asset_type_to_display].copy()

        print(f"\n--- {asset_type_to_display} Statistics ---")
        if not type_df.empty:
            type_df_sorted = type_df.sort_values(
                by=["Active", "Combined_Score"], ascending=[False, False]
            )

            # Create a DataFrame with only the columns to be displayed
            df_to_print = type_df_sorted[actual_display_cols].copy()

            # Create a specific rename map for only the columns present in df_to_print
            rename_map_for_display = {
                original_name: config.COLUMN_NAME_ABBREVIATIONS.get(
                    original_name, original_name
                )
                for original_name in df_to_print.columns  # Iterate over columns actually in df_to_print
            }
            df_to_print.rename(columns=rename_map_for_display, inplace=True)

            print(df_to_print.to_string(index=False, na_rep="N/A"))
        else:
            print(f"No {asset_type_to_display.lower()} data to display.")

    # Reset pandas display options
    pd.reset_option("display.max_rows")
    pd.reset_option("display.width")
    pd.reset_option("display.max_colwidth")
    pd.reset_option("display.float_format")


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
        logger.warning("No active assets available for optimization.")
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

        display_cols_original_names = [  # Use original names here
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
        actual_display_cols = [
            col for col in display_cols_original_names if col in optimal_team_df.columns
        ]

        df_to_print = optimal_team_df[actual_display_cols].copy()
        # Rename only the columns that are present in df_to_print and in the abbreviation map
        rename_map = {
            k: v
            for k, v in config.COLUMN_NAME_ABBREVIATIONS.items()
            if k in df_to_print.columns
        }
        df_to_print.rename(columns=rename_map, inplace=True)

        with pd.option_context(
            "display.max_rows",
            None,
            "display.width",
            200,  # Adjust
            "display.max_colwidth",
            None,
            "display.float_format",
            "{:.2f}".format,
        ):
            print(df_to_print.to_string(index=False, na_rep="NaN"))

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
        logger.error("Asset data is not available. Cannot optimize limitless team.")
        return

    active_assets = all_assets_df[all_assets_df["Active"]].copy()
    if "Combined_Score" not in active_assets.columns:
        logger.error(
            "'%s' not found in asset data. Please ensure data is processed correctly.",
            "Combined_Score",
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
        logger.warning(
            "Not enough active drivers or constructors available to form a full limitless team."
        )
        logger.warning(
            "Found %d drivers and %d constructors.",
            len(limitless_drivers),
            len(limitless_constructors),
        )

    limitless_team_df = pd.concat(
        [limitless_drivers, limitless_constructors], ignore_index=True
    )

    print(f"\n--- Optimal Limitless Team (Ignoring Budget) ---")
    if limitless_team_df.empty:
        logger.warning(
            "Could not form a limitless team (no active assets found or criteria not met)."
        )
        return

    display_cols_original_names = [  # Use original names here
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
    actual_display_cols = [
        col for col in display_cols_original_names if col in limitless_team_df.columns
    ]

    df_to_print = limitless_team_df[actual_display_cols].copy()
    # Rename only the columns that are present in df_to_print and in the abbreviation map
    rename_map = {
        k: v
        for k, v in config.COLUMN_NAME_ABBREVIATIONS.items()
        if k in df_to_print.columns
    }
    df_to_print.rename(columns=rename_map, inplace=True)

    with pd.option_context(
        "display.max_rows",
        None,
        "display.width",
        200,  # Adjust
        "display.max_colwidth",
        None,
        "display.float_format",
        "{:.2f}".format,
    ):
        print(df_to_print.to_string(index=False, na_rep="NaN"))

    total_hypothetical_cost = limitless_team_df["Price"].sum()
    total_score = limitless_team_df["Combined_Score"].sum()

    print(
        f"\nTotal Hypothetical Team Cost (if budget applied): ${total_hypothetical_cost:.2f}M"
    )
    print(f"Total Team Combined Score: {total_score:.2f}")


def configure_argparse() -> argparse.ArgumentParser:
    """Configures the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="F1 Fantasy Assistant: Helps with team management and optimization.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "-p",
        "--profile",
        type=str.lower,  # Convert input to lowercase for matching keys
        choices=list(
            CLI_PROFILE_CHOICES_MAP.keys()
        ),  # Use keys from our map as valid choices
        default=DEFAULT_CLI_PROFILE_CHOICE,
        help=(
            "Select the weighting profile for Combined_Score calculation.\n"
            f"Options: {', '.join(CLI_PROFILE_HELP_OPTIONS)}.\n"
            f"Default: {DEFAULT_CLI_PROFILE_CHOICE} ({CLI_PROFILE_CHOICES_MAP[DEFAULT_CLI_PROFILE_CHOICE].replace('_', ' ').title()})."
        ),
    )

    parser.add_argument(
        "-m",
        "--mode",
        type=str.lower,  # Convert input to lowercase
        choices=list(CLI_MODE_CHOICES_MAP.keys()),  # Use keys from our map
        default=DEFAULT_CLI_MODE_CHOICE,
        help=(
            "Select the mode of operation.\n"
            + "Options:\n  "
            + "\n  ".join(CLI_MODE_HELP_OPTIONS)
            + "\n"
            f"Default: {DEFAULT_CLI_MODE_CHOICE} ({CLI_MODE_CHOICES_MAP[DEFAULT_CLI_MODE_CHOICE].replace('_', ' ').title()})."
        ),
    )

    target_swap_group = parser.add_argument_group(
        "Target Swap Mode Options (--mode target/ts)"
    )
    target_swap_group.add_argument(
        "--sell",
        dest="sell_id",
        type=str.upper,  # Convert input to uppercase for ID matching
        help="ID of the asset to SELL.",
    )
    target_swap_group.add_argument(
        "--buy",
        dest="buy_id",
        type=str.upper,  # Convert input to uppercase
        help="ID of the asset to BUY.",
    )

    optional_group = parser.add_argument_group("Optional Parameters")
    optional_group.add_argument(
        "-t",
        "--transfers",
        type=int,
        default=config.DEFAULT_FREE_TRANSFERS,
        help=f"Number of free transfers for normal/weekly mode. Default: {config.DEFAULT_FREE_TRANSFERS}",
    )
    optional_group.add_argument(
        "-B",
        "--budget",
        type=float,
        default=config.INITIAL_BUDGET,
        help=f"Initial budget for wildcard mode. Default: {config.INITIAL_BUDGET}",
    )
    optional_group.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug level logging for more detailed output.",
    )
    optional_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose level logging for more detailed output.",
    )
    return parser


def configure_logging(args: argparse.ArgumentParser):  # Added type hint and parameter
    """Configures logging for the application."""
    log_level = logging.DEBUG
    if args.debug:
        log_level = logging.DEBUG  # Set to DEBUG if debug flag is enabled
    elif args.verbose:
        log_level = logging.INFO  # Set to INFO if verbose flag is enabled
    else:
        log_level = logging.WARNING  # Default to WARNING if neither is set

    # Get your named logger
    # logger = logging.getLogger('F1FantasyAssistant') # This is already global in your script

    logger.setLevel(log_level)

    # Configure console handler (ch) if not already configured
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:  # If handlers exist, just update their level if needed
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):  # Target console handler
                handler.setLevel(log_level)

    # Optional: FileHandler setup (can also respect debug_enabled for its level)
    # try:
    #     file_handler = logging.FileHandler('f1_fantasy_assistant.log', mode='a')
    #     file_handler.setFormatter(formatter) # Use the same formatter
    #     file_handler.setLevel(logging.DEBUG) # Always log DEBUG to file for instance
    #     if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    #         logger.addHandler(file_handler)
    # except Exception as e:
    #     logger.error("Failed to set up file logging: %s", e, exc_info=True)

    logger.debug("Logging enabled at level: %s", logging.getLevelName(log_level))


def main() -> int:
    parser = configure_argparse()
    args = parser.parse_args()
    configure_logging(args)

    print("Starting F1 Fantasy Assistant")
    logger.debug("Parsed arguments: %s", args)

    # --- Process Profile Argument using centralized map ---
    selected_profile_cli_choice = (
        args.profile
    )  # This is already lowercased by type=str.lower
    selected_profile_internal_key = CLI_PROFILE_CHOICES_MAP.get(
        selected_profile_cli_choice,  # User's input (e.g., "a" or "aggressive")
        CLI_PROFILE_CHOICES_MAP[
            DEFAULT_CLI_PROFILE_CHOICE
        ],  # Default if somehow invalid
    )
    selected_weights = config.WEIGHT_PROFILES.get(
        selected_profile_internal_key,
        config.WEIGHT_PROFILES["balanced"],  # Absolute fallback
    )
    logger.info(
        "Using '%s' weighting profile.",
        selected_profile_internal_key.replace("_", " ").title(),
    )

    # --- Process Mode Argument using centralized map ---
    selected_mode_cli_choice = args.mode  # Already lowercased
    internal_mode = CLI_MODE_CHOICES_MAP.get(
        selected_mode_cli_choice,
        CLI_MODE_CHOICES_MAP[DEFAULT_CLI_MODE_CHOICE],  # Default if somehow invalid
    )
    logger.info("Executing Mode: %s", internal_mode.replace("_", " ").title())

    # --- Data Loading ---
    all_assets_df, my_team_df, _ = (
        load_and_process_data(  # warnings string not actively used here
            config.ASSET_DATA_URL,
            config.MY_TEAM_URL,
            config.MANUAL_ADJUSTMENTS_URL,
            selected_weights,
        )
    )

    if all_assets_df is None:
        logger.critical("Exiting due to critical error in loading asset data.")
        return 1

    dynamic_budget_val = args.budget
    current_team_value_val = 0.0

    if internal_mode in ["weekly", "target_swap"]:
        # ... (your existing logic to get dynamic_budget_val and current_team_value_val) ...
        if my_team_df is not None and not my_team_df.empty:
            dynamic_budget_val, current_team_value_val = display_team_and_budget_info(
                my_team_df, config.INITIAL_BUDGET, ""
            )
        elif (
            config.MY_TEAM_URL
            and config.MY_TEAM_URL != "YOUR_GOOGLE_SHEET_URL_FOR_MY_TEAM_CSV"
        ):
            logger.error(  # Log error if team data is required but couldn't be loaded
                "Could not load team data from %s. Mode '%s' requires team data.",
                config.MY_TEAM_URL,
                internal_mode,
            )
            return 1  # Exit with an error status

    # --- Mode Execution (uses internal_mode) ---
    if internal_mode == "weekly":
        # ... (your existing logic for weekly mode, using args.transfers) ...
        if my_team_df is None or my_team_df.empty:
            logger.error(
                "Cannot proceed with weekly transfers: Team data is missing or empty."
            )
            return 1
        # ...
        mandatory_transfers_df = identify_mandatory_transfers(my_team_df)
        num_mandatory_transfers = len(mandatory_transfers_df)
        print(f"\nYou have {num_mandatory_transfers} mandatory transfer(s).")
        print(
            f"The system will consider up to {args.transfers} total transfers based on a team budget cap of ${dynamic_budget_val:.2f}M."
        )
        suggest_swaps(
            all_assets_df,
            my_team_df,
            mandatory_transfers_df,
            dynamic_budget_val,
            current_team_value_val,
            args.transfers,
            num_mandatory_transfers,
        )

    elif internal_mode == "wildcard":
        optimize_wildcard_team(all_assets_df, args.budget)

    elif internal_mode == "target_swap":
        if my_team_df is None or my_team_df.empty:
            logger.error(
                "Cannot proceed with target-based swap: Team data is missing or empty."
            )
            return 1

        if not args.sell_id or not args.buy_id:
            # argparse will handle 'required' if we set it, or we show usage.
            # For now, parser.error is a good way to exit if specific args for a mode are needed.
            # However, these args are not 'required' globally, only for this mode.
            # It's better to check here.
            logger.error(
                "For 'target_swap' mode, both --sell (sell_id) and --buy (buy_id) must be provided."
            )
            print("Example: --mode target --sell YOUR_SELL_ID --buy YOUR_BUY_ID")
            return 1
        else:
            # Sell/Buy IDs are already uppercased by type=str.upper in argparse if you use that,
            # otherwise, ensure they are processed as needed.
            # My previous suggestion used dest="sell_id" and dest="buy_id" which get uppercased.
            # If you used str.upper, great. Otherwise, do it here:
            sell_id = args.sell_id  # .strip().upper() if not done by argparse
            buy_id = args.buy_id  # .strip().upper() if not done by argparse
            print(
                f"\n--- Target-Based Double Transfer for Sell: {sell_id}, Buy: {buy_id} ---"
            )
            suggest_target_based_double_swap(
                sell_id,
                buy_id,
                all_assets_df,
                my_team_df,
                dynamic_budget_val,
                current_team_value_val,
            )

    elif internal_mode == "limitless":
        optimize_limitless_team(all_assets_df)

    elif internal_mode == "all_stats":
        display_all_asset_stats(all_assets_df)

    else:
        logger.error("Invalid internal mode resolved: %s. Exiting.", internal_mode)
        return 1

    logger.info("Script finished successfully.")
    return 0


if __name__ == "__main__":
    # main() # Call main and potentially exit with its status
    exit_code = main()
    if exit_code != 0:
        # Optional: print a generic error message or handle exit
        logging.error("Script exited with status %s", exit_code)
    sys.exit(exit_code)  # If you want to propagate exit code to shell
