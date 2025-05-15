# This script is a Python module for managing and analyzing F1 Fantasy teams.

import itertools
import os
import pulp
import pandas as pd
import numpy as np

# --- Configuration ---
INITIAL_BUDGET = 100.0  # Standard initial budget in F1 Fantasy (in millions)
ASSET_DATA_FILE = "asset_data.csv"
MY_TEAM_FILE = "my_team.csv"
MANUAL_ADJUSTMENTS_FILE = "manual_adjustments.csv"

# Define the expected metadata columns in asset_data.csv
# All other columns will be treated as GP points columns
METADATA_COLUMNS = ["ID", "Name", "Type", "Constructor", "Price", "Active"]
DEFAULT_FREE_TRANSFERS = 2  # Standard free transfers per week, can be adjusted

# Weights for Combined Score
# (Aim for these to sum to 1.0 for balanced contribution of normalized scores)
WEIGHT_RECENT_FORM = 0.5  # For User_Adjusted_Avg_Points_Last_3_Races
WEIGHT_LAST_RACE = 0.2  # For Points_Last_Race
WEIGHT_PPM = 0.3  # For PPM_Current


def _load_raw_asset_df(asset_file_path):
    """Loads and performs initial validation on asset_data.csv."""
    try:
        df = pd.read_csv(asset_file_path)
        df.columns = df.columns.str.strip()
        for col in METADATA_COLUMNS:
            if col not in df.columns:
                raise ValueError(
                    f"Essential metadata column '{col}' not found in {asset_file_path}. Required: {METADATA_COLUMNS}"
                )
        return df, ""
    except FileNotFoundError:
        return None, f"\nError: The asset data file {asset_file_path} was not found."
    except ValueError as e:
        return None, f"\nError validating asset data columns in {asset_file_path}: {e}"
    except Exception as e:
        return None, f"\nUnexpected error loading {asset_file_path}: {e}"


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


def _apply_manual_adjustments(df, adjustments_file_path):
    """Applies manual point adjustments from a CSV file."""
    warnings = ""
    df["Point_Adjustment_Avg3Races"] = 0.0  # Initialize

    if os.path.exists(adjustments_file_path):
        try:
            adj_df = pd.read_csv(adjustments_file_path)
            adj_df.columns = adj_df.columns.str.strip()
            if (
                "ID" in adj_df.columns
                and "Point_Adjustment_Avg3Races" in adj_df.columns
            ):
                adj_df["Point_Adjustment_Avg3Races"] = pd.to_numeric(
                    adj_df["Point_Adjustment_Avg3Races"], errors="coerce"
                ).fillna(0)

                # Use a temporary column for the merge result to avoid direct assignment issues
                df = pd.merge(
                    df,
                    adj_df[["ID", "Point_Adjustment_Avg3Races"]],
                    on="ID",
                    how="left",
                    suffixes=("", "_adj"),
                )

                # Fill NaNs from merge and assign to the final column
                df["Point_Adjustment_Avg3Races"] = df[
                    "Point_Adjustment_Avg3Races_adj"
                ].fillna(0)
                df.drop(
                    columns=["Point_Adjustment_Avg3Races_adj"],
                    errors="ignore",
                    inplace=True,
                )

                print("Successfully loaded and merged manual adjustments.")
            else:
                warnings += f"\nWarning: {adjustments_file_path} is missing 'ID' or 'Point_Adjustment_Avg3Races' column. Adjustments not applied."
        except Exception as e:
            warnings += f"\nError loading or processing {adjustments_file_path}: {e}. Adjustments not applied."
    else:
        print(
            f"Info: Manual adjustments file '{adjustments_file_path}' not found. No adjustments will be applied."
        )

    # Ensure the column exists even if file loading failed or file not found
    if "Point_Adjustment_Avg3Races" not in df.columns:
        df["Point_Adjustment_Avg3Races"] = 0.0

    df["User_Adjusted_Avg_Points_Last_3_Races"] = (
        df["Avg_Points_Last_3_Races"] + df["Point_Adjustment_Avg3Races"]
    )
    return df, warnings


def _calculate_derived_scores(
    df,
):  # df here is asset_data_df after previous processing steps
    """Calculates PPM_Current and Combined_Score including Points_Last_Race."""
    warnings = ""
    # Calculate PPM_Current
    df["PPM_Current"] = 0.0
    if "Price" in df.columns and "Total_Points_So_Far" in df.columns:
        # Ensure Price is numeric and notna before this step
        non_zero_price_mask = (df["Price"].notna()) & (df["Price"] != 0)
        df.loc[non_zero_price_mask, "PPM_Current"] = (
            df.loc[non_zero_price_mask, "Total_Points_So_Far"]
            / df.loc[non_zero_price_mask, "Price"]
        )
        df["PPM_Current"] = df["PPM_Current"].replace([np.inf, -np.inf], 0).fillna(0)
    else:
        warnings += "\nWarning: 'Price' or 'Total_Points_So_Far' missing for PPM_Current calculation. PPM_Current set to 0."

    # Initialize score columns
    df["Combined_Score"] = 0.0
    df["Norm_User_Adjusted_Avg_Points_Last_3"] = 0.5  # Renamed for clarity
    df["Norm_Points_Last_Race"] = 0.5  # New normalized column
    df["Norm_PPM"] = 0.5

    for asset_type in ["Driver", "Constructor"]:
        type_mask = df["Type"] == asset_type
        if type_mask.sum() > 0:
            # Normalize User_Adjusted_Avg_Points_Last_3_Races
            # This column should exist from _apply_manual_adjustments
            if "User_Adjusted_Avg_Points_Last_3_Races" in df.columns:
                avg_points_series = df.loc[
                    type_mask, "User_Adjusted_Avg_Points_Last_3_Races"
                ].fillna(0)
                df.loc[type_mask, "Norm_User_Adjusted_Avg_Points_Last_3"] = (
                    normalize_series(avg_points_series)
                )
            else:
                warnings += f"\nWarning: 'User_Adjusted_Avg_Points_Last_3_Races' not found for {asset_type}. Using default normalization."
                # Norm_User_Adjusted_Avg_Points_Last_3 remains 0.5

            # Normalize Points_Last_Race
            # This column should exist from _calculate_points_metrics
            if "Points_Last_Race" in df.columns:
                last_race_series = df.loc[type_mask, "Points_Last_Race"].fillna(0)
                df.loc[type_mask, "Norm_Points_Last_Race"] = normalize_series(
                    last_race_series
                )
            else:
                warnings += f"\nWarning: 'Points_Last_Race' not found for {asset_type}. Using default normalization."
                # Norm_Points_Last_Race remains 0.5

            # Normalize PPM_Current
            ppm_series = df.loc[type_mask, "PPM_Current"].fillna(
                0
            )  # PPM_Current is calculated above
            df.loc[type_mask, "Norm_PPM"] = normalize_series(ppm_series)

            # Calculate Combined Score using .loc for assignment
            # Fill NaNs for normalized columns just in case (e.g., if a type had only one asset, normalize_series returns all 0.5)
            norm_avg3 = df.loc[
                type_mask, "Norm_User_Adjusted_Avg_Points_Last_3"
            ].fillna(0.5)
            norm_last_race = df.loc[type_mask, "Norm_Points_Last_Race"].fillna(0.5)
            norm_ppm = df.loc[type_mask, "Norm_PPM"].fillna(0.5)

            df.loc[type_mask, "Combined_Score"] = (
                WEIGHT_RECENT_FORM * norm_avg3
                + WEIGHT_LAST_RACE * norm_last_race  # Added new component
                + WEIGHT_PPM * norm_ppm
            )

    df["Combined_Score"] = df["Combined_Score"].fillna(0)  # Final safety fill
    return df, warnings


def _load_and_process_team_df(team_file_path, all_assets_df_processed):
    """Loads team data, merges with asset data, and handles Purchase_Price."""
    warnings = ""
    my_team_df = None
    purchase_price_was_missing_in_file = False

    if not team_file_path:
        warnings += "\nInfo: No team file specified."
        return None, warnings

    if all_assets_df_processed is None:
        warnings += "\nError: Asset data is not available, cannot process team data."
        return None, warnings

    try:
        my_team_df_raw = pd.read_csv(team_file_path)
        my_team_df_raw.columns = my_team_df_raw.columns.str.strip()

        cols_to_select_from_raw = ["ID"]
        if "Purchase_Price" not in my_team_df_raw.columns:
            purchase_price_was_missing_in_file = True
            warnings += (
                f"\nWarning: 'Purchase_Price' column not found in {team_file_path}."
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
            "Norm_Avg_Points_Last_3",
            "Norm_PPM",
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
                else:  # Should not happen if Price is in actual_cols_to_merge
                    my_team_df["Purchase_Price"] = 0.0
                warnings += f"\n 'Purchase_Price' defaulted to Current Price. Effective budget cap: ~${INITIAL_BUDGET:.2f}M."

            my_team_df["Purchase_Price"] = pd.to_numeric(
                my_team_df["Purchase_Price"], errors="coerce"
            ).fillna(0)
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

            if my_team_df.isnull().values.any():
                warnings += "\nWarning: Some assets in your team file may have missing details after merge."
                # Fill NaNs for key display/logic columns
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
                ]
                for col in fill_cols:
                    if col in my_team_df.columns:
                        my_team_df[col] = my_team_df[col].fillna(
                            0 if my_team_df[col].dtype != "bool" else False
                        )
                    elif col not in [
                        "Purchase_Price",
                        "PPM_on_Purchase",
                    ]:  # Ensure column exists before filling
                        my_team_df[col] = False if col == "Active" else 0.0

        elif my_team_df is not None and my_team_df.empty:
            warnings += f"\nWarning: Team data is empty after merge (check IDs in {team_file_path})."

        if my_team_df is None:
            warnings += "\nError: Could not create final team dataframe."

    except FileNotFoundError:
        warnings += f"\nError: Your team file {team_file_path} was not found."
    except Exception as e:
        warnings += f"\nUnexpected error loading team data from {team_file_path}: {e}\n{traceback.format_exc()}"

    return my_team_df, warnings


def load_and_process_data(asset_file_path, team_file_path, adjustments_file_path):
    """
    Orchestrates loading and processing of all data by calling helper functions.
    """
    overall_warnings = []  # Use a list to collect warnings

    # 1. Load and validate raw asset data
    asset_data_df, warn = _load_raw_asset_df(asset_file_path)
    if warn:
        overall_warnings.append(warn)
    if asset_data_df is None:
        final_warning_msg = "\n".join(filter(None, overall_warnings))
        if final_warning_msg.strip():
            print(
                f"\n--- Data Loading Log ---\n{final_warning_msg.strip()}\n------------------------"
            )
        return None, None, final_warning_msg

    # 2. Preprocess metadata like Price, Active
    # Use .copy() to avoid SettingWithCopyWarning on slices later
    asset_data_df, warn = _preprocess_asset_attributes(asset_data_df.copy())
    if warn:
        overall_warnings.append(warn)

    # 3. Calculate base points metrics (Total, Avg3, LastRace)
    asset_data_df, warn = _calculate_points_metrics(
        asset_data_df.copy(), METADATA_COLUMNS
    )
    if warn:
        overall_warnings.append(warn)

    # 4. Apply manual adjustments
    asset_data_df, warn = _apply_manual_adjustments(
        asset_data_df.copy(), adjustments_file_path
    )
    if warn:
        overall_warnings.append(warn)

    # 5. Calculate derived scores (PPM, Combined_Score) - This becomes the final all_assets_df
    all_assets_df, warn = _calculate_derived_scores(asset_data_df.copy())
    if warn:
        overall_warnings.append(warn)
    if all_assets_df is None:  # Should not happen if previous steps worked
        final_warning_msg = "\n".join(filter(None, overall_warnings))
        if final_warning_msg.strip():
            print(
                f"\n--- Data Loading Log ---\n{final_warning_msg.strip()}\n------------------------"
            )
        return None, None, final_warning_msg

    # 6. Load and process team data
    my_team_df, warn = _load_and_process_team_df(team_file_path, all_assets_df)
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
                            "sell_score": sell_combined_score,
                            "sell_avg_points_raw": asset_to_sell_row[
                                "Avg_Points_Last_3_Races"
                            ],
                            "sell_last_race_raw": asset_to_sell_row["Points_Last_Race"],
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
                            "improvement_score": highest_improvement_score_this_iteration,
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


def display_suggestions(
    suggestion_list, suggestion_type_name, dynamic_budget=None
):  # Added dynamic_budget parameter
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
                f"\n{i+1}. For mandatory sale of: {asset_to_sell['Name']} (ID: {asset_to_sell['ID']}, Price: ${asset_to_sell['Price']:.1f}M)"
            )
            if message:
                print(f"   {message}")
            elif not options.empty:
                print("   Potential replacements (ranked by Avg Points Last 3 Races):")
                # Ensure all display columns exist for options
                opt_display_cols = [
                    "ID",
                    "Name",
                    "Price",
                    "Avg_Points_Last_3_Races",
                    "Total_Points_So_Far",
                ]
                for col in opt_display_cols:
                    if col not in options.columns:
                        options[col] = "N/A"  # Or np.nan if numeric
                print(options[opt_display_cols].to_string(index=False))
            else:
                print(
                    "   No replacement options found (this case should ideally be covered by 'message')."
                )

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
            sell_score_display = (
                f"{swap.get('sell_score', 'N/A'):.2f}"
                if isinstance(swap.get("sell_score"), (int, float))
                else "N/A"
            )
            buy_score_display = (
                f"{swap.get('buy_score', 'N/A'):.2f}"
                if isinstance(swap.get("buy_score"), (int, float))
                else "N/A"
            )
            improvement_score_display = (
                f"{swap.get('improvement_score', 'N/A'):.2f}"
                if isinstance(swap.get("improvement_score"), (int, float))
                else "N/A"
            )

            sell_avg_raw_display = (
                f"{swap.get('sell_avg_points_raw', 'N/A'):.2f}"
                if isinstance(swap.get("sell_avg_points_raw"), (int, float))
                else "N/A"
            )
            buy_avg_raw_display = (
                f"{swap.get('buy_avg_points_raw', 'N/A'):.2f}"
                if isinstance(swap.get("buy_avg_points_raw"), (int, float))
                else "N/A"
            )

            # Fetch Points_Last_Race for display (assuming it's in the swap dictionary, which suggest_swaps should add)
            sell_last_race_raw_display = (
                f"{swap.get('sell_last_race_raw', 'N/A'):.1f}"
                if isinstance(swap.get("sell_last_race_raw"), (int, float))
                else "N/A"
            )
            buy_last_race_raw_display = (
                f"{swap.get('buy_last_race_raw', 'N/A'):.1f}"
                if isinstance(swap.get("buy_last_race_raw"), (int, float))
                else "N/A"
            )

            print(
                f"\n{i+1}. Swap Out: {swap['sell_name']} (ID: {swap['sell_id']}, Price: ${swap['sell_price']:.1f}M, Score: {sell_score_display}, AvgL3: {sell_avg_raw_display}, LastR: {sell_last_race_raw_display})"
            )
            print(
                f"   Swap In:  {swap['buy_name']} (ID: {swap['buy_id']}, Price: ${swap['buy_price']:.1f}M, Score: {buy_score_display}, AvgL3: {buy_avg_raw_display}, LastR: {buy_last_race_raw_display})"
            )
            print(f"   Combined Score Improvement: +{improvement_score_display}")
            print(
                f"   Resulting Team Value: ${swap['new_team_value']:.2f}M / Money Left Under Cap: ${swap['money_left_under_cap']:.2f}M"
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

        for i, swap in enumerate(suggestion_list):
            s1 = swap["sell1"]
            s2 = swap["sell2"]
            b1 = swap["buy1"]
            b2 = swap["buy2"]

            print(f"\n{i+1}. Sell Pair:")
            print(
                f"   Out: {s1['Name']} (ID: {s1['ID']}, Price: ${s1['Price']:.1f}M, Score: {s1['Combined_Score']:.2f})"
            )
            print(
                f"   Out: {s2['Name']} (ID: {s2['ID']}, Price: ${s2['Price']:.1f}M, Score: {s2['Combined_Score']:.2f})"
            )
            print(f"   Combined Sell Price: ${swap['sell_pair_price']:.1f}M")
            print(f"   Buy Pair:")
            print(
                f"   In:  {b1['Name']} (ID: {b1['ID']}, Price: ${b1['Price']:.1f}M, Score: {b1['Combined_Score']:.2f})"
            )
            print(
                f"   In:  {b2['Name']} (ID: {b2['ID']}, Price: ${b2['Price']:.1f}M, Score: {b2['Combined_Score']:.2f})"
            )
            print(f"   Combined Buy Price: ${swap['buy_pair_price']:.1f}M")
            print(f"   Combined Score Improvement: +{swap['improvement_score']:.2f}")
            print(
                f"   Resulting Team Value: ${swap['new_team_value']:.2f}M / Money Left Under Cap: ${swap['money_left_under_cap']:.2f}M"
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


def main():
    all_assets_df, my_team_df, warning_msg = load_and_process_data(
        ASSET_DATA_FILE, MY_TEAM_FILE, MANUAL_ADJUSTMENTS_FILE
    )

    if all_assets_df is None:  # Critical error during data loading
        print("Exiting due to critical error in data loading.")
        return

    print("\nChoose mode:")
    print("1. Weekly Transfer Suggestions")
    print("2. Wildcard Team Optimizer (Build new team from scratch)")

    mode_choice = input("Enter choice (1 or 2): ")

    if mode_choice == "1":
        if my_team_df is None or my_team_df.empty:
            print(
                "Cannot proceed with weekly transfers: Team data is missing or empty."
            )
            return

        dynamic_budget, current_team_value = display_team_and_budget_info(
            my_team_df, INITIAL_BUDGET, warning_msg
        )

        mandatory_transfers_df = identify_mandatory_transfers(my_team_df)
        num_mandatory_transfers = len(mandatory_transfers_df)

        print(f"\nYou have {num_mandatory_transfers} mandatory transfer(s).")
        print(
            f"The system will consider up to {DEFAULT_FREE_TRANSFERS} total transfers based on a team budget cap of ${dynamic_budget:.2f}M."
        )

        suggested_swaps = suggest_swaps(
            all_assets_df,
            my_team_df,
            mandatory_transfers_df,
            dynamic_budget,
            current_team_value,
            DEFAULT_FREE_TRANSFERS,
            num_mandatory_transfers,
        )

        # Check if any suggestions were made (using .get for safety)
        no_mandatory = not suggested_swaps.get("mandatory")
        no_discretionary_seq = not suggested_swaps.get("discretionary_sequence")
        no_double_swaps = not suggested_swaps.get(
            "true_double_swaps"
        )  # Assuming this key exists

        if (
            no_mandatory
            and no_discretionary_seq
            and no_double_swaps
            and num_mandatory_transfers == 0
        ):
            print(
                "\nNo specific swap suggestions at this time based on current criteria and budget."
            )

    elif mode_choice == "2":
        optimize_wildcard_team(
            all_assets_df, INITIAL_BUDGET
        )  # Using INITIAL_BUDGET for wildcard
    else:
        print("Invalid choice. Exiting.")


if __name__ == "__main__":
    main()
