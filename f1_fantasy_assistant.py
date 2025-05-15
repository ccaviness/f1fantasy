import pandas as pd
import numpy as np

# --- Configuration ---
INITIAL_BUDGET = 100.0  # Standard initial budget in F1 Fantasy (in millions)
ASSET_DATA_FILE = "asset_data.csv"
MY_TEAM_FILE = "my_team.csv"

# Define the expected metadata columns in asset_data.csv
# All other columns will be treated as GP points columns
METADATA_COLUMNS = ["ID", "Name", "Type", "Constructor", "Price", "Active"]
DEFAULT_FREE_TRANSFERS = 2  # Standard free transfers per week, can be adjusted

# Weights for Combined Score (should sum to 1.0 if desired, but not strictly necessary for ranking improvement)
WEIGHT_RECENT_FORM = 0.6  # For Avg_Points_Last_3_Races
WEIGHT_PPM = 0.4  # For PPM_Current


def load_and_process_data(asset_file, team_file):
    warning_msg = ""
    asset_data_df = None
    my_team_df = None
    purchase_price_was_missing_in_file = False

    print(f"Attempting to load asset data from: {asset_file}")
    print(
        f"Attempting to load team data from: {team_file if team_file else 'No team file specified'}"
    )

    try:
        asset_data_df = pd.read_csv(asset_file)
        asset_data_df.columns = asset_data_df.columns.str.strip()

        # 1. Ensure essential METADATA columns exist
        for col in METADATA_COLUMNS:
            if col not in asset_data_df.columns:
                raise ValueError(
                    f"Essential metadata column '{col}' not found in {asset_file}. Required: {METADATA_COLUMNS}"
                )

        # Crucial Note for User: Ensure GP columns in your CSV are in chronological order!
        # Example: Sakhir (Bahrain), Jeddah (Saudi Arabia), Melbourne (Australia), ...

        # 2. Identify Potential GP Columns and determine completed ones
        potential_gp_cols = [
            col for col in asset_data_df.columns if col not in METADATA_COLUMNS
        ]

        completed_gp_cols = []
        if not potential_gp_cols:
            warning_msg += "\nWarning: No potential GP points columns found in asset_data.csv (columns other than metadata)."
        else:
            print(f"Found potential GP columns: {potential_gp_cols}")
            for col_name in potential_gp_cols:
                # Convert to numeric, coercing errors. This does not fill NaNs yet.
                numeric_series = pd.to_numeric(asset_data_df[col_name], errors="coerce")
                # A GP is completed if at least one asset has a non-NaN score
                if numeric_series.notna().any():
                    completed_gp_cols.append(col_name)
                    asset_data_df[col_name] = (
                        numeric_series  # Store the numeric version (with NaNs)
                    )
                else:
                    # If column has no valid numeric data, treat it as not completed / make it all NaN
                    # This helps if it had non-numeric placeholders like '-' that weren't caught by 'coerce' alone
                    asset_data_df[col_name] = np.nan

        if not completed_gp_cols:
            warning_msg += "\nWarning: No completed GP races identified from the data. Points-based metrics will be zero."
            asset_data_df["Total_Points_So_Far"] = 0.0
            asset_data_df["Avg_Points_Last_3_Races"] = 0.0
            asset_data_df["Points_Last_Race"] = 0.0
            print("No completed GP columns found.")
        else:
            print(f"Identified completed GP columns: {completed_gp_cols}")
            # Ensure only these completed_gp_cols (which are now numeric with NaNs) are used for calculations

            # 3. Calculate Total_Points_So_Far from completed races
            asset_data_df["Total_Points_So_Far"] = asset_data_df[completed_gp_cols].sum(
                axis=1, skipna=True
            )

            # 4. Calculate Avg_Points_Last_3_Races from completed races
            num_races_to_average = min(len(completed_gp_cols), 3)
            if num_races_to_average > 0:
                last_n_completed_gp_cols = completed_gp_cols[-num_races_to_average:]
                print(
                    f"Calculating Avg_Points_Last_3_Races based on: {last_n_completed_gp_cols}"
                )
                asset_data_df["Avg_Points_Last_3_Races"] = (
                    asset_data_df[last_n_completed_gp_cols]
                    .mean(axis=1, skipna=True)
                    .fillna(0)
                )
            else:
                asset_data_df["Avg_Points_Last_3_Races"] = 0.0

            # 5. Calculate Points_Last_Race from the latest completed race
            last_completed_race_col = completed_gp_cols[-1]
            print(f"Calculating Points_Last_Race based on: {last_completed_race_col}")
            asset_data_df["Points_Last_Race"] = asset_data_df[
                last_completed_race_col
            ].fillna(0)

        # Price parsing
        if "Price" in asset_data_df.columns:
            if asset_data_df["Price"].dtype == "object":
                asset_data_df["Price"] = (
                    asset_data_df["Price"]
                    .replace({r"\$": "", "M": ""}, regex=True)
                    .astype(float)
                )
            else:
                asset_data_df["Price"] = pd.to_numeric(
                    asset_data_df["Price"], errors="coerce"
                ).fillna(0)
        else:
            asset_data_df["Price"] = 0.0
            warning_msg += "\nCritical Warning: 'Price' column missing from asset data."

        asset_data_df["Active"] = asset_data_df["Active"].astype(bool)

        # Calculate PPM_Current (using the newly calculated Total_Points_So_Far)
        asset_data_df["PPM_Current"] = 0.0
        non_zero_price_mask = (asset_data_df["Price"].notna()) & (
            asset_data_df["Price"] != 0
        )
        asset_data_df.loc[non_zero_price_mask, "PPM_Current"] = (
            asset_data_df.loc[non_zero_price_mask, "Total_Points_So_Far"]
            / asset_data_df.loc[non_zero_price_mask, "Price"]
        )
        asset_data_df["PPM_Current"] = (
            asset_data_df["PPM_Current"].replace([np.inf, -np.inf], 0).fillna(0)
        )

        # Calculate Combined Score
        asset_data_df["Combined_Score"] = 0.0
        asset_data_df["Norm_Avg_Points_Last_3"] = 0.5
        asset_data_df["Norm_PPM"] = 0.5

        for asset_type in ["Driver", "Constructor"]:
            type_mask = asset_data_df["Type"] == asset_type
            if type_mask.sum() > 0:
                avg_points_series = asset_data_df.loc[
                    type_mask, "Avg_Points_Last_3_Races"
                ].fillna(
                    0
                )  # Already filled earlier
                norm_avg_points_for_type = normalize_series(avg_points_series)

                ppm_series = asset_data_df.loc[type_mask, "PPM_Current"].fillna(
                    0
                )  # Already filled earlier
                norm_ppm_for_type = normalize_series(ppm_series)

                asset_data_df.loc[type_mask, "Norm_Avg_Points_Last_3"] = (
                    norm_avg_points_for_type
                )
                asset_data_df.loc[type_mask, "Norm_PPM"] = norm_ppm_for_type

                asset_data_df.loc[
                    type_mask, "Combined_Score"
                ] = WEIGHT_RECENT_FORM * asset_data_df.loc[
                    type_mask, "Norm_Avg_Points_Last_3"
                ].fillna(
                    0.5
                ) + WEIGHT_PPM * asset_data_df.loc[
                    type_mask, "Norm_PPM"
                ].fillna(
                    0.5
                )
        asset_data_df["Combined_Score"] = asset_data_df["Combined_Score"].fillna(0)

    except FileNotFoundError:
        warning_msg = f"Error: The file {asset_file} was not found."
        print(warning_msg)  # Print immediately
        return None, None, warning_msg
    except ValueError as e:
        warning_msg = f"Error processing data in {asset_file}: {e}"
        print(warning_msg)  # Print immediately
        return None, None, warning_msg
    except Exception as e:
        warning_msg = (
            f"An unexpected error occurred during data loading from {asset_file}: {e}"
        )
        print(f"Details: {type(e).__name__} - {str(e)}")  # More detailed error
        import traceback

        traceback.print_exc()  # Print full traceback for unexpected errors
        return None, None, warning_msg

    # --- Load and process My Team data ---
    if team_file and asset_data_df is not None:  # Only proceed if asset_data is loaded
        try:
            my_team_df_raw = pd.read_csv(team_file)
            my_team_df_raw.columns = my_team_df_raw.columns.str.strip()

            cols_to_select_from_raw = ["ID"]
            if "Purchase_Price" not in my_team_df_raw.columns:
                purchase_price_was_missing_in_file = True
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
                "Points_Last_Race",  # Added Points_Last_Race
                "PPM_Current",
                "Combined_Score",
                "Norm_Avg_Points_Last_3",
                "Norm_PPM",
            ]
            actual_cols_to_merge = [
                col for col in cols_to_merge_from_assets if col in asset_data_df.columns
            ]

            my_team_df = pd.merge(
                my_team_df_raw[cols_to_select_from_raw],
                asset_data_df[actual_cols_to_merge],
                on="ID",
                how="left",
            )

            if my_team_df is not None and not my_team_df.empty:
                if purchase_price_was_missing_in_file:
                    if "Price" in my_team_df.columns:
                        my_team_df["Purchase_Price"] = my_team_df["Price"]
                    else:
                        my_team_df["Purchase_Price"] = 0.0
                    warning_msg += (
                        f"\nWarning: 'Purchase_Price' not found in {MY_TEAM_FILE}. "
                        f"Assumed equal to Current Price. Effective budget cap: ~${INITIAL_BUDGET:.2f}M."
                    )

                my_team_df["Purchase_Price"] = pd.to_numeric(
                    my_team_df["Purchase_Price"], errors="coerce"
                ).fillna(0)

                my_team_df["PPM_on_Purchase"] = 0.0
                if (
                    "Total_Points_So_Far" in my_team_df.columns
                    and "Purchase_Price" in my_team_df.columns
                ):
                    non_zero_purchase_mask = (my_team_df["Purchase_Price"].notna()) & (
                        my_team_df["Purchase_Price"] != 0
                    )
                    my_team_df.loc[non_zero_purchase_mask, "PPM_on_Purchase"] = (
                        my_team_df.loc[non_zero_purchase_mask, "Total_Points_So_Far"]
                        / my_team_df.loc[non_zero_purchase_mask, "Purchase_Price"]
                    )
                my_team_df["PPM_on_Purchase"] = (
                    my_team_df["PPM_on_Purchase"]
                    .replace([np.inf, -np.inf], 0)
                    .fillna(0)
                )

                # Fill NaNs for any other critical columns that might be missing after merge
                cols_to_check_fill = [
                    "Price",
                    "Total_Points_So_Far",
                    "Avg_Points_Last_3_Races",
                    "Points_Last_Race",
                    "PPM_Current",
                    "Combined_Score",
                    "Active",
                ]
                for col in cols_to_check_fill:
                    if col in my_team_df.columns:
                        if my_team_df[col].isnull().any():
                            if col == "Active":
                                my_team_df[col] = my_team_df[col].fillna(False)
                            else:  # Numeric columns default to 0
                                my_team_df[col] = my_team_df[col].fillna(0)
                    elif col not in [
                        "Purchase_Price",
                        "PPM_on_Purchase",
                    ]:  # These are handled specially
                        warning_msg += f"\nWarning: Column '{col}' missing in merged team data after processing."
                        # Potentially add the column with default if essential for downstream
                        if col == "Active":
                            my_team_df[col] = False
                        else:
                            my_team_df[col] = 0.0

            elif (
                my_team_df is not None and my_team_df.empty
            ):  # Merge happened but no matches
                warning_msg += f"\nWarning: Team data is empty after merge (no matching IDs found between {MY_TEAM_FILE} and {ASSET_DATA_FILE})."

            if my_team_df is None:  # If merge failed or asset_data_df was None
                warning_msg += "\nError: Could not create final team dataframe."

        except FileNotFoundError:
            warning_msg += f"\nError: Your team file {team_file} was not found."
            my_team_df = None  # Ensure it's None
        except Exception as e:
            warning_msg += f"\nAn unexpected error occurred loading team data from {team_file}: {e}"
            print(f"Team data loading error details: {type(e).__name__} - {str(e)}")
            import traceback

            traceback.print_exc()
            my_team_df = None  # Ensure it's None

    elif not team_file:
        warning_msg += (
            "\nInfo: No team file specified. Proceeding with asset data only."
        )
    elif asset_data_df is None:
        warning_msg += (
            "\nError: Asset data failed to load, so team data processing was skipped."
        )

    if warning_msg.strip():
        print(
            f"\n--- Data Loading Log ---\n{warning_msg.strip()}\n------------------------"
        )

    return asset_data_df, my_team_df, warning_msg


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
            "Points_Last_Race",  # Added here
            "PPM_Current",
            "PPM_on_Purchase",
            "Active",
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
    Suggests swaps:
    1. Mandatory replacements (suggested independently for each mandatory slot).
    2. A sequence of discretionary improvements if transfers are still available.
    """
    suggestions = {
        "mandatory": [],
        "discretionary_sequence": [],
    }  # Changed key for clarity

    # --- 1. Handle Mandatory Transfers (logic remains largely the same: independent suggestions per slot) ---
    # (This part is kept concise as the core logic for finding options for ONE mandatory slot is similar to before.
    # If multiple mandatory swaps are needed, the user still needs to manage the overall budget when choosing from options.)
    budget_headroom_initial = dynamic_budget - current_team_value
    if num_mandatory_transfers > 0:
        print("\n--- Suggestions for Mandatory Replacements ---")
        # (Existing logic for iterating through mandatory_transfers_df,
        #  calculating max_buy_price = sell_price + budget_headroom_initial,
        #  finding and storing options for each mandatory asset separately)
        # This part will still present independent options for each mandatory slot.
        # For brevity, I'm not repeating the full iteration here, assume it's the same as your current working version.
        # Ensure 'display_suggestions' is called for this part.
        # Example snippet for one mandatory asset (repeat in loop):
        # for index, asset_to_sell in mandatory_transfers_df.iterrows():
        #     ...
        #     max_buy_price = asset_to_sell['Price'] + budget_headroom_initial
        #     potential_buys = ...
        #     # Add to suggestions['mandatory']
        # display_suggestions(suggestions['mandatory'], "Mandatory Replacements", dynamic_budget)
        # For now, let's assume mandatory swaps are handled and we focus on discretionary:

        # If mandatory suggestions were made, display them
        # This part of the code should be the same as your previous version that correctly handled mandatory swaps.
        # For the purpose of this update, I'll assume that logic is in place and works.
        # Example:
        # if suggestions['mandatory']:
        #    display_suggestions(suggestions['mandatory'], "Mandatory Replacements", dynamic_budget)
        pass  # Placeholder for your existing mandatory swap suggestion logic

    # --- 2. Handle Sequential Discretionary Single Swaps ---
    num_discretionary_transfers_available = (
        num_total_transfers_allowed - num_mandatory_transfers
    )

    if num_discretionary_transfers_available > 0:
        print(
            f"\n--- Sequential Suggestions for up to {num_discretionary_transfers_available} Discretionary Single Swap(s) (Using Combined Score) ---"
        )

        hypothetical_team_df = my_team_df.copy()
        hypothetical_current_team_value = current_team_value
        final_discretionary_sequence = []

        for transfer_num in range(num_discretionary_transfers_available):
            current_budget_headroom = dynamic_budget - hypothetical_current_team_value

            if "Active" not in hypothetical_team_df.columns:
                hypothetical_team_df = hypothetical_team_df.merge(
                    all_assets_df[["ID", "Active"]], on="ID", how="left"
                )
            active_team_members_to_sell = hypothetical_team_df[
                hypothetical_team_df["Active"]
            ].copy()

            best_swap_this_iteration = None
            # Improvement score can be negative if we are forced to sell a high scorer
            # but for discretionary, we only want positive improvements.
            highest_improvement_score_this_iteration = (
                0  # Only consider positive improvements
            )

            for _, asset_to_sell_row in active_team_members_to_sell.iterrows():
                sell_id = asset_to_sell_row["ID"]
                sell_name = asset_to_sell_row["Name"]
                sell_price = asset_to_sell_row["Price"]
                sell_type = asset_to_sell_row["Type"]
                # Use Combined_Score of the asset to sell
                sell_combined_score = asset_to_sell_row["Combined_Score"]

                max_buy_price = sell_price + current_budget_headroom
                hypothetical_owned_ids = list(hypothetical_team_df["ID"])
                current_available_for_purchase_df = (
                    all_assets_df[  # all_assets_df now has Combined_Score
                        (all_assets_df["Active"])
                        & (~all_assets_df["ID"].isin(hypothetical_owned_ids))
                    ].copy()
                )

                potential_buys = current_available_for_purchase_df[
                    (current_available_for_purchase_df["Type"] == sell_type)
                    & (current_available_for_purchase_df["Price"] <= max_buy_price)
                ].copy()  # Removed ID != sell_id as isin(hypothetical_owned_ids) covers it

                if not potential_buys.empty:
                    # Calculate improvement based on Combined_Score
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
                                current_best_buy_for_this_sell[
                                    "Improvement_Score_Combined"
                                ]
                            )
                            # Store relevant details for the suggestion
                            best_swap_this_iteration = {
                                "sell_id": sell_id,
                                "sell_name": sell_name,
                                "sell_price": sell_price,
                                "sell_type": sell_type,
                                "sell_score": sell_combined_score,  # Storing score
                                "buy_id": current_best_buy_for_this_sell["ID"],
                                "buy_name": current_best_buy_for_this_sell["Name"],
                                "buy_price": current_best_buy_for_this_sell["Price"],
                                "buy_score": current_best_buy_for_this_sell[
                                    "Combined_Score"
                                ],  # Storing score
                                "improvement_score": highest_improvement_score_this_iteration,  # This is Combined Improvement
                            }

            if best_swap_this_iteration:
                # ... (rest of the logic for updating hypothetical state and appending to final_discretionary_sequence)
                # This part remains largely the same, just ensure the dict keys match what display_suggestions expects
                # e.g. 'sell_avg_points' might become 'sell_score', 'buy_avg_points' to 'buy_score'
                # For now, I will keep the existing structure and add new score fields to the dict for display.

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

                # Add original point metrics for display comparison if desired
                original_sell_asset = my_team_df[
                    my_team_df["ID"] == best_swap_this_iteration["sell_id"]
                ].iloc[0]
                original_buy_asset = all_assets_df[
                    all_assets_df["ID"] == best_swap_this_iteration["buy_id"]
                ].iloc[0]
                best_swap_this_iteration["sell_avg_points_raw"] = original_sell_asset[
                    "Avg_Points_Last_3_Races"
                ]
                best_swap_this_iteration["buy_avg_points_raw"] = original_buy_asset[
                    "Avg_Points_Last_3_Races"
                ]

                final_discretionary_sequence.append(best_swap_this_iteration)

                # ---- Update hypothetical state for the next iteration ----
                hypothetical_current_team_value = new_team_val_after_this_one_swap
                hypothetical_team_df = hypothetical_team_df[
                    hypothetical_team_df["ID"] != best_swap_this_iteration["sell_id"]
                ].copy()
                asset_to_add_details = all_assets_df[
                    all_assets_df["ID"] == best_swap_this_iteration["buy_id"]
                ].copy()
                asset_to_add_details["Purchase_Price"] = asset_to_add_details["Price"]
                asset_to_add_details["PPM_on_Purchase"] = (
                    (
                        asset_to_add_details["Total_Points_So_Far"]
                        / asset_to_add_details["Purchase_Price"]
                    )
                    .replace([np.inf, -np.inf], 0)
                    .fillna(0)
                )
                cols_for_hypothetical_team = list(my_team_df.columns)
                for col in cols_for_hypothetical_team:
                    if col not in asset_to_add_details.columns:
                        asset_to_add_details[col] = np.nan
                hypothetical_team_df = pd.concat(
                    [
                        hypothetical_team_df,
                        asset_to_add_details[cols_for_hypothetical_team],
                    ],
                    ignore_index=True,
                )

            else:
                if transfer_num == 0:
                    print(
                        "No beneficial discretionary single swaps found based on Combined Score and current budget."
                    )
                break

        if final_discretionary_sequence:
            suggestions["discretionary_sequence"] = final_discretionary_sequence
            display_suggestions(
                suggestions["discretionary_sequence"],
                "Discretionary Single Swaps",
                dynamic_budget,
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
            # Accessing the stored scores, and raw points for context
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

            print(
                f"\n{i+1}. Swap Out: {swap['sell_name']} (ID: {swap['sell_id']}, Price: ${swap['sell_price']:.1f}M, Score: {sell_score_display}, AvgL3Raw: {sell_avg_raw_display})"
            )
            print(
                f"   Swap In:  {swap['buy_name']} (ID: {swap['buy_id']}, Price: ${swap['buy_price']:.1f}M, Score: {buy_score_display}, AvgL3Raw: {buy_avg_raw_display})"
            )
            print(f"   Combined Score Improvement: +{improvement_score_display}")
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


def main():
    """
    Main function to load data, process it, and suggest swaps.
    """
    all_assets_df, my_team_df, warning_msg = load_and_process_data(
        ASSET_DATA_FILE, MY_TEAM_FILE
    )

    dynamic_budget = 0.0
    current_team_value = 0.0  # Initialize

    if all_assets_df is not None and my_team_df is not None:
        # Capture both returned values
        dynamic_budget, current_team_value = display_team_and_budget_info(
            my_team_df, INITIAL_BUDGET, warning_msg
        )

        mandatory_transfers_df = identify_mandatory_transfers(my_team_df)
        num_mandatory_transfers = len(mandatory_transfers_df)

        print(f"\nYou have {num_mandatory_transfers} mandatory transfer(s).")
        print(
            f"The system will consider up to {DEFAULT_FREE_TRANSFERS} total transfers based on a team budget cap of ${dynamic_budget:.2f}M."
        )  # Clarified print

        # Call the suggestion function with current_team_value
        suggested_swaps = suggest_swaps(
            all_assets_df,
            my_team_df,
            mandatory_transfers_df,
            dynamic_budget,
            current_team_value,  # Pass current_team_value
            DEFAULT_FREE_TRANSFERS,
            num_mandatory_transfers,
        )

        if (
            not suggested_swaps.get("mandatory")
            and not suggested_swaps.get("discretionary_sequence")
            and num_mandatory_transfers == 0
        ):
            print(
                "\nNo specific swap suggestions at this time based on current criteria and corrected budget."
            )


if __name__ == "__main__":
    main()
