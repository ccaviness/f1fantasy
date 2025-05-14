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


def load_and_process_data(asset_file_path, team_file_path):
    """
    Loads asset and team data from CSV files, processes it, and calculates
    initial metrics.
    """
    print(f"Loading data from {asset_file_path} and {team_file_path}...")

    # --- 1. Load asset_data.csv ---
    try:
        all_assets_df = pd.read_csv(asset_file_path)
    except FileNotFoundError:
        print(
            f"ERROR: File not found: {asset_file_path}. Please ensure it's in the same directory as the script."
        )
        return None, None, ""

    # Validate essential metadata columns
    missing_cols = [col for col in METADATA_COLUMNS if col not in all_assets_df.columns]
    if missing_cols:
        print(
            f"ERROR: Missing essential columns in {asset_file_path}: {', '.join(missing_cols)}"
        )
        return None, None, ""

    # --- 2. Preprocess asset_data.csv ---
    # Convert data types
    all_assets_df["Price"] = pd.to_numeric(all_assets_df["Price"], errors="coerce")
    all_assets_df["Active"] = all_assets_df["Active"].astype(bool)

    # Identify GP points columns (any column not in METADATA_COLUMNS)
    gp_points_columns = [
        col for col in all_assets_df.columns if col not in METADATA_COLUMNS
    ]

    # Convert GP points columns to numeric, coercing errors to NaN (for blanks)
    for col in gp_points_columns:
        all_assets_df[col] = pd.to_numeric(all_assets_df[col], errors="coerce")

    # --- 3. Calculate points-based metrics for all_assets_df ---
    all_assets_df["Total_Points_So_Far"] = all_assets_df[gp_points_columns].sum(
        axis=1, skipna=True
    )

    # Calculate Points_Last_Race and Avg_Points_Last_3_Races
    # This needs to find the *last non-NaN* columns for each row more dynamically

    valid_gp_cols_by_asset = []
    for index, row in all_assets_df.iterrows():
        asset_gp_cols = [col for col in gp_points_columns if pd.notna(row[col])]
        valid_gp_cols_by_asset.append(asset_gp_cols)

    all_assets_df["Points_Last_Race"] = [
        all_assets_df.loc[i, cols[-1]] if cols else 0
        for i, cols in enumerate(valid_gp_cols_by_asset)
    ]

    avg_last_3 = []
    for i, cols in enumerate(valid_gp_cols_by_asset):
        if len(cols) >= 3:
            avg_last_3.append(all_assets_df.loc[i, cols[-3:]].mean())
        elif cols:  # 1 or 2 races
            avg_last_3.append(all_assets_df.loc[i, cols].mean())
        else:  # 0 races
            avg_last_3.append(0)
    all_assets_df["Avg_Points_Last_3_Races"] = avg_last_3

    all_assets_df["PPM_Current"] = (
        (all_assets_df["Total_Points_So_Far"] / all_assets_df["Price"])
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )

    # --- 4. Load my_team.csv ---
    try:
        my_team_ids_df = pd.read_csv(team_file_path)
    except FileNotFoundError:
        print(
            f"ERROR: File not found: {team_file_path}. Please ensure it's in the same directory as the script."
        )
        return None, None, ""

    if "ID" not in my_team_ids_df.columns:
        print(f"ERROR: 'ID' column missing in {team_file_path}.")
        return None, None, ""

    # --- 5. Merge team data with asset data ---
    # We use the IDs from my_team_ids_df to select rows from all_assets_df
    # The 'Price' column in my_team.csv is the current price, which we'll effectively ignore
    # in favor of the 'Price' from all_assets_df for consistency.
    my_team_details_df = all_assets_df[
        all_assets_df["ID"].isin(my_team_ids_df["ID"])
    ].copy()

    if len(my_team_details_df) != len(my_team_ids_df):
        print(
            "WARNING: Some IDs in my_team.csv were not found in asset_data.csv. Check your IDs."
        )
        # Find missing IDs
        missing_ids_in_assets = set(my_team_ids_df["ID"]) - set(all_assets_df["ID"])
        if missing_ids_in_assets:
            print(
                f"IDs in {team_file_path} but not in {asset_file_path}: {missing_ids_in_assets}"
            )

    # --- 6. Handle Purchase_Price for the current team ---
    # As actual purchase prices are not available, assume Purchase_Price = Current_Price
    # The 'Price' column in all_assets_df is Current_Price
    my_team_details_df["Purchase_Price"] = my_team_details_df["Price"]

    purchase_price_warning = (
        "IMPORTANT:\n"
        "Historical purchase prices for your current team members were not available.\n"
        "Therefore, their 'Purchase_Price' has been defaulted to their 'Current_Price'.\n"
        "This means your initial dynamic budget will equal the standard INITIAL_BUDGET.\n"
        "The tool will track budget changes accurately based on future price fluctuations.\n"
        "For future transfers, try to record the actual purchase price in your my_team.csv \n"
        "(e.g., by changing its columns to 'ID' and 'Purchase_Price') for more accurate historical budget tracking.\n"
    )

    # Calculate PPM on purchase price for owned assets
    my_team_details_df["PPM_on_Purchase"] = (
        (
            my_team_details_df["Total_Points_So_Far"]
            / my_team_details_df["Purchase_Price"]
        )
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )

    print("Data loaded and processed successfully.")
    return all_assets_df, my_team_details_df, purchase_price_warning


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
        display_columns = [
            "ID",
            "Name",
            "Type",
            "Constructor",
            "Price",
            "Purchase_Price",
            "Total_Points_So_Far",
            "Avg_Points_Last_3_Races",
            "Points_Last_Race",
            "PPM_Current",
            "PPM_on_Purchase",
            "Active",
        ]
        for col in display_columns:  # Ensure columns exist
            if col not in team_df.columns:
                team_df[col] = np.nan
        print(team_df[display_columns].to_string(index=False))
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
            f"\n--- Sequential Suggestions for up to {num_discretionary_transfers_available} Discretionary Single Swap(s) ---"
        )

        # Start with the actual current team state for the sequence
        hypothetical_team_df = my_team_df.copy()
        hypothetical_current_team_value = current_team_value
        # Ensure all_assets_df has 'Purchase_Price' and 'PPM_on_Purchase' if needed for hypothetical_team_df consistency
        # Typically, these are calculated when a team is formed or an asset is bought.
        # For this simulation, if adding from all_assets_df, we might need to compute them.
        # However, PPM_on_Purchase for a *newly bought hypothetical asset* isn't relevant for its *selection criteria*.
        # We need 'Price', 'Type', 'Avg_Points_Last_3_Races', 'ID', 'Name'.
        # Let's ensure columns for hypothetical_team_df align with my_team_df.

        final_discretionary_sequence = []

        for transfer_num in range(num_discretionary_transfers_available):
            current_budget_headroom = dynamic_budget - hypothetical_current_team_value

            # Ensure hypothetical_team_df has an 'Active' column, if not already present from my_team_df copy
            if (
                "Active" not in hypothetical_team_df.columns
            ):  # Should be there from my_team_df
                hypothetical_team_df = hypothetical_team_df.merge(
                    all_assets_df[["ID", "Active"]], on="ID", how="left"
                )

            active_team_members_to_sell = hypothetical_team_df[
                hypothetical_team_df["Active"]
            ].copy()

            best_swap_this_iteration = None
            highest_improvement_score_this_iteration = -float(
                "inf"
            )  # Start very low to capture any positive

            # Find the single best swap from the current hypothetical state
            for _, asset_to_sell_row in active_team_members_to_sell.iterrows():
                sell_id = asset_to_sell_row["ID"]
                sell_name = asset_to_sell_row["Name"]
                sell_price = asset_to_sell_row["Price"]
                sell_type = asset_to_sell_row["Type"]
                sell_avg_points = asset_to_sell_row["Avg_Points_Last_3_Races"]

                max_buy_price = sell_price + current_budget_headroom

                # Available for purchase: active, not on *hypothetical* team
                hypothetical_owned_ids = list(hypothetical_team_df["ID"])
                current_available_for_purchase_df = all_assets_df[
                    (all_assets_df["Active"])
                    & (~all_assets_df["ID"].isin(hypothetical_owned_ids))
                ].copy()

                potential_buys = current_available_for_purchase_df[
                    (current_available_for_purchase_df["Type"] == sell_type)
                    & (current_available_for_purchase_df["Price"] <= max_buy_price)
                    & (
                        current_available_for_purchase_df["ID"] != sell_id
                    )  # Should be redundant due to isin(hypothetical_owned_ids)
                ].copy()

                if not potential_buys.empty:
                    potential_buys["Improvement_Score_Avg3"] = (
                        potential_buys["Avg_Points_Last_3_Races"] - sell_avg_points
                    )
                    # Filter for actual improvements
                    improved_options = potential_buys[
                        potential_buys["Improvement_Score_Avg3"] > 0
                    ].sort_values(by="Improvement_Score_Avg3", ascending=False)

                    if not improved_options.empty:
                        current_best_buy_for_this_sell = improved_options.iloc[0]
                        if (
                            current_best_buy_for_this_sell["Improvement_Score_Avg3"]
                            > highest_improvement_score_this_iteration
                        ):
                            highest_improvement_score_this_iteration = (
                                current_best_buy_for_this_sell["Improvement_Score_Avg3"]
                            )
                            best_swap_this_iteration = {
                                "sell_id": sell_id,
                                "sell_name": sell_name,
                                "sell_price": sell_price,
                                "sell_type": sell_type,
                                "sell_avg_points": sell_avg_points,
                                "buy_id": current_best_buy_for_this_sell["ID"],
                                "buy_name": current_best_buy_for_this_sell["Name"],
                                "buy_price": current_best_buy_for_this_sell["Price"],
                                "buy_avg_points": current_best_buy_for_this_sell[
                                    "Avg_Points_Last_3_Races"
                                ],
                                "improvement_score": highest_improvement_score_this_iteration,
                            }

            if best_swap_this_iteration:
                # Record this swap in the sequence
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

                # ---- Update hypothetical state for the next iteration ----
                # 1. Update team value
                hypothetical_current_team_value = new_team_val_after_this_one_swap

                # 2. Update team DataFrame: remove sold player, add bought player
                # Remove sold player
                hypothetical_team_df = hypothetical_team_df[
                    hypothetical_team_df["ID"] != best_swap_this_iteration["sell_id"]
                ].copy()

                # Get full details of bought player from all_assets_df
                # We need to ensure the structure matches my_team_df for concat, especially 'Purchase_Price' and 'PPM_on_Purchase'
                asset_to_add_details = all_assets_df[
                    all_assets_df["ID"] == best_swap_this_iteration["buy_id"]
                ].copy()

                # For hypothetical addition, set Purchase_Price = Current Price
                asset_to_add_details["Purchase_Price"] = asset_to_add_details["Price"]
                # Calculate PPM_on_Purchase for the new asset
                asset_to_add_details["PPM_on_Purchase"] = (
                    (
                        asset_to_add_details["Total_Points_So_Far"]
                        / asset_to_add_details["Purchase_Price"]
                    )
                    .replace([np.inf, -np.inf], 0)
                    .fillna(0)
                )

                # Select columns that match my_team_df (which hypothetical_team_df is a copy of)
                # This assumes my_team_df has all necessary columns also present in the processed all_assets_df
                # plus Purchase_Price and PPM_on_Purchase.
                cols_for_hypothetical_team = list(
                    my_team_df.columns
                )  # Get target columns from original my_team_df structure

                # Ensure asset_to_add_details has all these columns before concat
                for col in cols_for_hypothetical_team:
                    if col not in asset_to_add_details.columns:
                        # This might happen if a calculated field in my_team_df isn't directly in all_assets_df
                        # For now, we assume essential ones are there or calculated.
                        # For safety, assign NaN or a default if a column is missing.
                        asset_to_add_details[col] = np.nan

                hypothetical_team_df = pd.concat(
                    [
                        hypothetical_team_df,
                        asset_to_add_details[cols_for_hypothetical_team],
                    ],
                    ignore_index=True,
                )
            else:
                # No beneficial swap found in this iteration, so break the sequence
                if (
                    transfer_num == 0
                ):  # If even the first discretionary swap isn't found
                    print(
                        "No beneficial discretionary single swaps found based on Avg_Points_Last_3_Races and current budget."
                    )
                break

        if final_discretionary_sequence:
            suggestions["discretionary_sequence"] = final_discretionary_sequence
            # Display this sequence. You might want to adjust the title in display_suggestions for clarity.
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
        # Title is now more dynamic based on whether dynamic_budget is available for display
        title_cap_info = (
            f" (Team Budget Cap: ${dynamic_budget:.2f}M)"
            if dynamic_budget is not None
            else ""
        )
        print(
            f"\nTop {len(suggestion_list)} Discretionary Single Swap Option(s){title_cap_info}:"
        )

        for i, swap in enumerate(suggestion_list):
            print(
                f"\n{i+1}. Swap Out: {swap['sell_name']} (ID: {swap['sell_id']}, Price: ${swap['sell_price']:.1f}M, AvgL3: {swap['sell_avg_points']:.2f})"
            )
            print(
                f"   Swap In:  {swap['buy_name']} (ID: {swap['buy_id']}, Price: ${swap['buy_price']:.1f}M, AvgL3: {swap['buy_avg_points']:.2f})"
            )
            print(f"   AvgL3 Points Improvement: +{swap['improvement_score']:.2f}")
            # Displaying the new team value and how much is left under the cap
            print(
                f"   Resulting Team Value: ${swap['new_team_value']:.2f}M / Money Left Under Cap: ${swap['money_left_under_cap']:.2f}M"
            )


def main():  # Assuming you have this structure
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
            not suggested_swaps["mandatory"]
            and not suggested_swaps["discretionary"]
            and num_mandatory_transfers == 0
        ):
            print(
                "\nNo specific swap suggestions at this time based on current criteria and corrected budget."
            )


if __name__ == "__main__":
    main()
