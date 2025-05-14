import pandas as pd
import numpy as np

# --- Configuration ---
INITIAL_BUDGET = 100.0  # Standard initial budget in F1 Fantasy (in millions)
ASSET_DATA_FILE = "asset_data.csv"
MY_TEAM_FILE = "my_team.csv"

# Define the expected metadata columns in asset_data.csv
# All other columns will be treated as GP points columns
METADATA_COLUMNS = ["ID", "Name", "Type", "Constructor", "Price", "Active"]
DEFAULT_FREE_TRANSFERS = 1  # Standard free transfers per week, can be adjusted


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

    return dynamic_budget


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


# --- Main Execution ---
def main():
    """Main function to execute the F1 Fantasy Assistant."""
    all_assets_df, my_team_df, warning_msg = load_and_process_data(
        ASSET_DATA_FILE, MY_TEAM_FILE
    )

    dynamic_budget = 0.0
    if all_assets_df is not None and my_team_df is not None:
        dynamic_budget = display_team_and_budget_info(
            my_team_df, INITIAL_BUDGET, warning_msg
        )

        mandatory_transfers = identify_mandatory_transfers(my_team_df)
        num_mandatory_transfers = len(mandatory_transfers)

        num_discretionary_transfers = DEFAULT_FREE_TRANSFERS - num_mandatory_transfers
        if num_discretionary_transfers < 0:
            num_discretionary_transfers = (
                0  # You can't have negative discretionary transfers
            )

        print(f"\nYou have {num_mandatory_transfers} mandatory transfer(s).")
        print(
            f"Assuming {DEFAULT_FREE_TRANSFERS} default free transfer(s) per week, you have {num_discretionary_transfers} discretionary transfer(s) available."
        )

        # Placeholder for next steps:
        # 1. Implement suggestion engine for mandatory replacements
        # 2. Implement suggestion engine for discretionary swaps
        print("\nNext step is to implement the swap suggestion engine.")


if __name__ == "__main__":
    main()
