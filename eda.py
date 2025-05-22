# ==============EDA SCRIPT FOR DOTA 2 MATCH DATA (V8 - ENHANCED DOCS & PLOT FIX & CSV OUTPUT)===
# DESCRIPTION:
# This script performs Exploratory Data Analysis (EDA) on Dota 2 data,
# including matches, player stats, final items, and detailed pick/ban information
# with draft order. It features a timeline-style plot for the draft sequence,
# corrected plot elements, comprehensive documentation, and saves tabular outputs to CSV.
# ==================================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter
from itertools import combinations
import itertools
import os
import re
from matplotlib.colors import ListedColormap, BoundaryNorm

# ==============CONFIGURATION===========================================
# DESCRIPTION:
# Define file paths to your data files. Adjust these as necessary.
# Define constants used throughout the EDA for thresholds and plotting.
# ========================================================================
BASE_DATA_DIR = './Data'
PLAYERS_FILE = os.path.join(BASE_DATA_DIR, 'player_matches_aug_2024.csv')
MATCHES_FILE = os.path.join(BASE_DATA_DIR, 'players_matches.csv')
PICKS_BANS_FILE = os.path.join(BASE_DATA_DIR, '2024', 'picks_bans.csv')
CONSTANTS_DIR = os.path.join(BASE_DATA_DIR, 'Constants')
ITEMS_FILE = os.path.join(CONSTANTS_DIR, 'Constants.ItemIDs.csv')
HEROES_FILE = os.path.join(CONSTANTS_DIR, 'Constants.Heroes.csv')

EDA_OUTPUT_DIR = './EDA_Outputs_V8_DocsAndFixes_CSV' # Updated for CSV outputs
os.makedirs(EDA_OUTPUT_DIR, exist_ok=True)

MIN_GAMES_THRESHOLD_HERO = 50      # Min games hero played for reliable win rate
MIN_GAMES_THRESHOLD_ITEM = 50      # Min occurrences of an item for reliable win rate
MIN_GAMES_THRESHOLD_HERO_PAIR = 30 # Min games hero pair played together for synergy
MIN_DRAFT_OCCURRENCES = 20         # Min pick/ban occurrences for some rate calculations
TOP_N_PLOT = 15                    # Number of top items/heroes to display in plots

def sanitize_name(name): # Helper function, less critical for EDA display but good for consistency
    if pd.isna(name): return "Unknown"
    s = str(name).strip(); s = re.sub(r'\s+', '_', s); s = re.sub(r'[^\w-]', '', s); return s if s else "Unknown"

print(f"EDA Output will be saved in: {EDA_OUTPUT_DIR}")

# ==============I. DATA LOADING AND INITIAL OVERVIEW====================
# CALCULATION:
# Loads all required CSV files (matches, players, picks/bans, hero constants, item constants)
# into pandas DataFrames.
# INTERPRETATION:
# Provides a foundational understanding of the raw data's structure (shape, columns, data types),
# size (number of rows), a sample of its content (head), and identifies the total number
# of unique entities like heroes, items, and matches for context in rate calculations.
# Outputs basic stats and samples to CSV.
# ========================================================================
print("\n============== I. DATA LOADING AND INITIAL OVERVIEW ==============")
try:
    df_matches = pd.read_csv(MATCHES_FILE)
    df_players = pd.read_csv(PLAYERS_FILE)
    df_picks_bans = pd.read_csv(PICKS_BANS_FILE)
    df_heroes_constants = pd.read_csv(HEROES_FILE)
    df_items_constants = pd.read_csv(ITEMS_FILE)
    print("All data files loaded successfully.")
except FileNotFoundError as e:
    print(f"ERROR: Could not load data files. {e}\nPlease ensure paths are correct."); exit()
except Exception as e:
    print(f"An unexpected error occurred during data loading: {e}"); exit()

# Saving initial overview data
initial_overview_data = []

print(f"\n--- Matches Data ('{os.path.basename(MATCHES_FILE)}') ---")
print(f"Shape: {df_matches.shape}"); df_matches.info(); print("Sample:\n", df_matches.head(2))
initial_overview_data.append({'DataFrame': 'Matches', 'Rows': df_matches.shape[0], 'Columns': df_matches.shape[1]})
df_matches.head().to_csv(os.path.join(EDA_OUTPUT_DIR, 'sample_matches_data.csv'), index=False)

print(f"\n--- Players Data ('{os.path.basename(PLAYERS_FILE)}') ---")
print(f"Shape: {df_players.shape}"); df_players.info(); print("Sample:\n", df_players.head(2))
initial_overview_data.append({'DataFrame': 'Players', 'Rows': df_players.shape[0], 'Columns': df_players.shape[1]})
df_players.head().to_csv(os.path.join(EDA_OUTPUT_DIR, 'sample_players_data.csv'), index=False)

print(f"\n--- Picks/Bans Data ('{os.path.basename(PICKS_BANS_FILE)}') ---")
print(f"Shape: {df_picks_bans.shape}"); df_picks_bans.info(); print("Sample:\n", df_picks_bans.head(2))
initial_overview_data.append({'DataFrame': 'Picks/Bans', 'Rows': df_picks_bans.shape[0], 'Columns': df_picks_bans.shape[1]})
df_picks_bans.head().to_csv(os.path.join(EDA_OUTPUT_DIR, 'sample_picks_bans_data.csv'), index=False)

print(f"\n--- Heroes Constants ('{os.path.basename(HEROES_FILE)}') ---")
print(f"Shape: {df_heroes_constants.shape}; Unique heroes defined: {df_heroes_constants['id'].nunique()}")
initial_overview_data.append({'DataFrame': 'Heroes Constants', 'Rows': df_heroes_constants.shape[0], 'Columns': df_heroes_constants.shape[1], 'Unique Heroes': df_heroes_constants['id'].nunique()})
df_heroes_constants.head().to_csv(os.path.join(EDA_OUTPUT_DIR, 'sample_heroes_constants.csv'), index=False)


print(f"\n--- Items Constants ('{os.path.basename(ITEMS_FILE)}') ---")
print(f"Shape: {df_items_constants.shape}; Unique items defined: {df_items_constants['id'].nunique()}")
initial_overview_data.append({'DataFrame': 'Items Constants', 'Rows': df_items_constants.shape[0], 'Columns': df_items_constants.shape[1], 'Unique Items': df_items_constants['id'].nunique()})
df_items_constants.head().to_csv(os.path.join(EDA_OUTPUT_DIR, 'sample_items_constants.csv'), index=False)

df_initial_overview = pd.DataFrame(initial_overview_data)
df_initial_overview.to_csv(os.path.join(EDA_OUTPUT_DIR, 'data_overview_summary.csv'), index=False)


# Create ID-to-Name mappings for heroes and items
hero_id_to_name = df_heroes_constants.set_index('id')['localized_name'].to_dict()
hero_id_to_name[0] = "Unknown_Hero" # Handle cases where hero_id might be 0
pd.Series(hero_id_to_name, name='HeroName').rename_axis('HeroID').reset_index().to_csv(os.path.join(EDA_OUTPUT_DIR, 'hero_id_to_name_mapping.csv'), index=False)

item_id_to_name = df_items_constants.set_index('id')['name'].to_dict() # Assuming 'name' column for items
item_id_to_name[0] = "Unknown_Item" # Handle cases where item_id might be 0
pd.Series(item_id_to_name, name='ItemName').rename_axis('ItemID').reset_index().to_csv(os.path.join(EDA_OUTPUT_DIR, 'item_id_to_name_mapping.csv'), index=False)


total_matches_in_dataset = df_matches['match_id'].nunique()
print(f"\nTotal unique matches in dataset (for rate calculations): {total_matches_in_dataset}")
pd.DataFrame({'Total Unique Matches': [total_matches_in_dataset]}).to_csv(os.path.join(EDA_OUTPUT_DIR, 'total_unique_matches.csv'), index=False)

# ==============II. TARGET VARIABLE EXPLORATION=========================
# CALCULATION:
# Counts the occurrences of each outcome in the 'radiant_win' column (0 for Dire win, 1 for Radiant win).
# Calculates the percentage distribution of these outcomes.
# INTERPRETATION:
# Shows the overall win rate balance between Radiant and Dire teams in the dataset.
# Important for understanding if there's a significant skew that might affect model training or interpretation.
# A value close to 50% for each indicates a balanced dataset in terms of outcomes.
# Saves the distribution to CSV.
# ========================================================================
print("\n============== II. TARGET VARIABLE EXPLORATION ==============")
if 'radiant_win' in df_matches.columns:
    plt.figure(figsize=(7, 5)) # Adjusted size
    sns.countplot(data=df_matches, x='radiant_win', hue='radiant_win', palette=['#FF6B6B', '#6ECB63'], legend=False) # Custom palette
    plt.title('Distribution of Match Outcomes', fontsize=14)
    plt.xlabel('Match Outcome', fontsize=12); plt.ylabel('Number of Matches', fontsize=12)
    plt.xticks([0, 1], ['Dire Win', 'Radiant Win'], fontsize=10)
    plot_path = os.path.join(EDA_OUTPUT_DIR, 'target_variable_distribution.png')
    plt.savefig(plot_path, bbox_inches='tight'); plt.close() # Added bbox_inches
    print(f"Target variable distribution plot saved to {plot_path}")

    match_outcome_distribution = df_matches['radiant_win'].value_counts(normalize=True).mul(100).round(2)
    print("Match Outcome Distribution (%):")
    print(match_outcome_distribution.astype(str) + '%')
    match_outcome_distribution.to_frame(name='Percentage').rename(index={0: 'Dire Win', 1: 'Radiant Win'}).rename_axis('Outcome').reset_index().to_csv(os.path.join(EDA_OUTPUT_DIR, 'match_outcome_distribution_percentage.csv'), index=False)
    df_matches['radiant_win'].value_counts().to_frame(name='Count').rename(index={0: 'Dire Win', 1: 'Radiant Win'}).rename_axis('Outcome').reset_index().to_csv(os.path.join(EDA_OUTPUT_DIR, 'match_outcome_distribution_counts.csv'), index=False)

else:
    print("ERROR: 'radiant_win' column not found in matches data.")

# ==============III. PICK/BAN PHASE EXPLORATION=======================
# CALCULATION: Various metrics based on pick/ban actions from `df_picks_bans`.
# This includes mapping hero IDs to names, then calculating frequencies and rates
# for bans, picks, and overall draft contest (pick+ban). Outputs metrics to CSV.
# INTERPRETATION: Reveals insights into the draft meta: which heroes are most valued (picked),
# feared (banned), or frequently involved in the draft (contested).
# ========================================================================
print("\n============== III. PICK/BAN PHASE EXPLORATION ==============")
if not df_picks_bans.empty and all(col in df_picks_bans.columns for col in ['hero_id', 'is_pick', 'team', 'order']):
    df_picks_bans['hero_name'] = df_picks_bans['hero_id'].map(hero_id_to_name).fillna('Unknown_Hero')
    df_picks_bans_known_heroes = df_picks_bans[df_picks_bans['hero_name'] != 'Unknown_Hero'].copy()

    df_picks_all = df_picks_bans_known_heroes[df_picks_bans_known_heroes['is_pick']].copy()
    df_bans_all = df_picks_bans_known_heroes[~df_picks_bans_known_heroes['is_pick']].copy()
    print(f"\nTotal Pick actions (known heroes): {len(df_picks_all)}, Total Ban actions (known heroes): {len(df_bans_all)}")
    pd.DataFrame({'Action': ['Picks (Known Heroes)', 'Bans (Known Heroes)'], 'Count': [len(df_picks_all), len(df_bans_all)]}).to_csv(os.path.join(EDA_OUTPUT_DIR, 'total_pick_ban_actions.csv'), index=False)


    # --- A. Ban Analysis ---
    print("\n--- Ban Analysis ---")
    banned_hero_counts = df_bans_all['hero_name'].value_counts()
    if not banned_hero_counts.empty:
        print(f"Top {TOP_N_PLOT} Most Banned Heroes (Counts):\n{banned_hero_counts.head(TOP_N_PLOT)}")
        banned_hero_counts.to_frame(name='BanCount').rename_axis('HeroName').reset_index().to_csv(os.path.join(EDA_OUTPUT_DIR, 'banned_hero_counts.csv'), index=False)
        
        plt.figure(figsize=(12, 8)); banned_hero_counts.head(TOP_N_PLOT).plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title(f'Top {TOP_N_PLOT} Most Banned Heroes', fontsize=14); plt.xlabel('Hero Name', fontsize=12); plt.ylabel('Number of Times Banned', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10); plt.yticks(fontsize=10); plt.tight_layout()
        plot_path = os.path.join(EDA_OUTPUT_DIR, 'top_heroes_banned.png'); plt.savefig(plot_path); plt.close()
        print(f"Top heroes banned plot saved to {plot_path}")
        
        hero_ban_rates = (banned_hero_counts / total_matches_in_dataset).sort_values(ascending=False)
        print(f"Top {TOP_N_PLOT} Hero Ban Rates:\n{(hero_ban_rates.head(TOP_N_PLOT) * 100).round(2).astype(str) + '%'}")
        (hero_ban_rates * 100).round(2).to_frame(name='BanRatePercentage').rename_axis('HeroName').reset_index().to_csv(os.path.join(EDA_OUTPUT_DIR, 'hero_ban_rates.csv'), index=False)
    else: print("No ban data for known heroes to analyze."); hero_ban_rates = pd.Series(dtype='float64')

    # --- B. Pick Analysis (from picks_bans.csv) ---
    print("\n--- Pick Analysis (from picks_bans.csv) ---")
    picked_hero_counts_pb = df_picks_all['hero_name'].value_counts()
    if not picked_hero_counts_pb.empty:
        print(f"Top {TOP_N_PLOT} Most Picked Heroes (Counts from picks_bans.csv):\n{picked_hero_counts_pb.head(TOP_N_PLOT)}")
        picked_hero_counts_pb.to_frame(name='PickCount_from_PicksBans').rename_axis('HeroName').reset_index().to_csv(os.path.join(EDA_OUTPUT_DIR, 'picked_hero_counts_from_pb.csv'), index=False)

        plt.figure(figsize=(12, 8)); picked_hero_counts_pb.head(TOP_N_PLOT).plot(kind='bar', color='lightgreen', edgecolor='black')
        plt.title(f'Top {TOP_N_PLOT} Most Picked Heroes (from Picks/Bans Data)', fontsize=14); plt.xlabel('Hero Name', fontsize=12); plt.ylabel('Number of Times Picked', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10); plt.yticks(fontsize=10); plt.tight_layout()
        plot_path = os.path.join(EDA_OUTPUT_DIR, 'top_heroes_picked_from_pb.png'); plt.savefig(plot_path); plt.close()
        print(f"Top heroes picked (from picks/bans) plot saved to {plot_path}")
        
        hero_pick_rates_pb = (picked_hero_counts_pb / total_matches_in_dataset).sort_values(ascending=False)
        print(f"Top {TOP_N_PLOT} Hero Pick Rates (from picks_bans.csv):\n{(hero_pick_rates_pb.head(TOP_N_PLOT) * 100).round(2).astype(str) + '%'}")
        (hero_pick_rates_pb * 100).round(2).to_frame(name='PickRatePercentage_from_PicksBans').rename_axis('HeroName').reset_index().to_csv(os.path.join(EDA_OUTPUT_DIR, 'hero_pick_rates_from_pb.csv'), index=False)
    else: print("No pick data for known heroes from picks_bans.csv."); hero_pick_rates_pb = pd.Series(dtype='float64')

    # --- C. Contested Heroes ---
    print("\n--- Contested Heroes ---")
    if not hero_pick_rates_pb.empty or not ('hero_ban_rates' in locals() and not hero_ban_rates.empty): # Ensure at least one is valid
        hero_contest_counts = (picked_hero_counts_pb.fillna(0) + banned_hero_counts.fillna(0))
        hero_contest_rates = (hero_contest_counts / total_matches_in_dataset).sort_values(ascending=False)
        print(f"Top {TOP_N_PLOT} Most Contested Heroes (Pick+Ban Rate):\n{(hero_contest_rates.head(TOP_N_PLOT) * 100).round(2).astype(str) + '%'}")
        df_contest_summary = pd.DataFrame({
            'HeroName': hero_contest_rates.index,
            'ContestCount': hero_contest_counts.reindex(hero_contest_rates.index).values,
            'ContestRatePercentage': (hero_contest_rates.values * 100).round(2)
        })
        df_contest_summary.to_csv(os.path.join(EDA_OUTPUT_DIR, 'hero_contest_rates.csv'), index=False)
        
        df_pick_ban_rates_scatter = pd.DataFrame({
            'pick_rate': hero_pick_rates_pb, 
            'ban_rate': hero_ban_rates if 'hero_ban_rates' in locals() else pd.Series(dtype='float64') # ensure hero_ban_rates exists
        }).reindex(hero_pick_rates_pb.index.union(hero_ban_rates.index if 'hero_ban_rates' in locals() and not hero_ban_rates.empty else pd.Index([]))).fillna(0)
        
        if not df_pick_ban_rates_scatter.empty:
            df_pick_ban_rates_scatter.rename_axis('HeroName').reset_index().to_csv(os.path.join(EDA_OUTPUT_DIR, 'hero_pick_vs_ban_rate_data.csv'), index=False)
            plt.figure(figsize=(10, 10)); sns.scatterplot(x=df_pick_ban_rates_scatter['pick_rate']*100, y=df_pick_ban_rates_scatter['ban_rate']*100, data=df_pick_ban_rates_scatter, s=70, alpha=0.7)
            plt.title('Hero Pick Rate vs. Ban Rate', fontsize=14); plt.xlabel('Pick Rate (%)', fontsize=12); plt.ylabel('Ban Rate (%)', fontsize=12); plt.grid(True); plt.tight_layout()
            plot_path = os.path.join(EDA_OUTPUT_DIR, 'hero_pick_vs_ban_rate.png'); plt.savefig(plot_path); plt.close(); print(f"Hero Pick vs. Ban Rate scatter plot saved to {plot_path}")
    else: print("Not enough pick/ban data to analyze contested heroes.")

    # --- D. Draft Order Analysis ---
    print("\n--- D. Draft Order Analysis ---")
    if 'order' in df_picks_bans_known_heroes.columns and 'team' in df_picks_bans_known_heroes.columns:
        max_order = df_picks_bans_known_heroes['order'].max()
        print(f"Maximum draft order value found (known heroes): {max_order}")
        pd.DataFrame({'Max Draft Order': [max_order]}).to_csv(os.path.join(EDA_OUTPUT_DIR, 'max_draft_order.csv'), index=False)


        df_pb_temp = df_picks_bans_known_heroes.copy()
        def map_action_type_detailed(row):
            team_str = "Radiant" if row['team'] == 0 else ("Dire" if row['team'] == 1 else f"Team_{row['team']}")
            action_str = "Pick" if row['is_pick'] else "Ban"
            return f"{team_str} {action_str}"
        df_pb_temp['action_type_full'] = df_pb_temp.apply(map_action_type_detailed, axis=1)

        action_order_counts = df_pb_temp.groupby(['order', 'action_type_full']).size().unstack(fill_value=0)
        action_type_legend_order = ['Radiant Pick', 'Dire Pick', 'Radiant Ban', 'Dire Ban']
        action_order_counts = action_order_counts.reindex(columns=[col for col in action_type_legend_order if col in action_order_counts.columns], fill_value=0)
        
        if not action_order_counts.empty:
            action_order_counts.rename_axis('DraftOrder').reset_index().to_csv(os.path.join(EDA_OUTPUT_DIR, 'draft_actions_by_order_team_type_counts.csv'), index=False)
            action_order_counts.plot(kind='bar', stacked=True, figsize=(18,10), colormap='Spectral', edgecolor='grey')
            plt.title('Distribution of Draft Actions by Order, Team, and Type (Counts)', fontsize=14)
            plt.xlabel(f'Draft Order (0 = first action, up to {max_order})', fontsize=12); plt.ylabel('Count of Actions', fontsize=12)
            plt.xticks(rotation=90, fontsize=10); plt.yticks(fontsize=10); plt.legend(title='Action Type', bbox_to_anchor=(1.02, 1), loc='upper left')
            plt.tight_layout(rect=[0, 0, 0.85, 1]); plot_path = os.path.join(EDA_OUTPUT_DIR, 'draft_actions_by_order_team_type_counts.png'); plt.savefig(plot_path); plt.close()
            print(f"Detailed draft flow (counts) plot saved to {plot_path}")

        print("\n--- Timeline-Style Dominant Action by Draft Order (Visual Flow) ---")
        if not action_order_counts.empty:
            draft_flow_plot_data = []
            all_orders_index_timeline = pd.RangeIndex(start=0, stop=max_order + 1, name='order')
            for order_val in all_orders_index_timeline:
                if order_val in action_order_counts.index:
                    dominant_action_series = action_order_counts.loc[order_val]
                    dominant_action_type = dominant_action_series.idxmax() if dominant_action_series.sum() > 0 else "None"
                else: dominant_action_type = "None"
                team_acting = "Radiant" if "Radiant" in dominant_action_type else ("Dire" if "Dire" in dominant_action_type else "None")
                is_a_pick = "Pick" in dominant_action_type if team_acting != "None" else False
                draft_flow_plot_data.append({'order': order_val, 'team': team_acting, 'is_pick': is_a_pick, 'dominant_action_type': dominant_action_type})
            
            df_draft_flow = pd.DataFrame(draft_flow_plot_data)
            df_draft_flow.to_csv(os.path.join(EDA_OUTPUT_DIR, 'draft_timeline_flow_data.csv'), index=False)

            fig, ax = plt.subplots(figsize=(max(15, max_order * 0.6), 4)) 
            y_radiant = 1.5; y_dire = 0.5; bar_height = 0.8; bar_width = 0.9
            timeline_color_map = {
                ('Radiant', False): ('#ADD8E6', 0.7, '#6495ED'), 
                ('Dire', False): ('#FFA07A', 0.7, '#DC143C'),   
                ('Radiant', True): ('#0000FF', 1.0, '#00008B'), 
                ('Dire', True): ('#FF0000', 1.0, '#8B0000')    
            }
            for _, row in df_draft_flow.iterrows():
                order, team, is_pick_action = row['order'], row['team'], row['is_pick']
                if team == 'None': continue
                y_pos = y_radiant if team == "Radiant" else y_dire
                color, alpha, edgecolor = timeline_color_map.get((team, is_pick_action), ('lightgrey', 0.3, 'grey'))
                rect = mpatches.Rectangle((order - bar_width/2, y_pos - bar_height/2), bar_width, bar_height, 
                                          facecolor=color, alpha=alpha, edgecolor=edgecolor, linewidth=0.7)
                ax.add_patch(rect)
            ax.set_xlim(-0.5, max_order + 0.5); ax.set_ylim(0, 2.3); ax.set_xticks(np.arange(0, max_order + 1, 1.0))
            ax.set_xlabel("Draft Order (0 = first action)", fontsize=12)
            ax.set_yticks([y_dire, y_radiant]); ax.set_yticklabels(["Dire", "Radiant"], fontsize=10); ax.tick_params(axis='y', length=0)
            for spine_pos in ['left', 'right', 'top']: ax.spines[spine_pos].set_visible(False)
            ax.spines['bottom'].set_position(('outward', 10)); ax.grid(axis='x', linestyle=':', alpha=0.5)
            legend_elements = [
                mpatches.Patch(facecolor=timeline_color_map[('Radiant',False)][0], alpha=timeline_color_map[('Radiant',False)][1], edgecolor=timeline_color_map[('Radiant',False)][2], label='Radiant Ban'),
                mpatches.Patch(facecolor=timeline_color_map[('Dire',False)][0], alpha=timeline_color_map[('Dire',False)][1], edgecolor=timeline_color_map[('Dire',False)][2], label='Dire Ban'),
                mpatches.Patch(facecolor=timeline_color_map[('Dire',True)][0], alpha=timeline_color_map[('Dire',True)][1], edgecolor=timeline_color_map[('Dire',True)][2], label='Dire Pick'),
                mpatches.Patch(facecolor=timeline_color_map[('Radiant',True)][0], alpha=timeline_color_map[('Radiant',True)][1], edgecolor=timeline_color_map[('Radiant',True)][2], label='Radiant Pick')
            ]
            ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=4, frameon=False, fontsize=10)
            plt.title("Timeline of Dominant Draft Actions by Team", fontsize=14, pad=20); plt.tight_layout(rect=[0, 0.1, 1, 0.95])
            plot_path = os.path.join(EDA_OUTPUT_DIR, 'draft_timeline_visualization.png'); plt.savefig(plot_path, dpi=150); plt.close()
            print(f"Draft timeline visualization saved to {plot_path}")
        else:
            print("action_order_counts was empty. Skipping timeline draft plot.")

        print("\n--- Investigation of Specific Draft Order Compositions (Percentages) ---")
        if not action_order_counts.empty:
            action_order_proportions_for_debug = action_order_counts.apply(lambda x: (x / x.sum() * 100) if x.sum() > 0 else x, axis=1).fillna(0)
            action_order_proportions_for_debug.rename_axis('DraftOrder').reset_index().to_csv(os.path.join(EDA_OUTPUT_DIR, 'draft_order_action_proportions.csv')) # Save the full proportions table
            
            investigate_orders = sorted(list(set([0, 1, 2, 3, 6, 7, 8, 9, max_order-1 if max_order >0 else 0, max_order])))
            specific_order_investigation_list = []
            for order_val_to_check in investigate_orders:
                if 0 <= order_val_to_check <= max_order and order_val_to_check in action_order_proportions_for_debug.index:
                    print(f"\n--- Detailed Actions at Order {order_val_to_check} (Percentages) ---")
                    order_details = action_order_proportions_for_debug.loc[order_val_to_check][action_order_proportions_for_debug.loc[order_val_to_check] > 0.01].sort_values(ascending=False)
                    print(order_details)
                    for action_type, percentage in order_details.items():
                        specific_order_investigation_list.append({'DraftOrder': order_val_to_check, 'ActionType': action_type, 'Percentage': percentage})
            if specific_order_investigation_list:
                pd.DataFrame(specific_order_investigation_list).to_csv(os.path.join(EDA_OUTPUT_DIR, 'specific_draft_order_investigation.csv'), index=False)
        
        print(f"\n--- Analyzing Picks/Bans by User-Defined Draft Phases ---")
        phase_bins = [-1, 8, 17, max_order] 
        phase_labels = ["First Phase (0-8)", "Second Phase (9-17)", f"Third Phase (18-{max_order})"]
        if max_order < 18: 
            if max_order <= 8: phase_bins = [-1, max_order]; phase_labels = [f"First Phase (0-{max_order})"]
            else: phase_bins = [-1, 8, max_order]; phase_labels = ["First Phase (0-8)", f"Second Phase (9-{max_order})"]
        print(f"Using Phases: {phase_labels} with right bin edges {phase_bins[1:]}")
        pd.DataFrame({'PhaseLabels': phase_labels, 'PhaseBins_RightEdge': phase_bins[1:]}).to_csv(os.path.join(EDA_OUTPUT_DIR, 'draft_phase_definitions.csv'), index=False)


        if len(phase_bins) -1 == len(phase_labels): 
            df_pb_temp['draft_stage_user'] = pd.cut(df_pb_temp['order'], bins=phase_bins, labels=phase_labels, right=True, include_lowest=True)
            print(f"Value counts for draft_stage_user:\n{df_pb_temp['draft_stage_user'].value_counts().sort_index()}")
            df_pb_temp['draft_stage_user'].value_counts().sort_index().to_frame(name='Count').rename_axis('DraftStage').reset_index().to_csv(os.path.join(EDA_OUTPUT_DIR, 'draft_stage_user_value_counts.csv'), index=False)
            
            all_phase_analysis_data = []
            for stage in phase_labels:
                for team_val, team_name_str in [(0, "Radiant"), (1, "Dire")]: 
                    stage_team_picks_df = df_pb_temp[(df_pb_temp['draft_stage_user'] == stage) & (df_pb_temp['team'] == team_val) & (df_pb_temp['is_pick'])]
                    stage_team_picks = stage_team_picks_df['hero_name'].value_counts().head(TOP_N_PLOT // 2)
                    if not stage_team_picks.empty:
                        print(f"\n--- Top {team_name_str} Picks in {stage} ({len(stage_team_picks_df)} total) ---"); print(stage_team_picks)
                        stage_team_picks_df_out = stage_team_picks.reset_index()
                        stage_team_picks_df_out.columns = ['HeroName', 'PickCount']
                        stage_team_picks_df_out['Team'] = team_name_str
                        stage_team_picks_df_out['Phase'] = stage
                        stage_team_picks_df_out['Action'] = 'Pick'
                        all_phase_analysis_data.append(stage_team_picks_df_out)

                        plt.figure(figsize=(10,6)); stage_team_picks.plot(kind='bar', color='mediumseagreen' if team_name_str == "Radiant" else "tomato", edgecolor='black')
                        plt.title(f'Top {team_name_str} Picks - {stage}', fontsize=14); plt.xlabel('Hero Name', fontsize=12); plt.ylabel('Number of Picks', fontsize=12); plt.xticks(rotation=45, ha='right', fontsize=10); plt.yticks(fontsize=10); plt.tight_layout(); plot_path = os.path.join(EDA_OUTPUT_DIR, f'top_{team_name_str.lower()}_picks_{stage.lower().replace(" ","_").replace("(","").replace(")","").replace("-","_to_")}_draft.png'); plt.savefig(plot_path); plt.close(); print(f"Plot for top {team_name_str} picks in {stage} draft saved to {plot_path}")
                    
                    stage_team_bans_df = df_pb_temp[(df_pb_temp['draft_stage_user'] == stage) & (df_pb_temp['team'] == team_val) & (~df_pb_temp['is_pick'])]
                    stage_team_bans = stage_team_bans_df['hero_name'].value_counts().head(TOP_N_PLOT // 2)
                    if not stage_team_bans.empty:
                        print(f"\n--- Top {team_name_str} Bans in {stage} ({len(stage_team_bans_df)} total) ---"); print(stage_team_bans)
                        stage_team_bans_df_out = stage_team_bans.reset_index()
                        stage_team_bans_df_out.columns = ['HeroName', 'BanCount']
                        stage_team_bans_df_out['Team'] = team_name_str
                        stage_team_bans_df_out['Phase'] = stage
                        stage_team_bans_df_out['Action'] = 'Ban'
                        all_phase_analysis_data.append(stage_team_bans_df_out)

                        plt.figure(figsize=(10,6)); stage_team_bans.plot(kind='bar', color='lightsteelblue' if team_name_str == "Radiant" else "lightpink", edgecolor='black')
                        plt.title(f'Top {team_name_str} Bans - {stage}', fontsize=14); plt.xlabel('Hero Name', fontsize=12); plt.ylabel('Number of Bans', fontsize=12); plt.xticks(rotation=45, ha='right', fontsize=10); plt.yticks(fontsize=10); plt.tight_layout(); plot_path = os.path.join(EDA_OUTPUT_DIR, f'top_{team_name_str.lower()}_bans_{stage.lower().replace(" ","_").replace("(","").replace(")","").replace("-","_to_")}_draft.png'); plt.savefig(plot_path); plt.close(); print(f"Plot for top {team_name_str} bans in {stage} draft saved to {plot_path}")
            if all_phase_analysis_data:
                df_all_phase_analysis = pd.concat(all_phase_analysis_data, ignore_index=True)
                df_all_phase_analysis.to_csv(os.path.join(EDA_OUTPUT_DIR, 'draft_phase_hero_picks_bans.csv'), index=False)
        else: print("Bin/label mismatch for user-defined phases. Skipping phased analysis.")
    else: print("'order', 'team', or 'is_pick' column not found in picks_bans data. Skipping detailed Draft Order Analysis.")
else: print("Picks/Bans data not available or key columns missing. Skipping most of Pick/Ban Phase Exploration.")


# ==============IV. HEROES DATA EXPLORATION (INDIVIDUAL PERFORMANCE)======
# CALCULATION: Hero Win Rate = (Games Won by Hero's Team) / (Total Games Played by Hero).
# Also merges with pick/ban rates for a combined view. Outputs metrics to CSV.
# INTERPRETATION: Shows which heroes tend to win more often when they are actually in the game,
# and how this relates to their popularity/avoidance in the draft phase.
print("\n============== IV. HEROES DATA EXPLORATION (INDIVIDUAL PERFORMANCE) ==============")
if 'hero_id' in df_players.columns:
    if 'hero_name' not in df_players.columns: df_players['hero_name'] = df_players['hero_id'].map(hero_id_to_name).fillna('Unknown_Hero')
    df_players_known_heroes = df_players[df_players['hero_name'] != 'Unknown_Hero'].copy()
    df_player_matches_perf = pd.merge(df_players_known_heroes[['match_id', 'hero_name', 'isRadiant']], df_matches[['match_id', 'radiant_win']], on='match_id', how='left')
    
    if not df_player_matches_perf.empty and 'radiant_win' in df_player_matches_perf.columns and 'hero_name' in df_player_matches_perf:
        df_player_matches_perf.dropna(subset=['radiant_win', 'hero_name'], inplace=True)
        df_player_matches_perf['player_team_won'] = ((df_player_matches_perf['isRadiant'] == 1) & (df_player_matches_perf['radiant_win'] == 1)) | ((df_player_matches_perf['isRadiant'] == 0) & (df_player_matches_perf['radiant_win'] == 0))
        df_player_matches_perf['player_team_won'] = df_player_matches_perf['player_team_won'].astype(int)
        
        hero_win_stats = df_player_matches_perf.groupby('hero_name')['player_team_won'].agg(wins='sum', games_played='count').reset_index()
        hero_win_stats_filtered = hero_win_stats[hero_win_stats['games_played'] >= MIN_GAMES_THRESHOLD_HERO].copy()
        
        if not hero_win_stats_filtered.empty:
            hero_win_stats_filtered['win_rate'] = hero_win_stats_filtered['wins'] / hero_win_stats_filtered['games_played']
            hero_win_stats_filtered = hero_win_stats_filtered.sort_values('win_rate', ascending=False)
            print(f"\nTop {TOP_N_PLOT} Heroes by Win Rate (when picked & played, min {MIN_GAMES_THRESHOLD_HERO} games):\n{hero_win_stats_filtered.head(TOP_N_PLOT)}")
            hero_win_stats_filtered.to_csv(os.path.join(EDA_OUTPUT_DIR, 'hero_win_rates_when_picked.csv'), index=False)
            
            if 'hero_pick_rates_pb' in locals() and 'hero_ban_rates' in locals() and isinstance(hero_pick_rates_pb, pd.Series) and isinstance(hero_ban_rates, pd.Series):
                df_hero_summary = hero_win_stats_filtered.set_index('hero_name').copy()
                all_drafted_heroes_index = hero_pick_rates_pb.index.union(hero_ban_rates.index if 'hero_ban_rates' in locals() and isinstance(hero_ban_rates, pd.Series) else pd.Index([]))
                df_hero_summary = df_hero_summary.reindex(all_drafted_heroes_index) # Reindex to include all heroes from pick/ban phase for comprehensive summary
                
                # Fill missing win_rate data for heroes that might have been drafted but not met MIN_GAMES_THRESHOLD_HERO
                # Or re-fetch win_stats for all drafted heroes if that's preferred (current logic prioritizes filtered win_stats)
                
                df_hero_summary['wins'] = df_hero_summary['wins'].fillna(0).astype(int)
                df_hero_summary['games_played'] = df_hero_summary['games_played'].fillna(0).astype(int)
                # Recalculate win_rate for all, or only for those where games_played is now non-zero after fillna if applicable
                df_hero_summary['win_rate'] = np.where(df_hero_summary['games_played'] > 0, df_hero_summary['wins'] / df_hero_summary['games_played'], 0.0).astype(float)
                
                df_hero_summary['pick_rate'] = df_hero_summary.index.map(hero_pick_rates_pb).fillna(0)
                df_hero_summary['ban_rate'] = df_hero_summary.index.map(hero_ban_rates if 'hero_ban_rates' in locals() else pd.Series(dtype='float64')).fillna(0)
                
                if 'picked_hero_counts_pb' in locals() and isinstance(picked_hero_counts_pb, pd.Series): 
                    df_hero_summary['games_picked_count'] = df_hero_summary.index.map(picked_hero_counts_pb).fillna(0)
                else: 
                    df_hero_summary['games_picked_count'] = (df_hero_summary['pick_rate'] * total_matches_in_dataset).round().astype(int)
                
                df_hero_summary.reset_index(inplace=True) # hero_name becomes a column
                df_hero_summary.rename(columns={'index':'hero_name'}, inplace=True) # ensure column name is hero_name

                if not df_hero_summary.empty:
                    df_hero_summary.to_csv(os.path.join(EDA_OUTPUT_DIR, 'hero_summary_win_pick_ban_rates.csv'), index=False)
                    plt.figure(figsize=(14, 10))
                    sizes_for_plot = df_hero_summary['games_picked_count'] + MIN_DRAFT_OCCURRENCES / 2
                    sizes_for_plot = np.maximum(sizes_for_plot.fillna(10), 10) 
                    sns.scatterplot(x='pick_rate', y='win_rate', data=df_hero_summary, hue='ban_rate', size=sizes_for_plot, sizes=(30, 700), palette="viridis_r", legend='auto', alpha=0.7)
                    plt.title('Hero Win Rate vs. Pick Rate (Colored by Ban Rate, Sized by Games Picked)', fontsize=14); plt.xlabel('Pick Rate (Overall)', fontsize=12); plt.ylabel('Win Rate (When Picked & Played)', fontsize=12); plt.axhline(0.5, color='grey', linestyle='--'); plt.grid(True); plt.tight_layout()
                    plot_path = os.path.join(EDA_OUTPUT_DIR, 'hero_win_vs_pick_rate_detailed.png'); plt.savefig(plot_path); plt.close(); print(f"Detailed Hero Win Rate vs. Pick Rate plot saved to {plot_path}")
            else: print("df_hero_summary could not be created or is empty for win vs pick rate plot.")
        else: print("Pick/Ban rates from Section III not available or not valid Series for combined plot with win rates.")
    else: print(f"No heroes met min games threshold ({MIN_GAMES_THRESHOLD_HERO}) for win rate stats.")
else: print("Could not calculate hero win rates (df_players or df_player_matches_perf empty or missing key columns).")


# ==============V. ITEMS DATA EXPLORATION (FINAL ITEMS)=================
# CALCULATION: Item Frequency = Count of item in all final inventories.
# Player Win Rate When Item Held = (Wins by players holding item) / (Total occurrences of item held by players).
# INTERPRETATION: Shows popular final items and their strong (leaky) correlation with player wins. Outputs to CSV.
print("\n============== V. ITEMS DATA EXPLORATION (FINAL ITEMS) ==============")
if 'hero_id' in df_players.columns: 
    if 'hero_name' not in df_players.columns: df_players['hero_name'] = df_players['hero_id'].map(hero_id_to_name).fillna('Unknown_Hero')
    item_cols_for_melt = [f'item_{i}' for i in range(6)]
    df_items_melted = df_players.melt(id_vars=['match_id', 'player_slot', 'isRadiant'], value_vars=item_cols_for_melt, var_name='item_slot_num', value_name='item_id')
    df_items_melted = df_items_melted[df_items_melted['item_id'] != 0]
    df_items_melted['item_name'] = df_items_melted['item_id'].map(item_id_to_name).fillna('Unknown_Item')
    df_items_melted = df_items_melted[df_items_melted['item_name'] != 'Unknown_Item']
    
    print("\n--- Final Item Frequency ---")
    item_counts = df_items_melted['item_name'].value_counts()
    if not item_counts.empty:
        print(f"Top {TOP_N_PLOT} Most Frequent Final Items:\n{item_counts.head(TOP_N_PLOT)}")
        item_counts.to_frame(name='Frequency').rename_axis('ItemName').reset_index().to_csv(os.path.join(EDA_OUTPUT_DIR, 'final_item_frequency.csv'), index=False)
        
        plt.figure(figsize=(12, 8)); item_counts.head(TOP_N_PLOT).plot(kind='bar', color='coral', edgecolor='black')
        plt.title(f'Top {TOP_N_PLOT} Most Frequent Final Items', fontsize=14); plt.xlabel('Item Name', fontsize=12); plt.ylabel('Times in Final Inventories', fontsize=12); plt.xticks(rotation=45, ha='right', fontsize=10); plt.yticks(fontsize=10); plt.tight_layout(); plot_path = os.path.join(EDA_OUTPUT_DIR, 'top_final_items_frequency.png'); plt.savefig(plot_path); plt.close(); print(f"Top final items frequency plot saved to {plot_path}")
    else: print("No item data found after melting.")
    
    print("\n--- Win Rates with Key Final Items (Leakage Indication) ---")
    df_item_matches = pd.merge(df_items_melted[['match_id', 'item_name', 'isRadiant']], df_matches[['match_id', 'radiant_win']], on='match_id', how='left')
    if not df_item_matches.empty and 'radiant_win' in df_item_matches.columns and 'item_name' in df_item_matches:
        df_item_matches.dropna(subset=['radiant_win', 'item_name'], inplace=True)
        df_item_matches['player_team_won'] = ((df_item_matches['isRadiant'] == 1) & (df_item_matches['radiant_win'] == 1)) | ((df_item_matches['isRadiant'] == 0) & (df_item_matches['radiant_win'] == 0))
        df_item_matches['player_team_won'] = df_item_matches['player_team_won'].astype(int)
        
        item_win_stats = df_item_matches.groupby('item_name')['player_team_won'].agg(wins_when_held='sum', occurrences='count').reset_index()
        item_win_stats_filtered = item_win_stats[item_win_stats['occurrences'] >= MIN_GAMES_THRESHOLD_ITEM].copy()
        
        if not item_win_stats_filtered.empty:
            item_win_stats_filtered['win_rate_when_held'] = item_win_stats_filtered['wins_when_held'] / item_win_stats_filtered['occurrences']
            item_win_stats_filtered = item_win_stats_filtered.sort_values('win_rate_when_held', ascending=False)
            print(f"\nTop {TOP_N_PLOT} Final Items by Player Win Rate when Held (min {MIN_GAMES_THRESHOLD_ITEM} occurrences):\n{item_win_stats_filtered.head(TOP_N_PLOT)}")
            item_win_stats_filtered.to_csv(os.path.join(EDA_OUTPUT_DIR, 'final_item_win_rates_when_held.csv'), index=False)
            top_n_items_by_win_rate = item_win_stats_filtered.head(TOP_N_PLOT)

            if not top_n_items_by_win_rate.empty:
                # Save a dedicated CSV for the Top N items by player win rate
                csv_path_top_n_win_rate = os.path.join(EDA_OUTPUT_DIR, f'top_{TOP_N_PLOT}_final_items_by_player_win_rate.csv')
                top_n_items_by_win_rate.to_csv(csv_path_top_n_win_rate, index=False)
                print(f"Data for top {TOP_N_PLOT} items by player win rate (when held) saved to {csv_path_top_n_win_rate}")

                # Plotting these Top N items by player win rate
                plt.figure(figsize=(12, 8))
                # Data for plotting is already sorted as top_n_items_by_win_rate is from the head of a sorted DataFrame
                plot_data_top_win_rate = top_n_items_by_win_rate.set_index('item_name')['win_rate_when_held']
                plot_data_top_win_rate.plot(kind='bar', color='mediumpurple', edgecolor='black') # You can choose any color
                plt.title(f'Top {TOP_N_PLOT} Final Items by Player Win Rate (Min {MIN_GAMES_THRESHOLD_ITEM} Occurrences)', fontsize=14)
                plt.xlabel('Item Name', fontsize=12)
                plt.ylabel('Win Rate When Item is Held', fontsize=12)
                plt.xticks(rotation=45, ha='right', fontsize=10)
                plt.yticks(fontsize=10)
                plt.axhline(0.5, color='grey', linestyle='--')
                plt.tight_layout()
                plot_path_top_win_rate = os.path.join(EDA_OUTPUT_DIR, f'top_{TOP_N_PLOT}_items_by_win_rate_when_held.png')
                plt.savefig(plot_path_top_win_rate); plt.close()
                print(f"Plot for top {TOP_N_PLOT} items by player win rate (when held) saved to {plot_path_top_win_rate}")
                print("NOTE: High win rates for final items strongly indicate data leakage (as noted before).")
            else:
                print(f"No items to plot or save for top {TOP_N_PLOT} by player win rate (data was empty after filtering and taking head).")

        else: # This 'else' corresponds to 'if not item_win_stats_filtered.empty:'
            print(f"No items met min games threshold ({MIN_GAMES_THRESHOLD_ITEM}) for item win rate calc, so no top items plot.")
    else: # This 'else' corresponds to 'if not df_item_matches.empty and ...'
        print("Could not calculate item win rates (df_item_matches empty or missing key columns).")
else: # This 'else' corresponds to 'if 'hero_id' in df_players.columns:'
    print("Players data (df_players) is missing, skipping item exploration.")


# ==============VI. HERO COMBINATION INSIGHTS (ALLY PAIRS)===============
# CALCULATION: Ally Pair Win Rate = (Games Won by Team with Pair) / (Total Games Played by Pair on Same Team).
# INTERPRETATION: Identifies hero duos that have a high success rate when drafted together, suggesting synergy.
# Requires careful handling of minimum occurrences for reliability. Outputs to CSV.
print("\n============== VI. HERO COMBINATION INSIGHTS (ALLY PAIRS) ==============")
if 'df_player_matches_perf' in locals() and isinstance(df_player_matches_perf, pd.DataFrame) and not df_player_matches_perf.empty and 'hero_name' in df_player_matches_perf.columns and 'isRadiant' in df_player_matches_perf.columns and 'radiant_win' in df_player_matches_perf.columns:
    match_teams_heroes = df_player_matches_perf.groupby(['match_id', 'isRadiant'])['hero_name'].apply(lambda x: sorted(list(set(x)))).reset_index()
    ally_pairs_data = []
    for _, row in match_teams_heroes.iterrows():
        match_id, is_radiant_team, team_heroes = row['match_id'], row['isRadiant'] == 1, row['hero_name']
        current_match_info = df_matches[df_matches['match_id'] == match_id]
        if current_match_info.empty: continue
        radiant_won_match = current_match_info['radiant_win'].iloc[0] == 1
        team_won = (is_radiant_team and radiant_won_match) or (not is_radiant_team and not radiant_won_match)
        if len(team_heroes) >= 2:
            for pair_tuple in combinations(team_heroes, 2): ally_pairs_data.append({'hero1': pair_tuple[0], 'hero2': pair_tuple[1], 'team_won': team_won, 'match_id':match_id}) # Added match_id for reference
    
    if ally_pairs_data:
        df_ally_pairs = pd.DataFrame(ally_pairs_data)
        df_ally_pairs['pair_tuple'] = df_ally_pairs.apply(lambda r: tuple(sorted((r['hero1'], r['hero2']))), axis=1)
        
        pair_win_stats = df_ally_pairs.groupby('pair_tuple')['team_won'].agg(wins='sum', games_played='count').reset_index()
        pair_win_stats_filtered = pair_win_stats[pair_win_stats['games_played'] >= MIN_GAMES_THRESHOLD_HERO_PAIR].copy()
        
        if not pair_win_stats_filtered.empty:
            pair_win_stats_filtered['win_rate'] = pair_win_stats_filtered['wins'] / pair_win_stats_filtered['games_played']
            pair_win_stats_filtered = pair_win_stats_filtered.sort_values('win_rate', ascending=False)
            pair_win_stats_filtered['pair_str'] = pair_win_stats_filtered['pair_tuple'].apply(lambda x: f"{x[0]} & {x[1]}")
            
            print(f"\nTop {TOP_N_PLOT} Ally Hero Pairs by Win Rate (min {MIN_GAMES_THRESHOLD_HERO_PAIR} games):\n{pair_win_stats_filtered[['pair_str', 'win_rate', 'games_played']].head(TOP_N_PLOT)}")
            pair_win_stats_filtered[['pair_str', 'win_rate', 'wins', 'games_played', 'pair_tuple']].to_csv(os.path.join(EDA_OUTPUT_DIR, 'ally_hero_pair_win_rates.csv'), index=False)

            top_pairs_for_plot = pair_win_stats_filtered.head(TOP_N_PLOT)
            if not top_pairs_for_plot.empty:
                plt.figure(figsize=(12, max(8, len(top_pairs_for_plot) * 0.5))); 
                sns.barplot(x='win_rate', y='pair_str', data=top_pairs_for_plot, hue='pair_str', dodge=False, palette='viridis', legend=False)
                plt.title(f'Top {TOP_N_PLOT} Ally Hero Pair Win Rates (Min {MIN_GAMES_THRESHOLD_HERO_PAIR} Games)', fontsize=14);
                plt.xlabel('Win Rate', fontsize=12); plt.ylabel('Hero Pair', fontsize=12)
                plt.axvline(0.5, color='red', linestyle='--', linewidth=1) 
                plt.xticks(fontsize=10); plt.yticks(fontsize=10); plt.tight_layout()
                plot_path = os.path.join(EDA_OUTPUT_DIR, 'top_ally_hero_pairs_win_rates.png'); plt.savefig(plot_path); plt.close(); print(f"Top ally hero pairs win rates plot saved to {plot_path}")
        else: print(f"No ally hero pairs met min games threshold ({MIN_GAMES_THRESHOLD_HERO_PAIR}).")
    else: print("No ally pair data generated from matches.")
else: print("Skipping Hero Combination analysis as relevant data (df_player_matches_perf) is not suitable or empty.")



# ==============VII. OBSERVED HERO COMBINATION COUNTS & ANALYSIS=======
# CALCULATION:
# Iterates through each team's hero lineup in every match of the dataset.
# Identifies unique pairs (2 heroes), triples (3 heroes), and full teams (5 heroes)
# that actually played together as allies. Uses sets for efficient uniqueness tracking.
# Saves the lists of unique observed combinations to CSV files.
# Plots the counts of these unique observed combinations.
# INTERPRETATION:
# Reveals which specific hero combinations are actually utilized in practice within
# the dataset. Compares the scale of observed combinations to the theoretical maximums
# calculated earlier (if Section VII was run). Shows how many distinct synergistic
# units (pairs, triples) and full team compositions were present in the data.
# This is more directly relevant to understanding the played meta than theoretical counts.
# ========================================================================
print("\n============== VII. OBSERVED HERO COMBINATION COUNTS & ANALYSIS ==============")

# Ensure necessary libraries are available (itertools should be imported already)
# from math import comb # Not needed here as we aren't calculating theoretical max

try:
    # Check if the required DataFrame from Section VI exists
    if 'match_teams_heroes' in locals() and isinstance(match_teams_heroes, pd.DataFrame) and not match_teams_heroes.empty:
        print(f"Processing {len(match_teams_heroes)} team lineups from matches...")

        # Use sets to store unique combinations (canonical form: sorted tuple)
        observed_pairs = set()
        observed_triples = set()
        observed_5_hero_teams = set()

        # Iterate through each team lineup in each match
        for _, row in match_teams_heroes.iterrows():
            # Ensure heroes are sorted to treat (A, B) the same as (B, A)
            team_heroes = sorted(row['hero_name'])
            num_heroes_on_team = len(team_heroes)

            # Generate and add observed pairs (if >= 2 heroes)
            if num_heroes_on_team >= 2:
                for pair in itertools.combinations(team_heroes, 2):
                    observed_pairs.add(pair) # Add tuple directly

            # Generate and add observed triples (if >= 3 heroes)
            if num_heroes_on_team >= 3:
                for triple in itertools.combinations(team_heroes, 3):
                    observed_triples.add(triple)

            # Add observed 5-hero teams (only if exactly 5 heroes)
            if num_heroes_on_team == 5:
                 # The list is already sorted, convert to tuple for set
                observed_5_hero_teams.add(tuple(team_heroes))

        # --- Calculate and Report Counts ---
        n_observed_pairs = len(observed_pairs)
        n_observed_triples = len(observed_triples)
        n_observed_5_hero_teams = len(observed_5_hero_teams)

        print(f"\nFound {n_observed_pairs:,} unique observed allied hero pairs.")
        print(f"Found {n_observed_triples:,} unique observed allied hero triples.")
        print(f"Found {n_observed_5_hero_teams:,} unique observed 5-hero team compositions.")

        # --- Save Observed Combinations to CSV ---
        # Save Pairs
        if n_observed_pairs > 0:
            df_observed_pairs = pd.DataFrame(list(observed_pairs), columns=['Hero_1', 'Hero_2'])
            obs_pairs_filename = os.path.join(EDA_OUTPUT_DIR, 'observed_ally_pair_combinations.csv')
            df_observed_pairs.to_csv(obs_pairs_filename, index=False)
            print(f"Saved unique observed pairs to {obs_pairs_filename}")
        else:
            print("No observed pairs found to save.")

        # Save Triples
        if n_observed_triples > 0:
            df_observed_triples = pd.DataFrame(list(observed_triples), columns=['Hero_1', 'Hero_2', 'Hero_3'])
            obs_triples_filename = os.path.join(EDA_OUTPUT_DIR, 'observed_ally_triple_combinations.csv')
            df_observed_triples.to_csv(obs_triples_filename, index=False)
            print(f"Saved unique observed triples to {obs_triples_filename}")
        else:
            print("No observed triples found to save.")

        # Save 5-Hero Teams
        if n_observed_5_hero_teams > 0:
            df_observed_5_hero_teams = pd.DataFrame(list(observed_5_hero_teams), columns=['Hero_1', 'Hero_2', 'Hero_3', 'Hero_4', 'Hero_5'])
            obs_5teams_filename = os.path.join(EDA_OUTPUT_DIR, 'observed_5hero_team_combinations.csv')
            df_observed_5_hero_teams.to_csv(obs_5teams_filename, index=False)
            print(f"Saved unique observed 5-hero teams to {obs_5teams_filename}")
        else:
            print("No observed 5-hero teams found to save.")

        # --- Plot Observed Counts ---
        print("\nGenerating bar plot of OBSERVED combination counts...")
        observed_counts_data = {
            'Combination Type': ['Observed Pairs', 'Observed Triples', 'Observed 5-Hero Teams'],
            'Number Found': [n_observed_pairs, n_observed_triples, n_observed_5_hero_teams]
        }
        df_observed_counts = pd.DataFrame(observed_counts_data)

        # Save observed counts data to CSV
        obs_counts_filename = os.path.join(EDA_OUTPUT_DIR, 'observed_combination_counts.csv')
        df_observed_counts.to_csv(obs_counts_filename, index=False)
        print(f"Saved observed combination counts data to {obs_counts_filename}")

        # Create the plot
        plt.figure(figsize=(10, 7))
        barplot_observed = sns.barplot(x='Combination Type', y='Number Found', data=df_observed_counts, palette='magma', hue='Combination Type', legend=False, dodge=False)

        # Add counts on top of bars
        for index, row in df_observed_counts.iterrows():
          y_pos = row['Number Found']
          label_text = f"{y_pos:,.0f}" # Comma formatting for readability
          plt.text(index, y_pos, label_text, color='black', ha="center", va='bottom', fontsize=10)

        plt.title(f'Number of Unique OBSERVED Hero Combinations in Dataset', fontsize=14)
        plt.ylabel('Count Found', fontsize=12) # Linear scale likely okay here unless counts vary wildly
        plt.xlabel('Type of Combination', fontsize=12)
        # plt.yscale('log') # Uncomment this if the counts differ by orders of magnitude
        plt.xticks(fontsize=10); plt.yticks(fontsize=10)
        plt.tight_layout() # Adjust layout

        # Save the Plot
        obs_plot_filename = os.path.join(EDA_OUTPUT_DIR, 'observed_combination_counts_plot.png')
        plt.savefig(obs_plot_filename)
        plt.close() # Close the plot
        print(f"Saved observed combination counts bar plot to {obs_plot_filename}")

    else:
        print("ERROR: Could not find 'match_teams_heroes' DataFrame from Section VI or it was empty.")
        print("Skipping observed hero combination analysis.")

except Exception as e:
    print(f"An error occurred during observed combination calculations: {e}")

# End of Section VIII


print(f"\n============== EDA SCRIPT FINISHED ==============")
print(f"All EDA outputs, plots, and CSVs should be saved in: {EDA_OUTPUT_DIR}")