import pandas as pd
import numpy as np
import time
import os
import gc
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, roc_auc_score,
                             roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, # Added
                             precision_recall_fscore_support) # Added
# Use KFold and cross_val_score for basic CV
from sklearn.model_selection import KFold, cross_val_score
# from imblearn.over_sampling import SMOTE # Kept if needed later, but not used in baseline
# Optional: Suppress warnings for cleaner output, use with caution
# import warnings
# warnings.filterwarnings('ignore')

###############################################################################
################################## DOCS #######################################
# This script aims to replicate a Dota 2 match prediction model potentially
# suffering from data leakage by using post-game information (final items).
# Key steps include:
# 1. Loading match and player data.
# 2. Feature Engineering: Creating a feature set with Hero IDs and final Item IDs
#    for each player slot, pivoted into one row per match.
# 3. Data Splitting: Using a time-based split (80% train, 20% test) for the *final* evaluation.
# 4. Cross-Validation: Applying standard K-Fold cross-validation (with shuffle)
#    on the training set for model assessment during development.
# 5. Modeling: Training a baseline Random Forest classifier.
# 6. Evaluation: Assessing model performance using Accuracy, ROC AUC,
#    Classification Report, plotting ROC curve, Confusion Matrix, and feature importances on the hold-out test set.
# 7. Output Saving: Saving intermediate dataframes, final results, plots,
#    and logs into structured directories.
###############################################################################
###############################################################################


################################## DOCS #######################################
# Configuration & Constants: Define file paths, column names, output directories,
# and helper functions/classes (like Colors and logging).
# BASE_DATA_DIR: Root directory containing the input CSV files.
# OUTPUT_DIR: Root directory where all results (CSVs, plots, logs) will be saved.
# SUBFOLDERS: Specific directories within OUTPUT_DIR for organized output.
###############################################################################
# === Configuration & Constants ===
BASE_DATA_DIR = './Data'
# <<< --- Make sure these filenames match your actual data files --- >>>
PLAYERS_FILE = os.path.join(BASE_DATA_DIR, 'player_matches_aug_2024.csv') # Example filename
MATCHES_FILE = os.path.join(BASE_DATA_DIR, 'players_matches.csv') # Example filename
# <<< ------------------------------------------------------------- >>>
CONSTANTS_DIR = os.path.join(BASE_DATA_DIR, 'Constants')
HEROES_FILE = os.path.join(CONSTANTS_DIR, 'Constants.Heroes.csv') # Needed for lookup if using names

# <<< --- Output Folders --- >>>
OUTPUT_DIR = './Baseline_Model' # Changed name slightly
SF_01_LOADED = os.path.join(OUTPUT_DIR, '01_Loaded_Data')
SF_02_PIVOTED = os.path.join(OUTPUT_DIR, '02_Pivoted_Data')
SF_03_FEATURES = os.path.join(OUTPUT_DIR, '03_Combined_Features')
SF_04_SPLIT = os.path.join(OUTPUT_DIR, '04_Split_Data')
SF_05_EVALUATION = os.path.join(OUTPUT_DIR, '05_Evaluation_Outputs')

# Create directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SF_01_LOADED, exist_ok=True)
os.makedirs(SF_02_PIVOTED, exist_ok=True)
os.makedirs(SF_03_FEATURES, exist_ok=True)
os.makedirs(SF_04_SPLIT, exist_ok=True)
os.makedirs(SF_05_EVALUATION, exist_ok=True)


# Columns needed for this specific feature set replication
matches_cols_rep = ['match_id', 'radiant_win', 'start_date_time']
players_cols_rep = ['match_id', 'player_slot', 'hero_id', 'item_0', 'item_1', 'item_2', 'item_3', 'item_4', 'item_5', 'isRadiant']
item_cols_names = [f'item_{i}' for i in range(6)]

# Add color codes for progress markers
class Colors:
    HEADER = '\033[95m'; BLUE = '\033[94m'; GREEN = '\033[92m'
    WARNING = '\033[93m'; FAIL = '\033[91m'; ENDC = '\033[0m'; BOLD = '\033[1m'

# Basic logging function
log_messages = []
def log_and_print(message, color=Colors.BLUE):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S") # Use current system time
    log_msg = f"[{timestamp}] [STATUS] {message}"
    print(f"{color}{log_msg}{Colors.ENDC}")
    log_messages.append(log_msg)

# Function to save logs
def save_log_file(filepath):
    try:
        with open(filepath, 'w') as f:
            for msg in log_messages:
                f.write(msg + '\n')
        log_and_print(f"Log saved to {filepath}", Colors.GREEN)
    except Exception as e:
        log_and_print(f"Error saving log file: {e}", Colors.FAIL)

# Function to safely save CSV
def save_csv(df, path, index=False):
     log_and_print(f"Saving DataFrame to {path}...", Colors.BLUE)
     try:
         df.to_csv(path, index=index)
         log_and_print(f"   Successfully saved {os.path.basename(path)} (Shape: {df.shape})", Colors.GREEN)
     except Exception as e:
         log_and_print(f"   Error saving {os.path.basename(path)}: {e}", Colors.WARNING)

################################## DOCS #######################################
# Feature Engineering Function: `create_leaky_replication_features`
# Takes raw matches and players dataframes.
# 1. Filters necessary columns.
# 2. Cleans data (fills NaNs, converts types).
# 3. Maps `player_slot` to understandable identifiers (e.g., 'radiant_player_1').
# 4. Pivots hero data: Creates columns like `radiant_player_1_hero`, etc.
# 5. Pivots item data: Creates columns like `radiant_player_1_item_1`, etc.
# 6. Joins hero, item, and target (`radiant_win`) data based on `match_id`.
# 7. Ensures all expected columns exist (filling missing ones with 0).
# 8. Saves intermediate pivoted dataframes to the `SF_02_PIVOTED` folder.
# Returns a dataframe where each row is a match and columns represent
# all player hero IDs and final item IDs.
###############################################################################
def create_leaky_replication_features(matches_df, players_df):
    """ Creates the 70-feature set (10 heroes + 60 items) with specific naming. """
    log_and_print("Starting creation of leaky feature set (Heroes + Final Items)...", Colors.HEADER)

    # Select necessary player columns
    player_features = players_df[players_cols_rep].copy()
    log_and_print(f"   Initial player data shape: {player_features.shape}")

    # Fill NaN item IDs with 0 & ensure int type
    for col in item_cols_names:
        player_features[col] = player_features[col].fillna(0).astype(int)
    player_features['hero_id'] = player_features['hero_id'].fillna(0).astype(int)
    player_features['player_slot'] = player_features['player_slot'].astype(int)
    log_and_print("   Filled NaN item/hero IDs and ensured integer types.")

    # Determine player identifiers (radiant_player_1 to _5, dire_player_1 to _5)
    slot_map = {
        0: 'radiant_player_1', 1: 'radiant_player_2', 2: 'radiant_player_3', 3: 'radiant_player_4', 4: 'radiant_player_5',
        128: 'dire_player_1', 129: 'dire_player_2', 130: 'dire_player_3', 131: 'dire_player_4', 132: 'dire_player_5'
    }
    player_features['player_id_mapped'] = player_features['player_slot'].map(slot_map)
    mapped_count = player_features['player_id_mapped'].notna().sum()
    original_count = len(player_features)
    # Drop rows where mapping failed (unexpected player_slot values)
    player_features = player_features.dropna(subset=['player_id_mapped'])
    log_and_print(f"   Mapped player slots to identifiers ({mapped_count}/{original_count} mapped successfully).")

    # --- Pivot to get one row per match ---
    log_and_print("   Pivoting data to get one row per match...", Colors.BLUE)
    # Pivot hero features
    hero_pivot = player_features.pivot_table(index='match_id', columns='player_id_mapped', values='hero_id', fill_value=0)
    hero_pivot.columns = [f"{col}_hero" for col in hero_pivot.columns] # Rename to radiant_player_1_hero etc.
    log_and_print(f"       Hero features pivoted. Shape: {hero_pivot.shape}")
    save_csv(hero_pivot, os.path.join(SF_02_PIVOTED, 'hero_pivot.csv'), index=True)

    # Pivot item features
    # Melt first to long format for easier pivoting
    items_melted = player_features.melt(id_vars=['match_id', 'player_id_mapped'], value_vars=item_cols_names,
                                        var_name='item_slot_num', value_name='item_id')
    # Create item column name like radiant_player_1_item_1, radiant_player_1_item_2 ... radiant_player_1_item_6
    items_melted['item_col_name'] = items_melted['player_id_mapped'] + '_item_' + (items_melted['item_slot_num'].str.split('_').str[1].astype(int) + 1).astype(str)

    item_pivot = items_melted.pivot_table(index='match_id', columns='item_col_name', values='item_id', fill_value=0)
    log_and_print(f"       Item features pivoted. Shape: {item_pivot.shape}")
    save_csv(item_pivot, os.path.join(SF_02_PIVOTED, 'item_pivot.csv'), index=True)

    # Combine hero and item pivots
    replication_features = hero_pivot.join(item_pivot, how='inner') # Use inner join to keep only matches with all player data pivoted
    log_and_print(f"   Joined Hero and Item features. Shape: {replication_features.shape}")

    # Add target variable and time for splitting
    matches_subset = matches_df[['match_id', 'radiant_win', 'start_date_time']].set_index('match_id')
    replication_features = replication_features.join(matches_subset, how='inner') # Inner join ensures we only have matches present in both features and original matches list
    log_and_print(f"   Joined with target variable and start time. Shape: {replication_features.shape}")

    # Ensure all expected columns based on map exist
    expected_hero_cols = [f"{slot_map[slot]}_hero" for slot in slot_map]
    expected_item_cols = [f"{slot_map[slot]}_item_{i+1}" for slot in slot_map for i in range(6)]
    expected_cols = expected_hero_cols + expected_item_cols
    all_expected_cols_with_meta = expected_cols + ['radiant_win', 'start_date_time']
    replication_features = replication_features.reindex(columns=all_expected_cols_with_meta, fill_value=0)
    log_and_print(f"   Reindexed columns to ensure all slots represented. Shape: {replication_features.shape}")

    # Final cleanup
    replication_features = replication_features.fillna(0)
    for col in expected_cols:
        replication_features[col] = pd.to_numeric(replication_features[col], errors='coerce').fillna(0).astype(int)
    replication_features['radiant_win'] = replication_features['radiant_win'].astype(int)

    log_and_print(f"Leaky feature set created successfully.", Colors.GREEN)
    save_csv(replication_features, os.path.join(SF_03_FEATURES, 'replication_features_combined.csv'), index=True)

    return replication_features

# === Main Script Execution ===

################################## DOCS #######################################
# Data Loading:
# Reads the matches and player-match data from CSV files.
###############################################################################
log_and_print("Loading data files...", Colors.HEADER)
try:
    matches_raw = pd.read_csv(MATCHES_FILE, usecols=matches_cols_rep, low_memory=False)
    players_raw = pd.read_csv(PLAYERS_FILE, usecols=players_cols_rep, low_memory=False)
    # heroes_df = pd.read_csv(HEROES_FILE, usecols=['id', 'localized_name']) # Optional
except FileNotFoundError as e: log_and_print(f"Error loading data: {e}. Make sure files are in {BASE_DATA_DIR}", Colors.FAIL); exit()
except ValueError as e: log_and_print(f"Error loading columns (check column names): {e}.", Colors.FAIL); exit()
except Exception as e: log_and_print(f"An unexpected error during data loading: {e}", Colors.FAIL); exit()
log_and_print(f"Loaded {len(matches_raw)} match rows and {len(players_raw)} player rows.", Colors.GREEN)

################################## DOCS #######################################
# Preprocessing (Matches Data):
# Converts types, sorts by time, removes duplicates.
###############################################################################
log_and_print("Preprocessing matches data...", Colors.BLUE)
matches_processed = matches_raw.copy()
matches_processed['radiant_win'] = matches_processed['radiant_win'].astype(np.int8)
matches_processed['start_date_time'] = pd.to_datetime(matches_processed['start_date_time'])
matches_processed = matches_processed.sort_values('start_date_time').reset_index(drop=True)
matches_processed = matches_processed.drop_duplicates(subset=['match_id'], keep='first')
log_and_print("   Match data preprocessed and sorted by time.", Colors.GREEN)
save_csv(matches_processed, os.path.join(SF_01_LOADED, 'matches_processed.csv'))
save_csv(players_raw, os.path.join(SF_01_LOADED, 'players_raw_loaded.csv'))

valid_match_ids = matches_processed['match_id'].unique()
players_filtered = players_raw[players_raw['match_id'].isin(valid_match_ids)].copy()
log_and_print(f"   Filtered players data to match processed matches. Player rows: {len(players_filtered)}", Colors.GREEN)
save_csv(players_filtered, os.path.join(SF_01_LOADED, 'players_filtered.csv'))

del matches_raw, players_raw
gc.collect()

################################## DOCS #######################################
# Feature Engineering Execution:
# Generates the final feature set.
###############################################################################
replication_data = create_leaky_replication_features(matches_processed, players_filtered)

del players_filtered, matches_processed
gc.collect()

################################## DOCS #######################################
# Data Splitting (Time-Based 80:20 for Final Test):
# Splits data for final model training and evaluation.
###############################################################################
log_and_print("Splitting data into Train/Test sets (Time-Based 80:20 for Final Evaluation)...", Colors.BLUE)
replication_data = replication_data.sort_values('start_date_time')
train_size_rep = int(len(replication_data) * 0.8)
train_df_rep = replication_data.iloc[:train_size_rep]
test_df_rep = replication_data.iloc[train_size_rep:]

X_train_rep = train_df_rep.drop(columns=['radiant_win', 'start_date_time'])
y_train_rep = train_df_rep['radiant_win']
X_test_rep = test_df_rep.drop(columns=['radiant_win', 'start_date_time'])
y_test_rep = test_df_rep['radiant_win']
log_and_print(f"Data split complete. Train shape: {X_train_rep.shape}, Test shape: {X_test_rep.shape}", Colors.GREEN)

log_and_print("Saving Train/Test splits...", Colors.BLUE)
save_csv(X_train_rep, os.path.join(SF_04_SPLIT, 'X_train_leaky.csv'), index=True)
save_csv(y_train_rep, os.path.join(SF_04_SPLIT, 'y_train_leaky.csv'), index=True)
save_csv(X_test_rep, os.path.join(SF_04_SPLIT, 'X_test_leaky.csv'), index=True)
save_csv(y_test_rep, os.path.join(SF_04_SPLIT, 'y_test_leaky.csv'), index=True)


################################## DOCS #######################################
# Cross-Validation Setup & Execution (Basic K-Fold):
# Uses standard KFold on the training data. Warns about time-series inappropriateness.
###############################################################################
log_and_print("Starting Basic Cross-Validation (K-Fold with Shuffle) on Training Data...", Colors.HEADER)
log_and_print(f"{Colors.WARNING}WARNING: Using KFold with shuffle on time-ordered data. Results may be optimistic due to ignoring time dependencies.{Colors.ENDC}", Colors.WARNING)

N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

baseline_params = {
    'n_estimators': 100, 'max_depth': 20, 'min_samples_split': 20,
    'min_samples_leaf': 10, 'max_features': 'sqrt', 'class_weight': 'balanced_subsample',
    'random_state': 42, 'n_jobs': -1
}
cv_model = RandomForestClassifier(**baseline_params)

cv_accuracies = []
cv_roc_aucs = []
avg_cv_acc, std_cv_acc, avg_cv_roc, std_cv_roc = np.nan, np.nan, np.nan, np.nan # Initialize

try:
    log_and_print(f"   Calculating CV Accuracy ({N_SPLITS} folds)...", Colors.BLUE)
    cv_accuracies = cross_val_score(cv_model, X_train_rep, y_train_rep, cv=kf, scoring='accuracy', n_jobs=-1)

    log_and_print(f"   Calculating CV ROC AUC ({N_SPLITS} folds)...", Colors.BLUE)
    cv_roc_aucs = cross_val_score(cv_model, X_train_rep, y_train_rep, cv=kf, scoring='roc_auc', n_jobs=-1)

    avg_cv_acc = np.mean(cv_accuracies)
    std_cv_acc = np.std(cv_accuracies)
    avg_cv_roc = np.mean(cv_roc_aucs)
    std_cv_roc = np.std(cv_roc_aucs)

    log_and_print("\n--- K-Fold Cross-Validation Summary (Shuffle=True) ---", Colors.HEADER)
    log_and_print(f"Individual Fold Accuracies: {[f'{acc:.4f}' for acc in cv_accuracies]}")
    log_and_print(f"Average Accuracy: {avg_cv_acc:.4f} (+/- {std_cv_acc:.4f})")
    log_and_print(f"Individual Fold ROC AUCs: {[f'{roc:.4f}' for roc in cv_roc_aucs]}")
    log_and_print(f"Average ROC AUC:  {avg_cv_roc:.4f} (+/- {std_cv_roc:.4f})")
    log_and_print("----------------------------------------------------------")

    cv_results_df = pd.DataFrame({
        'Fold': list(range(1, N_SPLITS + 1)),
        'Accuracy': cv_accuracies,
        'ROC_AUC': cv_roc_aucs
    })
    save_csv(cv_results_df, os.path.join(SF_05_EVALUATION, 'kfold_cross_validation_results.csv'))

except Exception as e:
        log_and_print(f"Error during Cross-Validation using cross_val_score: {e}", Colors.FAIL)


################################## DOCS #######################################
# Final Model Training & Evaluation on Test Set:
# Trains on full training set, evaluates on held-out test set.
# Calculates metrics, plots ROC, Confusion Matrix, and feature importances.
###############################################################################
log_and_print("Training final model on full training set (80%)...", Colors.WARNING)
final_model = RandomForestClassifier(**baseline_params)
try:
    final_model.fit(X_train_rep, y_train_rep)
    log_and_print("Final model training complete.", Colors.GREEN)
except Exception as e:
    log_and_print(f"Error during final model fitting: {e}", Colors.FAIL); exit()

# --- Evaluation on Test Set ---
log_and_print("Evaluating final model performance on the held-out Test Set (20%)...", Colors.GREEN)
training_cols = X_train_rep.columns
X_test_rep_aligned = X_test_rep.reindex(columns=training_cols, fill_value=0)

y_pred_test = final_model.predict(X_test_rep_aligned)
y_proba_test = final_model.predict_proba(X_test_rep_aligned)[:, 1]

test_accuracy = accuracy_score(y_test_rep, y_pred_test)
test_roc_auc = roc_auc_score(y_test_rep, y_proba_test)
report_str = classification_report(y_test_rep, y_pred_test)

log_and_print("\n--- Final Test Set Evaluation (Time-Based Hold-Out) ---")
log_and_print(f"Accuracy: {test_accuracy:.4f}")
log_and_print(f"ROC-AUC: {test_roc_auc:.4f}")
log_and_print("\nClassification Report (Test Set):")
print(report_str)
log_messages.append("\nClassification Report (Test Set):\n" + report_str)
log_and_print("---------------------------------------------------------------------")

# --- Plot ROC Curve ---
log_and_print("Plotting ROC Curve for Test Set...", Colors.BLUE)
try:
    fpr, tpr, thresholds = roc_curve(y_test_rep, y_proba_test)
    roc_auc_value = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_value:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve - Final Test Set')
    plt.legend(loc="lower right")
    plt.grid(True)
    plot_path = os.path.join(SF_05_EVALUATION, 'roc_curve_test_set.png')
    plt.savefig(plot_path)
    log_and_print(f"ROC Curve plot saved to {plot_path}", Colors.GREEN)
    plt.close()
except Exception as e:
    log_and_print(f"Could not plot/save ROC curve: {e}", Colors.WARNING)

# --- Plot Confusion Matrix ---
log_and_print("Plotting Confusion Matrix for Test Set...", Colors.BLUE)
try:
    cm = confusion_matrix(y_test_rep, y_pred_test, labels=final_model.classes_)
    # Ensure display_labels match the order of final_model.classes_ (typically [0, 1])
    # If your labels are 'Dire Win' (for 0) and 'Radiant Win' (for 1)
    display_labels = ['Dire Win (0)', 'Radiant Win (1)'] if np.array_equal(final_model.classes_, [0,1]) else final_model.classes_

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d') # 'd' for integer format
    ax.set_title('Confusion Matrix - Final Test Set')
    plot_path_cm = os.path.join(SF_05_EVALUATION, 'confusion_matrix_test_set.png')
    plt.savefig(plot_path_cm)
    log_and_print(f"Confusion Matrix plot saved to {plot_path_cm}", Colors.GREEN)
    plt.close(fig)
except Exception as e:
    log_and_print(f"Could not plot/save Confusion Matrix: {e}", Colors.WARNING)


# --- Feature Importance ---
log_and_print("Calculating feature importances for the final model...", Colors.BLUE)
feature_imp = None # Initialize
try:
    importances = final_model.feature_importances_
    feature_names = X_train_rep.columns
    feature_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_imp = feature_imp.sort_values('Importance', ascending=False).reset_index(drop=True)

    log_and_print("\nTop 20 Most Important Features (Final Model):")
    print(feature_imp.head(20).to_string())
    log_messages.append("\nTop 20 Most Important Features (Final Model):\n" + feature_imp.head(20).to_string())

    imp_path = os.path.join(SF_05_EVALUATION, 'final_model_feature_importances.csv')
    save_csv(feature_imp, imp_path)

    N_TOP_FEATURES = 30
    plt.figure(figsize=(10, max(6, N_TOP_FEATURES // 2)))
    plt.barh(feature_imp['Feature'][:N_TOP_FEATURES], feature_imp['Importance'][:N_TOP_FEATURES], align='center')
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"Top {N_TOP_FEATURES} Feature Importances (Leaky Replication - Final Model)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plot_path_imp = os.path.join(SF_05_EVALUATION, 'final_model_feature_importances_plot.png')
    plt.savefig(plot_path_imp)
    log_and_print(f"Feature importance plot saved to {plot_path_imp}", Colors.GREEN)
    plt.close()

except Exception as e:
    log_and_print(f"Could not calculate/save feature importances: {e}", Colors.WARNING)


################################## DOCS #######################################
# Saving Final Results & Reporting:
# Saves key metrics, detailed classification metrics, and generates a summary report.
###############################################################################
log_and_print("Saving final metrics and report summary...", Colors.BLUE)

# --- Save Detailed Classification Metrics for Test Set ---
log_and_print("Saving detailed classification metrics for the Test Set...", Colors.BLUE)
try:
    # Ensure classes are [0, 1] for indexing, adjust if your classes are different
    # This assumes final_model.classes_ is [0, 1] where 0 is the first class, 1 is the second.
    class_labels_dict = {0: 'Class 0 (e.g., Dire Win)', 1: 'Class 1 (e.g., Radiant Win)'} # Customize as needed
    
    # Get precision, recall, fscore, support for each class
    # Explicitly set labels to handle cases where one class might not be predicted in a small test set, though unlikely here.
    p, r, f1, s = precision_recall_fscore_support(y_test_rep, y_pred_test, average=None, labels=final_model.classes_, zero_division=0)
    
    # Get macro and weighted averages
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test_rep, y_pred_test, average='macro', zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test_rep, y_pred_test, average='weighted', zero_division=0)

    metrics_list = []
    # Class-specific metrics
    for i, label_val in enumerate(final_model.classes_):
        metrics_list.append({
            'Category': 'Class-specific',
            'Class Label': class_labels_dict.get(label_val, f"Class {label_val}"),
            'Metric': 'Precision',
            'Value': p[i]
        })
        metrics_list.append({
            'Category': 'Class-specific',
            'Class Label': class_labels_dict.get(label_val, f"Class {label_val}"),
            'Metric': 'Recall',
            'Value': r[i]
        })
        metrics_list.append({
            'Category': 'Class-specific',
            'Class Label': class_labels_dict.get(label_val, f"Class {label_val}"),
            'Metric': 'F1-Score',
            'Value': f1[i]
        })
        metrics_list.append({
            'Category': 'Class-specific',
            'Class Label': class_labels_dict.get(label_val, f"Class {label_val}"),
            'Metric': 'Support',
            'Value': s[i]
        })
        
    # Averaged metrics
    avg_metrics = [
        ('Macro Avg', 'Precision', p_macro), ('Macro Avg', 'Recall', r_macro), ('Macro Avg', 'F1-Score', f1_macro),
        ('Weighted Avg', 'Precision', p_weighted), ('Weighted Avg', 'Recall', r_weighted), ('Weighted Avg', 'F1-Score', f1_weighted)
    ]
    for cat, met, val in avg_metrics:
         metrics_list.append({'Category': cat, 'Class Label': '', 'Metric': met, 'Value': val})

    # Overall metrics
    metrics_list.append({'Category': 'Overall', 'Class Label': '', 'Metric': 'Accuracy', 'Value': test_accuracy})
    metrics_list.append({'Category': 'Overall', 'Class Label': '', 'Metric': 'ROC AUC', 'Value': test_roc_auc})

    detailed_metrics_df = pd.DataFrame(metrics_list)
    detailed_metrics_path = os.path.join(SF_05_EVALUATION, 'classification_metrics_detailed_test_set.csv')
    save_csv(detailed_metrics_df, detailed_metrics_path)
    log_and_print(f"Detailed classification metrics saved to {detailed_metrics_path}", Colors.GREEN)

except Exception as e:
    log_and_print(f"Error saving detailed classification metrics: {e}", Colors.WARNING)


# Save the summary including CV results (your original metrics_final_df style)
summary_metrics_df = pd.DataFrame({
   'Metric': ['Final Test Accuracy', 'Final Test ROC AUC',
              'Avg KFold CV Accuracy (Shuffle)', 'Std KFold CV Accuracy (Shuffle)',
              'Avg KFold CV ROC AUC (Shuffle)', 'Std KFold CV ROC AUC (Shuffle)'],
   'Value': [test_accuracy, test_roc_auc, avg_cv_acc, std_cv_acc, avg_cv_roc, std_cv_roc]
})
summary_metrics_path = os.path.join(SF_05_EVALUATION, 'final_model_metrics_summary.csv')
save_csv(summary_metrics_df, summary_metrics_path)
log_and_print(f"Summary metrics (including CV) saved to {summary_metrics_path}", Colors.GREEN)


# Create a simple markdown report summary
report_summary_path = os.path.join(OUTPUT_DIR, 'replication_summary_report_basic_cv_detailed.md')
try:
    with open(report_summary_path, 'w') as f:
        f.write("# Dota 2 Leaky Model Replication Report (Basic K-Fold CV - Detailed)\n\n")
        f.write("## Objective\n")
        f.write("Replicated a feature set using Hero IDs + Final Item IDs to establish baseline performance, demonstrate potential data leakage effects, and implement K-Fold cross-validation for assessment. This version includes detailed metrics and a confusion matrix.\n\n")

        f.write("## Methodology\n")
        f.write(f"* **Data Source:** Matches (`{os.path.basename(MATCHES_FILE)}`), Players (`{os.path.basename(PLAYERS_FILE)}`)\n")
        f.write(f"* **Feature Set:** 10 Hero IDs + 60 Final Item IDs per match (pivoted).\n")
        f.write(f"* **Total Features:** {X_train_rep.shape[1]}\n")
        f.write(f"* **Final Evaluation Split:** Time-based 80% Train ({len(X_train_rep)} matches) / 20% Test ({len(X_test_rep)} matches).\n")
        f.write(f"* **Cross-Validation:** {N_SPLITS}-Fold KFold (with `shuffle=True`, `random_state=42`) performed on the 80% training set.\n")
        f.write(f"   * **Note:** Standard KFold ignores time-ordering and is generally unsuitable for time-series data, used here per request.\n")
        f.write(f"* **Algorithm:** Baseline Random Forest\n")
        f.write(f"   ```json\n   {baseline_params}\n   ```\n\n")

        f.write("## K-Fold Cross-Validation Results (on Training Set, Shuffle=True)\n")
        f.write(f"* **Average Accuracy:** {avg_cv_acc:.4f} (+/- {std_cv_acc:.4f})\n")
        f.write(f"* **Average ROC AUC:** {avg_cv_roc:.4f} (+/- {std_cv_roc:.4f})\n")
        f.write(f"* **Individual Fold Results:** See `{os.path.join(os.path.basename(SF_05_EVALUATION), 'kfold_cross_validation_results.csv')}`\n\n")

        f.write("## Final Model Evaluation (on Chronological Hold-Out Test Set)\n")
        f.write(f"* **Overall Accuracy:** {test_accuracy:.4f}\n")
        f.write(f"* **Overall ROC AUC:** {test_roc_auc:.4f}\n")
        f.write(f"* **Confusion Matrix Plot:** See `{os.path.join(os.path.basename(SF_05_EVALUATION),'confusion_matrix_test_set.png')}`\n")
        f.write(f"* **ROC Curve Plot:** See `{os.path.join(os.path.basename(SF_05_EVALUATION),'roc_curve_test_set.png')}`\n")
        f.write(f"* **Detailed Classification Metrics CSV:** See `{os.path.join(os.path.basename(SF_05_EVALUATION),'classification_metrics_detailed_test_set.csv')}`\n\n")

        f.write("### Classification Report (Test Set)\n")
        f.write("```\n")
        f.write(report_str)
        f.write("\n```\n\n")

        f.write("### Top 20 Feature Importances (Final Model)\n")
        if feature_imp is not None and isinstance(feature_imp, pd.DataFrame):
              f.write("```\n")
              f.write(feature_imp.head(20).to_string())
              f.write("\n```\n")
              f.write(f"* **Full Importances:** See `{os.path.join(os.path.basename(SF_05_EVALUATION),'final_model_feature_importances.csv')}`\n")
              f.write(f"* **Importance Plot:** See `{os.path.join(os.path.basename(SF_05_EVALUATION),'final_model_feature_importances_plot.png')}`\n\n")
        else:
              f.write("Feature importance calculation failed or was skipped.\n\n")

        f.write("## Analysis\n")
        f.write("The model achieved high performance on the final chronological test set (Accuracy ~{:.2f}, ROC AUC ~{:.2f}). ".format(test_accuracy, test_roc_auc))
        f.write("The K-Fold Cross-Validation (with shuffling) on the training data yielded similarly high average performance (ROC AUC ~{:.2f}).\n\n".format(avg_cv_roc if not np.isnan(avg_cv_roc) else 0.0)) # Handle NaN if CV fails
        f.write("Critically, **both the high CV scores and the high final test score are primarily driven by data leakage** inherent in using **final item builds** as input features. These items are known only post-match and correlate strongly with the outcome.\n\n")
        f.write("The use of standard K-Fold (with shuffling) for cross-validation further exacerbates potential issues by breaking the time sequence during evaluation on the training set. While the results are similar to the final test set in this specific leaky scenario, this CV method would typically provide an unreliable, overly optimistic estimate for a genuine time-series prediction task.\n\n")
        f.write("Feature importance analysis confirms the leakage, showing item features highly ranked. This experiment serves as a **flawed baseline**, demonstrating the impact of leakage and highlighting the importance of feature selection (using only pre-game info) and appropriate validation techniques (like TimeSeriesSplit) for valid win prediction modeling.\n")

    log_and_print(f"Summary report saved to {report_summary_path}", Colors.GREEN)
except Exception as e:
    log_and_print(f"Error saving summary report: {e}", Colors.WARNING)


# --- Save Logs ---
log_file_path = os.path.join(OUTPUT_DIR, 'experiment_log.txt')
save_log_file(log_file_path)

log_and_print("Script finished.", Colors.BOLD + Colors.GREEN)