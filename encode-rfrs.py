import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
import os
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
import gc
import matplotlib
matplotlib.use('Agg') # Explicitly set non-interactive backend
import matplotlib.pyplot as plt
import re
from sklearn.tree import plot_tree
import json

# ==============Configuration & Constants=================================
BASE_DATA_DIR = './Data'
PLAYERS_FILE = os.path.join(BASE_DATA_DIR, 'player_matches_aug_2024.csv')
MATCHES_FILE = os.path.join(BASE_DATA_DIR, 'players_matches.csv')
CONSTANTS_DIR = os.path.join(BASE_DATA_DIR, 'Constants')
ITEMS_FILE = os.path.join(CONSTANTS_DIR, 'Constants.ItemIDs.csv')
HEROES_FILE = os.path.join(CONSTANTS_DIR, 'Constants.Heroes.csv')

# --- Output Directory Structure ---
OUTPUT_DIR_NAME = 'encoded_rfrs'
OUTPUT_DIR = f'./{OUTPUT_DIR_NAME}'

# Transformations sub-directories
TRANSFORMATIONS_DIR = os.path.join(OUTPUT_DIR, 'transformations')
LOADED_DATA_DIR = os.path.join(TRANSFORMATIONS_DIR, '01_loaded_data')
PREPROCESSED_DATA_DIR = os.path.join(TRANSFORMATIONS_DIR, '02_preprocessed_data')
PIVOTED_DATA_DIR = os.path.join(TRANSFORMATIONS_DIR, '03_pivoted_data')
COMBINED_FEATURES_DIR = os.path.join(TRANSFORMATIONS_DIR, '04_combined_features')
SPLIT_DATA_DIR = os.path.join(TRANSFORMATIONS_DIR, '05_split_data')

# Plot output sub-directories
EVALUATION_MATRIX_DIR = os.path.join(OUTPUT_DIR, 'evaluation_matrix')
ROC_AUC_CURVES_DIR = os.path.join(OUTPUT_DIR, 'roc_auc_curves')
FEATURE_IMPORTANCE_PLOTS_DIR = os.path.join(OUTPUT_DIR, 'feature_importance_plots')
TREE_PLOT_DIR = os.path.join(OUTPUT_DIR, 'sample_trees') # This was already well-named

# Create all necessary directories
dirs_to_create = [
    OUTPUT_DIR, TRANSFORMATIONS_DIR, LOADED_DATA_DIR, PREPROCESSED_DATA_DIR,
    PIVOTED_DATA_DIR, COMBINED_FEATURES_DIR, SPLIT_DATA_DIR,
    EVALUATION_MATRIX_DIR, ROC_AUC_CURVES_DIR, FEATURE_IMPORTANCE_PLOTS_DIR,
    TREE_PLOT_DIR
]
for D in dirs_to_create:
    os.makedirs(D, exist_ok=True)

# Columns to select
matches_cols_rep = ['match_id', 'radiant_win', 'start_date_time']
players_cols_rep = ['match_id', 'player_slot', 'hero_id', 'item_0', 'item_1', 'item_2', 'item_3', 'item_4', 'item_5', 'isRadiant']
item_cols_names = [f'item_{i}' for i in range(6)]

RADIANT_TEAM_ID = 0
DIRE_TEAM_ID = 1

# ==============Logging and Utility Functions==============================
class Colors:
    HEADER = '\033[95m'; BLUE = '\033[94m'; GREEN = '\033[92m'
    WARNING = '\033[93m'; FAIL = '\033[91m'; ENDC = '\033[0m'; BOLD = '\033[1m'

log_messages = []
def log_and_print(message, color=Colors.BLUE):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] [STATUS] {message}"
    print(f"{color}{log_msg}{Colors.ENDC}")
    log_messages.append(log_msg)

def save_log_file(filepath):
    try:
        with open(filepath, 'w') as f:
            for msg in log_messages:
                f.write(msg + '\n')
        log_and_print(f"Log saved to {filepath}", Colors.GREEN)
    except Exception as e:
        log_and_print(f"Error saving log file: {e}", Colors.FAIL)

def sanitize_name(name):
    if pd.isna(name): return "Unknown"
    s = str(name).strip()
    s = re.sub(r'\s+', '_', s)
    s = re.sub(r'[^\w-]', '', s)
    return s if s else "Unknown"

# ==============Custom Callback Function for RandomizedSearchCV Progress=======
def log_search_progress_callback(search_instance):
    current_iteration = len(search_instance.cv_results_['params'])
    if current_iteration > 0:
        last_idx = current_iteration - 1
        params_just_evaluated = search_instance.cv_results_['params'][last_idx]
        mean_score_just_evaluated = search_instance.cv_results_['mean_test_score'][last_idx]
        std_score_just_evaluated = search_instance.cv_results_['std_test_score'][last_idx]
        log_and_print(
            f"[Search Progress] Trial {current_iteration}/{search_instance.n_iter} Completed. "
            f"Mean CV Score: {mean_score_just_evaluated:.4f} (std: {std_score_just_evaluated:.4f}). "
            f"Params: {params_just_evaluated}",
            Colors.WARNING
        )

# ==============Feature Engineering Function==============================
def create_plus_minus_named_features(matches_df, players_df, hero_map, item_map, pivoted_data_save_dir, combined_features_save_dir):
    log_and_print("Creating +1/-1 named feature set...", Colors.BLUE)
    player_data = players_df[players_cols_rep].copy()
    log_and_print(f"  Initial player data shape: {player_data.shape}")

    for col in item_cols_names:
        player_data[col] = player_data[col].fillna(0).astype(int)
    ## Type Casting
    player_data['hero_id'] = player_data['hero_id'].fillna(0).astype(int)
    player_data['isRadiant'] = player_data['isRadiant'].astype(np.int8)
    player_data['player_slot'] = player_data['player_slot'].astype(int)
    log_and_print("  Filled NaNs and ensured integer types for player data.")

    player_data['hero_name'] = player_data['hero_id'].map(hero_map).fillna('Unknown_Hero')
    player_data = player_data[player_data['hero_name'] != 'Unknown_Hero']
    log_and_print("  Mapped hero IDs to names and removed unknown heroes from player data.")

    # --- Hero Features ---
    log_and_print("  Processing hero features (named columns)...", Colors.BLUE)
    hero_data = player_data[['match_id', 'hero_name', 'isRadiant']].copy()
    # If player is Radiant, they are 1, if Dire, -1
    hero_data['value'] = hero_data['isRadiant'].apply(lambda x: 1 if x == 1 else -1)
    hero_pivot = hero_data.pivot_table(index='match_id', columns='hero_name', values='value', aggfunc='sum', fill_value=0)
    all_hero_names = [name for name in hero_map.values() if name != 'Unknown_Hero']
    hero_pivot = hero_pivot.reindex(columns=all_hero_names, fill_value=0)
    log_and_print(f"    Hero features pivoted. Shape: {hero_pivot.shape}")
    hero_pivot.to_csv(os.path.join(pivoted_data_save_dir, 'hero_pivot.csv'))
    log_and_print(f"    Saved hero_pivot to {os.path.join(pivoted_data_save_dir, 'hero_pivot.csv')}", Colors.GREEN)


    # --- Item Features ---
    log_and_print("  Processing item features (named columns)...", Colors.BLUE)
    item_data_for_melt = player_data[['match_id', 'isRadiant'] + item_cols_names].copy()
    item_data_melted = item_data_for_melt.melt(id_vars=['match_id', 'isRadiant'], value_vars=item_cols_names, var_name='item_slot_num', value_name='item_id')
    item_data_melted = item_data_melted[item_data_melted['item_id'] != 0]
    item_data_melted['item_name'] = item_data_melted['item_id'].map(item_map).fillna('Unknown_Item')
    item_data_melted = item_data_melted[item_data_melted['item_name'] != 'Unknown_Item']
    log_and_print("    Mapped item IDs to names and removed unknown items.")
    item_data_melted['value'] = item_data_melted['isRadiant'].apply(lambda x: 1 if x == 1 else -1)
    item_pivot = item_data_melted.pivot_table(index='match_id', columns='item_name', values='value', aggfunc='sum', fill_value=0)
    all_item_names = [name for name in item_map.values() if name != 'Unknown_Item']
    item_pivot = item_pivot.reindex(columns=all_item_names, fill_value=0)
    log_and_print(f"    Item features pivoted. Shape: {item_pivot.shape}")
    item_pivot.to_csv(os.path.join(pivoted_data_save_dir, 'item_pivot.csv'))
    log_and_print(f"    Saved item_pivot to {os.path.join(pivoted_data_save_dir, 'item_pivot.csv')}", Colors.GREEN)

    # --- Combine Features & Target ---
    log_and_print("  Combining features...", Colors.BLUE)
    final_features = pd.DataFrame(index=matches_df['match_id'].unique())
    final_features.index.name = 'match_id'
    final_features = final_features.join(hero_pivot, how='left')
    final_features = final_features.join(item_pivot, how='left')
    final_features = final_features.join(matches_df.set_index('match_id')[['radiant_win', 'start_date_time']], how='inner')
    final_features = final_features.fillna(0)
    
    feature_cols_final_check = list(all_hero_names) + list(all_item_names)
    for col in feature_cols_final_check:
        if col in final_features.columns:
            final_features[col] = pd.to_numeric(final_features[col], errors='coerce').fillna(0).astype(int)
        else:
            final_features[col] = 0
            log_and_print(f"Warning: Column {col} was not in final_features after joins, added as all zeros.", Colors.WARNING)
    
    log_and_print(f"+/- Named feature set created successfully. Shape: {final_features.shape}", Colors.GREEN)
    combined_features_path = os.path.join(combined_features_save_dir, 'combined_features_before_pruning.csv')
    # Save with start_date_time for now, will be dropped before final model feature saving if needed by that step
    final_features.to_csv(combined_features_path, index=True)
    log_and_print(f"  Saved combined features (before pruning) to {combined_features_path}", Colors.GREEN)
    
    return final_features, all_hero_names, all_item_names

# ==============Main Script Execution====================================
# --- Data Loading ---
log_and_print("Loading data files...", Colors.HEADER)
try:
    matches_2024 = pd.read_csv(MATCHES_FILE, usecols=matches_cols_rep, low_memory=False)
    players_2024 = pd.read_csv(PLAYERS_FILE, usecols=players_cols_rep, low_memory=False)
    heroes_df = pd.read_csv(HEROES_FILE, usecols=['id', 'localized_name'])
    items_df = pd.read_csv(ITEMS_FILE, usecols=['id', 'name'])
except FileNotFoundError as e:
    log_and_print(f"Error loading data: {e}. Please ensure data files are in {BASE_DATA_DIR} or update paths.", Colors.FAIL)
    exit()
except Exception as e:
    log_and_print(f"An unexpected error occurred during data loading: {e}", Colors.FAIL)
    exit()
log_and_print(f"Loaded {len(matches_2024)} matches and {len(players_2024)} player rows.", Colors.GREEN)

# Save loaded data to '01_loaded_data'
matches_2024.to_csv(os.path.join(LOADED_DATA_DIR, 'loaded_matches_2024.csv'), index=False)
players_2024.to_csv(os.path.join(LOADED_DATA_DIR, 'loaded_players_2024.csv'), index=False)
log_and_print(f"Saved loaded data to {LOADED_DATA_DIR}", Colors.GREEN)


# --- Create ID to Name Mappings ---
log_and_print("Creating ID-to-Name mappings...", Colors.BLUE)
heroes_df['clean_name'] = heroes_df['localized_name'].apply(sanitize_name)
items_df['clean_name'] = items_df['name'].apply(sanitize_name)
hero_id_to_name = heroes_df.set_index('id')['clean_name'].to_dict()
item_id_to_name = items_df.set_index('id')['clean_name'].to_dict()
hero_id_to_name[0] = 'Unknown_Hero'
item_id_to_name[0] = 'Unknown_Item'
log_and_print("Mappings created.", Colors.GREEN)

# --- Preprocessing Matches Data ---
log_and_print("Preprocessing matches data...", Colors.BLUE)
matches_2024['radiant_win'] = matches_2024['radiant_win'].astype(np.int8)
matches_2024['start_date_time'] = pd.to_datetime(matches_2024['start_date_time'])
matches_2024 = matches_2024.sort_values('start_date_time').reset_index(drop=True)
matches_2024 = matches_2024.drop_duplicates(subset=['match_id'], keep='first')
log_and_print("  Match data preprocessed.", Colors.GREEN)
matches_2024.to_csv(os.path.join(PREPROCESSED_DATA_DIR, 'preprocessed_matches_2024.csv'), index=False)
log_and_print(f"Saved preprocessed matches data to {PREPROCESSED_DATA_DIR}", Colors.GREEN)

# --- Feature Engineering ---
replication_data_pm_named, engineered_hero_names, engineered_item_names = create_plus_minus_named_features(
    matches_2024, players_2024, hero_id_to_name, item_id_to_name, PIVOTED_DATA_DIR, COMBINED_FEATURES_DIR
)

# --- Remove Zero-Variance Features ---
log_and_print("Checking for and removing zero-variance features...", Colors.BLUE)
feature_columns_to_check = engineered_hero_names + engineered_item_names
cols_to_drop = []
if not replication_data_pm_named.empty:
    for col in feature_columns_to_check:
        if col in replication_data_pm_named.columns:
            if (replication_data_pm_named[col] == 0).all():
                cols_to_drop.append(col)
        else:
            log_and_print(f"  Warning: Expected feature column '{col}' not found during zero-variance check.", Colors.WARNING)
    if cols_to_drop:
        replication_data_pm_named = replication_data_pm_named.drop(columns=cols_to_drop)
        log_and_print(f"  Removed {len(cols_to_drop)} zero-variance feature columns: {cols_to_drop}", Colors.GREEN)
        engineered_hero_names = [name for name in engineered_hero_names if name not in cols_to_drop]
        engineered_item_names = [name for name in engineered_item_names if name not in cols_to_drop]
    else:
        log_and_print("  No zero-variance features found to remove.", Colors.GREEN)
else:
    log_and_print("  Skipping zero-variance check as feature data is empty.", Colors.WARNING)

# --- Save Intermediate Features (After Pruning) ---
feature_file_path_after_pruning = os.path.join(COMBINED_FEATURES_DIR, 'combined_features_after_pruning.csv')
log_and_print(f"Saving combined features (after pruning) to {feature_file_path_after_pruning}...", Colors.BLUE)
try:
    # Drop 'start_date_time' before saving this version as it's for model input
    if 'start_date_time' in replication_data_pm_named.columns:
        replication_data_pm_named.drop(columns=['start_date_time']).to_csv(feature_file_path_after_pruning, index=True)
    else:
        replication_data_pm_named.to_csv(feature_file_path_after_pruning, index=True)
    log_and_print("  Combined features (after pruning) saved successfully.", Colors.GREEN)
except Exception as e:
    log_and_print(f"  Error saving combined features (after pruning): {e}", Colors.WARNING)

# Clean up
del players_2024, matches_2024, heroes_df, items_df
gc.collect()

# --- Data Splitting (Time-Based 80:20) ---
log_and_print("Splitting data into Train/Test sets (Time-Based 80:20)...", Colors.BLUE)
if 'start_date_time' in replication_data_pm_named.columns:
    replication_data_pm_named = replication_data_pm_named.sort_values('start_date_time')
else:
    log_and_print("Warning: 'start_date_time' not in columns for sorting before split. Assuming pre-sorted.", Colors.WARNING)

train_size_rep = int(len(replication_data_pm_named) * 0.8)
train_df_rep = replication_data_pm_named.iloc[:train_size_rep]
test_df_rep = replication_data_pm_named.iloc[train_size_rep:]

feature_columns_for_X = [col for col in replication_data_pm_named.columns if col not in ['radiant_win', 'start_date_time']]
X_train_rep = train_df_rep[feature_columns_for_X]
y_train_rep = train_df_rep['radiant_win']
X_test_rep = test_df_rep[feature_columns_for_X]
y_test_rep = test_df_rep['radiant_win']
log_and_print(f"Data split complete. Train shape: {X_train_rep.shape}, Test shape: {X_test_rep.shape}", Colors.GREEN)

# --- Save Split Data ---
X_train_path = os.path.join(SPLIT_DATA_DIR, 'X_train.csv')
y_train_path = os.path.join(SPLIT_DATA_DIR, 'y_train.csv')
X_test_path = os.path.join(SPLIT_DATA_DIR, 'X_test.csv')
y_test_path = os.path.join(SPLIT_DATA_DIR, 'y_test.csv')
log_and_print(f"Saving Train/Test splits to {SPLIT_DATA_DIR}...", Colors.BLUE)
try:
    X_train_rep.to_csv(X_train_path, index=True)
    y_train_rep.to_csv(y_train_path, index=True, header=['radiant_win'])
    X_test_rep.to_csv(X_test_path, index=True)
    y_test_rep.to_csv(y_test_path, index=True, header=['radiant_win'])
    log_and_print("  Train/Test splits saved.", Colors.GREEN)
except Exception as e:
    log_and_print(f"  Error saving splits: {e}", Colors.WARNING)

# --- Modeling ---
X_train_final, y_train_final = X_train_rep.copy(), y_train_rep.copy()
X_test_final, y_test_final = X_test_rep.copy(), y_test_rep.copy()
del X_train_rep, y_train_rep, X_test_rep, y_test_rep, train_df_rep, test_df_rep, replication_data_pm_named
gc.collect()

log_and_print("Aligning train/test columns after split...", Colors.BLUE)
common_cols = X_train_final.columns.union(X_test_final.columns)
X_train_final = X_train_final.reindex(columns=common_cols, fill_value=0)
X_test_final = X_test_final.reindex(columns=common_cols, fill_value=0)
log_and_print(f"Aligned Train shape: {X_train_final.shape}, Aligned Test shape: {X_test_final.shape}", Colors.GREEN)

# --- Hyperparameter Tuning with RandomizedSearchCV ---
log_and_print("Performing Random Search for hyperparameter tuning...", Colors.BLUE)
param_dist = {
    'n_estimators': [100, 200, 300, 400], 'max_depth': [10, 20, 30, 40, None],
    'min_samples_split': [5, 10, 20, 30, 50], 'min_samples_leaf': [3, 5, 10, 20, 30],
    'max_features': ['sqrt', 0.5, 0.7, None], 'class_weight': ['balanced_subsample', 'balanced', None]
}
N_ITER_SEARCH = 50 # Reduced for faster testing, increase for real runs
CV_FOLDS = 3

# --- Define and Train Baseline Model ---
log_and_print("Defining and Training Baseline Model...", Colors.HEADER)
base_rf = RandomForestClassifier(random_state=42, n_jobs=-1)
try:
    base_rf.fit(X_train_final, y_train_final)
    log_and_print("Baseline RandomForestClassifier fitted successfully.", Colors.GREEN)
except Exception as e:
    log_and_print(f"Error fitting baseline RandomForestClassifier: {e}. Exiting.", Colors.FAIL)
    log_file_path_error = os.path.join(OUTPUT_DIR, f'experiment_ERROR_baseline_fit.txt')
    save_log_file(log_file_path_error)
    exit()

# --- Baseline Model Evaluation ---
log_and_print("Evaluating Baseline Model Performance...", Colors.HEADER)
report_str_base = "Baseline classification report not generated."
test_accuracy_base, test_roc_auc_base = np.nan, np.nan
y_pred_base, y_proba_base = None, None
try:
    y_pred_base = base_rf.predict(X_test_final)
    y_proba_base = base_rf.predict_proba(X_test_final)[:, 1]
    test_accuracy_base = accuracy_score(y_test_final, y_pred_base)
    test_roc_auc_base = roc_auc_score(y_test_final, y_proba_base)
    report_str_base = classification_report(y_test_final, y_pred_base, target_names=['Dire Win', 'Radiant Win'])
    log_and_print("\n--- Baseline Model Test Set Evaluation ---", color=Colors.HEADER)
    log_and_print(f"Accuracy: {test_accuracy_base:.4f}", color=Colors.GREEN)
    log_and_print(f"ROC-AUC: {test_roc_auc_base:.4f}", color=Colors.GREEN)
    log_and_print("\nClassification Report:", color=Colors.GREEN)
    print(Colors.GREEN + f"\n{report_str_base}" + Colors.ENDC)
    log_messages.append(f"\nBaseline Model Classification Report:\n{report_str_base}")
    log_and_print("---------------------------------------------", color=Colors.HEADER)
except Exception as e:
    log_and_print(f"Error during baseline model prediction/evaluation: {e}.", Colors.FAIL)

# --- Baseline Model Confusion Matrix ---
log_and_print("Generating and saving Baseline Model Confusion Matrix plot...", Colors.BLUE)
if y_pred_base is not None:
    try:
        cm_base = confusion_matrix(y_test_final, y_pred_base, labels=base_rf.classes_)
        disp_base = ConfusionMatrixDisplay(confusion_matrix=cm_base, display_labels=['Dire Win', 'Radiant Win'])
        fig, ax = plt.subplots(figsize=(8, 6))
        disp_base.plot(ax=ax, cmap=plt.cm.Greens, values_format='d')
        ax.set_title(f'Confusion Matrix - Baseline Model ({OUTPUT_DIR_NAME})', fontsize=14)
        plt.xticks(fontsize=10); plt.yticks(fontsize=10)
        ax.xaxis.label.set_fontsize(12); ax.yaxis.label.set_fontsize(12)
        cm_plot_path_base = os.path.join(EVALUATION_MATRIX_DIR, 'baseline_confusion_matrix.png')
        plt.savefig(cm_plot_path_base, dpi=150, bbox_inches='tight'); plt.close(fig)
        log_and_print(f"Baseline Confusion Matrix plot saved to {cm_plot_path_base}", Colors.GREEN)
        cm_log_message_base = f"\nBaseline Confusion Matrix (Test Set):\n{cm_base}\n"
        print(Colors.GREEN + cm_log_message_base + Colors.ENDC); log_messages.append(cm_log_message_base)
    except Exception as e:
        log_and_print(f"Could not generate/save Baseline Confusion Matrix plot: {e}", Colors.WARNING)
else:
    log_and_print("Skipping Baseline Confusion Matrix plot: y_pred_base not defined.", Colors.WARNING)

# --- Baseline Model ROC AUC Curve ---
log_and_print("Generating and saving Baseline Model ROC AUC curve...", Colors.BLUE)
if y_proba_base is not None and not np.isnan(test_roc_auc_base):
    try:
        fpr_base, tpr_base, _ = roc_curve(y_test_final, y_proba_base)
        plt.figure(figsize=(10, 8))
        plt.plot(fpr_base, tpr_base, color='green', lw=2, label=f'Baseline ROC curve (AUC = {test_roc_auc_base:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)', fontsize=12); plt.ylabel('True Positive Rate (TPR)', fontsize=12)
        plt.title(f'ROC Curve - Baseline Model ({OUTPUT_DIR_NAME})', fontsize=14)
        plt.legend(loc="lower right", fontsize=10); plt.grid(True, linestyle='--', alpha=0.7)
        roc_plot_path_base = os.path.join(ROC_AUC_CURVES_DIR, 'baseline_roc_auc_plot.png')
        plt.savefig(roc_plot_path_base, dpi=150, bbox_inches='tight'); plt.close()
        log_and_print(f"Baseline ROC AUC plot saved to {roc_plot_path_base}", Colors.GREEN)
    except Exception as e:
        log_and_print(f"Could not generate/save Baseline ROC AUC plot: {e}", Colors.WARNING)
else:
    log_and_print("Skipping Baseline ROC AUC plot: y_proba_base not defined or test_roc_auc_base is NaN.", Colors.WARNING)

# --- Baseline Model Cross-Validation Scores ---
log_and_print("\nCalculating Baseline Model Cross-Validation Scores (on Training Data)...", Colors.BLUE)
cv_mean_score_base, cv_std_score_base = np.nan, np.nan
try:
    cv_model_base = RandomForestClassifier(**base_rf.get_params()) # n_jobs=-1 is already in base_rf params
    if not X_train_final.empty and not y_train_final.empty:
        cv_scores_base = cross_val_score(cv_model_base, X_train_final, y_train_final, cv=3, scoring='roc_auc', n_jobs=-1)
        cv_mean_score_base = cv_scores_base.mean(); cv_std_score_base = cv_scores_base.std()
        log_and_print(f"Baseline CV ROC-AUC Scores: {cv_scores_base}", Colors.GREEN)
        log_and_print(f"Baseline Avg CV ROC-AUC: {cv_mean_score_base:.4f} (+/- {cv_std_score_base * 2:.4f})", Colors.GREEN)
    else: log_and_print("Skipping Baseline CV: Training data empty.", Colors.WARNING)
except Exception as e: log_and_print(f"Could not calculate Baseline CV: {e}", Colors.WARNING)

# --- Baseline Model Feature Importance ---
log_and_print("Calculating Baseline Model feature importances...", Colors.BLUE)
feature_imp_df_base = pd.DataFrame()
try:
    importances_base = base_rf.feature_importances_
    feature_names_from_model_base = X_train_final.columns
    feature_imp_df_base = pd.DataFrame({'Feature': feature_names_from_model_base, 'Importance': importances_base})
    feature_imp_df_base = feature_imp_df_base.sort_values('Importance', ascending=False).reset_index(drop=True)
    log_and_print("\nTop 20 Most Important Features (Baseline Model):", color=Colors.HEADER)
    print(Colors.GREEN + feature_imp_df_base.head(20).to_string() + Colors.ENDC)
    log_messages.append("\nTop 20 Features (Baseline):\n" + feature_imp_df_base.head(20).to_string())
    imp_path_base = os.path.join(OUTPUT_DIR, 'feature_importances_baseline.csv') # Simplified filename
    feature_imp_df_base.to_csv(imp_path_base, index=False)
    log_and_print(f"Baseline full feature importances saved to {imp_path_base}", Colors.GREEN)
    plt.figure(figsize=(12, 10)); top_n = 20
    plt.barh(feature_imp_df_base['Feature'][:top_n], feature_imp_df_base['Importance'][:top_n], color='darkseagreen')
    plt.xlabel("Importance"); plt.ylabel("Feature")
    plt.title(f"Top {top_n} Feature Importances - Baseline ({OUTPUT_DIR_NAME})", fontsize=14)
    plt.gca().invert_yaxis(); plt.tight_layout()
    plot_path_base_imp = os.path.join(FEATURE_IMPORTANCE_PLOTS_DIR, 'baseline_feature_importances_plot.png')
    plt.savefig(plot_path_base_imp, dpi=150); plt.close()
    log_and_print(f"Baseline feature importance plot saved to {plot_path_base_imp}", Colors.GREEN)
except Exception as e: log_and_print(f"Baseline feature importance error: {e}", Colors.WARNING)

# --- Save Baseline Model Final Metrics ---
log_and_print("Saving baseline model final metrics...", Colors.BLUE)
metrics_summary_df_base = pd.DataFrame({
    'metric': ['baseline_test_accuracy', 'baseline_test_roc_auc', 'baseline_cv_train_mean_roc_auc', 'baseline_cv_train_std_roc_auc'],
    'value': [test_accuracy_base, test_roc_auc_base, cv_mean_score_base, cv_std_score_base]
})
metrics_path_base = os.path.join(OUTPUT_DIR, 'model_metrics_baseline.csv') # Simplified filename
metrics_summary_df_base.to_csv(metrics_path_base, index=False)
log_and_print(f"Baseline metrics saved to {metrics_path_base}", Colors.GREEN)

# --- Hyperparameter Tuning ---
rs = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=1), param_distributions=param_dist,
    n_iter=N_ITER_SEARCH, scoring='roc_auc', cv=CV_FOLDS, random_state=42, verbose=3, n_jobs=4,
    # callbacks=[log_search_progress_callback] # Uncomment if desired
)
baseline_best_params = {
    'n_estimators': 200, 'max_depth': 30, 'min_samples_split': 20, 'min_samples_leaf': 10,
    'max_features': 'sqrt', 'class_weight': 'balanced_subsample', 'random_state': 42, 'n_jobs': -1
}
best_params = baseline_best_params.copy()
try:
    rs.fit(X_train_final, y_train_final)
    if rs.best_params_:
        best_params_from_search = rs.best_params_
        log_and_print(f"Best parameters from RandomizedSearch: {best_params_from_search}", Colors.GREEN)
        best_params = {**best_params_from_search, 'random_state': 42, 'n_jobs': -1}
        pd.DataFrame([best_params]).to_csv(os.path.join(OUTPUT_DIR, 'best_params_tuned.csv'), index=False) # Simplified
        log_and_print("Best parameters saved.", Colors.GREEN)
    else:
        log_and_print("RandomizedSearchCV found no best_params_. Using baseline.", Colors.WARNING)
except Exception as e:
    log_and_print(f"Error during RandomizedSearch: {e}. Using baseline params.", Colors.FAIL)

# --- Train Final Model ---
log_and_print("Training final model with selected parameters...", Colors.WARNING)
final_model = RandomForestClassifier(**best_params)
try:
    final_model.fit(X_train_final, y_train_final)
    log_and_print("Model training complete.", Colors.GREEN)
except Exception as e: log_and_print(f"Error fitting final model: {e}. Exiting.", Colors.FAIL); exit()

# --- Evaluation (Tuned Model) ---
log_and_print("Evaluating tuned model performance...", Colors.GREEN)
report_str, test_accuracy, test_roc_auc = "Report N/A", np.nan, np.nan
y_proba = None
try:
    y_pred = final_model.predict(X_test_final)
    y_proba = final_model.predict_proba(X_test_final)[:, 1]
    test_accuracy = accuracy_score(y_test_final, y_pred)
    test_roc_auc = roc_auc_score(y_test_final, y_proba)
    report_str = classification_report(y_test_final, y_pred, target_names=['Dire Win', 'Radiant Win'])
    log_and_print("\n--- Tuned Model Test Set Evaluation ---", color=Colors.HEADER)
    log_and_print(f"Accuracy: {test_accuracy:.4f}", color=Colors.GREEN)
    log_and_print(f"ROC-AUC: {test_roc_auc:.4f}", color=Colors.GREEN)
    log_and_print("\nClassification Report:", color=Colors.GREEN)
    print(Colors.GREEN + f"\n{report_str}" + Colors.ENDC); log_messages.append(f"\n{report_str}")
    log_and_print("---------------------------------------------", color=Colors.HEADER)
except Exception as e: log_and_print(f"Error during tuned model evaluation: {e}.", Colors.FAIL)

# --- ROC AUC Curve (Tuned Model) ---
log_and_print("Generating and saving Tuned Model ROC AUC curve...", Colors.BLUE)
if y_proba is not None and not np.isnan(test_roc_auc):
    try:
        fpr, tpr, _ = roc_curve(y_test_final, y_proba)
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {test_roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('FPR'); plt.ylabel('TPR')
        plt.title(f'ROC Curve - Tuned Model ({OUTPUT_DIR_NAME})', fontsize=14)
        plt.legend(loc="lower right"); plt.grid(True, linestyle='--', alpha=0.7)
        roc_plot_path = os.path.join(ROC_AUC_CURVES_DIR, 'tuned_roc_auc_plot.png') # Simplified
        plt.savefig(roc_plot_path, dpi=150, bbox_inches='tight'); plt.close()
        log_and_print(f"Tuned ROC AUC plot saved to {roc_plot_path}", Colors.GREEN)
    except Exception as e: log_and_print(f"Could not generate/save Tuned ROC AUC plot: {e}", Colors.WARNING)
else: log_and_print("Skipping Tuned ROC AUC plot: y_proba or test_roc_auc invalid.", Colors.WARNING)

# --- Cross-Validation Scores (Tuned Model) ---
log_and_print("\nCalculating Tuned Model CV Scores (Training Data)...", Colors.BLUE)
cv_mean_score, cv_std_score = np.nan, np.nan
try:
    cv_model = RandomForestClassifier(**best_params)
    cv_scores = cross_val_score(cv_model, X_train_final, y_train_final, cv=3, scoring='roc_auc', n_jobs=-1)
    cv_mean_score = cv_scores.mean(); cv_std_score = cv_scores.std()
    log_and_print(f"Tuned CV ROC-AUC Scores: {cv_scores}", Colors.GREEN)
    log_and_print(f"Tuned Avg CV ROC-AUC: {cv_mean_score:.4f} (+/- {cv_std_score*2:.4f})", Colors.GREEN)
except Exception as e: log_and_print(f"Could not calculate Tuned CV: {e}", Colors.WARNING)

# --- Save Sample Decision Tree Plots (Tuned Model) ---
log_and_print(f"Saving sample tree plots to {TREE_PLOT_DIR}...", Colors.BLUE)
n_trees_to_plot = 3 # Reduced for brevity
if hasattr(final_model, 'estimators_') and len(final_model.estimators_) >= 1:
    for i in range(min(n_trees_to_plot, len(final_model.estimators_))):
        try:
            plt.figure(figsize=(35, 25))
            plot_tree(final_model.estimators_[i], feature_names=X_train_final.columns.tolist(),
                      class_names=['Dire Win', 'Radiant Win'], filled=True, impurity=True,
                      rounded=True, precision=3, max_depth=3, fontsize=10)
            plot_filename = f"tuned_sample_tree_{i}_plot.png" # Simplified
            plt.title(f"Sample Tree {i} - Tuned Model ({OUTPUT_DIR_NAME}, Max Depth 3)", fontsize=18)
            plt.savefig(os.path.join(TREE_PLOT_DIR, plot_filename), dpi=100, bbox_inches='tight'); plt.close()
            log_and_print(f"  Saved plot for tree {i}", Colors.GREEN)
        except Exception as e: log_and_print(f"  Error plotting tree {i}: {e}", Colors.WARNING)
else: log_and_print(f"  Could not plot trees: no estimators found.", Colors.WARNING)

# --- Feature Importance (Tuned Model) ---
log_and_print("Calculating Tuned Model feature importances...", Colors.BLUE)
feature_imp_df = pd.DataFrame()
try:
    importances = final_model.feature_importances_
    feature_imp_df = pd.DataFrame({'Feature': X_train_final.columns, 'Importance': importances})
    feature_imp_df = feature_imp_df.sort_values('Importance', ascending=False).reset_index(drop=True)
    log_and_print("\nTop 20 Features (Tuned Model):", color=Colors.HEADER)
    print(Colors.GREEN + feature_imp_df.head(20).to_string() + Colors.ENDC)
    log_messages.append("\nTop 20 Features (Tuned):\n" + feature_imp_df.head(20).to_string())
    imp_path = os.path.join(OUTPUT_DIR, 'feature_importances_tuned.csv') # Simplified
    feature_imp_df.to_csv(imp_path, index=False)
    log_and_print(f"Tuned feature importances saved to {imp_path}", Colors.GREEN)
    plt.figure(figsize=(12, 10)); top_n = 20
    plt.barh(feature_imp_df['Feature'][:top_n], feature_imp_df['Importance'][:top_n], color='mediumpurple')
    plt.xlabel("Importance"); plt.ylabel("Feature")
    plt.title(f"Top {top_n} Feature Importances - Tuned Model ({OUTPUT_DIR_NAME})", fontsize=14)
    plt.gca().invert_yaxis(); plt.tight_layout()
    plot_path = os.path.join(FEATURE_IMPORTANCE_PLOTS_DIR, 'tuned_feature_importances_plot.png') # Simplified
    plt.savefig(plot_path, dpi=150); plt.close()
    log_and_print(f"Tuned feature importance plot saved to {plot_path}", Colors.GREEN)
except Exception as e: log_and_print(f"Tuned feature importance error: {e}", Colors.WARNING)

# --- Confusion Matrix (Tuned Model) ---
log_and_print("Generating and saving Tuned Model Confusion Matrix plot...", Colors.BLUE)
if 'y_pred' in locals() and y_pred is not None:
    try:
        cm = confusion_matrix(y_test_final, y_pred, labels=final_model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Dire Win', 'Radiant Win'])
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
        ax.set_title(f'Confusion Matrix - Tuned Model ({OUTPUT_DIR_NAME})', fontsize=14) # Simplified
        plt.xticks(fontsize=10); plt.yticks(fontsize=10)
        ax.xaxis.label.set_fontsize(12); ax.yaxis.label.set_fontsize(12)
        cm_plot_path = os.path.join(EVALUATION_MATRIX_DIR, 'tuned_confusion_matrix.png') # Simplified
        plt.savefig(cm_plot_path, dpi=150, bbox_inches='tight'); plt.close(fig)
        log_and_print(f"Tuned Confusion Matrix plot saved to {cm_plot_path}", Colors.GREEN)
        cm_log_message = f"\nConfusion Matrix (Tuned Test Set):\n{cm}\n"
        print(Colors.GREEN + cm_log_message + Colors.ENDC); log_messages.append(cm_log_message)
    except Exception as e: log_and_print(f"Could not generate/save Tuned CM plot: {e}", Colors.WARNING)
else: log_and_print("Skipping Tuned CM plot: y_pred not defined.", Colors.WARNING)

# --- Save Final Metrics & Report Summary (Tuned Model) ---
log_and_print("Saving final tuned model metrics and report summary...", Colors.BLUE)
metrics_summary_df = pd.DataFrame({
    'metric': ['tuned_test_accuracy', 'tuned_test_roc_auc', 'tuned_cv_train_mean_roc_auc', 'tuned_cv_train_std_roc_auc'],
    'value': [test_accuracy, test_roc_auc, cv_mean_score, cv_std_score]
})
metrics_path = os.path.join(OUTPUT_DIR, 'model_metrics_tuned.csv') # Simplified
metrics_summary_df.to_csv(metrics_path, index=False)
log_and_print(f"Tuned metrics saved to {metrics_path}", Colors.GREEN)

report_summary_path = os.path.join(OUTPUT_DIR, 'summary_report.md') # Simplified
try:
    X_train_shape_str = str(X_train_final.shape) if 'X_train_final' in locals() else "N/A"
    len_X_train_str = str(len(X_train_final.index)) if 'X_train_final' in locals() else "N/A"
    len_X_test_str = str(len(X_test_final.index)) if 'X_test_final' in locals() else "N/A"
    best_params_str = json.dumps(best_params, indent=4) if 'best_params' in locals() else "{}"
    param_dist_str = json.dumps(param_dist, indent=4)
    test_accuracy_str = f"{test_accuracy:.4f}" if not np.isnan(test_accuracy) else "N/A"
    test_roc_auc_str = f"{test_roc_auc:.4f}" if not np.isnan(test_roc_auc) else "N/A"
    cv_mean_score_str = f"{cv_mean_score:.4f}" if not np.isnan(cv_mean_score) else "N/A"
    cv_std_dev_str = f"{cv_std_score*2:.4f}" if not np.isnan(cv_std_score) else "N/A"
    report_str_md = report_str if report_str != "Report N/A" else "Classification report not available."
    feature_imp_head_str = feature_imp_df.head(20).to_string() if not feature_imp_df.empty else "N/A"
    cols_to_drop_len_str = str(len(cols_to_drop)) if 'cols_to_drop' in locals() else "Some"

    with open(report_summary_path, 'w', encoding='utf-8') as f:
        f.write(f"# Dota 2 Match Outcome Analysis: {OUTPUT_DIR_NAME}\n\n")
        f.write("## 1. Abstract\n")
        f.write("This project evaluates an improved feature encoding strategy (+/- named features for heroes and items) and optimizes a Random Forest model using RandomizedSearchCV for Dota 2 match outcome prediction. This analysis intentionally uses features including final items (data known at match conclusion) to understand their impact and explore predictability under \"leaky\" conditions. The aim is to critically examine feature representation, data leakage, and model interpretation, not to build a pre-match predictive tool.\n\n")
        f.write("## 2. Introduction\n")
        f.write(f"* **Data Scope:** Matches from `{MATCHES_FILE.split('/')[-1]}`, players from `{PLAYERS_FILE.split('/')[-1]}`.\n\n")
        f.write("## 3. Methodology\n\n")
        f.write("### 3.1. Data Preprocessing & Feature Engineering\n")
        f.write(f"* **Output Directory:** `{OUTPUT_DIR}`\n")
        f.write(f"* **Transformations Saved in:** `{TRANSFORMATIONS_DIR}`\n")
        f.write(f"  * Loaded Data: `{LOADED_DATA_DIR}`\n")
        f.write(f"  * Preprocessed Data: `{PREPROCESSED_DATA_DIR}`\n")
        f.write(f"  * Pivoted Features: `{PIVOTED_DATA_DIR}` (hero_pivot.csv, item_pivot.csv)\n")
        f.write(f"  * Combined Features: `{COMBINED_FEATURES_DIR}` (before_pruning.csv, after_pruning.csv)\n")
        f.write(f"* **Zero-Variance Pruning:** {cols_to_drop_len_str} features removed.\n")
        f.write(f"* **Final Training Feature Shape:** `{X_train_shape_str}`.\n\n")
        f.write("### 3.2. Data Splitting\n")
        f.write(f"* **Strategy:** Time-based 80% train ({len_X_train_str} matches), 20% test ({len_X_test_str} matches).\n")
        f.write(f"* **Split Data Saved in:** `{SPLIT_DATA_DIR}` (X_train.csv, y_train.csv, etc.)\n\n")
        f.write("### 3.3. Model Selection and Hyperparameter Tuning\n")
        f.write("* **Algorithm:** RandomForestClassifier.\n")
        f.write(f"* **Tuning:** RandomizedSearchCV (roc_auc scoring).\n")
        f.write(f"  * **Parameter Grid:**\n```json\n{param_dist_str}\n```\n")
        f.write(f"  * **Best Parameters (Tuned Model):**\n```json\n{best_params_str}\n```\n\n")
        f.write("## 4. Results and Discussion\n\n")
        f.write("### 4.1. Baseline Model Performance\n")
        f.write(f"* Test Accuracy: {test_accuracy_base:.4f}\n")
        f.write(f"* Test ROC AUC: {test_roc_auc_base:.4f}\n")
        f.write(f"* Avg Training CV ROC AUC: {cv_mean_score_base:.4f} (+/- {cv_std_score_base*2:.4f})\n")
        f.write(f"* Plots: Baseline Confusion Matrix in `{EVALUATION_MATRIX_DIR}`, ROC Curve in `{ROC_AUC_CURVES_DIR}`, Feature Importances in `{FEATURE_IMPORTANCE_PLOTS_DIR}`.\n\n")
        f.write("### 4.2. Tuned Model Performance on Test Set\n")
        f.write(f"* **Test Accuracy:** {test_accuracy_str}\n")
        f.write(f"* **Test ROC AUC:** {test_roc_auc_str}\n")
        f.write(f"* **Avg Training CV ROC AUC:** {cv_mean_score_str} (+/- {cv_std_dev_str})\n")
        f.write(f"* **Plots Saved:**\n")
        f.write(f"  * Confusion Matrix: `evaluation_matrix/tuned_confusion_matrix.png`\n")
        f.write(f"  * ROC AUC Curve: `roc_auc_curves/tuned_roc_auc_plot.png`\n")
        f.write(f"  * Feature Importances: `feature_importance_plots/tuned_feature_importances_plot.png`\n")
        f.write(f"  * Sample Trees: `sample_trees/tuned_sample_tree_i_plot.png`\n\n")
        f.write("### 4.3. Classification Report (Tuned Model - Test Set)\n")
        f.write(f"```\n{report_str_md}\n```\n\n")
        f.write("### 4.4. Top 20 Feature Importances (Tuned Model)\n")
        f.write(f"```\n{feature_imp_head_str}\n```\n\n")
        f.write("### 4.5. Data Leakage Discussion\n")
        f.write("The high performance is largely due to data leakage from final item builds. This model is not for pre-match prediction but illustrates encoding impact under leaky conditions.\n\n")
        f.write("## 5. Conclusion\n")
        f.write(f"The '+/- named feature' encoding with an optimized Random Forest achieved high performance (Test ROC AUC: {test_roc_auc_str}) using leaky data. This highlights feature definition, encoding, and leakage awareness.\n\n")
    log_and_print(f"Summary report saved to {report_summary_path}", Colors.GREEN)
except Exception as e:
    log_and_print(f"Error saving summary report: {e}", Colors.WARNING)

# --- Save Logs ---
log_file_path = os.path.join(OUTPUT_DIR, 'experiment_log.txt') # Simplified
save_log_file(log_file_path)
log_and_print(f"Script finished for {OUTPUT_DIR}.", Colors.BOLD + Colors.GREEN)