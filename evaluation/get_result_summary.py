import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def normalize_smw_label(label_idx):
    return 5 if label_idx == 6 else label_idx


def build_results_dataframe(file_paths, true_labels, pred_labels, label_names, extra_columns=None):
    results_df = pd.DataFrame({
        'file_path': file_paths,
        'true_label': true_labels,
        'pred_label': pred_labels,
    })

    if extra_columns is not None:
        for column_name, values in extra_columns.items():
            results_df[column_name] = values

    idx_to_class = {idx: cls for idx, cls in enumerate(label_names)}
    results_df['true_label_name'] = results_df['true_label'].map(idx_to_class)
    results_df['pred_label_name'] = results_df['pred_label'].map(idx_to_class)

    return results_df


def summarize_classification_results(results_df, label_names, display_order=None):
    y_true = results_df['true_label'].tolist()
    y_pred = results_df['pred_label'].tolist()

    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')

    class_to_idx = {cls: idx for idx, cls in enumerate(label_names)}
    ordered_class_names = [cls for cls in (display_order or label_names) if cls in class_to_idx]

    if len(ordered_class_names) != len(class_to_idx):
        ordered_class_names += [cls for cls in label_names if cls not in ordered_class_names]

    ordered_label_indices = [class_to_idx[cls] for cls in ordered_class_names]
    cm = confusion_matrix(y_true, y_pred, labels=ordered_label_indices)
    cm_df = pd.DataFrame(cm, index=ordered_class_names, columns=ordered_class_names)

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'cm_df': cm_df,
    }


def save_results_dataframe(results_df, save_path):
    results_df.to_csv(save_path, index=False)
    return save_path