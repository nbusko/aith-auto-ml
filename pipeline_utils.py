"""
Утилиты для предобработки данных в pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, LabelEncoder


def identify_feature_types(df, target_col='class', id_col='id'):
    """
    Идентифицирует типы признаков.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Датасет
    target_col : str
        Название целевой переменной
    id_col : str
        Название столбца с ID
        
    Returns:
    --------
    dict
        Словарь с типами признаков
    """
    feature_cols = [col for col in df.columns if col not in [target_col, id_col]]
    
    numeric_features = []
    categorical_features = []
    
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_features.append(col)
        else:
            categorical_features.append(col)
    
    return {
        'numeric': numeric_features,
        'categorical': categorical_features
    }


def preprocess_data(train_df, val_df, numeric_features, categorical_features, target_col='class'):
    """
    Предобрабатывает данные.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Тренировочный датасет
    val_df : pd.DataFrame
        Validation датасет
    numeric_features : list
        Список численных признаков
    categorical_features : list
        Список категориальных признаков
    target_col : str
        Название целевой переменной
        
    Returns:
    --------
    tuple
        (X_train_processed, X_val_processed, y_train, y_val, feature_names)
    """
    # Обработка численных признаков
    numeric_imputer = SimpleImputer(strategy='median')
    numeric_scaler = RobustScaler()
    
    X_train_num = train_df[numeric_features].copy()
    X_val_num = val_df[numeric_features].copy()
    
    X_train_num_imputed = pd.DataFrame(
        numeric_imputer.fit_transform(X_train_num),
        columns=numeric_features,
        index=X_train_num.index
    )
    X_val_num_imputed = pd.DataFrame(
        numeric_imputer.transform(X_val_num),
        columns=numeric_features,
        index=X_val_num.index
    )
    
    X_train_num_scaled = pd.DataFrame(
        numeric_scaler.fit_transform(X_train_num_imputed),
        columns=numeric_features,
        index=X_train_num_imputed.index
    )
    X_val_num_scaled = pd.DataFrame(
        numeric_scaler.transform(X_val_num_imputed),
        columns=numeric_features,
        index=X_val_num_imputed.index
    )
    
    # Обработка категориальных признаков (one-hot encoding)
    X_train_cat = train_df[categorical_features].copy()
    X_val_cat = val_df[categorical_features].copy()
    
    # Заполнение пропусков
    X_train_cat = X_train_cat.fillna('missing')
    X_val_cat = X_val_cat.fillna('missing')
    
    # One-hot encoding
    X_train_cat_encoded = pd.get_dummies(X_train_cat, columns=categorical_features, prefix=categorical_features)
    X_val_cat_encoded = pd.get_dummies(X_val_cat, columns=categorical_features, prefix=categorical_features)
    
    # Выравнивание столбцов (на случай разных значений в train и val)
    all_cat_cols = set(X_train_cat_encoded.columns) | set(X_val_cat_encoded.columns)
    
    # Создаем новые датафреймы с правильными столбцами
    train_cat_final = pd.DataFrame(0, index=X_train_cat_encoded.index, columns=sorted(all_cat_cols))
    val_cat_final = pd.DataFrame(0, index=X_val_cat_encoded.index, columns=sorted(all_cat_cols))
    
    for col in X_train_cat_encoded.columns:
        train_cat_final[col] = X_train_cat_encoded[col]
    for col in X_val_cat_encoded.columns:
        val_cat_final[col] = X_val_cat_encoded[col]
    
    # Объединение численных и категориальных признаков
    X_train_processed = pd.concat([X_train_num_scaled, train_cat_final], axis=1)
    X_val_processed = pd.concat([X_val_num_scaled, val_cat_final], axis=1)
    
    # Выравнивание столбцов
    common_cols = X_train_processed.columns.intersection(X_val_processed.columns)
    X_train_processed = X_train_processed[common_cols]
    X_val_processed = X_val_processed[common_cols]
    
    # Целевая переменная
    le = LabelEncoder()
    y_train = le.fit_transform(train_df[target_col])
    y_val = le.transform(val_df[target_col])
    
    return X_train_processed, X_val_processed, y_train, y_val, list(X_train_processed.columns)

