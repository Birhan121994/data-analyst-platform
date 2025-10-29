import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.ensemble import IsolationForest
from scipy import stats
import json
import re
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self):
        self.df = None
        self.original_df = None
        self.cleaning_report = {}
        self.original_dtypes = {}
        self.transformations_applied = []
        self.feature_engineering_applied = []
    
    def load_data(self, file_path, file_type):
        """Load data from various sources"""
        try:
            if file_type == 'csv':
                self.df = pd.read_csv(file_path, low_memory=False)
            elif file_type == 'excel':
                self.df = pd.read_excel(file_path)
            elif file_type == 'json':
                self.df = pd.read_json(file_path)
            
            # Store original data and dtypes
            self.original_df = self.df.copy()
            self.original_dtypes = self.df.dtypes.astype(str).to_dict()
            
            # Initialize cleaning report
            self.cleaning_report = {
                'initial_rows': len(self.df),
                'initial_columns': len(self.df.columns),
                'initial_memory_usage': self.df.memory_usage(deep=True).sum(),
                'cleaning_steps': [],
                'transformations_applied': [],
                'rows_removed': 0,
                'columns_removed': 0
            }
            return True
        except Exception as e:
            return str(e)
    
    def get_basic_info(self):
        """Get comprehensive information about the dataset"""
        if self.df is None:
            return None
        
        try:
            # Convert dtypes to string to avoid serialization issues
            dtypes_dict = {}
            for col in self.df.columns:
                dtypes_dict[col] = str(self.df[col].dtype)
            
            info = {
                'shape': self.df.shape,
                'columns': list(self.df.columns),
                'data_types': dtypes_dict,
                'missing_values': self.df.isnull().sum().to_dict(),
                'missing_percentage': (self.df.isnull().sum() / len(self.df) * 100).round(2).to_dict(),
                'memory_usage': self.df.memory_usage(deep=True).sum(),
                'duplicate_rows': self.df.duplicated().sum(),
                'numeric_columns': self.df.select_dtypes(include=[np.number]).columns.tolist(),
                'categorical_columns': self.df.select_dtypes(include=['object', 'category']).columns.tolist(),
                'datetime_columns': self.df.select_dtypes(include=['datetime64']).columns.tolist(),
                'boolean_columns': self.df.select_dtypes(include=['bool']).columns.tolist(),
            }
            return info
        except Exception as e:
            print(f"Error in get_basic_info: {e}")
            return None
    
    def detect_outliers(self, column):
        """Detect outliers using multiple methods"""
        if column not in self.df.columns or not np.issubdtype(self.df[column].dtype, np.number):
            return None
        
        try:
            # IQR method
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound_iqr = Q1 - 1.5 * IQR
            upper_bound_iqr = Q3 + 1.5 * IQR
            
            outliers_iqr = self.df[(self.df[column] < lower_bound_iqr) | (self.df[column] > upper_bound_iqr)]
            outlier_count_iqr = len(outliers_iqr)
            
            # Z-score method
            z_scores = np.abs(stats.zscore(self.df[column].dropna()))
            outlier_count_z = len(z_scores[z_scores > 3])
            
            # Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            if len(self.df[column].dropna()) > 0:
                preds = iso_forest.fit_predict(self.df[[column]].dropna())
                outlier_count_iso = len(preds[preds == -1])
            else:
                outlier_count_iso = 0
            
            return {
                'count_iqr': outlier_count_iqr,
                'percentage_iqr': round((outlier_count_iqr / len(self.df) * 100), 2),
                'count_zscore': outlier_count_z,
                'percentage_zscore': round((outlier_count_z / len(self.df) * 100), 2),
                'count_isolation': outlier_count_iso,
                'percentage_isolation': round((outlier_count_iso / len(self.df) * 100), 2),
                'lower_bound_iqr': float(lower_bound_iqr),
                'upper_bound_iqr': float(upper_bound_iqr)
            }
        except Exception as e:
            return {
                'count_iqr': 0,
                'percentage_iqr': 0.0,
                'count_zscore': 0,
                'percentage_zscore': 0.0,
                'count_isolation': 0,
                'percentage_isolation': 0.0,
                'lower_bound_iqr': None,
                'upper_bound_iqr': None,
                'error': str(e)
            }
    
    def clean_data(self, strategies=None, remove_duplicates=True, handle_outliers=True, standardize_text=True):
        """Comprehensive data cleaning with multiple strategies"""
        if self.df is None:
            return False
        
        original_shape = self.df.shape
        cleaning_steps = []
        
        try:
            # Step 1: Handle missing values with strategies
            if strategies:
                for column, strategy in strategies.items():
                    if column in self.df.columns:
                        missing_count = self.df[column].isnull().sum()
                        if missing_count > 0:
                            if strategy == 'drop_column':
                                self.df.drop(column, axis=1, inplace=True)
                                cleaning_steps.append(f"Dropped column {column} ({missing_count} missing values)")
                            
                            elif strategy == 'drop_rows':
                                before_count = len(self.df)
                                self.df = self.df.dropna(subset=[column])
                                after_count = len(self.df)
                                removed = before_count - after_count
                                if removed > 0:
                                    cleaning_steps.append(f"Dropped {removed} rows with missing values in {column}")
                            
                            elif strategy == 'mean' and np.issubdtype(self.df[column].dtype, np.number):
                                imputer = SimpleImputer(strategy='mean')
                                self.df[column] = imputer.fit_transform(self.df[[column]]).ravel()
                                cleaning_steps.append(f"Filled {missing_count} missing values in {column} with mean")
                            
                            elif strategy == 'median' and np.issubdtype(self.df[column].dtype, np.number):
                                imputer = SimpleImputer(strategy='median')
                                self.df[column] = imputer.fit_transform(self.df[[column]]).ravel()
                                cleaning_steps.append(f"Filled {missing_count} missing values in {column} with median")
                            
                            elif strategy == 'mode':
                                mode_value = self.df[column].mode()
                                if len(mode_value) > 0:
                                    self.df[column].fillna(mode_value[0], inplace=True)
                                    cleaning_steps.append(f"Filled {missing_count} missing values in {column} with mode")
                            
                            elif strategy == 'knn':
                                if np.issubdtype(self.df[column].dtype, np.number):
                                    imputer = KNNImputer(n_neighbors=5)
                                    self.df[column] = imputer.fit_transform(self.df[[column]]).ravel()
                                    cleaning_steps.append(f"Filled {missing_count} missing values in {column} using KNN")
                            
                            elif strategy == 'forward_fill':
                                self.df[column].fillna(method='ffill', inplace=True)
                                cleaning_steps.append(f"Forward filled {missing_count} missing values in {column}")
                            
                            elif strategy == 'backward_fill':
                                self.df[column].fillna(method='bfill', inplace=True)
                                cleaning_steps.append(f"Backward filled {missing_count} missing values in {column}")
                            
                            elif strategy == 'zero':
                                self.df[column].fillna(0, inplace=True)
                                cleaning_steps.append(f"Filled {missing_count} missing values in {column} with zero")
                            
                            elif strategy == 'unknown':
                                self.df[column].fillna('Unknown', inplace=True)
                                cleaning_steps.append(f"Filled {missing_count} missing values in {column} with 'Unknown'")
            
            # Step 2: Remove duplicates
            if remove_duplicates:
                duplicates_before = self.df.duplicated().sum()
                self.df = self.df.drop_duplicates()
                duplicates_removed = duplicates_before - self.df.duplicated().sum()
                if duplicates_removed > 0:
                    cleaning_steps.append(f"Removed {duplicates_removed} duplicate rows")
            
            # Step 3: Handle outliers
            if handle_outliers:
                numeric_columns = self.df.select_dtypes(include=[np.number]).columns
                outliers_handled = 0
                
                for column in numeric_columns:
                    outlier_info = self.detect_outliers(column)
                    if outlier_info and outlier_info['count_iqr'] > 0:
                        Q1 = self.df[column].quantile(0.25)
                        Q3 = self.df[column].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        # Cap outliers (winsorization)
                        before_outliers = outlier_info['count_iqr']
                        self.df[column] = np.where(self.df[column] < lower_bound, lower_bound, self.df[column])
                        self.df[column] = np.where(self.df[column] > upper_bound, upper_bound, self.df[column])
                        outliers_handled += before_outliers
                
                if outliers_handled > 0:
                    cleaning_steps.append(f"Capped {outliers_handled} outliers in {len(numeric_columns)} numeric columns")
            
            # Step 4: Standardize text data
            if standardize_text:
                text_columns = self.df.select_dtypes(include=['object']).columns
                for column in text_columns:
                    if column in self.df.columns:
                        # Remove extra whitespace
                        self.df[column] = self.df[column].astype(str).str.strip()
                        # Replace multiple spaces with single space
                        self.df[column] = self.df[column].str.replace(r'\s+', ' ', regex=True)
                        # Standardize case
                        self.df[column] = self.df[column].str.title()
                
                if len(text_columns) > 0:
                    cleaning_steps.append(f"Standardized text in {len(text_columns)} columns")
            
            # Step 5: Fix data types
            for column in self.df.columns:
                # Convert numeric strings to numbers
                if self.df[column].dtype == 'object':
                    try:
                        self.df[column] = pd.to_numeric(self.df[column], errors='ignore')
                    except:
                        pass
                
                # Convert date strings to datetime
                if self.df[column].dtype == 'object':
                    try:
                        self.df[column] = pd.to_datetime(self.df[column], errors='ignore')
                    except:
                        pass
            
            # Update cleaning report
            self.cleaning_report['final_rows'] = len(self.df)
            self.cleaning_report['final_columns'] = len(self.df.columns)
            self.cleaning_report['rows_removed'] = original_shape[0] - len(self.df)
            self.cleaning_report['columns_removed'] = original_shape[1] - len(self.df.columns)
            self.cleaning_report['cleaning_steps'] = cleaning_steps
            self.cleaning_report['final_memory_usage'] = self.df.memory_usage(deep=True).sum()
            
            return True
            
        except Exception as e:
            print(f"Error in clean_data: {e}")
            return False
    
    def transform_data(self, transformations):
        """Apply various data transformations"""
        if self.df is None:
            return False
        
        transformation_steps = []
        
        for transformation in transformations:
            column = transformation.get('column')
            operation = transformation.get('operation')
            parameters = transformation.get('parameters', {})
            
            if column not in self.df.columns:
                continue
            
            try:
                if operation == 'normalize':
                    scaler = StandardScaler()
                    self.df[column] = scaler.fit_transform(self.df[[column]])
                    transformation_steps.append(f"Normalized {column} using StandardScaler")
                
                elif operation == 'minmax_scale':
                    scaler = MinMaxScaler()
                    self.df[column] = scaler.fit_transform(self.df[[column]])
                    transformation_steps.append(f"Scaled {column} using MinMaxScaler")
                
                elif operation == 'robust_scale':
                    scaler = RobustScaler()
                    self.df[column] = scaler.fit_transform(self.df[[column]])
                    transformation_steps.append(f"Scaled {column} using RobustScaler")
                
                elif operation == 'log_transform':
                    min_val = self.df[column].min()
                    if min_val <= 0:
                        shift = abs(min_val) + 1
                        self.df[column] = np.log1p(self.df[column] + shift)
                        transformation_steps.append(f"Applied log transformation to {column} (shifted by {shift})")
                    else:
                        self.df[column] = np.log1p(self.df[column])
                        transformation_steps.append(f"Applied log transformation to {column}")
                
                elif operation == 'square_root':
                    min_val = self.df[column].min()
                    if min_val < 0:
                        shift = abs(min_val)
                        self.df[column] = np.sqrt(self.df[column] + shift)
                        transformation_steps.append(f"Applied square root transformation to {column} (shifted by {shift})")
                    else:
                        self.df[column] = np.sqrt(self.df[column])
                        transformation_steps.append(f"Applied square root transformation to {column}")
                
                elif operation == 'box_cox':
                    if self.df[column].min() > 0:
                        self.df[column], _ = stats.boxcox(self.df[column])
                        transformation_steps.append(f"Applied Box-Cox transformation to {column}")
                
                elif operation == 'encode_categorical':
                    encoder = LabelEncoder()
                    self.df[column] = encoder.fit_transform(self.df[column].astype(str))
                    transformation_steps.append(f"Label encoded {column}")
                
                elif operation == 'one_hot_encode':
                    encoded_cols = pd.get_dummies(self.df[column], prefix=column, drop_first=True)
                    self.df = pd.concat([self.df, encoded_cols], axis=1)
                    self.df.drop(column, axis=1, inplace=True)
                    transformation_steps.append(f"One-hot encoded {column} into {len(encoded_cols.columns)} columns")
                
                elif operation == 'binning':
                    bins = parameters.get('bins', 5)
                    labels = parameters.get('labels', None)
                    if labels is None:
                        labels = [f'Bin_{i+1}' for i in range(bins)]
                    
                    try:
                        self.df[f'{column}_binned'] = pd.cut(self.df[column], bins=bins, labels=labels, duplicates='drop')
                        transformation_steps.append(f"Binned {column} into {bins} categories")
                    except:
                        # If cutting fails, use qcut
                        self.df[f'{column}_binned'] = pd.qcut(self.df[column], q=bins, labels=labels, duplicates='drop')
                        transformation_steps.append(f"Binned {column} into {bins} quantiles")
                
                elif operation == 'datetime_extract':
                    if self.df[column].dtype == 'object':
                        self.df[column] = pd.to_datetime(self.df[column], errors='coerce')
                    
                    if pd.api.types.is_datetime64_any_dtype(self.df[column]):
                        self.df[f'{column}_year'] = self.df[column].dt.year
                        self.df[f'{column}_month'] = self.df[column].dt.month
                        self.df[f'{column}_day'] = self.df[column].dt.day
                        self.df[f'{column}_dayofweek'] = self.df[column].dt.dayofweek
                        self.df[f'{column}_quarter'] = self.df[column].dt.quarter
                        self.df[f'{column}_is_weekend'] = (self.df[column].dt.dayofweek >= 5).astype(int)
                        transformation_steps.append(f"Extracted datetime features from {column}")
                
                elif operation == 'text_clean':
                    self.df[column] = self.df[column].astype(str).apply(
                        lambda x: re.sub(r'[^a-zA-Z\s]', '', x).strip()
                    )
                    transformation_steps.append(f"Cleaned text in {column}")
                
                elif operation == 'remove_column':
                    self.df.drop(column, axis=1, inplace=True)
                    transformation_steps.append(f"Removed column {column}")
                
                elif operation == 'rename_column':
                    new_name = parameters.get('new_name')
                    if new_name and new_name not in self.df.columns:
                        self.df.rename(columns={column: new_name}, inplace=True)
                        transformation_steps.append(f"Renamed {column} to {new_name}")
                
                elif operation == 'clip_values':
                    min_val = parameters.get('min')
                    max_val = parameters.get('max')
                    if min_val is not None or max_val is not None:
                        self.df[column] = self.df[column].clip(lower=min_val, upper=max_val)
                        transformation_steps.append(f"Clipped {column} to range [{min_val}, {max_val}]")
                
            except Exception as e:
                transformation_steps.append(f"Failed to transform {column}: {str(e)}")
                continue
        
        self.transformations_applied.extend(transformation_steps)
        return True
    
    def feature_engineering(self, operations):
        """Create new features from existing data"""
        if self.df is None:
            return False
        
        engineering_steps = []
        
        for operation in operations:
            try:
                op_type = operation['type']
                
                if op_type == 'interaction':
                    col1 = operation['column1']
                    col2 = operation['column2']
                    if col1 in self.df.columns and col2 in self.df.columns:
                        if (np.issubdtype(self.df[col1].dtype, np.number) and 
                            np.issubdtype(self.df[col2].dtype, np.number)):
                            self.df[f'{col1}_x_{col2}'] = self.df[col1] * self.df[col2]
                            engineering_steps.append(f"Created interaction feature {col1}_x_{col2}")
                
                elif op_type == 'polynomial':
                    column = operation['column']
                    degree = operation.get('degree', 2)
                    if (column in self.df.columns and 
                        np.issubdtype(self.df[column].dtype, np.number)):
                        for d in range(2, degree + 1):
                            self.df[f'{column}_power_{d}'] = self.df[column] ** d
                        engineering_steps.append(f"Created polynomial features for {column} up to degree {degree}")
                
                elif op_type == 'aggregation':
                    group_by = operation['group_by']
                    aggregate_column = operation['aggregate_column']
                    method = operation['method']
                    
                    if (group_by in self.df.columns and 
                        aggregate_column in self.df.columns and
                        np.issubdtype(self.df[aggregate_column].dtype, np.number)):
                        
                        if method == 'mean':
                            agg_values = self.df.groupby(group_by)[aggregate_column].transform('mean')
                        elif method == 'sum':
                            agg_values = self.df.groupby(group_by)[aggregate_column].transform('sum')
                        elif method == 'count':
                            agg_values = self.df.groupby(group_by)[aggregate_column].transform('count')
                        elif method == 'max':
                            agg_values = self.df.groupby(group_by)[aggregate_column].transform('max')
                        elif method == 'min':
                            agg_values = self.df.groupby(group_by)[aggregate_column].transform('min')
                        elif method == 'std':
                            agg_values = self.df.groupby(group_by)[aggregate_column].transform('std')
                        
                        self.df[f'{aggregate_column}_{method}_by_{group_by}'] = agg_values
                        engineering_steps.append(f"Created aggregation feature for {aggregate_column} by {group_by}")
                
                elif op_type == 'datetime_feature':
                    column = operation['column']
                    if pd.api.types.is_datetime64_any_dtype(self.df[column]):
                        self.df[f'{column}_is_weekend'] = self.df[column].dt.dayofweek.isin([5, 6]).astype(int)
                        self.df[f'{column}_quarter'] = self.df[column].dt.quarter
                        self.df[f'{column}_is_month_start'] = self.df[column].dt.is_month_start.astype(int)
                        self.df[f'{column}_is_month_end'] = self.df[column].dt.is_month_end.astype(int)
                        self.df[f'{column}_hour'] = self.df[column].dt.hour
                        engineering_steps.append(f"Created datetime features for {column}")
                
                elif op_type == 'bin_count':
                    column = operation['column']
                    if column in self.df.columns:
                        self.df[f'{column}_bin_count'] = pd.cut(self.df[column], bins=10).astype(str)
                        engineering_steps.append(f"Created bin count feature for {column}")
                
                elif op_type == 'ratio':
                    col1 = operation['column1']
                    col2 = operation['column2']
                    if (col1 in self.df.columns and col2 in self.df.columns and
                        np.issubdtype(self.df[col1].dtype, np.number) and
                        np.issubdtype(self.df[col2].dtype, np.number)):
                        
                        # Avoid division by zero
                        self.df[f'{col1}_to_{col2}_ratio'] = self.df[col1] / (self.df[col2].replace(0, np.nan))
                        engineering_steps.append(f"Created ratio feature {col1}_to_{col2}_ratio")
                
            except Exception as e:
                engineering_steps.append(f"Failed feature engineering {op_type}: {str(e)}")
                continue
        
        self.feature_engineering_applied.extend(engineering_steps)
        return True
    
    def get_data_quality_report(self):
        """Generate comprehensive data quality report"""
        if self.df is None:
            return None
        
        try:
            report = {
                'basic_info': self.get_basic_info(),
                'cleaning_report': self.cleaning_report,
                'transformations_applied': self.transformations_applied,
                'feature_engineering_applied': self.feature_engineering_applied,
                'column_analysis': {},
                'data_quality_score': 0
            }
            
            quality_scores = []
            
            for column in self.df.columns:
                col_info = {
                    'dtype': str(self.df[column].dtype),
                    'missing_values': int(self.df[column].isnull().sum()),
                    'missing_percentage': round(float(self.df[column].isnull().sum() / len(self.df) * 100), 2),
                    'unique_values': int(self.df[column].nunique()),
                    'unique_percentage': round(float(self.df[column].nunique() / len(self.df) * 100), 2),
                }
                
                # Calculate column quality score (0-100)
                col_quality = 100
                
                # Penalize for missing values
                col_quality -= col_info['missing_percentage']
                
                # Penalize for low variance (for numeric columns)
                if np.issubdtype(self.df[column].dtype, np.number):
                    if self.df[column].std() == 0:
                        col_quality -= 20
                
                col_quality = max(0, col_quality)
                col_info['quality_score'] = round(col_quality, 2)
                quality_scores.append(col_quality)
                
                if np.issubdtype(self.df[column].dtype, np.number):
                    col_stats = {
                        'mean': float(self.df[column].mean()) if not self.df[column].isnull().all() else None,
                        'std': float(self.df[column].std()) if not self.df[column].isnull().all() else None,
                        'min': float(self.df[column].min()) if not self.df[column].isnull().all() else None,
                        'max': float(self.df[column].max()) if not self.df[column].isnull().all() else None,
                        'median': float(self.df[column].median()) if not self.df[column].isnull().all() else None,
                        'skewness': float(self.df[column].skew()) if not self.df[column].isnull().all() else None,
                        'outliers': self.detect_outliers(column)
                    }
                    col_info.update(col_stats)
                else:
                    col_info.update({
                        'most_frequent': str(self.df[column].mode().iloc[0]) if not self.df[column].mode().empty else None,
                        'frequency': int(self.df[column].value_counts().iloc[0]) if not self.df[column].value_counts().empty else 0,
                        'top_categories': self.df[column].value_counts().head(5).to_dict()
                    })
                
                report['column_analysis'][column] = col_info
            
            # Calculate overall data quality score
            if quality_scores:
                report['data_quality_score'] = round(sum(quality_scores) / len(quality_scores), 2)
            
            return report
            
        except Exception as e:
            print(f"Error in get_data_quality_report: {e}")
            return None
    
    def get_sample_data(self, n=10):
        """Get sample data for preview"""
        if self.df is None:
            return None
        
        try:
            # Replace NaN with None for JSON serialization
            sample_df = self.df.head(n).copy()
            
            # Convert all columns to compatible types
            for col in sample_df.columns:
                if pd.api.types.is_datetime64_any_dtype(sample_df[col]):
                    sample_df[col] = sample_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                elif sample_df[col].dtype == 'object':
                    sample_df[col] = sample_df[col].astype(str)
            
            return sample_df.replace({np.nan: None, pd.NaT: None}).to_dict('records')
        except Exception as e:
            print(f"Error in get_sample_data: {e}")
            return None
    
    def export_cleaned_data(self, file_path, file_type='csv'):
        """Export cleaned and transformed data"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            if file_type == 'csv':
                self.df.to_csv(file_path, index=False)
            elif file_type == 'excel':
                self.df.to_excel(file_path, index=False)
            elif file_type == 'json':
                self.df.to_json(file_path, orient='records', indent=2)
            
            return True
        except Exception as e:
            return str(e)
    
    def get_cleaning_suggestions(self):
        """Generate automated cleaning suggestions"""
        if self.df is None:
            return {}
        
        suggestions = {}
        basic_info = self.get_basic_info()
        
        if not basic_info:
            return suggestions
        
        for column in basic_info['columns']:
            col_suggestions = []
            missing_pct = basic_info['missing_percentage'][column]
            
            # Missing value suggestions
            if missing_pct > 50:
                col_suggestions.append("Consider dropping this column (high missing values)")
            elif missing_pct > 10:
                if column in basic_info['numeric_columns']:
                    col_suggestions.append("Fill missing values with median")
                else:
                    col_suggestions.append("Fill missing values with mode or 'Unknown'")
            
            # Data type suggestions
            if column in basic_info['categorical_columns']:
                unique_count = basic_info.get('unique_values', {}).get(column, 0)
                if unique_count > 50:
                    col_suggestions.append("Consider one-hot encoding or target encoding")
                else:
                    col_suggestions.append("Consider label encoding")
            
            # Outlier suggestions
            if column in basic_info['numeric_columns']:
                outlier_info = self.detect_outliers(column)
                if outlier_info and outlier_info['percentage_iqr'] > 5:
                    col_suggestions.append("Consider capping outliers")
            
            if col_suggestions:
                suggestions[column] = col_suggestions
        
        return suggestions
    
    def get_automated_cleaning_strategies(self):
        """Generate automated cleaning strategies based on data analysis"""
        if self.df is None:
            return {}
        
        strategies = {}
        basic_info = self.get_basic_info()
        
        if not basic_info:
            return strategies
        
        for column in basic_info['columns']:
            missing_pct = basic_info['missing_percentage'][column]
            
            if missing_pct > 70:
                strategies[column] = 'drop_column'
            elif missing_pct > 20:
                if column in basic_info['numeric_columns']:
                    strategies[column] = 'median'
                else:
                    strategies[column] = 'mode'
            elif missing_pct > 0:
                if column in basic_info['numeric_columns']:
                    strategies[column] = 'mean'
                else:
                    strategies[column] = 'forward_fill'
        
        return strategies
    
    def get_automated_transformations(self):
        """Generate automated transformations for modeling readiness"""
        if self.df is None:
            return []
        
        transformations = []
        basic_info = self.get_basic_info()
        
        if not basic_info:
            return transformations
        
        for column in basic_info['numeric_columns']:
            # Check for skewness
            if (self.df[column].dtype in [np.float64, np.int64] and 
                not self.df[column].isnull().all()):
                skewness = self.df[column].skew()
                if abs(skewness) > 1:  # Highly skewed
                    transformations.append({
                        'column': column,
                        'operation': 'log_transform'
                    })
        
        # Encode categorical variables with few categories
        for column in basic_info['categorical_columns']:
            unique_count = self.df[column].nunique()
            if unique_count <= 10:
                transformations.append({
                    'column': column,
                    'operation': 'one_hot_encode'
                })
            else:
                transformations.append({
                    'column': column,
                    'operation': 'encode_categorical'
                })
        
        return transformations
    
    def get_automated_feature_engineering(self):
        """Generate automated feature engineering operations"""
        if self.df is None:
            return []
        
        operations = []
        basic_info = self.get_basic_info()
        
        if not basic_info:
            return operations
        
        numeric_columns = basic_info['numeric_columns']
        
        # Create interaction features for highly correlated numeric columns
        if len(numeric_columns) >= 2:
            # Calculate correlations
            corr_matrix = self.df[numeric_columns].corr().abs()
            
            # Find highly correlated pairs
            for i in range(len(numeric_columns)):
                for j in range(i+1, len(numeric_columns)):
                    col1, col2 = numeric_columns[i], numeric_columns[j]
                    if corr_matrix.loc[col1, col2] > 0.7:
                        operations.append({
                            'type': 'interaction',
                            'column1': col1,
                            'column2': col2
                        })
        
        # Create polynomial features for important numeric columns
        for column in numeric_columns[:3]:  # Top 3 numeric columns
            operations.append({
                'type': 'polynomial',
                'column': column,
                'degree': 2
            })
        
        return operations
    
    def is_ready_for_modeling(self):
        """Check if data is ready for machine learning modeling"""
        if self.df is None or len(self.df) == 0:
            return False, "No data available"
        
        basic_info = self.get_basic_info()
        if not basic_info:
            return False, "Could not analyze data"
        
        # Check for missing values
        total_missing = sum(basic_info['missing_values'].values())
        if total_missing > 0:
            return False, f"Data still has {total_missing} missing values"
        
        # Check for infinite values
        if any(self.df.select_dtypes(include=[np.number]).map(np.isinf).any()):
            return False, "Data contains infinite values"
        
        # Check data types
        if len(basic_info['numeric_columns']) == 0:
            return False, "No numeric columns available for modeling"
        
        return True, "Data is ready for modeling"