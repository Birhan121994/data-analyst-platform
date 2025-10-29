import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
import json
warnings.filterwarnings('ignore')

class DataAnalyzer:
    def __init__(self, dataframe):
        self.df = dataframe
        self.results = {}
    
    def convert_to_serializable(self, obj):
        """Convert numpy/pandas types to Python native types for JSON serialization"""
        if pd.isna(obj) or obj is None:
            return None
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: self.convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self.convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    def safe_descriptive_stats(self, series):
        """Safely compute descriptive statistics handling different data types"""
        stats_dict = {}
        
        # Basic stats that work for all types
        stats_dict['count'] = series.count()
        stats_dict['unique'] = series.nunique() if series.dtype == 'object' else None
        
        # For numeric columns
        if np.issubdtype(series.dtype, np.number):
            try:
                stats_dict.update({
                    'mean': series.mean(),
                    'std': series.std(),
                    'min': series.min(),
                    'p25': series.quantile(0.25),
                    'p50': series.quantile(0.50),
                    'p75': series.quantile(0.75),
                    'max': series.max(),
                    'skewness': series.skew(),
                    'kurtosis': series.kurtosis()
                })
            except Exception as e:
                # If any calculation fails, set to None
                stats_dict.update({
                    'mean': None, 'std': None, 'min': None, 'p25': None,
                    'p50': None, 'p75': None, 'max': None, 'skewness': None, 'kurtosis': None
                })
        else:
            # For non-numeric columns
            try:
                stats_dict.update({
                    'top': series.mode().iloc[0] if not series.mode().empty else None,
                    'freq': series.value_counts().iloc[0] if not series.value_counts().empty else 0,
                    'mean': None, 'std': None, 'min': None, 'p25': None,
                    'p50': None, 'p75': None, 'max': None, 'skewness': None, 'kurtosis': None
                })
            except Exception as e:
                stats_dict.update({
                    'top': None, 'freq': 0, 'mean': None, 'std': None, 'min': None, 'p25': None,
                    'p50': None, 'p75': None, 'max': None, 'skewness': None, 'kurtosis': None
                })
        
        # Convert all values to serializable format
        return {k: self.convert_to_serializable(v) for k, v in stats_dict.items()}
    
    def descriptive_statistics(self):
        """Generate descriptive statistics with proper type handling"""
        desc_stats = {}
        
        for col in self.df.columns:
            try:
                desc_stats[col] = self.safe_descriptive_stats(self.df[col])
            except Exception as e:
                desc_stats[col] = {'error': f"Could not compute statistics: {str(e)}"}
        
        # Return with the expected key name
        return {'descriptive_stats': desc_stats}
    
    def correlation_analysis(self):
        """Perform correlation analysis on numeric columns only"""
        # Select only numeric columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) < 2:
            return {
                "correlation_matrix": {"error": "Need at least 2 numerical columns for correlation analysis"},
                "available_numeric_columns": list(numerical_cols)
            }
        
        try:
            correlation_matrix = self.df[numerical_cols].corr()
            
            # Convert to serializable format
            corr_dict = {}
            for col in correlation_matrix.columns:
                corr_dict[col] = {
                    k: self.convert_to_serializable(v) 
                    for k, v in correlation_matrix[col].items()
                }
            
            # Find high correlations
            high_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7 and not pd.isna(corr_value):  # Threshold for high correlation
                        high_correlations.append({
                            'variable1': correlation_matrix.columns[i],
                            'variable2': correlation_matrix.columns[j],
                            'correlation': self.convert_to_serializable(corr_value)
                        })
            
            results = {
                'correlation_matrix': corr_dict,
                'high_correlations': high_correlations,
                'numeric_columns_used': list(numerical_cols)
            }
            
            return results
            
        except Exception as e:
            return {
                "correlation_matrix": {"error": f"Correlation analysis failed: {str(e)}"}
            }
    
    def regression_analysis(self, target_column):
        """Perform regression analysis with proper error handling"""
        if target_column not in self.df.columns:
            return {"regression": {"error": f"Target column '{target_column}' not found"}}
        
        # Select only numeric columns for features
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numerical_cols if col != target_column]
        
        if len(feature_cols) == 0:
            return {"regression": {"error": "No numeric feature columns available for regression"}}
        
        # Check if target is numeric
        if not np.issubdtype(self.df[target_column].dtype, np.number):
            return {"regression": {"error": f"Target column '{target_column}' must be numeric for regression"}}
        
        try:
            X = self.df[feature_cols]
            y = self.df[target_column]
            
            # Handle missing values
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())
            
            # Remove any remaining NaN values
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[mask]
            y = y[mask]
            
            if len(X) == 0:
                return {"regression": {"error": "No valid data points after cleaning"}}
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = model.score(X_test, y_test)
            
            coefficients = {
                feature: self.convert_to_serializable(coef) 
                for feature, coef in zip(feature_cols, model.coef_)
            }
            
            results = {
                'mse': self.convert_to_serializable(mse),
                'r_squared': self.convert_to_serializable(r2),
                'coefficients': coefficients,
                'intercept': self.convert_to_serializable(model.intercept_),
                'features_used': feature_cols,
                'target_column': target_column
            }
            
            return {'regression': results}
            
        except Exception as e:
            return {"regression": {"error": f"Regression analysis failed: {str(e)}"}}
    
    def clustering_analysis(self, n_clusters=3, columns=None):
        """Perform clustering analysis with proper data handling"""
        if columns is None:
            # Use only numeric columns by default
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(columns) < 2:
            return {"clustering": {"error": "Need at least 2 numeric columns for clustering"}}
        
        # Filter to only numeric columns
        numeric_columns = [col for col in columns if np.issubdtype(self.df[col].dtype, np.number)]
        
        if len(numeric_columns) < 2:
            return {
                "clustering": {
                    "error": "Need at least 2 numeric columns for clustering",
                    "available_numeric_columns": numeric_columns
                }
            }
        
        try:
            X = self.df[numeric_columns].copy()
            
            # Handle missing values
            X = X.fillna(X.mean())
            
            # Remove any rows with remaining NaN values
            X = X.dropna()
            
            if len(X) < n_clusters:
                return {"clustering": {"error": f"Not enough data points ({len(X)}) for {n_clusters} clusters"}}
            
            # Standardize the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            
            silhouette_avg = silhouette_score(X_scaled, clusters)
            
            # Add clusters to dataframe for reference
            self.df['cluster'] = pd.Series(clusters, index=X.index)
            
            results = {
                'n_clusters': n_clusters,
                'silhouette_score': self.convert_to_serializable(silhouette_avg),
                'cluster_sizes': pd.Series(clusters).value_counts().to_dict(),
                'cluster_centers': [center.tolist() for center in kmeans.cluster_centers_],
                'features_used': numeric_columns,
                'n_samples': len(X)
            }
            
            # Convert cluster sizes to serializable
            results['cluster_sizes'] = {
                str(k): self.convert_to_serializable(v) 
                for k, v in results['cluster_sizes'].items()
            }
            
            return {'clustering': results}
            
        except Exception as e:
            return {"clustering": {"error": f"Clustering analysis failed: {str(e)}"}}
    
    def get_numeric_columns(self):
        """Get list of numeric columns in the dataset"""
        return self.df.select_dtypes(include=[np.number]).columns.tolist()
    
    def get_categorical_columns(self):
        """Get list of categorical columns in the dataset"""
        return self.df.select_dtypes(include=['object']).columns.tolist()