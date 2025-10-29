import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix, classification_report,
    silhouette_score
)
import joblib
import json
from datetime import datetime

class ModelTrainer:
    def __init__(self, dataframe):
        self.df = dataframe
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model = None
        self.results = {}
    
    def prepare_data(self, target_column, feature_columns, test_size=0.2, random_state=42):
        """Prepare data for model training"""
        try:
            # Check if target and features exist
            if target_column not in self.df.columns:
                return {"error": f"Target column '{target_column}' not found"}
            
            missing_features = [col for col in feature_columns if col not in self.df.columns]
            if missing_features:
                return {"error": f"Feature columns not found: {missing_features}"}
            
            # Handle missing values
            data = self.df[feature_columns + [target_column]].copy()
            data = data.dropna()
            
            if len(data) == 0:
                return {"error": "No data available after cleaning"}
            
            # Separate features and target
            X = data[feature_columns]
            y = data[target_column]
            
            # Handle categorical features
            for column in X.columns:
                if X[column].dtype == 'object':
                    le = LabelEncoder()
                    X[column] = le.fit_transform(X[column].astype(str))
                    self.label_encoders[column] = le
            
            # Check if target needs encoding
            is_classification = False
            if y.dtype == 'object' or y.nunique() < 10:  # Assume classification for few unique values
                is_classification = True
                le_target = LabelEncoder()
                y = le_target.fit_transform(y.astype(str))
                self.label_encoders[target_column] = le_target
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=random_state, 
                stratify=y if is_classification else None
            )
            
            return {
                'X_train': X_train, 'X_test': X_test, 
                'y_train': y_train, 'y_test': y_test,
                'feature_names': feature_columns,
                'target_name': target_column,
                'is_classification': is_classification,
                'original_target': data[target_column]
            }
            
        except Exception as e:
            return {"error": f"Data preparation failed: {str(e)}"}
    
    def train_linear_regression(self, data):
        """Train linear regression model"""
        try:
            model = LinearRegression()
            model.fit(data['X_train'], data['y_train'])
            
            # Predictions
            y_pred = model.predict(data['X_test'])
            
            # Metrics
            mse = mean_squared_error(data['y_test'], y_pred)
            r2 = r2_score(data['y_test'], y_pred)
            
            # Feature importance (coefficients)
            feature_importance = dict(zip(data['feature_names'], model.coef_))
            
            return {
                'model': model,
                'predictions': y_pred.tolist(),
                'metrics': {
                    'mean_squared_error': float(mse),
                    'r2_score': float(r2),
                    'rmse': float(np.sqrt(mse))
                },
                'feature_importance': feature_importance,
                'intercept': float(model.intercept_)
            }
        except Exception as e:
            return {"error": f"Linear regression training failed: {str(e)}"}
    
    def train_logistic_regression(self, data):
        """Train logistic regression model"""
        try:
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(data['X_train'], data['y_train'])
            
            # Predictions
            y_pred = model.predict(data['X_test'])
            y_pred_proba = model.predict_proba(data['X_test'])
            
            # Metrics
            accuracy = accuracy_score(data['y_test'], y_pred)
            precision = precision_score(data['y_test'], y_pred, average='weighted', zero_division=0)
            recall = recall_score(data['y_test'], y_pred, average='weighted', zero_division=0)
            f1 = f1_score(data['y_test'], y_pred, average='weighted', zero_division=0)
            
            # Feature importance (coefficients)
            if len(model.coef_) == 1:  # Binary classification
                feature_importance = dict(zip(data['feature_names'], model.coef_[0]))
            else:  # Multiclass
                feature_importance = {name: np.mean(coef) for name, coef in zip(data['feature_names'], model.coef_.T)}
            
            # Class labels
            if data['target_name'] in self.label_encoders:
                class_labels = self.label_encoders[data['target_name']].classes_.tolist()
            else:
                class_labels = list(range(len(model.classes_)))
            
            return {
                'model': model,
                'predictions': y_pred.tolist(),
                'prediction_probabilities': y_pred_proba.tolist(),
                'metrics': {
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1)
                },
                'feature_importance': feature_importance,
                'class_labels': class_labels,
                'confusion_matrix': confusion_matrix(data['y_test'], y_pred).tolist()
            }
        except Exception as e:
            return {"error": f"Logistic regression training failed: {str(e)}"}
    
    def train_random_forest(self, data):
        """Train random forest model"""
        try:
            if data['is_classification']:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            model.fit(data['X_train'], data['y_train'])
            
            # Predictions
            y_pred = model.predict(data['X_test'])
            
            # Metrics
            if data['is_classification']:
                accuracy = accuracy_score(data['y_test'], y_pred)
                precision = precision_score(data['y_test'], y_pred, average='weighted', zero_division=0)
                recall = recall_score(data['y_test'], y_pred, average='weighted', zero_division=0)
                f1 = f1_score(data['y_test'], y_pred, average='weighted', zero_division=0)
                metrics = {
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1)
                }
            else:
                mse = mean_squared_error(data['y_test'], y_pred)
                r2 = r2_score(data['y_test'], y_pred)
                metrics = {
                    'mean_squared_error': float(mse),
                    'r2_score': float(r2),
                    'rmse': float(np.sqrt(mse))
                }
            
            # Feature importance
            feature_importance = dict(zip(data['feature_names'], model.feature_importances_))
            
            return {
                'model': model,
                'predictions': y_pred.tolist(),
                'metrics': metrics,
                'feature_importance': feature_importance,
                'n_estimators': model.n_estimators
            }
        except Exception as e:
            return {"error": f"Random forest training failed: {str(e)}"}
    
    def train_decision_tree(self, data):
        """Train decision tree model"""
        try:
            if data['is_classification']:
                model = DecisionTreeClassifier(random_state=42)
            else:
                model = DecisionTreeRegressor(random_state=42)
            
            model.fit(data['X_train'], data['y_train'])
            
            # Predictions
            y_pred = model.predict(data['X_test'])
            
            # Metrics
            if data['is_classification']:
                accuracy = accuracy_score(data['y_test'], y_pred)
                metrics = {'accuracy': float(accuracy)}
            else:
                mse = mean_squared_error(data['y_test'], y_pred)
                r2 = r2_score(data['y_test'], y_pred)
                metrics = {
                    'mean_squared_error': float(mse),
                    'r2_score': float(r2),
                    'rmse': float(np.sqrt(mse))
                }
            
            # Feature importance
            feature_importance = dict(zip(data['feature_names'], model.feature_importances_))
            
            return {
                'model': model,
                'predictions': y_pred.tolist(),
                'metrics': metrics,
                'feature_importance': feature_importance,
                'tree_depth': model.get_depth()
            }
        except Exception as e:
            return {"error": f"Decision tree training failed: {str(e)}"}
    
    def train_svm(self, data):
        """Train support vector machine model"""
        try:
            if data['is_classification']:
                model = SVC(random_state=42, probability=True)
            else:
                model = SVR()
            
            model.fit(data['X_train'], data['y_train'])
            
            # Predictions
            y_pred = model.predict(data['X_test'])
            
            # Metrics
            if data['is_classification']:
                accuracy = accuracy_score(data['y_test'], y_pred)
                metrics = {'accuracy': float(accuracy)}
            else:
                mse = mean_squared_error(data['y_test'], y_pred)
                r2 = r2_score(data['y_test'], y_pred)
                metrics = {
                    'mean_squared_error': float(mse),
                    'r2_score': float(r2),
                    'rmse': float(np.sqrt(mse))
                }
            
            return {
                'model': model,
                'predictions': y_pred.tolist(),
                'metrics': metrics
            }
        except Exception as e:
            return {"error": f"SVM training failed: {str(e)}"}
    
    def train_kmeans(self, data, n_clusters=3):
        """Train K-means clustering model"""
        try:
            model = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = model.fit_predict(data['X_train'])
            
            # Metrics
            silhouette_avg = silhouette_score(data['X_train'], clusters)
            
            # Predict on test set
            test_clusters = model.predict(data['X_test'])
            
            return {
                'model': model,
                'train_clusters': clusters.tolist(),
                'test_clusters': test_clusters.tolist(),
                'metrics': {
                    'silhouette_score': float(silhouette_avg),
                    'inertia': float(model.inertia_),
                    'n_clusters': n_clusters
                },
                'cluster_centers': model.cluster_centers_.tolist()
            }
        except Exception as e:
            return {"error": f"K-means training failed: {str(e)}"}
    
    def train_neural_network(self, data):
        """Train neural network model"""
        try:
            if data['is_classification']:
                model = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
            else:
                model = MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
            
            model.fit(data['X_train'], data['y_train'])
            
            # Predictions
            y_pred = model.predict(data['X_test'])
            
            # Metrics
            if data['is_classification']:
                accuracy = accuracy_score(data['y_test'], y_pred)
                metrics = {'accuracy': float(accuracy)}
            else:
                mse = mean_squared_error(data['y_test'], y_pred)
                r2 = r2_score(data['y_test'], y_pred)
                metrics = {
                    'mean_squared_error': float(mse),
                    'r2_score': float(r2),
                    'rmse': float(np.sqrt(mse))
                }
            
            return {
                'model': model,
                'predictions': y_pred.tolist(),
                'metrics': metrics,
                'n_layers': len(model.hidden_layer_sizes) + 2,  # +2 for input and output layers
                'training_loss': model.loss_curve_[-1] if hasattr(model, 'loss_curve_') else None
            }
        except Exception as e:
            return {"error": f"Neural network training failed: {str(e)}"}
    
    def train_model(self, model_type, target_column, feature_columns, **kwargs):
        """Main training function"""
        try:
            # Prepare data
            data = self.prepare_data(target_column, feature_columns, 
                                   kwargs.get('test_size', 0.2), 
                                   kwargs.get('random_state', 42))
            
            if 'error' in data:
                return data
            
            # Train specific model
            if model_type == 'linear_regression':
                result = self.train_linear_regression(data)
            elif model_type == 'logistic_regression':
                result = self.train_logistic_regression(data)
            elif model_type == 'random_forest':
                result = self.train_random_forest(data)
            elif model_type == 'decision_tree':
                result = self.train_decision_tree(data)
            elif model_type == 'svm':
                result = self.train_svm(data)
            elif model_type == 'kmeans':
                result = self.train_kmeans(data, kwargs.get('n_clusters', 3))
            elif model_type == 'neural_network':
                result = self.train_neural_network(data)
            else:
                return {"error": f"Unknown model type: {model_type}"}
            
            if 'error' in result:
                return result
            
            # Add common information
            result['data_info'] = {
                'n_train_samples': len(data['X_train']),
                'n_test_samples': len(data['X_test']),
                'n_features': len(feature_columns),
                'target_column': target_column,
                'feature_columns': feature_columns,
                'model_type': model_type,
                'training_time': datetime.now().isoformat()
            }
            
            self.model = result['model']
            self.results = result
            
            return result
            
        except Exception as e:
            return {"error": f"Model training failed: {str(e)}"}
    
    def save_model(self, file_path):
        """Save trained model to file with all preprocessing objects"""
        try:
            if self.model is None:
                return {"error": "No model trained yet"}
            
            # Save model and preprocessing objects
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'results': self.results,
                'feature_columns': self.results.get('data_info', {}).get('feature_columns', []),
                'target_column': self.results.get('data_info', {}).get('target_column', '')
            }
            
            joblib.dump(model_data, file_path)
            return {"success": f"Model saved to {file_path}"}
            
        except Exception as e:
            return {"error": f"Model saving failed: {str(e)}"}    
    def predict(self, new_data):
        """Make predictions with trained model"""
        try:
            if self.model is None:
                return {"error": "No model trained yet"}
            
            # Preprocess new data
            X_new = new_data.copy()
            
            for column in X_new.columns:
                if column in self.label_encoders:
                    X_new[column] = self.label_encoders[column].transform(X_new[column].astype(str))
            
            X_new_scaled = self.scaler.transform(X_new)
            
            # Make predictions
            predictions = self.model.predict(X_new_scaled)
            
            # Convert back to original labels if classification
            if hasattr(self.model, 'classes_') and 'data_info' in self.results:
                target_column = self.results['data_info']['target_column']
                if target_column in self.label_encoders:
                    predictions = self.label_encoders[target_column].inverse_transform(predictions)
            
            return {
                'predictions': predictions.tolist(),
                'n_predictions': len(predictions)
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}