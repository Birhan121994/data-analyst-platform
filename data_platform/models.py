from django.db import models
from django.contrib.auth.models import User

class DataSource(models.Model):
    DATA_TYPES = [
        ('csv', 'CSV'),
        ('excel', 'Excel'),
        ('json', 'JSON'),
        ('sql', 'SQL Database'),
    ]
    
    name = models.CharField(max_length=200)
    data_type = models.CharField(max_length=10, choices=DATA_TYPES)
    file = models.FileField(upload_to='datasets/', null=True, blank=True)
    database_connection = models.TextField(null=True, blank=True)
    description = models.TextField(blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    
    def __str__(self):
        return self.name

class DataAnalysis(models.Model):
    ANALYSIS_TYPES = [
        ('descriptive', 'Descriptive Statistics'),
        ('correlation', 'Correlation Analysis'),
        ('regression', 'Regression Analysis'),
        ('clustering', 'Clustering'),
        ('classification', 'Classification'),
    ]
    
    name = models.CharField(max_length=200)
    data_source = models.ForeignKey(DataSource, on_delete=models.CASCADE)
    analysis_type = models.CharField(max_length=20, choices=ANALYSIS_TYPES)
    parameters = models.JSONField(default=dict)
    results = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.name} - {self.analysis_type}"

class Visualization(models.Model):
    CHART_TYPES = [
        ('line', 'Line Chart'),
        ('bar', 'Bar Chart'),
        ('scatter', 'Scatter Plot'),
        ('histogram', 'Histogram'),
        ('box', 'Box Plot'),
        ('heatmap', 'Heatmap'),
    ]
    
    name = models.CharField(max_length=200)
    data_source = models.ForeignKey(DataSource, on_delete=models.CASCADE)
    chart_type = models.CharField(max_length=20, choices=CHART_TYPES)
    configuration = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name

class PredictiveModel(models.Model):
    MODEL_TYPES = [
        ('linear_regression', 'Linear Regression'),
        ('logistic_regression', 'Logistic Regression'),
        ('random_forest', 'Random Forest'),
        ('decision_tree', 'Decision Tree'),
        ('svm', 'Support Vector Machine'),
        ('kmeans', 'K-Means Clustering'),
        ('neural_network', 'Neural Network'),
    ]
    
    STATUS_CHOICES = [
        ('training', 'Training'),
        ('trained', 'Trained'),
        ('failed', 'Failed'),
        ('deployed', 'Deployed'),
    ]
    
    name = models.CharField(max_length=200)
    model_type = models.CharField(max_length=30, choices=MODEL_TYPES)
    data_source = models.ForeignKey(DataSource, on_delete=models.CASCADE)
    parameters = models.JSONField(default=dict)
    model_file = models.FileField(upload_to='models/', null=True, blank=True)
    accuracy = models.FloatField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='training')
    training_time = models.FloatField(null=True, blank=True)  # in seconds
    feature_importance = models.JSONField(default=dict, blank=True)
    metrics = models.JSONField(default=dict, blank=True)
    description = models.TextField(blank=True)
    is_public = models.BooleanField(default=False)
    version = models.IntegerField(default=1)
    parent_model = models.ForeignKey('self', on_delete=models.SET_NULL, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} - {self.model_type}"
    
    def get_metric_display(self):
        """Get the main metric for display based on model type"""
        if self.model_type in ['linear_regression', 'random_forest', 'decision_tree', 'svm', 'neural_network']:
            return self.metrics.get('r2_score', self.metrics.get('accuracy'))
        elif self.model_type == 'kmeans':
            return self.metrics.get('silhouette_score')
        else:
            return self.metrics.get('accuracy')
    
    def get_metric_name(self):
        """Get the name of the main metric"""
        if self.model_type in ['linear_regression', 'random_forest', 'decision_tree', 'svm', 'neural_network']:
            return 'R² Score' if 'r2_score' in self.metrics else 'Accuracy'
        elif self.model_type == 'kmeans':
            return 'Silhouette Score'
        else:
            return 'Accuracy'
    
    def get_feature_importance_list(self, limit=10):
        """Get top feature importances"""
        if not self.feature_importance:
            return []
        items = list(self.feature_importance.items())
        items.sort(key=lambda x: abs(x[1]), reverse=True)
        return items[:limit]

class ModelPrediction(models.Model):
    model = models.ForeignKey(PredictiveModel, on_delete=models.CASCADE)
    input_data = models.JSONField()
    prediction = models.JSONField()
    confidence = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']

class ModelVersion(models.Model):
    model = models.ForeignKey(PredictiveModel, on_delete=models.CASCADE, related_name='versions')
    version_number = models.IntegerField()
    changes = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['model', 'version_number']
        ordering = ['-version_number']