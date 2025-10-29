from django.contrib import admin
from .models import DataSource, DataAnalysis, Visualization, PredictiveModel

@admin.register(DataSource)
class DataSourceAdmin(admin.ModelAdmin):
    list_display = ['name', 'data_type', 'user', 'uploaded_at']
    list_filter = ['data_type', 'uploaded_at']
    search_fields = ['name', 'description']
    readonly_fields = ['uploaded_at']

@admin.register(DataAnalysis)
class DataAnalysisAdmin(admin.ModelAdmin):
    list_display = ['name', 'analysis_type', 'data_source', 'created_at']
    list_filter = ['analysis_type', 'created_at']
    search_fields = ['name', 'parameters']
    readonly_fields = ['created_at']

@admin.register(Visualization)
class VisualizationAdmin(admin.ModelAdmin):
    list_display = ['name', 'chart_type', 'data_source', 'created_at']
    list_filter = ['chart_type', 'created_at']
    search_fields = ['name', 'configuration']
    readonly_fields = ['created_at']

@admin.register(PredictiveModel)
class PredictiveModelAdmin(admin.ModelAdmin):
    list_display = ['name', 'model_type', 'data_source', 'accuracy', 'created_at']
    list_filter = ['model_type', 'created_at']
    search_fields = ['name', 'parameters']
    readonly_fields = ['created_at']