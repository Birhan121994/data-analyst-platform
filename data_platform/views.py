from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, FileResponse
from django.contrib import messages
from django.conf import settings
import json
import pandas as pd
import numpy as np
import os

from .models import DataSource, DataAnalysis, Visualization, PredictiveModel, ModelPrediction, ModelVersion
from .forms import DataUploadForm, DataAnalysisForm, VisualizationForm, PredictiveModelForm, ModelUpdateForm, PredictionForm
from .utils.data_processing import DataProcessor
from .utils.analysis import DataAnalyzer
from .utils.visualization import DataVisualizer
from django.views.decorators.http import require_POST

from django.contrib.auth import login, authenticate, logout
from .forms import CustomUserCreationForm, CustomAuthenticationForm

def signup_view(request):
    """Modern signup view"""
    if request.user.is_authenticated:
        return redirect('dashboard')
    
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, f'Welcome to DataInsight, {user.username}! Your account has been created successfully.')
            return redirect('dashboard')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = CustomUserCreationForm()
    
    context = {
        'form': form,
        'title': 'Create Account'
    }
    return render(request, 'auth/signup.html', context)

def login_view(request):
    """Modern login view"""
    if request.user.is_authenticated:
        return redirect('dashboard')
    
    if request.method == 'POST':
        form = CustomAuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, f'Welcome back, {user.username}!')
                next_url = request.GET.get('next', 'dashboard')
                return redirect(next_url)
        else:
            messages.error(request, 'Invalid username or password.')
    else:
        form = CustomAuthenticationForm()
    
    context = {
        'form': form,
        'title': 'Sign In'
    }
    return render(request, 'auth/login.html', context)

@login_required
def logout_view(request):
    """Logout view"""
    logout(request)
    messages.success(request, 'You have been successfully logged out.')
    return redirect('login')

def landing_view(request):
    """Landing page for non-authenticated users"""
    if request.user.is_authenticated:
        return redirect('dashboard')
    
    return render(request, 'landing.html')

@login_required
def dashboard(request):
    """Main dashboard view"""
    data_sources = DataSource.objects.filter(user=request.user)
    analyses = DataAnalysis.objects.filter(data_source__user=request.user)
    visualizations = Visualization.objects.filter(data_source__user=request.user)
    models = PredictiveModel.objects.filter(data_source__user=request.user)
    
    context = {
        'data_sources': data_sources,
        'analyses': analyses,
        'visualizations': visualizations,
        'models': models,
    }
    return render(request, 'dashboard.html', context)

@login_required
def data_upload(request):
    """Handle data upload and initial processing"""
    if request.method == 'POST':
        form = DataUploadForm(request.POST, request.FILES)
        if form.is_valid():
            data_source = form.save(commit=False)
            data_source.user = request.user
            
            # Save the instance first to get the file path
            data_source.save()
            
            try:
                # Process the uploaded file
                processor = DataProcessor()
                file_path = data_source.file.path
                success = processor.load_data(file_path, data_source.data_type)
                
                if success is True:
                    messages.success(request, 'Data uploaded successfully!')
                    return redirect('data_exploration', data_source_id=data_source.id)
                else:
                    # Delete the data source if processing fails
                    data_source.delete()
                    messages.error(request, f'Error loading data: {success}')
            except Exception as e:
                # Delete the data source if any exception occurs
                data_source.delete()
                messages.error(request, f'Error processing file: {str(e)}')
    else:
        form = DataUploadForm()
    
    return render(request, 'data_upload.html', {'form': form})

@login_required
def data_exploration(request, data_source_id):
    """Explore and clean uploaded data"""
    data_source = get_object_or_404(DataSource, id=data_source_id, user=request.user)
    processor = DataProcessor()
    
    # Load data
    success = processor.load_data(data_source.file.path, data_source.data_type)
    
    if success is not True:
        messages.error(request, f'Error loading data: {success}')
        return redirect('dashboard')
    
    basic_info = processor.get_basic_info()
    sample_data = processor.get_sample_data(10)
    
    if request.method == 'POST':
        if 'clean_data' in request.POST:
            strategies_json = request.POST.get('cleaning_strategies', '{}')
            try:
                strategies = json.loads(strategies_json)
                processor.clean_data(strategies)
                messages.success(request, 'Data cleaned successfully!')
            except json.JSONDecodeError:
                messages.error(request, 'Invalid cleaning strategies format')
        
        # Update basic info after operations
        basic_info = processor.get_basic_info()
        sample_data = processor.get_sample_data(10)
    
    context = {
        'data_source': data_source,
        'basic_info': basic_info,
        'sample_data': sample_data,
        'cleaning_report': processor.cleaning_report,
    }
    return render(request, 'data_exploration.html', context)


@login_required
def data_analysis(request, data_source_id):
    """Perform various data analyses with proper error handling"""
    data_source = get_object_or_404(DataSource, id=data_source_id, user=request.user)
    processor = DataProcessor()
    processor.load_data(data_source.file.path, data_source.data_type)
    
    if processor.df is None:
        messages.error(request, 'Could not load data for analysis')
        return redirect('data_exploration', data_source_id=data_source_id)
    
    analyzer = DataAnalyzer(processor.df)
    analysis_results = {}
    
    if request.method == 'POST':
        analysis_type = request.POST.get('analysis_type')
        
        try:
            if analysis_type == 'descriptive':
                analysis_results = analyzer.descriptive_statistics()
            
            elif analysis_type == 'correlation':
                analysis_results = analyzer.correlation_analysis()
            
            elif analysis_type == 'regression':
                target_column = request.POST.get('target_column')
                if target_column:
                    analysis_results = analyzer.regression_analysis(target_column)
                else:
                    analysis_results = {"error": "Target column is required for regression analysis"}
            
            elif analysis_type == 'clustering':
                n_clusters = int(request.POST.get('n_clusters', 3))
                columns = request.POST.getlist('columns')
                # If no columns selected, use all numeric columns
                if not columns:
                    columns = analyzer.get_numeric_columns()
                analysis_results = analyzer.clustering_analysis(n_clusters, columns)
            
            # Save analysis to database if successful
            if 'error' not in analysis_results:
                try:
                    DataAnalysis.objects.create(
                        name=f"{analysis_type.title()} Analysis - {data_source.name}",
                        data_source=data_source,
                        analysis_type=analysis_type,
                        parameters=json.dumps(request.POST.dict(), default=str),
                        results=json.dumps(analysis_results, default=str)
                    )
                    messages.success(request, f'{analysis_type.title()} analysis completed successfully!')
                except Exception as e:
                    messages.warning(request, f'Analysis completed but could not save results: {str(e)}')
            else:
                messages.error(request, f'Analysis failed: {analysis_results["error"]}')
                
        except Exception as e:
            analysis_results = {"error": f"Analysis failed: {str(e)}"}
            messages.error(request, f'Analysis failed: {str(e)}')
    
    # Get column information for the form
    numeric_columns = analyzer.get_numeric_columns() if processor.df is not None else []
    all_columns = processor.df.columns.tolist() if processor.df is not None else []
    
    context = {
        'data_source': data_source,
        'analysis_results': analysis_results,
        'columns': all_columns,
        'numeric_columns': numeric_columns,
    }
    return render(request, 'analysis.html', context)

@login_required
def data_visualization(request, data_source_id):
    """Create visualizations"""
    data_source = get_object_or_404(DataSource, id=data_source_id, user=request.user)
    processor = DataProcessor()
    processor.load_data(data_source.file.path, data_source.data_type)
    
    visualizer = DataVisualizer(processor.df)
    plot_json = None
    
    if request.method == 'POST':
        chart_type = request.POST.get('chart_type')
        x_col = request.POST.get('x_column')
        y_col = request.POST.get('y_column')
        color_col = request.POST.get('color_column')
        
        if chart_type == 'line':
            plot_json = visualizer.create_line_chart(x_col, y_col)
        elif chart_type == 'bar':
            plot_json = visualizer.create_bar_chart(x_col, y_col)
        elif chart_type == 'scatter':
            plot_json = visualizer.create_scatter_plot(x_col, y_col, color_col)
        elif chart_type == 'histogram':
            plot_json = visualizer.create_histogram(x_col)
        elif chart_type == 'box':
            plot_json = visualizer.create_box_plot(x_col, y_col)
        elif chart_type == 'heatmap':
            plot_json = visualizer.create_heatmap()
        elif chart_type == 'dashboard':
            plot_json = visualizer.create_dashboard()
        
        if plot_json:
            Visualization.objects.create(
                name=f"{chart_type.title()} Visualization",
                data_source=data_source,
                chart_type=chart_type,
                configuration=request.POST.dict()
            )
    
    context = {
        'data_source': data_source,
        'plot_json': plot_json,
        'columns': processor.df.columns.tolist() if processor.df is not None else [],
    }
    return render(request, 'visualization.html', context)

import os
from datetime import datetime
from .utils.model_training import ModelTrainer
@login_required
def predictive_modeling(request, data_source_id):
    """Build and train predictive models"""
    data_source = get_object_or_404(DataSource, id=data_source_id, user=request.user)
    processor = DataProcessor()
    
    # Load data
    success = processor.load_data(data_source.file.path, data_source.data_type)
    if success is not True:
        messages.error(request, f'Error loading data: {success}')
        return redirect('data_exploration', data_source_id=data_source.id)
    
    # Get ALL columns and their data types for the form
    basic_info = processor.get_basic_info()
    all_columns = basic_info['columns'] if basic_info else []
    
    # Get column data types to help with model selection
    column_info = []
    if processor.df is not None:
        for col in all_columns:
            dtype = str(processor.df[col].dtype)
            unique_count = processor.df[col].nunique()
            column_info.append({
                'name': col,
                'dtype': dtype,
                'unique_count': unique_count,
                'is_numeric': pd.api.types.is_numeric_dtype(processor.df[col]),
                'is_categorical': processor.df[col].dtype == 'object' or unique_count < 20
            })
    
    trained_model = None
    training_results = None
    
    if request.method == 'POST':
        form = PredictiveModelForm(request.POST)
        if form.is_valid():
            try:
                predictive_model = form.save(commit=False)
                predictive_model.data_source = data_source
                predictive_model.status = 'training'
                predictive_model.save()
                
                # Get parameters from form
                target_column = request.POST.get('target_column')
                feature_columns = request.POST.getlist('feature_columns')
                test_size = int(request.POST.get('test_size', 20)) / 100
                random_state = int(request.POST.get('random_state', 42))
                
                # Validate inputs
                if not target_column:
                    messages.error(request, 'Please select a target column')
                    predictive_model.status = 'failed'
                    predictive_model.save()
                    return redirect('predictive_modeling', data_source_id=data_source.id)
                
                if not feature_columns:
                    messages.error(request, 'Please select at least one feature column')
                    predictive_model.status = 'failed'
                    predictive_model.save()
                    return redirect('predictive_modeling', data_source_id=data_source.id)
                
                # Filter out target column from features
                filtered_features = [col for col in feature_columns if col != target_column]
                if len(filtered_features) < len(feature_columns):
                    messages.warning(request, f'Target column "{target_column}" was automatically removed from features.')
                    feature_columns = filtered_features
                
                # Validate that we have at least one feature after filtering
                if not feature_columns:
                    messages.error(request, 'No features selected after removing target column. Please select different features.')
                    predictive_model.status = 'failed'
                    predictive_model.save()
                    return redirect('predictive_modeling', data_source_id=data_source.id)
                
                # Validate model type suitability
                target_dtype = str(processor.df[target_column].dtype)
                target_unique = processor.df[target_column].nunique()
                
                # Auto-detect problem type and suggest appropriate models
                is_classification_problem = (target_dtype == 'object' or 
                                           (pd.api.types.is_numeric_dtype(processor.df[target_column]) and target_unique < 10))
                
                # Warn about model type mismatch
                if is_classification_problem and predictive_model.model_type in ['linear_regression']:
                    messages.warning(request, 
                        f'Target column "{target_column}" appears to be categorical ({target_unique} unique values). '
                        f'Consider using classification models like Logistic Regression or Random Forest for better results.'
                    )
                elif not is_classification_problem and predictive_model.model_type in ['logistic_regression']:
                    messages.warning(request, 
                        f'Target column "{target_column}" appears to be continuous. '
                        f'Consider using regression models like Linear Regression or Random Forest Regressor.'
                    )
                
                # Initialize model trainer
                trainer = ModelTrainer(processor.df)
                
                # Train model
                training_params = {
                    'test_size': test_size,
                    'random_state': random_state
                }
                
                if predictive_model.model_type == 'kmeans':
                    training_params['n_clusters'] = int(request.POST.get('n_clusters', 3))
                
                start_time = datetime.now()
                training_results = trainer.train_model(
                    predictive_model.model_type, target_column, feature_columns, **training_params
                )
                training_time = (datetime.now() - start_time).total_seconds()
                
                if 'error' in training_results:
                    messages.error(request, f'Model training failed: {training_results["error"]}')
                    predictive_model.status = 'failed'
                    predictive_model.save()
                    return redirect('predictive_modeling', data_source_id=data_source.id)
                
                # Update model with results
                predictive_model.parameters = {
                    'target_column': target_column,
                    'feature_columns': feature_columns,
                    'test_size': test_size,
                    'random_state': random_state,
                    'training_time_seconds': training_time,
                    'target_unique_values': int(target_unique),
                    'target_dtype': target_dtype,
                    'problem_type': 'classification' if is_classification_problem else 'regression'
                }
                
                predictive_model.metrics = training_results.get('metrics', {})
                predictive_model.feature_importance = training_results.get('feature_importance', {})
                
                # Calculate appropriate accuracy metric based on model type
                if predictive_model.model_type == 'kmeans':
                    predictive_model.accuracy = training_results.get('metrics', {}).get('silhouette_score', 0.0)
                elif is_classification_problem:
                    predictive_model.accuracy = training_results.get('metrics', {}).get('accuracy', 0.0)
                else:
                    predictive_model.accuracy = training_results.get('metrics', {}).get('r2_score', 0.0)
                
                predictive_model.training_time = training_time
                predictive_model.status = 'trained'
                
                # Save model file with correct path
                models_dir = os.path.join(settings.MEDIA_ROOT, 'trained_models')
                os.makedirs(models_dir, exist_ok=True)
                model_filename = f"model_{data_source.id}_{predictive_model.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                model_filepath = os.path.join(models_dir, model_filename)
                
                save_result = trainer.save_model(model_filepath)
                if 'error' not in save_result:
                    # Store relative path from MEDIA_ROOT
                    predictive_model.model_file.name = os.path.join('trained_models', model_filename)
                else:
                    messages.warning(request, f'Model trained but could not save file: {save_result["error"]}')
                
                predictive_model.save()
                
                # Success message with appropriate metrics
                if predictive_model.model_type == 'kmeans':
                    metric_msg = f'Silhouette Score: {predictive_model.accuracy:.3f}'
                elif is_classification_problem:
                    metric_msg = f'Accuracy: {predictive_model.accuracy:.3f}'
                else:
                    metric_msg = f'R² Score: {predictive_model.accuracy:.3f}'
                
                messages.success(request, 
                    f'{predictive_model.model_type.replace("_", " ").title()} model trained successfully! '
                    f'{metric_msg} | Time: {training_time:.2f}s'
                )
                
                trained_model = predictive_model
                
            except Exception as e:
                messages.error(request, f'Model training failed: {str(e)}')
                if 'predictive_model' in locals():
                    predictive_model.status = 'failed'
                    predictive_model.save()
        else:
            messages.error(request, 'Please correct the form errors.')
    else:
        form = PredictiveModelForm()
    
    # Get all trained models for this data source
    trained_models = PredictiveModel.objects.filter(data_source=data_source, status='trained').order_by('-created_at')
    
    context = {
        'data_source': data_source,
        'form': form,
        'all_columns': all_columns,  # Changed from numeric_columns to all_columns
        'column_info': column_info,  # Added detailed column info
        'trained_model': trained_model,
        'training_results': training_results,
        'trained_models': trained_models,
    }
    return render(request, 'predictive_modeling.html', context)

@login_required
def data_cleaning(request, data_source_id):
    """Comprehensive data cleaning interface"""
    data_source = get_object_or_404(DataSource, id=data_source_id, user=request.user)
    processor = DataProcessor()
    
    # Load data
    success = processor.load_data(data_source.file.path, data_source.data_type)
    if success is not True:
        messages.error(request, f'Error loading data: {success}')
        return redirect('data_exploration', data_source_id=data_source.id)
    
    basic_info = processor.get_basic_info()
    data_quality_report = processor.get_data_quality_report()
    
    if request.method == 'POST':
        # Handle cleaning strategies
        if 'clean_data' in request.POST:
            strategies = {}
            
            # Get cleaning strategies for each column
            for column in basic_info['columns']:
                strategy = request.POST.get(f'cleaning_strategy_{column}')
                if strategy and strategy != 'none':
                    strategies[column] = strategy
            
            # Get global cleaning options
            remove_duplicates = request.POST.get('remove_duplicates') == 'on'
            handle_outliers = request.POST.get('handle_outliers') == 'on'
            standardize_text = request.POST.get('standardize_text') == 'on'
            
            # Apply cleaning
            success = processor.clean_data(strategies, remove_duplicates, handle_outliers, standardize_text)
            if success:
                messages.success(request, 'Data cleaned successfully!')
                
                # Save cleaned data back to the data source
                cleaned_file_path = data_source.file.path.replace('.', '_cleaned.')
                export_success = processor.export_cleaned_data(cleaned_file_path)
                
                if export_success:
                    # Update the data source with cleaned file
                    data_source.file.name = data_source.file.name.replace('.', '_cleaned.')
                    data_source.save()
            else:
                messages.error(request, 'Data cleaning failed!')
        
        # Handle transformations
        elif 'transform_data' in request.POST:
            transformations_json = request.POST.get('transformations', '[]')
            try:
                transformations = json.loads(transformations_json)
                if transformations:
                    success = processor.transform_data(transformations)
                    if success:
                        messages.success(request, 'Data transformations applied successfully!')
                        
                        # Save transformed data
                        transformed_file_path = data_source.file.path.replace('.', '_transformed.')
                        export_success = processor.export_cleaned_data(transformed_file_path)
                        
                        if export_success:
                            data_source.file.name = data_source.file.name.replace('.', '_transformed.')
                            data_source.save()
                    else:
                        messages.error(request, 'Data transformations failed!')
            except json.JSONDecodeError:
                messages.error(request, 'Invalid transformations format!')
        
        # Handle feature engineering
        elif 'engineer_features' in request.POST:
            operations_json = request.POST.get('feature_operations', '[]')
            try:
                operations = json.loads(operations_json)
                if operations:
                    success = processor.feature_engineering(operations)
                    if success:
                        messages.success(request, 'Feature engineering completed successfully!')
                        
                        # Save engineered data
                        engineered_file_path = data_source.file.path.replace('.', '_engineered.')
                        export_success = processor.export_cleaned_data(engineered_file_path)
                        
                        if export_success:
                            data_source.file.name = data_source.file.name.replace('.', '_engineered.')
                            data_source.save()
                    else:
                        messages.error(request, 'Feature engineering failed!')
            except json.JSONDecodeError:
                messages.error(request, 'Invalid feature operations format!')
        
        # Export cleaned data
        elif 'export_data' in request.POST:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            export_dir = "media/cleaned_datasets/"
            os.makedirs(export_dir, exist_ok=True)
            export_path = f"{export_dir}cleaned_{data_source.name}_{timestamp}.csv"
            
            success = processor.export_cleaned_data(export_path)
            if success is True:
                messages.success(request, f'Data exported successfully to {export_path}!')
            else:
                messages.error(request, f'Export failed: {success}')
        
        # Reset to original data
        elif 'reset_data' in request.POST:
            # Reload original data
            original_file_path = data_source.file.path.replace('_cleaned', '').replace('_transformed', '').replace('_engineered', '')
            if os.path.exists(original_file_path):
                success = processor.load_data(original_file_path, data_source.data_type)
                if success:
                    messages.success(request, 'Data reset to original state!')
            
        # Update info after operations
        basic_info = processor.get_basic_info()
        data_quality_report = processor.get_data_quality_report()
    
    # Get current data state for display
    sample_data = processor.get_sample_data(10)
    cleaning_suggestions = processor.get_cleaning_suggestions()
    
    context = {
        'data_source': data_source,
        'basic_info': basic_info,
        'data_quality_report': data_quality_report,
        'cleaning_report': processor.cleaning_report,
        'transformations_applied': processor.transformations_applied,
        'sample_data': sample_data,
        'cleaning_suggestions': cleaning_suggestions,
    }
    return render(request, 'data_cleaning.html', context)

@login_required
def automated_cleaning(request, data_source_id):
    """Apply automated data cleaning pipeline"""
    data_source = get_object_or_404(DataSource, id=data_source_id, user=request.user)
    processor = DataProcessor()
    
    success = processor.load_data(data_source.file.path, data_source.data_type)
    if success is not True:
        messages.error(request, f'Error loading data: {success}')
        return redirect('data_cleaning', data_source_id=data_source.id)
    
    # Get automated cleaning strategies
    strategies = processor.get_automated_cleaning_strategies()
    
    # Apply automated cleaning with all options enabled
    processor.clean_data(
        strategies=strategies, 
        remove_duplicates=True, 
        handle_outliers=True,
        standardize_text=True
    )
    
    # Apply common transformations for modeling
    transformations = processor.get_automated_transformations()
    if transformations:
        processor.transform_data(transformations)
    
    # Apply feature engineering for modeling
    feature_operations = processor.get_automated_feature_engineering()
    if feature_operations:
        processor.feature_engineering(feature_operations)
    
    # Save the cleaned data
    cleaned_file_path = data_source.file.path.replace('.', '_cleaned_auto.')
    export_success = processor.export_cleaned_data(cleaned_file_path)
    
    if export_success:
        data_source.file.name = data_source.file.name.replace('.', '_cleaned_auto.')
        data_source.save()
        messages.success(request, 'Automated data cleaning and preprocessing completed successfully! Data is now ready for modeling.')
    else:
        messages.error(request, 'Automated cleaning completed but failed to save data.')
    
    return redirect('data_cleaning', data_source_id=data_source.id)

@login_required
def get_cleaning_preview(request, data_source_id):
    """Get preview of cleaning results"""
    if request.method == 'POST':
        data_source = get_object_or_404(DataSource, id=data_source_id, user=request.user)
        processor = DataProcessor()
        
        success = processor.load_data(data_source.file.path, data_source.data_type)
        if success is not True:
            return JsonResponse({'error': 'Failed to load data'}, status=400)
        
        try:
            strategies_json = request.POST.get('strategies', '{}')
            strategies = json.loads(strategies_json)
            
            # Create a copy for preview without modifying original
            preview_processor = DataProcessor()
            preview_processor.df = processor.df.copy()
            
            # Apply strategies to preview
            remove_duplicates = request.POST.get('remove_duplicates') == 'true'
            handle_outliers = request.POST.get('handle_outliers') == 'true'
            
            preview_processor.clean_data(strategies, remove_duplicates, handle_outliers)
            
            # Get preview data
            preview_info = preview_processor.get_basic_info()
            preview_sample = preview_processor.get_sample_data(5)
            
            return JsonResponse({
                'preview_info': preview_info,
                'preview_sample': preview_sample,
                'cleaning_report': preview_processor.cleaning_report
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    
    return JsonResponse({'error': 'Invalid request'}, status=400)



@login_required
def model_management(request):
    """Main model management dashboard"""
    user_models = PredictiveModel.objects.filter(data_source__user=request.user).select_related('data_source')
    
    # Statistics
    total_models = user_models.count()
    trained_models = user_models.filter(status='trained').count()
    deployed_models = user_models.filter(status='deployed').count()
    
    # Recent activity
    recent_models = user_models.order_by('-created_at')[:5]
    recent_predictions = ModelPrediction.objects.filter(model__data_source__user=request.user).select_related('model').order_by('-created_at')[:5]
    
    context = {
        'total_models': total_models,
        'trained_models': trained_models,
        'deployed_models': deployed_models,
        'recent_models': recent_models,
        'recent_predictions': recent_predictions,
        'all_models': user_models,
    }
    return render(request, 'model_management.html', context)

@login_required
def model_detail(request, model_id):
    """Detailed view of a specific model"""
    model = get_object_or_404(PredictiveModel, id=model_id, data_source__user=request.user)
    predictions = ModelPrediction.objects.filter(model=model).order_by('-created_at')[:10]
    versions = ModelVersion.objects.filter(model=model).order_by('-version_number')
    
    # Prediction form
    prediction_form = PredictionForm()
    
    context = {
        'model': model,
        'predictions': predictions,
        'versions': versions,
        'prediction_form': prediction_form,
    }
    return render(request, 'model_detail.html', context)

@login_required
@require_POST
def delete_model(request, model_id):
    """Delete a trained model"""
    model = get_object_or_404(PredictiveModel, id=model_id, data_source__user=request.user)
    
    # Delete associated file
    if model.model_file:
        if os.path.isfile(model.model_file.path):
            os.remove(model.model_file.path)
    
    model_name = model.name
    model.delete()
    
    messages.success(request, f'Model "{model_name}" has been deleted successfully.')
    return redirect('model_management')

@login_required
@require_POST
def deploy_model(request, model_id):
    """Deploy a model for predictions"""
    model = get_object_or_404(PredictiveModel, id=model_id, data_source__user=request.user)
    
    # Undeploy other models from the same data source
    PredictiveModel.objects.filter(data_source=model.data_source, status='deployed').update(status='trained')
    
    # Deploy this model
    model.status = 'deployed'
    model.save()
    
    messages.success(request, f'Model "{model.name}" has been deployed successfully.')
    return redirect('model_detail', model_id=model.id)

@login_required
@require_POST
def make_prediction(request, model_id):
    """Make a prediction using a trained model"""
    
    def convert_value(value):
        """Convert input values to appropriate data types"""
        if isinstance(value, (int, float)):
            return value
        try:
            return float(value)
        except (ValueError, TypeError):
            try:
                return int(value)
            except (ValueError, TypeError):
                return str(value)
    
    # Check if this is an AJAX request
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
    
    try:
        model = get_object_or_404(PredictiveModel, id=model_id, data_source__user=request.user)
        
        if model.status != 'trained' and model.status != 'deployed':
            error_msg = 'This model is not ready for predictions.'
            if is_ajax:
                return JsonResponse({'success': False, 'error': error_msg})
            messages.error(request, error_msg)
            return redirect('model_detail', model_id=model.id)
        
        form = PredictionForm(request.POST)
        if not form.is_valid():
            error_msg = 'Invalid form data. Please check your input.'
            if is_ajax:
                return JsonResponse({'success': False, 'error': error_msg, 'form_errors': form.errors})
            messages.error(request, error_msg)
            return redirect('model_detail', model_id=model.id)
        
        input_data = json.loads(form.cleaned_data['input_data'])
        
        # Convert input values to appropriate data types
        for key, value in input_data.items():
            input_data[key] = convert_value(value)
        
        # Check if model file exists
        if not model.model_file:
            error_msg = 'Model file not found. Please retrain the model.'
            if is_ajax:
                return JsonResponse({'success': False, 'error': error_msg})
            messages.error(request, error_msg)
            return redirect('model_detail', model_id=model.id)
        
        # Try multiple possible file paths
        possible_paths = [
            model.model_file.path,
            os.path.join(settings.MEDIA_ROOT, model.model_file.name),
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if not model_path:
            error_msg = 'Model file not found. Please retrain the model.'
            if is_ajax:
                return JsonResponse({'success': False, 'error': error_msg})
            messages.error(request, error_msg)
            return redirect('model_detail', model_id=model.id)
        
        # Load the trained model
        try:
            import joblib
            model_data = joblib.load(model_path)
            trained_model = model_data['model']
            scaler = model_data['scaler']
            label_encoders = model_data.get('label_encoders', {})
        except Exception as e:
            error_msg = f'Failed to load model: {str(e)}'
            if is_ajax:
                return JsonResponse({'success': False, 'error': error_msg})
            messages.error(request, error_msg)
            return redirect('model_detail', model_id=model.id)
        
        # Get model parameters
        model_params = model.parameters
        feature_columns = model_params.get('feature_columns', [])
        target_column = model_params.get('target_column')
        
        # Validate input data
        if not isinstance(input_data, dict):
            error_msg = 'Input data must be a JSON object with feature names as keys.'
            if is_ajax:
                return JsonResponse({'success': False, 'error': error_msg})
            messages.error(request, error_msg)
            return redirect('model_detail', model_id=model.id)
        
        # Check required features
        missing_features = [col for col in feature_columns if col not in input_data]
        if missing_features:
            error_msg = f'Missing features: {", ".join(missing_features)}'
            if is_ajax:
                return JsonResponse({'success': False, 'error': error_msg})
            messages.error(request, error_msg)
            return redirect('model_detail', model_id=model.id)
        
        # Create input DataFrame
        try:
            input_df = pd.DataFrame([input_data])[feature_columns]
        except Exception as e:
            error_msg = f'Error creating input data: {str(e)}'
            if is_ajax:
                return JsonResponse({'success': False, 'error': error_msg})
            messages.error(request, error_msg)
            return redirect('model_detail', model_id=model.id)
        
        # Store original input
        original_input = input_df.to_dict('records')[0]
        
        # Preprocess input data
        try:
            # Handle categorical features
            for column in input_df.columns:
                if column in label_encoders:
                    le = label_encoders[column]
                    input_df[column] = input_df[column].astype(str)
                    # Handle unseen categories
                    input_df[column] = input_df[column].apply(
                        lambda x: x if x in le.classes_ else le.classes_[0]
                    )
                    input_df[column] = le.transform(input_df[column])
            
            # Scale features
            input_scaled = scaler.transform(input_df)
            
        except Exception as e:
            error_msg = f'Data preprocessing failed: {str(e)}'
            if is_ajax:
                return JsonResponse({'success': False, 'error': error_msg})
            messages.error(request, error_msg)
            return redirect('model_detail', model_id=model.id)
        
        # Make prediction
        try:
            prediction_result = {
                'timestamp': datetime.now().isoformat(),
                'model_type': model.model_type,
                'model_name': model.name,
                'input_features': original_input,
            }
            
            if hasattr(trained_model, 'predict_proba'):
                # Classification model
                prediction = trained_model.predict(input_scaled)[0]
                probabilities = trained_model.predict_proba(input_scaled)[0]
                confidence = float(max(probabilities))
                
                # Get class labels
                if target_column in label_encoders:
                    le_target = label_encoders[target_column]
                    prediction_label = le_target.inverse_transform([prediction])[0]
                    class_probabilities = {
                        le_target.inverse_transform([i])[0]: {
                            'probability': float(prob),
                            'percentage': float(prob) * 100
                        }
                        for i, prob in enumerate(probabilities)
                    }
                else:
                    prediction_label = str(prediction)
                    class_probabilities = {
                        str(i): {
                            'probability': float(prob),
                            'percentage': float(prob) * 100
                        }
                        for i, prob in enumerate(probabilities)
                    }
                
                # Sort by probability
                sorted_classes = sorted(
                    class_probabilities.items(),
                    key=lambda x: x[1]['probability'],
                    reverse=True
                )
                
                prediction_result.update({
                    'prediction': prediction_label,
                    'confidence': confidence,
                    'confidence_percentage': confidence * 100,
                    'sorted_predictions': [
                        {
                            'class': class_name,
                            'probability': prob_info['probability'],
                            'percentage': prob_info['percentage']
                        }
                        for class_name, prob_info in sorted_classes
                    ],
                    'is_classification': True,
                })
                
            else:
                # Regression model
                prediction = trained_model.predict(input_scaled)[0]
                prediction_result.update({
                    'prediction': float(prediction),
                    'is_classification': False,
                })
            
            # For clustering models
            if model.model_type == 'kmeans':
                cluster = trained_model.predict(input_scaled)[0]
                distance_to_centers = trained_model.transform(input_scaled)[0]
                
                prediction_result.update({
                    'cluster_assignment': int(cluster),
                    'distance_to_centers': [float(d) for d in distance_to_centers],
                    'is_clustering': True
                })
            
        except Exception as e:
            error_msg = f'Prediction failed: {str(e)}'
            if is_ajax:
                return JsonResponse({'success': False, 'error': error_msg})
            messages.error(request, error_msg)
            return redirect('model_detail', model_id=model.id)
        
        # Save prediction to database
        prediction_obj = ModelPrediction.objects.create(
            model=model,
            input_data=input_data,
            prediction=prediction_result,
            confidence=prediction_result.get('confidence')
        )
        
        # Return response
        if is_ajax:
            return JsonResponse({
                'success': True,
                'prediction': prediction_result,
                'prediction_id': prediction_obj.id,
                'message': f'Prediction successful! Result: {prediction_result["prediction"]}'
            })
        else:
            messages.success(request, f'Prediction made successfully! Result: {prediction_result["prediction"]}')
            return redirect('model_detail', model_id=model.id)
            
    except json.JSONDecodeError:
        error_msg = 'Invalid JSON format in input data.'
        if is_ajax:
            return JsonResponse({'success': False, 'error': error_msg})
        messages.error(request, error_msg)
        return redirect('model_detail', model_id=model_id)
    except Exception as e:
        error_msg = f'Unexpected error: {str(e)}'
        if is_ajax:
            return JsonResponse({'success': False, 'error': error_msg})
        messages.error(request, error_msg)
        return redirect('model_detail', model_id=model_id)

@login_required
def create_model_version(request, model_id):
    """Create a new version of a model"""
    original_model = get_object_or_404(PredictiveModel, id=model_id, data_source__user=request.user)
    
    if request.method == 'POST':
        form = PredictiveModelForm(request.POST)
        if form.is_valid():
            try:
                new_model = form.save(commit=False)
                new_model.data_source = original_model.data_source
                new_model.parent_model = original_model
                new_model.version = original_model.version + 1
                new_model.status = 'training'
                new_model.save()
                
                # Create version record
                changes = request.POST.get('changes', 'Initial version')
                ModelVersion.objects.create(
                    model=new_model,
                    version_number=new_model.version,
                    changes=changes
                )
                
                messages.success(request, f'New version {new_model.version} created for model "{original_model.name}"')
                return redirect('model_detail', model_id=new_model.id)
                
            except Exception as e:
                messages.error(request, f'Failed to create model version: {str(e)}')
        else:
            # Show form errors
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f'{field}: {error}')
    
    else:
        form = PredictiveModelForm(initial={
            'name': f"{original_model.name} v{original_model.version + 1}",
            'model_type': original_model.model_type,
            'description': original_model.description
        })
    
    context = {
        'form': form,
        'original_model': original_model,
    }
    return render(request, 'create_model_version.html', context)

@login_required
def model_comparison(request):
    """Compare multiple models"""
    model_ids = request.GET.getlist('models')
    models = PredictiveModel.objects.filter(
        id__in=model_ids, 
        data_source__user=request.user,
        status__in=['trained', 'deployed']
    ).order_by('-accuracy')  # Sort by accuracy descending
    
    # Get common metrics across all models
    common_metrics = set()
    for model in models:
        if model.metrics:
            common_metrics.update(model.metrics.keys())
    common_metrics = sorted(common_metrics)
    
    # Get all user models for the selection form
    user_models = PredictiveModel.objects.filter(
        data_source__user=request.user,
        status__in=['trained', 'deployed']
    ).select_related('data_source')
    
    context = {
        'models': models,
        'common_metrics': common_metrics,
        'user_models': user_models,
    }
    return render(request, 'model_comparison.html', context)


@login_required
def predictive_modeling_redirect(request):
    """Redirect to predictive modeling with the latest data source or show available data sources"""
    # Get user's most recent data source
    latest_data_source = DataSource.objects.filter(user=request.user).order_by('-uploaded_at').first()
    
    if latest_data_source:
        return redirect('predictive_modeling', data_source_id=latest_data_source.id)
    else:
        messages.info(request, 'Please upload a dataset first to train models.')
        return redirect('upload_data')


@login_required
def download_model(request, model_id):
    """Download trained model file"""
    model = get_object_or_404(PredictiveModel, id=model_id, data_source__user=request.user)
    
    if not model.model_file:
        messages.error(request, 'Model file not found.')
        return redirect('model_detail', model_id=model.id)
    
    # Check if file exists
    if not os.path.exists(model.model_file.path):
        messages.error(request, 'Model file not found on server.')
        return redirect('model_detail', model_id=model.id)
    
    # Serve the file for download
    response = FileResponse(open(model.model_file.path, 'rb'))
    response['Content-Type'] = 'application/octet-stream'
    response['Content-Disposition'] = f'attachment; filename="{model.name}_v{model.version}.pkl"'
    
    return response


@login_required
def get_prediction_details(request, prediction_id):
    """Get prediction details for AJAX requests"""
    try:
        prediction = get_object_or_404(
            ModelPrediction, 
            id=prediction_id, 
            model__data_source__user=request.user
        )
        
        return JsonResponse({
            'success': True,
            'prediction': prediction.prediction
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Failed to load prediction: {str(e)}'
        }, status=500)