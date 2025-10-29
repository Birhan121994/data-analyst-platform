from django import template
import math

register = template.Library()

@register.filter
def get_item(dictionary, key):
    """Template filter to get dictionary item by key"""
    if isinstance(dictionary, dict):
        return dictionary.get(key)
    return None

@register.filter
def divide(value, arg):
    """Divide the value by the argument"""
    try:
        return float(value) / float(arg)
    except (ValueError, ZeroDivisionError):
        return None

@register.filter
def multiply(value, arg):
    """Multiply the value by the argument"""
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return None

@register.filter
def filesizeformat(bytes_value):
    """Format bytes to human readable format"""
    try:
        bytes_value = float(bytes_value)
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} TB"
    except (ValueError, TypeError):
        return "0 B"

@register.filter
def sum_values(dictionary):
    """Sum all values in a dictionary"""
    try:
        if isinstance(dictionary, dict):
            return sum(float(v) for v in dictionary.values() if v is not None)
        return 0
    except (ValueError, TypeError):
        return 0

@register.filter
def sum_list(value_list):
    """Sum all values in a list"""
    try:
        if isinstance(value_list, (list, tuple)):
            return sum(float(v) for v in value_list if v is not None)
        return 0
    except (ValueError, TypeError):
        return 0

@register.filter
def get_missing_values(dictionary):
    """Get missing values count from data quality report"""
    try:
        if isinstance(dictionary, dict) and 'missing_values' in dictionary:
            return sum(int(v) for v in dictionary['missing_values'].values() if v is not None)
        return 0
    except (ValueError, TypeError):
        return 0

@register.filter
def get_duplicate_rows(dictionary):
    """Get duplicate rows count from data quality report"""
    try:
        if isinstance(dictionary, dict) and 'duplicate_rows' in dictionary:
            return int(dictionary['duplicate_rows'])
        return 0
    except (ValueError, TypeError):
        return 0

@register.filter
def get_numeric_columns_count(dictionary):
    """Get numeric columns count from data quality report"""
    try:
        if isinstance(dictionary, dict) and 'numeric_columns' in dictionary:
            return len(dictionary['numeric_columns'])
        return 0
    except (ValueError, TypeError):
        return 0

@register.filter
def percentage(value, total):
    """Calculate percentage"""
    try:
        if total and float(total) > 0:
            return (float(value) / float(total)) * 100
        return 0
    except (ValueError, TypeError):
        return 0

@register.filter
def round_value(value, digits=2):
    """Round a value to specified digits"""
    try:
        return round(float(value), digits)
    except (ValueError, TypeError):
        return value

@register.filter
def is_regression_model(model_type):
    """Check if model type is for regression (uses R² score)"""
    regression_models = ['linear_regression', 'random_forest', 'decision_tree', 'svm', 'neural_network']
    return model_type in regression_models

@register.filter
def replace_underscores(value):
    """Replace underscores with spaces"""
    return value.replace('_', ' ')

@register.filter
def format_model_type(value):
    """Format model type for display"""
    return value.replace('_', ' ').title()

@register.filter
def replace(value, arg):
    """Replace characters in string"""
    old, new = arg.split(',')
    return value.replace(old, new)

@register.filter
def format_metric_name(value):
    """Format metric name for display - replace underscores and title case"""
    if not value:
        return ""
    return value.replace('_', ' ').title()

@register.filter
def dict_values_sum(dictionary):
    """Sum all values in a dictionary"""
    return sum(dictionary.values())

@register.filter
def get_item_nested(dictionary, key):
    """Get nested dictionary item"""
    if dictionary and key in dictionary:
        return dictionary[key]
    return {}

@register.filter
def sum_values(dictionary):
    """Sum all values in a dictionary"""
    if isinstance(dictionary, dict):
        return sum(dictionary.values())
    return 0

@register.filter
def is_ready_for_modeling(basic_info):
    """Check if data is ready for modeling based on basic info"""
    if not basic_info:
        return (False, "No data available")
    
    # Check for missing values
    missing_values = basic_info.get('missing_values', {})
    total_missing = sum(missing_values.values())
    if total_missing > 0:
        return (False, f"Data has {total_missing} missing values")
    
    # Check for numeric columns
    numeric_columns = basic_info.get('numeric_columns', [])
    if len(numeric_columns) == 0:
        return (False, "No numeric columns available for modeling")
    
    return (True, "Data is ready for modeling")



@register.filter
def filesizeformat_mb(value):
    """Format filesize in MB"""
    try:
        return filesizeformat(value)
    except:
        return "0 MB"