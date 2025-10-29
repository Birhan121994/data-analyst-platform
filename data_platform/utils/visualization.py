import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import pandas as pd
import numpy as np
import json
from plotly.utils import PlotlyJSONEncoder

class DataVisualizer:
    def __init__(self, dataframe):
        self.df = dataframe
    
    def create_line_chart(self, x_col, y_col, title="Line Chart"):
        """Create a line chart"""
        fig = px.line(self.df, x=x_col, y=y_col, title=title)
        return json.dumps(fig, cls=PlotlyJSONEncoder)
    
    def create_bar_chart(self, x_col, y_col, title="Bar Chart"):
        """Create a bar chart"""
        fig = px.bar(self.df, x=x_col, y=y_col, title=title)
        return json.dumps(fig, cls=PlotlyJSONEncoder)
    
    def create_scatter_plot(self, x_col, y_col, color_col=None, title="Scatter Plot"):
        """Create a scatter plot"""
        fig = px.scatter(self.df, x=x_col, y=y_col, color=color_col, title=title)
        return json.dumps(fig, cls=PlotlyJSONEncoder)
    
    def create_histogram(self, column, title="Histogram"):
        """Create a histogram"""
        fig = px.histogram(self.df, x=column, title=title)
        return json.dumps(fig, cls=PlotlyJSONEncoder)
    
    def create_box_plot(self, x_col, y_col, title="Box Plot"):
        """Create a box plot"""
        fig = px.box(self.df, x=x_col, y=y_col, title=title)
        return json.dumps(fig, cls=PlotlyJSONEncoder)
    
    def create_heatmap(self, title="Correlation Heatmap"):
        """Create a correlation heatmap"""
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) < 2:
            return None
        
        corr_matrix = self.df[numerical_cols].corr()
        fig = px.imshow(corr_matrix, title=title, aspect="auto")
        return json.dumps(fig, cls=PlotlyJSONEncoder)
    
    def create_dashboard(self, columns=None):
        """Create a comprehensive dashboard"""
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()[:4]
        
        fig = sp.make_subplots(
            rows=2, cols=2,
            subplot_titles=['Distribution', 'Correlation', 'Trend', 'Box Plot']
        )
        
        # Histogram
        if len(columns) > 0:
            fig.add_trace(go.Histogram(x=self.df[columns[0]], name='Distribution'), row=1, col=1)
        
        # Correlation heatmap
        if len(columns) > 1:
            corr_data = self.df[columns[:4]].corr()
            fig.add_trace(go.Heatmap(z=corr_data.values, 
                                   x=corr_data.columns, 
                                   y=corr_data.index), row=1, col=2)
        
        # Line chart
        if len(columns) > 2 and len(self.df) > 10:
            sample_df = self.df.head(50)  # Sample for performance
            fig.add_trace(go.Scatter(x=sample_df.index, y=sample_df[columns[2]], 
                                   mode='lines', name='Trend'), row=2, col=1)
        
        # Box plot
        if len(columns) > 3:
            fig.add_trace(go.Box(y=self.df[columns[3]], name='Box Plot'), row=2, col=2)
        
        fig.update_layout(height=600, title_text="Data Analysis Dashboard")
        return json.dumps(fig, cls=PlotlyJSONEncoder)