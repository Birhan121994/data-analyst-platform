# Data Analyst Platform

A modern, web-based data analysis platform built with Django that allows users to upload, clean, analyze, visualize data and train a machine learning model and manage trained models through an intuitive interface.

![Data Analyst Platform](https://img.shields.io/badge/Django-4.2.7-green) ![Python](https://img.shields.io/badge/Python-3.10-blue) ![License](https://img.shields.io/badge/License-MIT-yellow) ![Deployment](https://img.shields.io/badge/Deployment-Render-blue)

## 🚀 Live Demo

Check out the live deployment: [Data Analyst Platform on Render](https://data-analyst-platform.onrender.com/)

**Demo Credentials:**
- Username: `admin`
- Password: `admin123`

## ✨ Features

### 📊 Data Management
- **Multi-format Support**: Upload CSV, Excel, and JSON files
- **Data Preview**: Instant preview of uploaded datasets
- **Column Analysis**: Automatic data type detection and statistics

### 🧹 Data Cleaning
- **Missing Value Handling**: Multiple strategies (mean, median, mode, drop)
- **Outlier Detection**: Automatic outlier detection and capping
- **Duplicate Removal**: Identify and remove duplicate rows
- **Text Standardization**: Clean and standardize text data

### 📈 Data Analysis
- **Descriptive Statistics**: Comprehensive statistical summaries
- **Correlation Analysis**: Identify relationships between variables
- **Regression Analysis**: Linear regression with performance metrics
- **Clustering**: K-means clustering with silhouette scores

### 📊 Visualization
- **Interactive Charts**: Line, bar, scatter, histogram, box plots
- **Correlation Heatmaps**: Visual correlation matrices
- **Dashboard Creation**: Combine multiple visualizations
- **Export Options**: Download charts and analysis results

## 🤖 Predictive Modeling & Model Management 

The **Predictive Modeling** module extends the platform into a lightweight machine learning environment.

### 🧩 Key Capabilities

#### 🧠 Model Training
Train supervised learning models directly in the web interface using **scikit-learn**.

**Supported models include:**
- Linear Regression  
- Logistic Regression  
- Decision Tree  
- Random Forest  
- Support Vector Machines (SVM)

#### 🔍 Feature Selection
Automatically detect numerical and categorical features for training.

#### ⚙️ Train/Test Split
Adjustable split ratios for model validation.

#### 📊 Performance Metrics
**Regression:**
- R² (Coefficient of Determination)  
- RMSE (Root Mean Squared Error)  
- MAE (Mean Absolute Error)

**Classification:**
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix

#### 📈 Visualization of Results
- Model performance comparison  
- Predicted vs Actual value plots  
- Feature importance charts (for tree-based models)

#### 💾 Model Persistence
Save trained models with metadata such as:
- Training date  
- Algorithm type  
- Performance metrics  
for easy reuse.

#### 🗂️ Model Management
- List, update, or delete trained models  
- Load saved models to make new predictions without retraining  
- Track model versions and updates

---

### ⚡ Example Workflow
1. **Upload your dataset**  
2. **Choose target and feature columns**  
3. **Select a machine learning algorithm**  
4. **Train and evaluate the model**  
5. **Save the model** for later use or prediction  
6. **Manage trained models** from the “Model Management” dashboard

---

### 🔐 User Management
- **Modern Authentication**: Clean signup and login interface
- **User Isolation**: Data privacy between users
- **Session Management**: Secure user sessions

## 🛠️ Technology Stack

- **Backend**: Django 4.2.7
- **Frontend**: Tailwind CSS, Plotly.js
- **Database**: SQLite (Development), PostgreSQL-ready
- **Deployment**: Render
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn, Plotly

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip (Python package manager)

