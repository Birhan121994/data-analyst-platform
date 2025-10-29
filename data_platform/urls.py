from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.dashboard, name='dashboard'),
    path('upload/', views.data_upload, name='data_upload'),
    path('data/<int:data_source_id>/explore/', views.data_exploration, name='data_exploration'),
    path('data/<int:data_source_id>/analyze/', views.data_analysis, name='data_analysis'),
    path('data/<int:data_source_id>/visualize/', views.data_visualization, name='data_visualization'),
    path('data/<int:data_source_id>/model/', views.predictive_modeling, name='predictive_modeling'),
    path('data/<int:data_source_id>/cleaning/', views.data_cleaning, name='data_cleaning'),
    path('data/<int:data_source_id>/automated-cleaning/', views.automated_cleaning, name='automated_cleaning'),

    # Model Management URLs
    path('models/', views.model_management, name='model_management'),
    path('models/<int:model_id>/', views.model_detail, name='model_detail'),
    path('models/<int:model_id>/delete/', views.delete_model, name='delete_model'),
    path('models/<int:model_id>/deploy/', views.deploy_model, name='deploy_model'),
    path('models/<int:model_id>/predict/', views.make_prediction, name='make_prediction'),
    path('models/<int:model_id>/version/', views.create_model_version, name='create_model_version'),
    path('models/compare/', views.model_comparison, name='model_comparison'),
    path('models/<int:model_id>/download/', views.download_model, name='download_model'),
    path('predictions/<int:prediction_id>/', views.get_prediction_details, name='get_prediction_details'),
    
    # Add a redirect URL for predictive modeling without data_source_id
    path('predictive-modeling/', views.predictive_modeling_redirect, name='predictive_modeling_redirect'),

    path('', views.landing_view, name='landing'),
    path('signup/', views.signup_view, name='signup'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
]

# Serve static and media files during development
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)