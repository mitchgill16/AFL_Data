from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='blog_home'),
    path('about/', views.about, name='blog_about'),
    path('history/', views.history, name='blog_history'),
    path('discussion/', views.discussion, name='blog_discussion')
]
