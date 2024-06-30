from django.urls import path, include
from . import views


urlpatterns = [
    path('', views.searchPage, name="searchPage"),
    path('overview/', views.overview, name="overview"),
    path('navigation/', views.navigation, name="navigation"),
    path('candlestick_chart/', views.candlestick_chart, name='candlestick_chart'),
]
