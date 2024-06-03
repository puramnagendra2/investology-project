from django.urls import path, include
from . import views


urlpatterns = [
    path('', views.searchPage, name="searchPage"),
    path('searching/', views.searching, name="searching"),
    path('navigation/', views.navigation, name="navigation"),
]
