from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='index'),
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),
    path('train/', views.train, name='train'),
    path('predict/', views.predict, name='predict'),
]