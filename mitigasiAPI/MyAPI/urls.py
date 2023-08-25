from django.urls import path, include
from . import views,viewtelegram
from rest_framework import routers

router = routers.DefaultRouter()

urlpatterns = [
	path('mitigasi/', views.classification, name='reportmsg'),
   path('hit-me/', viewtelegram.send_message_view, name='my_view'),
   # path('get_analisis/', viewtelegram.get_anallisis, name='my_view'),
   
] 
