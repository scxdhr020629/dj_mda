from django.shortcuts import render
from dj_map import models
# Create your views here.

def ip_list(request):
    ip_queryset = models.