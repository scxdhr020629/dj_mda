from django.db import models

# Create your models here.
class IPCount(models.Model):
    ip = models.CharField(max_length=255)
    location = models.CharField(max_length=255)
    ISP = models.CharField(max_length=255)