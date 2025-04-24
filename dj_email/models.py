# Create your models here.
from django.db import models


class UploadedFile(models.Model):
    file = models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)


class Node(models.Model):
    NODE_TYPES = (
        ('drug', 'Drug'),
        ('rna', 'RNA'),
    )

    node_id = models.CharField(max_length=100, unique=True)
    name = models.CharField(max_length=255)
    node_type = models.CharField(max_length=10, choices=NODE_TYPES)
    properties = models.JSONField(default=dict, blank=True)

    def __str__(self):
        return f"{self.node_type}: {self.name}"


class Relationship(models.Model):
    source = models.ForeignKey(Node, related_name='source_relationships', on_delete=models.CASCADE)
    target = models.ForeignKey(Node, related_name='target_relationships', on_delete=models.CASCADE)
    relationship_type = models.CharField(max_length=100)
    properties = models.JSONField(default=dict, blank=True)

    def __str__(self):
        return f"{self.source.name} --[{self.relationship_type}]--> {self.target.name}"