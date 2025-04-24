# api/serializers.py
from rest_framework import serializers
from .models import UploadedFile, Node, Relationship


class UploadedFileSerializer(serializers.ModelSerializer):
    class Meta:
        model = UploadedFile
        fields = ('id', 'file', 'uploaded_at')


class NodeSerializer(serializers.ModelSerializer):
    class Meta:
        model = Node
        fields = ('id', 'node_id', 'name', 'node_type', 'properties')


class RelationshipSerializer(serializers.ModelSerializer):
    source_name = serializers.CharField(source='source.name', read_only=True)
    target_name = serializers.CharField(source='target.name', read_only=True)
    source_id = serializers.CharField(source='source.node_id', read_only=True)
    target_id = serializers.CharField(source='target.node_id', read_only=True)

    class Meta:
        model = Relationship
        fields = ('id', 'source', 'target', 'source_name', 'target_name',
                  'source_id', 'target_id', 'relationship_type', 'properties')