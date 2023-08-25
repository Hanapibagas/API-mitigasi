from rest_framework import serializers

class getClassification(serializers.Serializer):
    reportmsg = serializers.CharField(max_length=250)