from rest_framework import serializers
from .models import Message


class UserMessageSerializer(serializers.Serializer):
    first_name = serializers.CharField(read_only=True)
    last_name = serializers.CharField(read_only=True)
    avatar = serializers.ImageField(read_only=True)


class MessageSerializer(serializers.ModelSerializer):

    class Meta:
        model = Message
        fields = '__all__'
        required_fields = ['content']
        read_only_fields = ['created', 'updated', 'user']

    def to_representation(self, instance):
        data = super().to_representation(instance)
        data["user"] = UserMessageSerializer(instance.user).data
        return data
