import json

from django.contrib.auth import get_user_model
from django.db.models import Q
from django.http import JsonResponse
from django.views import View
from rest_framework.generics import ListAPIView, RetrieveUpdateDestroyAPIView, RetrieveAPIView, ListCreateAPIView, \
    DestroyAPIView, CreateAPIView
from message.models import Message
from user.serializers import UserSerializer


User = get_user_model()


class ListUserView(ListAPIView):
    """
    get:
    Lists all users.
    """
    serializer_class = UserSerializer
    # permission_classes = [IsAuthenticated]

    def get_queryset(self):
        query = self.request.GET.get('search', '')  # search is the params and '' the default value
        queryset = User.objects.filter(
            Q(email__contains=query) | Q(first_name__contains=query) | Q(last_name__contains=query))
        return queryset


class LoggedInUserView(RetrieveUpdateDestroyAPIView):
    """
        get:
        Retrieves logged-in User and displays all details.

        patch:
        Updates details of logged-in User.

        delete:
        Deletes logged-in User.

    """
    http_method_names = ['get', 'patch', 'delete']  # disallow put as we don't use it
    serializer_class = UserSerializer
    # permission_classes = [IsAuthenticated]

    def get_object(self):
        return self.request.user


class RetrieveUserView(RetrieveAPIView):
    """
        get:
        Retrieves a specific User by ID and displays all the information about it.

        patch:
        Updates the status of a specific User.

        delete:
        Deletes a User by ID.

    """
    lookup_field = 'id'
    lookup_url_kwarg = 'id_user'
    queryset = User.objects.all()
    serializer_class = UserSerializer
    # permission_classes = [IsAuthenticated]


class CreateUserView(CreateAPIView):
    user_serializer = UserSerializer
    http_method_names = ['post']
    def post(self, request, *args, **kwargs):
        data = json.loads(request.body)
        user_email = data.get('email')
        user = User.objects.create(email=user_email)
        return JsonResponse({'message': 'User created successfully'})


class DeleteUserView(DestroyAPIView):
    user_serializer = UserSerializer
    http_method_names = ['post']

    def post(self, request, *args, **kwargs):
        data = json.loads(request.body)
        user_email = data.get('email')
        try:
            user = User.objects.get(email=user_email)
            messages = Message.objects.filter(user=user)
            messages.delete()  # Delete messages associated to the user
            user.delete()  # Delete the user
            return JsonResponse({'message': 'User deleted successfully'})
        except User.DoesNotExist:
            return JsonResponse({'error': 'User not found'}, status=404)