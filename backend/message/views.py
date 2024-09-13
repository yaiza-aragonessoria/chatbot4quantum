from django.db.models import Q
from rest_framework import status
from rest_framework.response import Response
from rest_framework.generics import ListCreateAPIView, RetrieveUpdateDestroyAPIView, get_object_or_404, ListAPIView

from message.models import Message
from message.serializers import MessageSerializer

from user.models import User




class ListCreateMessageView(ListCreateAPIView):
    """
    get: Lists all messages of the logged-in in chronological order.

    post: Creates a new Message.
    """
    serializer_class = MessageSerializer

    def get_queryset(self):
        return Message.objects.order_by("created")

    def perform_create(self, serializer):
        # Get the user information from the request's body
        user_email = self.request.data.get('user_email')

        if not user_email:
            return Response({"error": "User email not provided in the request body"},
                            status=status.HTTP_400_BAD_REQUEST)

        try:
            # Retrieve the user based on the provided email
            user = User.objects.get(email=user_email)
        except User.DoesNotExist:
            return Response({"error": f"User with email {user_email} not found"}, status=status.HTTP_404_NOT_FOUND)

        # Associate the retrieved user with the new message
        serializer.save(user=user)


class RetrieveUpdateDeleteMessageView(RetrieveUpdateDestroyAPIView):
    """
        get: Retrieves a specific Message.
        patch: Updates a specific Message.
        delete: Deletes a Message.
    """
    queryset = Message.objects.all()
    serializer_class = MessageSerializer
    lookup_field = 'id'  # field in the database
    lookup_url_kwarg = 'id_message'  # field in the request
    http_method_names = ['get', 'patch', 'delete']  # disallow put as we don't use it



class ListUserMessagesView(ListAPIView):
    """
    get: Lists all messages of the logged-in in chronological order.

    post: Creates a new Message.
    """
    serializer_class = MessageSerializer

    def get_queryset(self):
        user_email = self.request.query_params.get('user_email')
        messages = Message.objects.filter(Q(user__email=user_email)).order_by("created")
        return messages

