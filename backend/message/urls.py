from django.urls import path

from message.views import ListCreateMessageView, RetrieveUpdateDeleteMessageView, ListUserMessagesView

urlpatterns = [
    path('', ListCreateMessageView.as_view()),
    path('<int:id_message>/', RetrieveUpdateDeleteMessageView.as_view()),
    path('user/', ListUserMessagesView.as_view()),
]
