from django.urls import path
from user.views import ListUserView, RetrieveUserView, LoggedInUserView, CreateUserView, DeleteUserView


urlpatterns = [
    path('', ListUserView.as_view()),
    path('<int:id_user>/', RetrieveUserView.as_view()),
    path('create/', CreateUserView.as_view()),
    path('delete/', DeleteUserView.as_view()),
    path('me/', LoggedInUserView.as_view()),
]
