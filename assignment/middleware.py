import logging

from django.contrib.auth import login
from django.http import HttpResponse

from assignment.models import User

logger = logging.getLogger(__name__)


# Preserved from Mantra's assignment template for /api/v1/user/me/ compatibility
class MockLoginUserMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.path.startswith("/api/v1/user"):
            username = request.headers.get("X-User-NAME")
            if username:
                logger.info("Mock login for user: %s", username)
                try:
                    user = User.objects.get(username=username)
                    login(request, user)
                except User.DoesNotExist:
                    return HttpResponse("User not found or invalid credentials.", status=401)
        return self.get_response(request)
