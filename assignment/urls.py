from django.contrib import admin
from django.urls import include, path
from rest_framework.routers import DefaultRouter

from assignment.views import (
    JudgeBulkView,
    JudgeSingleView,
    UserViewSet,
    health_check,
    initialize_data,
    playground_view,
)

router = DefaultRouter()
router.register(r"user", UserViewSet, basename="user")

urlpatterns = [
    path("admin/", admin.site.urls),
    path("health", health_check, name="health"),
    path("health/", health_check),
    path("playground", playground_view, name="playground"),
    path("playground/", playground_view, name="playground-slash"),
    path("init_data/", initialize_data, name="initialize_data"),
    # Canonical F1-F2 endpoints (with & without trailing slash)
    path("judge", JudgeSingleView.as_view(), name="judge-single"),
    path("judge/", JudgeSingleView.as_view()),
    path("judge/bulk", JudgeBulkView.as_view(), name="judge-bulk"),
    path("judge/bulk/", JudgeBulkView.as_view()),
    # Preserved template router (/api/v1/user/me/)
    path("api/v1/", include(router.urls)),
]
