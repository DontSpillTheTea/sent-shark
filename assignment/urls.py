from django.contrib import admin
from django.urls import include, path
from rest_framework.routers import DefaultRouter

from assignment.views import (
    JudgeBulkView,
    JudgeEvaluateView,
    JudgeSingleView,
    JudgeStatsView,
    UserViewSet,
    health_check,
    initialize_data,
)

router = DefaultRouter()
router.register(r"user", UserViewSet, basename="user")

urlpatterns = [
    path("admin/", admin.site.urls),
    path("health", health_check, name="health"),
    path("health/", health_check),
    path("init_data/", initialize_data, name="initialize_data"),
    # Canonical F1-F3 endpoints (with & without trailing slash)
    path("judge", JudgeSingleView.as_view(), name="judge-single"),
    path("judge/", JudgeSingleView.as_view()),
    path("judge/bulk", JudgeBulkView.as_view(), name="judge-bulk"),
    path("judge/bulk/", JudgeBulkView.as_view()),
    path("judge/stats", JudgeStatsView.as_view(), name="judge-stats"),
    path("judge/stats/", JudgeStatsView.as_view()),
    path("judge/evaluate", JudgeEvaluateView.as_view(), name="judge-evaluate"),
    path("judge/evaluate/", JudgeEvaluateView.as_view()),
    # Versioned /api/v1/ aliases
    path("api/v1/", include(router.urls)),
    path("api/v1/judge", JudgeSingleView.as_view(), name="api-judge-single"),
    path("api/v1/judge/", JudgeSingleView.as_view()),
    path("api/v1/judge/bulk", JudgeBulkView.as_view(), name="api-judge-bulk"),
    path("api/v1/judge/bulk/", JudgeBulkView.as_view()),
    path("api/v1/judge/stats", JudgeStatsView.as_view(), name="api-judge-stats"),
    path("api/v1/judge/stats/", JudgeStatsView.as_view()),
    path("api/v1/judge/evaluate", JudgeEvaluateView.as_view(), name="api-judge-evaluate"),
    path("api/v1/judge/evaluate/", JudgeEvaluateView.as_view()),
]
