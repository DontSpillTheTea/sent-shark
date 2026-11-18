import logging
import math
import time

from django.conf import settings
from django.core.management import call_command
from django.db import connection
from rest_framework import status, viewsets
from rest_framework.decorators import action, api_view
from rest_framework.response import Response
from rest_framework.views import APIView

from assignment.models import RequestAuditLog
from assignment.rate_limit import get_client_ip, rate_limiter
from assignment.serializers import (
    BulkJudgeWrapperSerializer,
    EvaluateRequestSerializer,
    JudgePairSerializer,
)
from assignment.services.judge_service import (
    _SQLITE_WRITE_LOCK,
    compute_stats,
    evaluate_ground_truth,
    judge_bulk,
    judge_single,
    prune_storage_if_needed,
)

logger = logging.getLogger(__name__)


def _record_audit(
    request,
    client_ip: str,
    status_code: int,
    started_at: float,
    payload_chars: int = 0,
    rate_limited: bool = False,
) -> None:
    latency_ms = round((time.perf_counter() - started_at) * 1000.0, 2)

    def _write():
        RequestAuditLog.objects.create(
            client_ip=client_ip[:64],
            method=request.method[:10],
            path=request.path[:255],
            status_code=status_code,
            payload_chars=payload_chars,
            latency_ms=latency_ms,
            rate_limited=rate_limited,
        )
        prune_storage_if_needed(judgments=False, audit_logs=True)

    try:
        if connection.vendor == "sqlite":
            with _SQLITE_WRITE_LOCK:
                _write()
        else:
            _write()
    except Exception as exc:
        logger.warning("Failed to write RequestAuditLog: %s", exc)


class _BaseJudgeView(APIView):
    """
    Base APIView handling per-IP sliding-window rate limiting (5 req/s -> 429)
    and security request audit logging for all Judge endpoints.
    """

    authentication_classes = []
    permission_classes = []

    def _check_rate_limit(self, request, started_at: float) -> tuple[str, Response | None]:
        client_ip = get_client_ip(request)
        rate_limiter.max_requests = int(getattr(settings, "RATE_LIMIT_REQUESTS", 5))
        rate_limiter.window_seconds = float(getattr(settings, "RATE_LIMIT_WINDOW_SECONDS", 1.0))
        decision = rate_limiter.check(client_ip)
        if not decision.allowed:
            _record_audit(
                request,
                client_ip=client_ip,
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                started_at=started_at,
                rate_limited=True,
            )
            retry_after_header = str(max(1, math.ceil(decision.retry_after)))
            return client_ip, Response(
                {
                    "error": (
                        f"Rate limit exceeded: max {rate_limiter.max_requests} requests "
                        f"per {rate_limiter.window_seconds:g}s per client_ip."
                    ),
                    "retry_after": round(decision.retry_after, 3),
                },
                status=status.HTTP_429_TOO_MANY_REQUESTS,
                headers={"Retry-After": retry_after_header},
            )
        return client_ip, None


class JudgeSingleView(_BaseJudgeView):
    """POST /judge — Evaluate entailment between two same-language sentences (F1)."""

    def post(self, request):
        started = time.perf_counter()
        client_ip, limited = self._check_rate_limit(request, started)
        if limited is not None:
            return limited

        serializer = JudgePairSerializer(data=request.data)
        if not serializer.is_valid():
            _record_audit(
                request,
                client_ip=client_ip,
                status_code=status.HTTP_400_BAD_REQUEST,
                started_at=started,
            )
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        s1 = serializer.validated_data["sentence1"]
        s2 = serializer.validated_data["sentence2"]
        result = judge_single(s1, s2)

        _record_audit(
            request,
            client_ip=client_ip,
            status_code=status.HTTP_200_OK,
            started_at=started,
            payload_chars=len(s1) + len(s2),
        )
        return Response(
            {"label": result.label, "score": result.score},
            status=status.HTTP_200_OK,
        )


class JudgeBulkView(_BaseJudgeView):
    """POST /judge/bulk — Batch entailment judgment for 1 to 100 pairs (F2)."""

    def post(self, request):
        started = time.perf_counter()
        client_ip, limited = self._check_rate_limit(request, started)
        if limited is not None:
            return limited

        raw_data = request.data
        if isinstance(raw_data, list):
            if not raw_data or len(raw_data) > 100:
                _record_audit(
                    request,
                    client_ip=client_ip,
                    status_code=status.HTTP_400_BAD_REQUEST,
                    started_at=started,
                )
                return Response(
                    {"error": "Bulk request must contain between 1 and 100 sentence pairs."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            item_serializer = JudgePairSerializer(data=raw_data, many=True)
            if not item_serializer.is_valid():
                _record_audit(
                    request,
                    client_ip=client_ip,
                    status_code=status.HTTP_400_BAD_REQUEST,
                    started_at=started,
                )
                return Response(item_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            validated_pairs = item_serializer.validated_data
        else:
            wrapper_serializer = BulkJudgeWrapperSerializer(data=raw_data)
            if not wrapper_serializer.is_valid():
                _record_audit(
                    request,
                    client_ip=client_ip,
                    status_code=status.HTTP_400_BAD_REQUEST,
                    started_at=started,
                )
                return Response(wrapper_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            validated_pairs = wrapper_serializer.validated_data["pairs"]

        pairs = [(item["sentence1"], item["sentence2"]) for item in validated_pairs]
        outputs = judge_bulk(pairs)
        total_chars = sum(len(s1) + len(s2) for s1, s2 in pairs)

        _record_audit(
            request,
            client_ip=client_ip,
            status_code=status.HTTP_200_OK,
            started_at=started,
            payload_chars=total_chars,
        )
        return Response(
            [{"label": out.label, "score": out.score} for out in outputs],
            status=status.HTTP_200_OK,
        )


class JudgeStatsView(_BaseJudgeView):
    """GET /judge/stats — Aggregate judgment metrics, latency, and accuracy summary (F3)."""

    def get(self, request):
        started = time.perf_counter()
        client_ip, limited = self._check_rate_limit(request, started)
        if limited is not None:
            return limited

        stats = compute_stats()
        _record_audit(
            request,
            client_ip=client_ip,
            status_code=status.HTTP_200_OK,
            started_at=started,
        )
        return Response(stats, status=status.HTTP_200_OK)


class JudgeEvaluateView(_BaseJudgeView):
    """POST /judge/evaluate — Evaluate against ground-truth pairs and return accuracy/F1 (F3)."""

    def post(self, request):
        started = time.perf_counter()
        client_ip, limited = self._check_rate_limit(request, started)
        if limited is not None:
            return limited

        serializer = EvaluateRequestSerializer(data=request.data)
        if not serializer.is_valid():
            _record_audit(
                request,
                client_ip=client_ip,
                status_code=status.HTTP_400_BAD_REQUEST,
                started_at=started,
            )
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        labeled = [
            (item["sentence1"], item["sentence2"], item["expected_label"])
            for item in serializer.validated_data["pairs"]
        ]
        report = evaluate_ground_truth(labeled)
        _record_audit(
            request,
            client_ip=client_ip,
            status_code=status.HTTP_200_OK,
            started_at=started,
            payload_chars=sum(len(s1) + len(s2) for s1, s2, _ in labeled),
        )
        return Response(report, status=status.HTTP_200_OK)


@api_view(["GET"])
def health_check(request):
    return Response({"status": "ok"}, status=status.HTTP_200_OK)


@api_view(["POST"])
def initialize_data(request):
    try:
        file_name = request.data.get("file", "MOCK_DATA.json")
        call_command("init_data", file=file_name)
        return Response(
            {"message": f"Data initialized successfully from {file_name}"},
            status=status.HTTP_200_OK,
        )
    except Exception as exc:
        return Response({"error": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class UserViewSet(viewsets.ViewSet):
    """ViewSet preserved from Mantra's assignment template."""

    @action(detail=False, methods=["get"])
    def me(self, request):
        if request.user.is_authenticated:
            return Response({"username": request.user.username}, status=status.HTTP_200_OK)
        return Response({"error": "User not authenticated"}, status=status.HTTP_401_UNAUTHORIZED)
