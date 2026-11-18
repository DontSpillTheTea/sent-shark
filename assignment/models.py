from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    """
    Custom User model preserved from the Mantra assignment template.
    """

    pass


class JudgmentRecord(models.Model):
    """
    Deduplicated entailment judgment record keyed by deterministic SHA-256 pair_hash.
    Ensures zero duplicate rows even under high concurrent writes (30+ parallel threads).
    """

    LABEL_ENTAIL = "ENTAIL"
    LABEL_NO_ENTAIL = "NO_ENTAIL"
    LABEL_CHOICES = [
        (LABEL_ENTAIL, "Entail"),
        (LABEL_NO_ENTAIL, "No Entail"),
    ]

    TIER_RULE = "tier1_rule"
    TIER_NLI = "tier2_nli"
    TIER_CHOICES = [
        (TIER_RULE, "Tier 1 Deterministic Rule"),
        (TIER_NLI, "Tier 2 NLI Cross-Encoder"),
    ]

    pair_hash = models.CharField(max_length=64, unique=True, db_index=True)
    sentence1 = models.TextField()
    sentence2 = models.TextField()
    label = models.CharField(max_length=16, choices=LABEL_CHOICES, db_index=True)
    score = models.FloatField()
    tier_used = models.CharField(max_length=32, choices=TIER_CHOICES, default=TIER_NLI)
    hit_count = models.PositiveIntegerField(default=1)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["label", "created_at"], name="idx_label_created"),
            models.Index(fields=["tier_used"], name="idx_tier_used"),
        ]

    def __str__(self) -> str:
        return f"{self.label} ({self.score:.2f}) [{self.pair_hash[:8]}]"


class RequestAuditLog(models.Model):
    """
    Security & observability audit trail for every incoming judge API request.
    """

    client_ip = models.CharField(max_length=64, db_index=True)
    method = models.CharField(max_length=10)
    path = models.CharField(max_length=255, db_index=True)
    status_code = models.IntegerField(db_index=True)
    payload_chars = models.IntegerField(default=0)
    latency_ms = models.FloatField(default=0.0)
    rate_limited = models.BooleanField(default=False, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["client_ip", "created_at"], name="idx_audit_ip_created"),
        ]

    def __str__(self) -> str:
        return f"{self.method} {self.path} -> {self.status_code} ({self.client_ip})"
