from django.apps import AppConfig
from django.db.backends.signals import connection_created


def _configure_sqlite_pragmas(sender, connection, **kwargs):
    """Enable WAL mode and busy timeout on SQLite for safe concurrent writes."""
    if connection.vendor == "sqlite":
        with connection.cursor() as cursor:
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA synchronous=NORMAL;")
            cursor.execute("PRAGMA busy_timeout=30000;")


class AssignmentConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "assignment"

    def ready(self):
        connection_created.connect(_configure_sqlite_pragmas)
