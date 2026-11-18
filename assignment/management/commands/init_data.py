import json
import os

from django.core.management.base import BaseCommand

from assignment.models import User


class Command(BaseCommand):
    help = "Initialize mock users and optional sample data."

    def add_arguments(self, parser):
        parser.add_argument(
            "--file", default="MOCK_DATA.json", help="JSON file name to load data from"
        )

    def handle(self, *args, **options):
        User.objects.all().delete()
        self.stdout.write(self.style.SUCCESS("All existing user data has been deleted"))

        try:
            file_name = options.get("file", "MOCK_DATA.json")
            json_file_path = os.path.join(os.path.dirname(__file__), file_name)

            with open(json_file_path, encoding="utf-8") as json_file:
                _ = json.load(json_file)

            User.objects.create_superuser(
                "testuser", email="testuser@example.com", password="testpassword"
            )
            for i in range(1, 6):
                User.objects.create_user(
                    f"testuser{i}",
                    email=f"testuser{i}@example.com",
                    password="testpassword",
                )

            self.stdout.write(self.style.SUCCESS(f"Mock data loaded successfully from {file_name}"))
        except Exception as exc:
            self.stdout.write(self.style.ERROR(f"Error loading data: {exc}"))
