from rest_framework import serializers

from assignment.services.normalizer import has_substantive_content, is_same_language_pair


class StrictStringField(serializers.CharField):
    """CharField rejecting non-string JSON primitives (int, float, bool, list, dict)."""

    def to_internal_value(self, data):
        if not isinstance(data, str):
            self.fail("invalid")
        return super().to_internal_value(data)


class JudgePairSerializer(serializers.Serializer):
    sentence1 = StrictStringField(max_length=2000, allow_blank=False, trim_whitespace=True)
    sentence2 = StrictStringField(max_length=2000, allow_blank=False, trim_whitespace=True)

    def validate(self, attrs):
        errors = {}
        for field_name in ("sentence1", "sentence2"):
            if not has_substantive_content(attrs[field_name]):
                errors[field_name] = "Sentence must contain at least one letter, number, or symbol."
        if errors:
            raise serializers.ValidationError(errors)
        if not is_same_language_pair(attrs["sentence1"], attrs["sentence2"]):
            raise serializers.ValidationError(
                "sentence1 and sentence2 must be written in the same language."
            )
        return attrs


class BulkJudgeWrapperSerializer(serializers.Serializer):
    pairs = JudgePairSerializer(many=True, min_length=1, max_length=100)
