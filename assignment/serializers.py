from rest_framework import serializers

from assignment.models import JudgmentRecord
from assignment.services.normalizer import is_same_language_pair


class JudgePairSerializer(serializers.Serializer):
    sentence1 = serializers.CharField(max_length=2000, allow_blank=False, trim_whitespace=True)
    sentence2 = serializers.CharField(max_length=2000, allow_blank=False, trim_whitespace=True)

    def validate(self, attrs):
        if not is_same_language_pair(attrs["sentence1"], attrs["sentence2"]):
            raise serializers.ValidationError(
                "sentence1 and sentence2 must be written in the same language."
            )
        return attrs


class BulkJudgeWrapperSerializer(serializers.Serializer):
    pairs = JudgePairSerializer(many=True, min_length=1, max_length=100)


class LabeledJudgePairSerializer(JudgePairSerializer):
    expected_label = serializers.ChoiceField(
        choices=[JudgmentRecord.LABEL_ENTAIL, JudgmentRecord.LABEL_NO_ENTAIL]
    )


class EvaluateRequestSerializer(serializers.Serializer):
    pairs = LabeledJudgePairSerializer(many=True, min_length=1, max_length=100)
