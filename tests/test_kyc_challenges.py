from app.services.challenge_generator import generate_challenge_sentence
from app.services.kyc_enrollment_service import KYCEnrollmentService


def test_dynamic_challenge_sentence_is_short_and_contains_tail_number():
    sentence = generate_challenge_sentence()

    assert sentence.endswith(".")
    assert "number" in sentence.lower()
    assert 8 <= len(sentence.rstrip(".").split()) <= 14
    assert any(token.rstrip(".").isdigit() for token in sentence.split())


def test_dynamic_challenge_generator_produces_fresh_sentences():
    sentences = {generate_challenge_sentence() for _ in range(30)}

    assert len(sentences) > 1


def test_kyc_transcript_match_scores_exact_phrase_as_pass():
    expected = "My UCO account confirms a safe login number 721."
    spoken = "my uco account confirms a safe login number 721"

    assert KYCEnrollmentService._match_score(spoken, expected) >= 0.75


def test_kyc_transcript_match_rejects_replayed_wrong_sentence():
    expected = "The secure branch verified the voice check number 918."
    replayed = "my banking assistant confirms a safe login number 217"

    assert KYCEnrollmentService._match_score(replayed, expected) < 0.75
