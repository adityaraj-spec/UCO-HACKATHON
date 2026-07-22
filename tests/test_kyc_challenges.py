from app.services.challenge_generator import generate_challenge_sentence
from app.services.kyc_enrollment_service import KYCEnrollmentService


def test_dynamic_challenge_sentence_is_short_and_contains_tail_number():
    sentence = generate_challenge_sentence()

    assert sentence.endswith(".")
    assert "number" in sentence.lower()
    assert 8 <= len(sentence.rstrip(".").split()) <= 16
    assert any(token.rstrip(".").isdigit() for token in sentence.split())


def test_dynamic_challenge_generator_produces_fresh_sentences():
    sentences = {generate_challenge_sentence() for _ in range(30)}

    assert len(sentences) > 1


def test_kyc_transcript_match_scores_exact_phrase_as_pass():
    expected = "I authorize a fund transfer of 5000 rupees for transaction number 721."
    spoken = "i authorize a fund transfer of 5000 rupees for transaction number 721"

    assert KYCEnrollmentService._match_score(spoken, expected) >= 0.75


def test_kyc_transcript_match_rejects_replayed_wrong_sentence():
    expected = "Confirm payment of 2500 rupees from my UCO account number 918."
    replayed = "authorize money transfer of 1000 rupees to beneficiary number 217"

    assert KYCEnrollmentService._match_score(replayed, expected) < 0.75

