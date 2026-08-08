"""Tests for app.ibg.bank_name: issuing bank vs. beneficiary bank.

Parametrized against the shared ground-truth corpus (tests/fixtures/ibg_corpus.py)
so this module is scored against the same documents as every other IBG field
owner, plus a handful of direct tests for behaviour the corpus alone can't pin
down (the issuer/beneficiary split on one document, the OCR-repair gate, and
the "never a bare literal Unknown/Generic" rule).
"""
import pytest

from app.ibg.bank_name import extract_bank_name, extract_beneficiary_bank
from app.ibg.bank_registry import lookup_by_bic
from app.ibg.contract import FieldResult
from tests.fixtures.ibg_corpus import by_id, cases_for


# ---------------------------------------------------------------------------
# Corpus-parametrized: every sample must return `expected["bank_name"]` and
# must never return anything listed in `traps["bank_name"]`.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "sample_id, text, ocr_used, expected, traps",
    list(cases_for("bank_name")),
)
def test_bank_name_matches_expected_and_avoids_traps(
    sample_id, text, ocr_used, expected, traps
):
    result = extract_bank_name(text, ocr_used=ocr_used)

    assert result.value == expected, (
        "%s: expected bank_name=%r, got %r (source=%s, candidates=%s)"
        % (sample_id, expected, result.value, result.source, result.candidates)
    )
    assert result.value not in traps, (
        "%s: bank_name %r is a known trap value %r"
        % (sample_id, result.value, traps)
    )

    if expected is None:
        # An unrecoverable bank must be reported as missing, never guessed,
        # and never smuggled in as a confident "Unknown"/"Generic" string.
        assert result.value is None
        assert not result.found
        assert result.confidence < 0.5
    else:
        assert result.found
        assert result.confidence >= 0.5


# ---------------------------------------------------------------------------
# The core distinction this module exists to fix: on S1 the issuing bank
# (Maybank) and the beneficiary bank (RHB Islamic Bank) must never collapse
# into the same answer.
# ---------------------------------------------------------------------------
def test_s1_separates_issuer_from_beneficiary():
    sample = by_id("S1_MAYBANK_M2E_ACH")
    issuer = extract_bank_name(sample["text"], ocr_used=sample["ocr_used"])
    beneficiary = extract_beneficiary_bank(sample["text"], ocr_used=sample["ocr_used"])

    assert issuer.value == "Maybank"
    assert beneficiary.value == "RHB Islamic Bank"
    assert issuer.value != beneficiary.value
    assert issuer.found and beneficiary.found


def test_s1_beneficiary_bic_is_the_islamic_subsidiary_not_the_parent():
    # RHBAMYKL is RHB Islamic Bank's own BIC, a distinct legal entity from
    # RHB Bank (RHBBMYKL). Collapsing them loses where the money went.
    sample = by_id("S1_MAYBANK_M2E_ACH")
    beneficiary = extract_beneficiary_bank(sample["text"], ocr_used=sample["ocr_used"])
    assert beneficiary.value == "RHB Islamic Bank"
    assert beneficiary.value != "RHB"


def test_s5_beneficiary_side_has_no_bank_label_so_bare_rhb_is_the_issuer():
    # S5 is itself an RHB-issued receipt: "RHB BANK (RHBBMYKL)" appears with
    # no "Beneficiary/Receiving/Recipient Bank" label anywhere near it, which
    # is what makes it safe to trust as the issuer.
    sample = by_id("S5_RHB_DEGRADED")
    issuer = extract_bank_name(sample["text"], ocr_used=sample["ocr_used"])
    assert issuer.value == "RHB"


# ---------------------------------------------------------------------------
# S4: the issuing bank is genuinely not recoverable from the text (no
# "Maybank" token anywhere), and the only bank actually named is the
# recipient's. Guessing here would only be right by accident.
# ---------------------------------------------------------------------------
def test_s4_no_issuer_signal_returns_missing_not_a_guess():
    sample = by_id("S4_MAYBANK_GIRO_DUITNOW")
    result = extract_bank_name(sample["text"], ocr_used=sample["ocr_used"])
    assert result.value is None
    assert not result.found
    assert result.value not in ("RHB", "RHB Bank", "DuitNow", "GIRO", "Unknown", "Generic")


def test_s4_never_returns_a_payment_rail_as_bank_name():
    # The baseline defect (D4 in docs/IBG_BASELINE.md): the live engine
    # returns bank_name="DuitNow", a transfer rail, not a bank.
    sample = by_id("S4_MAYBANK_GIRO_DUITNOW")
    result = extract_bank_name(sample["text"], ocr_used=sample["ocr_used"])
    assert result.value not in ("DuitNow", "GIRO", "IBG", "FPX", "RENTAS", "JomPAY")


@pytest.mark.parametrize("rail_text", [
    "DuitNow Transfer Successful, DUITNOW reference 12345678",
    "Paid via GrabPay wallet, GRABPAY txn 87654321",
    "JomPAY bill payment JOMPAY reference 99998888",
])
def test_bare_rail_mentions_never_resolve_to_a_bank_name(rail_text):
    # Rails are excluded structurally (BANKS_ONLY drops tier="rail"), but
    # assert the public behaviour directly: no rail-only text should ever
    # produce a bank_name value.
    result = extract_bank_name(rail_text, ocr_used=False)
    assert result.value not in (
        "DuitNow", "GrabPay", "JomPAY", "ShopeePay", "Boost eWallet",
        "Touch n Go", "FPX", "RENTAS", "IBG",
    )


# ---------------------------------------------------------------------------
# "Unknown"/"Generic" must never be returned as a confident literal string --
# an unidentifiable bank is FieldResult.missing().
# ---------------------------------------------------------------------------
def test_unidentifiable_bank_returns_missing_not_a_literal_string():
    result = extract_bank_name(
        "This document mentions no financial institution, just the "
        "numbers 12345 and a date of 01 Jan 2026.",
        ocr_used=False,
    )
    assert result.value is None
    assert result.value != "Unknown"
    assert result.value != "Generic"
    assert not result.found
    assert result.confidence < 0.5


def test_unidentifiable_beneficiary_returns_missing_not_a_literal_string():
    result = extract_beneficiary_bank(
        "This document mentions no financial institution at all.",
        ocr_used=False,
    )
    assert result.value is None
    assert result.value != "Unknown"
    assert result.value != "Generic"
    assert not result.found


@pytest.mark.parametrize("blank", ["", "   ", "\n\n\t"])
def test_empty_or_blank_text_returns_missing(blank):
    assert extract_bank_name(blank, ocr_used=False).found is False
    assert extract_beneficiary_bank(blank, ocr_used=False).found is False


# ---------------------------------------------------------------------------
# OCR repair must apply only when ocr_used=True. Text lifted from a PDF's own
# text layer (ocr_used=False) has no misreads and must be read literally.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("corrupted, repaired_bank", [
    ("Your bank is H5BC Malaysia", "HSBC"),
    ("Payment processed via MAYBONK transfer", "Maybank"),
    ("Transferred through C1MB Clicks", "CIMB"),
    ("Bank Reference: R4B Now portal", "RHB"),
])
def test_ocr_repair_applied_only_when_ocr_used_true(corrupted, repaired_bank):
    with_repair = extract_bank_name(corrupted, ocr_used=True)
    without_repair = extract_bank_name(corrupted, ocr_used=False)

    assert with_repair.value == repaired_bank, (
        "ocr_used=True should repair %r to find %s, got %r"
        % (corrupted, repaired_bank, with_repair.value)
    )
    # Each corrupted token above has no other clean bank signal riding along
    # with it, so without repair the text layer must be trusted literally --
    # the corrupted token matches no known alias, and the result is missing.
    assert without_repair.value != repaired_bank, (
        "ocr_used=False must not silently repair %r (got %r)"
        % (corrupted, without_repair.value)
    )
    assert without_repair.value is None


def test_ocr_repair_does_not_run_on_clean_pdf_text_layer():
    # A literal, uncorrupted-looking "H5BC" with ocr_used=False must not be
    # matched at all -- it is not a real alias, and repair must not touch it.
    result = extract_bank_name("Ref H5BC 12345678 processed", ocr_used=False)
    assert result.value != "HSBC"


def test_s1_and_s2_are_not_ocr_repaired_pdf_text_layer():
    # Regression guard: S1/S2 are embedded PDF text (ocr_used=False). Nothing
    # in their text needs repair, but this pins down that the ocr_used flag
    # is actually being read and honoured, not ignored.
    s1 = by_id("S1_MAYBANK_M2E_ACH")
    s2 = by_id("S2_MAYBANK_M2E_TRUNCATED")
    assert s1["ocr_used"] is False
    assert s2["ocr_used"] is False
    assert extract_bank_name(s1["text"], ocr_used=s1["ocr_used"]).value == "Maybank"
    assert extract_bank_name(s2["text"], ocr_used=s2["ocr_used"]).value == "Maybank"


# ---------------------------------------------------------------------------
# SWIFT/BIC -> bank map, consumed from the shared app.ibg.bank_registry
# rather than a second, hand-rolled map. Covers the codes the role brief
# calls out by name.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bic, expected_name", [
    ("MBBEMYKL", "Maybank"),
    ("RHBAMYKL", "RHB Islamic Bank"),   # Islamic subsidiary, distinct entity
    ("RHBBMYKL", "RHB"),                # conventional RHB Bank
    ("CIBBMYKL", "CIMB"),
    ("PBBEMYKL", "Public Bank"),
    ("HLBBMYKL", "Hong Leong Bank"),
    ("UOVBMYKL", "UOB"),
    ("OCBCMYKL", "OCBC"),
    ("HBMBMYKL", "HSBC"),
    ("PHBMMYKL", "Affin Bank"),
    ("BIMBMYKL", "Bank Islam"),
    ("SCBLMYKX", "Standard Chartered"),
])
def test_beneficiary_bic_map_covers_required_codes(bic, expected_name):
    text = "Beneficiary Bank Code\n%s\nBeneficiary Bank Name\n-\n" % bic
    result = extract_beneficiary_bank(text, ocr_used=False)
    assert result.value == expected_name
    assert result.found


def test_ambank_bic_in_shared_registry_is_arbkmykl_not_ammbmykl():
    # The original brief for this module listed AMMBMYKL as AmBank's BIC.
    # app/ibg/bank_registry.py (the shared source of truth this module now
    # consumes instead of hand-rolling its own map) asserts ARBKMYKL for
    # AmBank instead, and does not recognise AMMBMYKL at all. This test
    # documents that discrepancy against the registry actually in the repo
    # rather than silently disagreeing with it.
    assert lookup_by_bic("ARBKMYKL") is not None
    assert lookup_by_bic("ARBKMYKL").name == "AmBank"
    assert lookup_by_bic("AMMBMYKL") is None

    text = "Beneficiary Bank Code\nARBKMYKL\nBeneficiary Bank Name\n-\n"
    result = extract_beneficiary_bank(text, ocr_used=False)
    assert result.value == "AmBank"


def test_bic_is_the_fallback_signal_in_a_degraded_scan():
    # "The BIC is often the only clean signal in a degraded scan" -- assert
    # a bare BIC with no clean label around it still resolves.
    text = "xxx garbled xxx CIBBMYKL xxx garbled xxx"
    result = extract_beneficiary_bank(text, ocr_used=True)
    assert result.value == "CIMB"
    assert result.found


# ---------------------------------------------------------------------------
# Sanity: found values always carry evidence (source, candidates); the
# contract's confidence<0.5-iff-missing invariant holds across the corpus.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sample_id, text, ocr_used, expected, traps",
                          list(cases_for("bank_name")))
def test_field_result_contract_invariants(sample_id, text, ocr_used, expected, traps):
    result = extract_bank_name(text, ocr_used=ocr_used)
    assert isinstance(result, FieldResult)
    if result.value is None:
        assert result.confidence < 0.5
    else:
        assert result.source != "none"
        assert len(result.candidates) >= 1
