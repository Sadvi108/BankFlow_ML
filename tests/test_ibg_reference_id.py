"""Tests for app/ibg/reference_id.py.

Two guarantees, and they are different:

1. `extract_references()` captures EVERY reference on the receipt, each tagged
   with the label that named it and the role that label implies.
2. `extract_reference_id()` returns only the bank-issued primary, because that
   is the value the ledger reconciles against. A payer-supplied reference is
   never promoted, no matter how official it looks -- that promotion is the bug
   this module exists to prevent.

Note the two interact: a value can be a legitimate reference AND a trap for the
scalar field. On the Maybank2u receipt `MXC12511726684` is the payer's own
"Recipient reference" -- correctly captured as `payer_supplied`, and correctly
refused as `reference_id`.
"""
import pytest

from app.ibg.contract import (
    ROLE_BANK_PRIMARY,
    ROLE_BANK_SECONDARY,
    ROLE_PAYER_SUPPLIED,
)
from app.ibg.reference_id import extract_reference_id, extract_references
from tests.fixtures.ibg_corpus import CORPUS, by_id, cases_for, reference_cases
from tests.fixtures.ibg_holdout import HOLDOUT

_SCALAR_CASES = list(cases_for("reference_id"))
_SCALAR_IDS = [c[0] for c in _SCALAR_CASES]
_REF_CASES = list(reference_cases())
_REF_IDS = [c[0] for c in _REF_CASES]


# ---------------------------------------------------------------------------
# The scalar primary
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "sample_id, text, ocr_used, expected, traps", _SCALAR_CASES, ids=_SCALAR_IDS,
)
def test_scalar_matches_expected(sample_id, text, ocr_used, expected, traps):
    result = extract_reference_id(text, ocr_used=ocr_used)
    assert result.value == expected, (
        "%s: expected %r, got %r (source=%s)"
        % (sample_id, expected, result.value, result.source)
    )


@pytest.mark.parametrize(
    "sample_id, text, ocr_used, expected, traps", _SCALAR_CASES, ids=_SCALAR_IDS,
)
def test_scalar_never_returns_a_trap(sample_id, text, ocr_used, expected, traps):
    result = extract_reference_id(text, ocr_used=ocr_used)
    assert result.value not in traps, (
        "%s: returned trap value %r" % (sample_id, result.value)
    )


@pytest.mark.parametrize(
    "sample_id, text, ocr_used, expected, traps", _SCALAR_CASES, ids=_SCALAR_IDS,
)
def test_missing_scalar_has_low_confidence(sample_id, text, ocr_used,
                                            expected, traps):
    """A None must be reviewable, never dressed up as a confident answer."""
    result = extract_reference_id(text, ocr_used=ocr_used)
    if result.value is None:
        assert result.confidence < 0.5, (
            "%s: value is None but confidence is %r"
            % (sample_id, result.confidence)
        )


def test_payer_reference_is_never_promoted():
    """The original defect: the payer's own reference outranking the bank's."""
    sample = by_id("S3_MAYBANK2U_OPEN_INTERBANK")
    result = extract_reference_id(sample["text"], ocr_used=sample["ocr_used"])
    assert result.value == "7901990048"          # Reference number (the bank's)
    assert result.value != "MXC12511726684"      # Recipient reference (payer's)


def test_scalar_is_none_when_only_payer_references_exist():
    text = (
        "Payment Confirmation\n"
        "Recipient Reference: INV-90210\n"
        "Other Payment Details: THANKS\n"
        "Amount: MYR 12.00\n"
    )
    result = extract_reference_id(text, ocr_used=False)
    assert result.value is None
    assert result.confidence < 0.5
    # ...but the payer references are still captured.
    roles = set(r.role for r in extract_references(text, ocr_used=False))
    assert roles == set([ROLE_PAYER_SUPPLIED])


# ---------------------------------------------------------------------------
# Multi-reference capture
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "sample_id, text, ocr_used, expected_refs", _REF_CASES, ids=_REF_IDS,
)
def test_every_expected_reference_is_captured(sample_id, text, ocr_used,
                                               expected_refs):
    found = set(
        (r.value, r.role)
        for r in extract_references(text, ocr_used=ocr_used)
    )
    missing = set((str(v), str(role)) for v, role in expected_refs) - found
    assert not missing, (
        "%s: did not capture %s (found %s)" % (sample_id, sorted(missing),
                                               sorted(found))
    )


@pytest.mark.parametrize(
    "sample_id, text, ocr_used, expected_refs", _REF_CASES, ids=_REF_IDS,
)
def test_at_most_one_bank_primary(sample_id, text, ocr_used, expected_refs):
    refs = extract_references(text, ocr_used=ocr_used)
    primaries = [r for r in refs if r.role == ROLE_BANK_PRIMARY]
    assert len(primaries) <= 1, (
        "%s: %d bank_primary references, expected at most 1: %s"
        % (sample_id, len(primaries), [r.value for r in primaries])
    )


@pytest.mark.parametrize(
    "sample_id, text, ocr_used, expected_refs", _REF_CASES, ids=_REF_IDS,
)
def test_every_reference_is_well_formed(sample_id, text, ocr_used,
                                         expected_refs):
    for ref in extract_references(text, ocr_used=ocr_used):
        assert ref.value and ref.value.strip() == ref.value
        assert ref.label, "%s: reference %r has no label" % (sample_id, ref.value)
        assert ref.role in (ROLE_BANK_PRIMARY, ROLE_BANK_SECONDARY,
                            ROLE_PAYER_SUPPLIED)
        assert 0.0 <= ref.confidence <= 1.0
        assert ref.source.startswith("label:")


def test_roles_are_distinguished_on_a_three_role_receipt():
    """Public Bank IBG prints all three kinds of reference at once."""
    sample = by_id("S9_PBB_IBG_APPROVAL")
    refs = extract_references(sample["text"], ocr_used=sample["ocr_used"])
    by_role = {}
    for ref in refs:
        by_role.setdefault(ref.role, set()).add(ref.value)

    assert "2608070370908566" in by_role.get(ROLE_BANK_PRIMARY, set())
    assert "559155" in by_role.get(ROLE_BANK_SECONDARY, set())
    assert "DEM - TAWU4057398" in by_role.get(ROLE_PAYER_SUPPLIED, set())


def test_scalar_equals_the_bank_primary_reference():
    """The scalar field and the roled list must never disagree."""
    for sample in CORPUS:
        refs = extract_references(sample["text"], ocr_used=sample["ocr_used"])
        scalar = extract_reference_id(sample["text"],
                                      ocr_used=sample["ocr_used"]).value
        primaries = [r.value for r in refs if r.role == ROLE_BANK_PRIMARY]
        if scalar is None:
            assert not primaries, (
                "%s: scalar is None but a bank_primary exists: %s"
                % (sample["id"], primaries)
            )
        else:
            assert scalar in primaries, (
                "%s: scalar %r is not among the bank_primary references %s"
                % (sample["id"], scalar, primaries)
            )


@pytest.mark.parametrize("sample", HOLDOUT, ids=[s["id"] for s in HOLDOUT])
def test_unseen_holdout_references_are_all_recovered(sample):
    """The bank vocabulary must generalize beyond the tuning corpus."""
    refs = extract_references(sample["text"], ocr_used=sample["ocr_used"])
    found = set((ref.value, ref.role) for ref in refs)
    expected = set(sample["expected"]["references"])
    assert not expected - found
    assert extract_reference_id(
        sample["text"], ocr_used=sample["ocr_used"]
    ).value == sample["expected"]["reference_id"]


# ---------------------------------------------------------------------------
# Rules, tested independently of the corpus
# ---------------------------------------------------------------------------

def test_a_label_never_spans_a_line_break():
    """"Transfer Advice" + "Reference Number" on separate lines are two things.

    Letting the label regex cross the newline glued them into one match and left
    "Number:" stranded inside the extracted value.
    """
    text = "Transfer Advice\nReference Number: ZZ11223344556\n"
    result = extract_reference_id(text, ocr_used=False)
    assert result.value == "ZZ11223344556"


def test_column_header_run_is_not_mistaken_for_a_value():
    """A table's header row must not become the reference."""
    text = (
        "Transaction Reference Number Transaction Status Transfer Type\n"
        "QQ99887766554 Approved Fund Transfer\n"
    )
    result = extract_reference_id(text, ocr_used=False)
    assert result.value == "QQ99887766554"


def test_multi_column_row_is_cut_at_the_date():
    text = "Transaction Reference Number\nZZ12345678901 - 01/02/2030\n"
    result = extract_reference_id(text, ocr_used=False)
    assert result.value == "ZZ12345678901"


def test_masked_account_value_is_not_adopted_as_a_reference():
    """A masked account number sits between the label and the real reference."""
    text = (
        "BENEFICIARY ACCOUNT NO. : 2*********0001\n"
        "REFERENCE NO. : 1234567890123456\n"
    )
    result = extract_reference_id(text, ocr_used=False)
    assert result.value == "1234567890123456"


def test_unmasked_account_number_is_never_returned_as_a_reference():
    """An explicit account label outweighs a nearby generic ref label."""
    text = (
        "Debit From Account Number: 81001234567890\n"
        "Reference No: 81001234567890\n"
        "Amount: MYR 21.00\n"
    )
    refs = extract_references(text, ocr_used=False)
    assert "81001234567890" not in [ref.value for ref in refs]
    assert extract_reference_id(text, ocr_used=False).value is None


def test_real_reference_survives_next_to_an_account_number():
    text = (
        "Beneficiary Account No: 21001234567890\n"
        "Transaction Reference No: TXN9988776655\n"
    )
    refs = extract_references(text, ocr_used=False)
    assert [ref.value for ref in refs] == ["TXN9988776655"]


def test_account_word_in_prose_does_not_suppress_next_reference():
    text = (
        "We have transferred the funds into your account.\n"
        "Reference No: 2609988776655\n"
    )
    assert extract_reference_id(text, ocr_used=False).value == "2609988776655"


@pytest.mark.parametrize("noise", ["1", "D D", "Today :", "Debit Amount"])
def test_short_or_form_label_noise_is_not_a_reference(noise):
    refs = extract_references(
        "Customer Ref: %s\nAmount: MYR 10.00\n" % noise,
        ocr_used=True,
    )
    assert refs == []


def test_sole_clearing_reference_is_promoted():
    """When the portal prints only a clearing reference, that IS the bank's."""
    text = (
        "Fund Transfer\n"
        "PayNet Reference No.\n"
        "20300101BANKMYKL010ABC12345678\n"
        "Your Reference No.\n"
        "MYREF-1\n"
    )
    refs = extract_references(text, ocr_used=False)
    primaries = [r for r in refs if r.role == ROLE_BANK_PRIMARY]
    assert len(primaries) == 1
    assert primaries[0].value == "20300101BANKMYKL010ABC12345678"


def test_hsbc_status_sentence_supplies_primary_reference():
    text = (
        "HSBCnet Priority Payment\n"
        "Your reference\n"
        "CUSTOMER-NOTE-7\n"
        "The status for payment 48291ID07ABC is: Received by Beneficiary Bank\n"
    )
    refs = extract_references(text, ocr_used=False)
    assert [(r.value, r.role) for r in refs] == [
        ("CUSTOMER-NOTE-7", ROLE_PAYER_SUPPLIED),
        ("48291ID07ABC", ROLE_BANK_PRIMARY),
    ]
    assert extract_reference_id(text, ocr_used=False).value == "48291ID07ABC"


def test_stronger_bank_label_wins_when_value_is_repeated():
    text = (
        "Customer reference\nLP ABC829632\n"
        "Bank reference\nLP ABC829632\n"
    )
    refs = extract_references(text, ocr_used=False)
    assert len(refs) == 1
    assert refs[0].value == "LP ABC829632"
    assert refs[0].role == ROLE_BANK_PRIMARY


def test_scrambled_payment_reference_can_precede_its_label():
    text = (
        "Payment Details\nFIN0708264455667\n"
        "Payee Account\nPayment Ref No.\n:\nPayee Bank\n"
    )
    assert extract_reference_id(text, ocr_used=False).value == "FIN0708264455667"


def test_official_receipt_number_is_captured_without_becoming_a_bank_id():
    text = (
        "Receipt No\n70000987654\nIssue Date\n5 Aug 2030\n"
        "OFFICIAL RECEIPT\n"
    )
    assert [r.value for r in extract_references(text, ocr_used=False)] == ["70000987654"]
    assert extract_reference_id(text, ocr_used=False).value is None


def test_clearing_reference_stays_secondary_when_a_primary_exists():
    text = (
        "Transaction Reference No. : CC1234567890\n"
        "DuitNow Reference No. : 20300101BANKMYKL010ABC12345678\n"
    )
    refs = dict((r.value, r.role) for r in extract_references(text, ocr_used=False))
    assert refs["CC1234567890"] == ROLE_BANK_PRIMARY
    assert refs["20300101BANKMYKL010ABC12345678"] == ROLE_BANK_SECONDARY


def test_placeholder_dash_is_not_a_reference():
    text = "Customer Ref No.\n-\nReference No.\nAB1234567890\n"
    refs = extract_references(text, ocr_used=False)
    assert "-" not in [r.value for r in refs]
    assert extract_reference_id(text, ocr_used=False).value == "AB1234567890"


def test_empty_and_whitespace_text():
    for text in ("", "   \n\t  \n"):
        assert extract_references(text) == []
        result = extract_reference_id(text)
        assert result.value is None and result.confidence < 0.5


def test_large_junk_input_terminates():
    junk = ("lorem ipsum 12345 " * 5000)
    result = extract_reference_id(junk, ocr_used=True)
    assert result.value is None or isinstance(result.value, str)


def test_no_corpus_literal_is_hardcoded():
    """The rules must generalize, not memorise this corpus."""
    import app.ibg.reference_id as module

    with open(module.__file__, "r") as handle:
        source = handle.read()

    literals = set()
    for sample in CORPUS:
        expected = sample["expected"].get("reference_id")
        if expected:
            literals.add(expected)
        for value, _role in sample["expected"].get("references", []):
            literals.add(value)
        for trap in sample.get("traps", {}).get("reference_id", []):
            literals.add(str(trap))

    leaked = sorted(lit for lit in literals if len(lit) >= 6 and lit in source)
    assert not leaked, "corpus literals hardcoded in the module: %s" % leaked


def test_straight2bank_remittance_table_extraction():
    """Straight2Bank payee advice with multiple header references and table row reference."""
    text = (
        "SCB Ref : MY00150Q0354490\n"
        "Customer Ref : CPDDC\n"
        "Date : 10/07/2026\n"
        "Straight2Bank\n"
        "PAYEE ADVICE\n"
        "To: D&D CONTROL (MALAYSIA) SDN BHD\n"
        "UTR Reference SB2596260710F399\n"
        "Remittance Advice\n"
        "Payment Details : S2505174689\n"
        "Reference Date Description Amount ( MYR )\n"
        "YML125117584 10/07/2026 10.00\n"
    )
    refs = extract_references(text, ocr_used=False)
    ref_map = {r.value: r.role for r in refs}
    assert ref_map.get("MY00150Q0354490") == ROLE_BANK_PRIMARY
    assert ref_map.get("SB2596260710F399") == ROLE_BANK_SECONDARY
    assert ref_map.get("CPDDC") == ROLE_PAYER_SUPPLIED
    assert ref_map.get("S2505174689") == ROLE_PAYER_SUPPLIED
    assert ref_map.get("YML125117584") == ROLE_PAYER_SUPPLIED


def test_straight2bank_split_column_layout():
    """When the PDF text extractor splits table columns onto separate lines,
    YML125117584 must still be captured (the real-world layout from embedded text)."""
    text = (
        "SCB Ref : MY00150Q0354490\n"
        "Customer Ref : CPDDC\n"
        "Date : 10/07/2026\n"
        "Straight2Bank\n"
        "PAYEE ADVICE\n"
        "To:\n"
        "D&D CONTROL (MALAYSIA) SDN BHD\n"
        "MALAYSIA\n"
        "Invoice Total\n"
        "10.00\n"
        "UTR Reference\n"
        "SB2596260710F399\n"
        "Remittance Advice\n"
        "Payment Details :\n"
        "S2505174689\n"
        "Reference\n"
        "Date\n"
        "Description\n"
        "Amount ( MYR )\n"
        "YML125117584\n"
        "10/07/2026\n"
        "10.00\n"
        "Page  1\n"
    )
    refs = extract_references(text, ocr_used=False)
    ref_map = {r.value: r.role for r in refs}
    assert ref_map.get("MY00150Q0354490") == ROLE_BANK_PRIMARY
    assert ref_map.get("SB2596260710F399") == ROLE_BANK_SECONDARY
    assert ref_map.get("CPDDC") == ROLE_PAYER_SUPPLIED
    assert ref_map.get("S2505174689") == ROLE_PAYER_SUPPLIED
    assert ref_map.get("YML125117584") == ROLE_PAYER_SUPPLIED, (
        "YML125117584 must be captured even when table columns are on separate lines"
    )
