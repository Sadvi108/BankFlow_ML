"""Tests for app/ibg/amount.py -- the IBG transaction-amount extractor.

Two kinds of coverage:
  - Corpus-parametrized: every real receipt sample in
    tests/fixtures/ibg_corpus.py must return its expected amount and must
    never return (or even surface as a candidate) any of its known trap
    values.
  - Synthetic unit tests for behavior the corpus doesn't happen to
    exercise directly: fee-vs-amount ranking, account-number-fragment
    rejection, the output-format contract, and a genuine "0.00" amount.
    These use invented figures rather than corpus literals, so they keep
    testing the *rule*, not the sample.

Run:  .venv/bin/python -m pytest tests/test_ibg_amount.py -q
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app.ibg.amount import _looks_like_account_fragment, extract_amount
from app.ibg.contract import FieldResult
from tests.fixtures.ibg_corpus import cases_for

_AMOUNT_CASES = list(cases_for("amount"))
_AMOUNT_IDS = [case[0] for case in _AMOUNT_CASES]

_OUTPUT_FORMAT_RE = re.compile(r'^\d+\.\d{2}$')


# ---------------------------------------------------------------------------
# Corpus-parametrized: ground truth from real receipts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "sample_id, text, ocr_used, expected, traps", _AMOUNT_CASES, ids=_AMOUNT_IDS,
)
def test_corpus_returns_expected_amount(sample_id, text, ocr_used, expected, traps):
    result = extract_amount(text, ocr_used=ocr_used)

    if expected is None:
        assert result.value is None, (
            f"{sample_id}: expected no amount, got {result.value!r}"
        )
    else:
        assert result.value == expected, (
            f"{sample_id}: expected {expected!r}, got {result.value!r} "
            f"(candidates={result.candidates!r})"
        )


@pytest.mark.parametrize(
    "sample_id, text, ocr_used, expected, traps", _AMOUNT_CASES, ids=_AMOUNT_IDS,
)
def test_corpus_never_returns_a_trap_value(sample_id, text, ocr_used, expected, traps):
    result = extract_amount(text, ocr_used=ocr_used)
    candidate_values = [c[0] for c in result.candidates]

    for trap in traps:
        assert result.value != trap, (
            f"{sample_id}: returned trap value {trap!r} as the answer"
        )

    # NOTE: this test used to also assert that no trap appeared anywhere in
    # `candidates`. That invariant belonged to the single-value design. Now that
    # the module also returns `fee` and `total_debit`, several amount-traps are
    # legitimately captured values: on the CIMB slip `106.10` is the trap for
    # `amount` and simultaneously the correct `total_debit`, and `0.10` is the
    # correct `fee`. Both must be considered, so both appear as candidates.
    # The guarantee that matters -- a trap is never the returned answer -- is
    # asserted above and is unchanged.
    assert candidate_values or result.value is None


@pytest.mark.parametrize(
    "sample_id, text, ocr_used, expected, traps", _AMOUNT_CASES, ids=_AMOUNT_IDS,
)
def test_corpus_result_shape_matches_contract(sample_id, text, ocr_used, expected, traps):
    result = extract_amount(text, ocr_used=ocr_used)
    assert isinstance(result, FieldResult)
    if result.value is None:
        assert result.confidence < 0.5, sample_id
    else:
        assert 0.0 < result.confidence <= 1.0, sample_id
        assert result.found is True, sample_id


# ---------------------------------------------------------------------------
# Output format -- plain decimal string, exactly two places, no separators,
# no currency, never a float.
# ---------------------------------------------------------------------------

def test_corpus_amounts_are_two_decimals_no_comma_no_currency():
    checked_any = False
    for sample_id, text, ocr_used, _expected, _traps in _AMOUNT_CASES:
        result = extract_amount(text, ocr_used=ocr_used)
        if result.value is None:
            continue
        checked_any = True
        assert isinstance(result.value, str), sample_id
        assert type(result.value) is str, sample_id  # never a float
        assert _OUTPUT_FORMAT_RE.match(result.value), (
            f"{sample_id}: {result.value!r} is not formatted as DIGITS.DD"
        )
        assert "," not in result.value, sample_id
        assert "RM" not in result.value.upper(), sample_id
        assert "MYR" not in result.value.upper(), sample_id
    assert checked_any, "no corpus sample produced a value to check format on"


def test_thousands_separator_is_stripped_from_output():
    text = "Transfer Amount\nRM 3,200.15\n"
    result = extract_amount(text, ocr_used=False)
    assert result.value == "3200.15"
    assert _OUTPUT_FORMAT_RE.match(result.value)


def test_output_value_type_is_str_not_float():
    text = "Transaction Amount\n88.50\n"
    result = extract_amount(text, ocr_used=False)
    assert type(result.value) is str
    assert result.value == "88.50"


# ---------------------------------------------------------------------------
# Fee vs. transaction-amount ranking
# ---------------------------------------------------------------------------

def test_fee_listed_before_amount_does_not_win():
    text = "Service Charge\n5.00\nTransaction Amount\n250.75\n"
    result = extract_amount(text, ocr_used=False)
    assert result.value == "250.75"


def test_fee_listed_after_amount_does_not_leak_into_the_answer():
    text = "Transaction Amount\n250.75\nService Charge\n5.00\n"
    result = extract_amount(text, ocr_used=False)
    assert result.value == "250.75"


def test_zero_fee_does_not_outrank_a_nonzero_real_amount():
    """A 0.00 fee sitting right next to the real amount must not look like
    the 'safe' pick -- label ranking decides, not the value itself."""
    text = "Service Charge\n0.00\nTransaction Amount\n250.75\n"
    result = extract_amount(text, ocr_used=False)
    assert result.value == "250.75"


def test_fee_only_document_returns_missing_never_the_fee_value():
    text = "Service Charge\n5.00\nGST\n0.30\n"
    result = extract_amount(text, ocr_used=False)
    assert result.value is None
    assert result.confidence < 0.5
    # The fees may surface for debugging, but never as the winning value.
    assert "5.00" not in [result.value]
    assert "0.30" not in [result.value]


def test_tax_amount_label_is_ranked_as_a_fee_not_the_transaction_amount():
    """'Tax Amount' contains the literal word 'Amount' -- it must still be
    ranked as a fee, not picked up by the bare-'Amount' rule."""
    text = "Amount\n500.00\nTax Amount\n30.00\n"
    result = extract_amount(text, ocr_used=False)
    assert result.value == "500.00"


def test_gst_amount_label_is_ranked_as_a_fee_not_the_transaction_amount():
    text = "Total Amount\n1200.00\nGST Amount\n72.00\n"
    result = extract_amount(text, ocr_used=False)
    assert result.value == "1200.00"


def test_fee_value_with_currency_prefix_still_loses_to_labeled_amount():
    text = "Transfer Amount\n3200.15\nBank Charges RM 12.00\n"
    result = extract_amount(text, ocr_used=False)
    assert result.value == "3200.15"
    assert result.candidates[0][0] == "3200.15"


# ---------------------------------------------------------------------------
# Account-number / not-money rejection
# ---------------------------------------------------------------------------

def test_looks_like_account_fragment_rejects_space_joined_digit_run():
    """Mirrors an OCR failure mode: a space inside a long account number
    misread as a decimal point, e.g. "...0099 205.00" where "205" is really
    the tail of "7788 3341 0099 205", not fifty sen over RM 205."""
    text = "Account No. 7788 3341 0099 205.00 more text"
    start = text.index("205.00")
    end = start + len("205.00")
    assert _looks_like_account_fragment(text, start, end) is True


def test_looks_like_account_fragment_accepts_a_clean_standalone_amount():
    text = "Transaction Amount\n88.50\nValue Date"
    start = text.index("88.50")
    end = start + len("88.50")
    assert _looks_like_account_fragment(text, start, end) is False


def test_account_number_decoy_between_label_and_real_amount_is_skipped():
    """Same shape as the corpus S4 trap (a space-separated account number
    sitting near the real amount), but forces the decoy to actually be
    money-shaped so the rejection rule is load-bearing here, rather than
    moot because the decoy never had a decimal point to begin with."""
    text = (
        "Transaction Amount\n"
        "Beneficiary Account No.\n"
        "7788 3341 0099 205.00\n"
        "88.50\n"
    )
    result = extract_amount(text, ocr_used=False)
    assert result.value == "88.50"
    assert "205.00" not in [c[0] for c in result.candidates]


def test_bare_account_number_and_postal_code_are_never_returned():
    text = "Beneficiary Account No.\n912233445566\nZip/Postal Code\n50450\n"
    result = extract_amount(text, ocr_used=False)
    assert result.value is None


# ---------------------------------------------------------------------------
# Zero amount
# ---------------------------------------------------------------------------

def test_genuine_zero_amount_is_returned_not_treated_as_missing():
    """0.00 is a legitimate transaction amount (e.g. a fully-waived
    transfer) and must survive -- a naive `if value:` truthiness check on
    a numeric zero would have swallowed it."""
    text = "Total Amount\n0.00\n"
    result = extract_amount(text, ocr_used=False)
    assert result.value == "0.00"
    assert result.found is True
    assert result.confidence > 0.0


def test_zero_amount_survives_alongside_a_nonzero_fee():
    text = "Transaction Amount\n0.00\nService Charge\n1.50\n"
    result = extract_amount(text, ocr_used=False)
    assert result.value == "0.00"


# ---------------------------------------------------------------------------
# Layout traps, isolated from the corpus text (invented figures)
# ---------------------------------------------------------------------------

def test_intervening_line_between_label_and_value():
    """Trap 1: the label and value are separated by another line."""
    text = "Transaction Amount\nIn Transaction Currency\n42.42\n"
    result = extract_amount(text, ocr_used=False)
    assert result.value == "42.42"


def test_column_scrambled_labels_then_values():
    """Trap 2: labels arrive as one block, values as another; the value
    sits several lines below its own label, right after an unrelated one."""
    text = (
        "Transaction Amount\n"
        "Payment Mode\n"
        "Beneficiary Name\n"
        "Destination Country\n"
        "Value Date (dd/mm/yyyy)\n"
        "99.15\n"
        "MY (MALAYSIA)\n"
        "05 Aug 2026\n"
    )
    result = extract_amount(text, ocr_used=False)
    assert result.value == "99.15"


def test_no_currency_prefix_is_required():
    """Trap 3: no RM/MYR anywhere near the amount."""
    text = "Total Amount\n77.25\n"
    result = extract_amount(text, ocr_used=False)
    assert result.value == "77.25"


# ---------------------------------------------------------------------------
# Missing / empty input
# ---------------------------------------------------------------------------

def test_empty_text_returns_missing():
    result = extract_amount("", ocr_used=False)
    assert result.value is None
    assert result.confidence < 0.5


def test_no_amount_signal_at_all_returns_missing():
    text = "Beneficiary Name\nJOHN DOE\nReceiving Bank\nSOME BANK\n"
    result = extract_amount(text, ocr_used=True)
    assert result.value is None
    assert result.confidence < 0.5
