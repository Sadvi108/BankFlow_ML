"""Tests for `app.ibg.transaction_date.extract_transaction_date`.

Two groups:

* `test_matches_corpus` -- parametrized off `ibg_corpus.cases_for
  ("transaction_date")`, i.e. every ground-truth receipt in
  `tests/fixtures/ibg_corpus.py`. This is the scoring harness: the
  extractor must return `expected["transaction_date"]` and must never
  return a value listed in that sample's `traps["transaction_date"]`. Note
  this calls `extract_transaction_date` with no `bank_key` -- exactly as
  the not-yet-built coordinator's absence means it is called in production
  today -- so it also exercises the text-based bank detection and
  same-document field-order inference the module falls back to.

* Everything else -- small, hand-built texts that isolate one rule each:
  the two-digit-year pivot, invalid-date rejection, day-first parsing, the
  format-hint override, print-stamp/administrative-stamp demotion
  (including the "nothing but a stamp on the page" case, which the corpus
  itself never exercises directly since every corpus sample with a stamp
  also has a real labelled date next to it), the `bank_key` /
  same-document date-order resolution, and the window-truncation digit-run
  guard.

None of these hand-built texts reproduce a corpus literal -- they are
independent regression pins for the underlying rule, not disguised copies
of `ibg_corpus.py`.
"""
import pytest

from app.ibg.contract import FieldResult
from app.ibg.transaction_date import (
    extract_transaction_date,
    _detect_bank_key,
    _extend_window,
    _infer_field_order,
    _LABEL_WINDOW,
    _normalize_day_monthname_year,
    _normalize_numeric,
    _resolve_two_digit_year,
    _to_iso,
)
from tests.fixtures import ibg_corpus


# ---------------------------------------------------------------------
# Corpus: the scoring harness.
# ---------------------------------------------------------------------
_CASES = list(ibg_corpus.cases_for("transaction_date"))


@pytest.mark.parametrize(
    "sample_id,text,ocr_used,expected,traps",
    _CASES,
    ids=[c[0] for c in _CASES],
)
def test_matches_corpus(sample_id, text, ocr_used, expected, traps):
    result = extract_transaction_date(text, ocr_used=ocr_used)

    assert result.value == expected, (
        f"{sample_id}: expected {expected!r}, got {result.value!r} "
        f"(source={result.source}, candidates={result.candidates})"
    )
    for trap_value in traps:
        assert result.value != trap_value, (
            f"{sample_id}: returned trap value {trap_value!r} "
            f"(source={result.source})"
        )

    # Contract: confidence must be < 0.5 whenever value is None, and always
    # within [0, 1].
    assert 0.0 <= result.confidence <= 1.0
    if result.value is None:
        assert result.confidence < 0.5
        assert result.found is False
    else:
        assert result.found is True


def test_corpus_has_at_least_one_none_and_one_found_case():
    """Sanity check on the corpus itself: both branches are exercised."""
    expected_values = [c[3] for c in _CASES]
    assert None in expected_values
    assert any(v is not None for v in expected_values)


# ---------------------------------------------------------------------
# Two-digit-year pivot.
# ---------------------------------------------------------------------
@pytest.mark.parametrize(
    "yy,reference_year,expected_year",
    [
        (25, 2026, 2025),   # within +10y window -> 2000s
        (0, 2026, 2000),    # boundary at the low end
        (36, 2026, 2036),   # exactly +10y from reference -> still 2000s
        (37, 2026, 1937),   # one past the pivot -> falls back a century
        (99, 2026, 1999),   # comfortably past the pivot -> 1900s
    ],
)
def test_two_digit_year_pivot(yy, reference_year, expected_year):
    assert _resolve_two_digit_year(yy, reference_year) == expected_year


def test_two_digit_year_pivot_end_to_end():
    # "15/06/25" -> 2025-06-15, never 1925 or 2125, using the real
    # reference year. Safe against future drift: this only stops holding
    # once "today" is earlier than 2015, which cannot happen from here on.
    result = extract_transaction_date("Value Date: 15/06/25", ocr_used=False)
    assert result.value == "2025-06-15"


# ---------------------------------------------------------------------
# Invalid-date rejection.
# ---------------------------------------------------------------------
@pytest.mark.parametrize(
    "day,month,year",
    [
        ("31", "02", "2025"),  # February has no 31st
        ("13", "13", "2025"),  # month 13 does not exist
        ("00", "01", "2025"),  # day 0 does not exist
        ("31", "04", "2025"),  # April has no 31st
        ("29", "02", "2025"),  # 2025 is not a leap year
    ],
)
def test_invalid_numeric_date_rejected(day, month, year):
    assert _normalize_numeric(day, month, year, reference_year=2026) is None


def test_invalid_month_name_date_rejected():
    assert _normalize_day_monthname_year("31", "Feb", "2025", reference_year=2026) is None


def test_to_iso_rejects_impossible_calendar_dates():
    assert _to_iso(2025, 2, 31) is None
    assert _to_iso(2025, 13, 1) is None
    assert _to_iso(2025, 4, 31) is None
    assert _to_iso(2025, 8, 3) == "2025-08-03"


def test_invalid_date_end_to_end_is_missing_not_coerced():
    # The only date-shaped token on the page is impossible. It must not be
    # silently coerced into some nearby valid date -- the honest answer is
    # "not found".
    result = extract_transaction_date(
        "Transaction Date: 31/02/2025\nNothing else here.", ocr_used=False
    )
    assert result.value is None
    assert result.confidence < 0.5


# ---------------------------------------------------------------------
# Day-first parsing (Malaysian receipts: dd/mm/yyyy, never mm/dd/yyyy).
# ---------------------------------------------------------------------
def test_day_first_is_the_default_with_no_hint():
    # 05/01/2025 with no format hint must be 5 January, not 1 May.
    result = extract_transaction_date("Transaction Date: 05/01/2025", ocr_used=False)
    assert result.value == "2025-01-05"


def test_day_first_disambiguated_when_day_exceeds_twelve():
    # 25/01/2025 can only be day-first (month 25 is not a month), so this
    # also pins day-first even though there's no ambiguity to resolve.
    result = extract_transaction_date("Payment Date: 25/01/2025", ocr_used=False)
    assert result.value == "2025-01-25"


def test_format_hint_is_honoured_and_can_override_day_first():
    # An explicit (mm/dd/yyyy) hint changes the interpretation of the same
    # digits: 05/01/2025 becomes 1 May, not 5 January.
    result = extract_transaction_date(
        "Transfer Date (mm/dd/yyyy): 05/01/2025", ocr_used=False
    )
    assert result.value == "2025-05-01"


def test_format_hint_text_itself_never_parses_as_a_date():
    # If the hint were the only thing on the page (no actual value after
    # it), there must be nothing to find -- "dd/mm/yyyy" has no digits and
    # cannot be mistaken for a date.
    result = extract_transaction_date("Value Date (dd/mm/yyyy)", ocr_used=False)
    assert result.value is None


# ---------------------------------------------------------------------
# Print-stamp demotion.
# ---------------------------------------------------------------------
def test_print_stamp_never_beats_a_labelled_date():
    # A render stamp (date, comma, bare HH:MM, no seconds, next to app
    # chrome) appears first in the text and is the only OTHER date on the
    # page -- exactly the shape that trips up a "first date wins" rule.
    text = (
        "15/09/2026, 06:00 SomeApp Portal\n"
        "https://example.com/portal/view\n"
        "Transaction Date: 05 Jan 2026\n"
    )
    result = extract_transaction_date(text, ocr_used=False)
    assert result.value == "2026-01-05"
    assert result.value != "2026-09-15"  # the stamp's own date, demoted
    assert result.source.startswith("label:")


def test_print_stamp_repeated_does_not_win_by_frequency_with_label():
    # The stamp repeats four times (once per page footer); the real date
    # appears once. A frequency/voting rule would pick the stamp.
    text = (
        "15/09/2026, 06:00 SomeApp Portal\n"
        "Value Date: 06 Jan 2026\n"
        "15/09/2026, 06:00 SomeApp Portal\n"
        "15/09/2026, 06:00 SomeApp Portal\n"
        "15/09/2026, 06:00 SomeApp Portal\n"
    )
    result = extract_transaction_date(text, ocr_used=False)
    assert result.value == "2026-01-06"


def test_print_stamp_repeated_does_not_win_by_frequency_unlabelled():
    # Same idea, but with no label at all, so the decision is forced
    # through the tier-3 unlabeled-fallback path rather than the label
    # path -- the stamp still must not win just because it outnumbers the
    # real date four to one.
    text = (
        "15/09/2026, 06:00 SomeApp Portal\n"
        "15/09/2026, 06:00 SomeApp Portal\n"
        "Some receipt body text with no date label at all: 06 Jan 2026.\n"
        "15/09/2026, 06:00 SomeApp Portal\n"
        "15/09/2026, 06:00 SomeApp Portal\n"
    )
    result = extract_transaction_date(text, ocr_used=False)
    assert result.value == "2026-01-06"
    assert result.source == "fallback:unlabeled_date"


def test_only_print_stamp_present_returns_missing():
    # No label anywhere, and the one date-shaped token on the page is
    # print-stamp shaped. The rule is: no unlabeled date beats a labelled
    # one, and a print stamp is never the answer -- so this must come back
    # as "not found", not as the stamp's date.
    text = "15/09/2026, 06:00 SomeApp Portal https://example.com/x\n"
    result = extract_transaction_date(text, ocr_used=False)
    assert result.value is None
    assert result.found is False
    assert result.confidence < 0.5
    assert result.source == "missing:only_print_stamp"


def test_unlabelled_genuine_timestamp_with_seconds_is_not_a_print_stamp():
    # A comma between date and time is not enough on its own to be a print
    # stamp -- seconds present means it is a genuine audit timestamp, even
    # with no label attached to it.
    text = "Receipt issued 12 Feb 2026, 09:15:42 for your records.\n"
    result = extract_transaction_date(text, ocr_used=True)
    assert result.value == "2026-02-12"
    assert result.source == "fallback:unlabeled_date"


# ---------------------------------------------------------------------
# Times and timezones: date-only output, no shifting.
# ---------------------------------------------------------------------
def test_time_and_timezone_suffix_is_stripped_not_shifted():
    text = "Date and Time\n07 Feb 2026 23:59:59 MY (UTC+08:00)\n"
    result = extract_transaction_date(text, ocr_used=False)
    # Must stay 2026-02-07 -- a naive UTC conversion would push this to
    # 2026-02-07 15:59:59Z, which is still the 7th, but a careless -8h
    # shift instead of -0h (already local) would wrongly land on the 6th.
    assert result.value == "2026-02-07"


# ---------------------------------------------------------------------
# Other formats / edge cases.
# ---------------------------------------------------------------------
def test_iso_format_is_parsed_directly():
    result = extract_transaction_date("Value Date: 2026-02-07", ocr_used=False)
    assert result.value == "2026-02-07"


def test_us_style_month_name_format_is_supported():
    result = extract_transaction_date("Payment Date: Feb 07, 2026", ocr_used=False)
    assert result.value == "2026-02-07"


def test_secondary_label_used_only_when_no_primary_label_exists():
    text = "Advice Date: 09 Mar 2026\n"
    result = extract_transaction_date(text, ocr_used=False)
    assert result.value == "2026-03-09"
    assert result.source.startswith("label_secondary:")


def test_primary_label_beats_secondary_label_when_both_present():
    text = (
        "Advice Date: 09 Mar 2026\n"
        "Transaction Date: 10 Mar 2026\n"
    )
    result = extract_transaction_date(text, ocr_used=False)
    assert result.value == "2026-03-10"
    assert result.source.startswith("label:")


def test_empty_text_returns_missing():
    result = extract_transaction_date("", ocr_used=False)
    assert result == FieldResult.missing(source="missing:empty_text")


def test_no_date_shaped_text_returns_missing():
    result = extract_transaction_date("No dates anywhere in this receipt body.", ocr_used=True)
    assert result.value is None
    assert result.source == "missing:no_date_candidates"


def test_value_is_always_iso_format_when_found():
    import re

    iso_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    for sample_id, text, ocr_used, expected, _traps in _CASES:
        result = extract_transaction_date(text, ocr_used=ocr_used)
        if result.value is not None:
            assert iso_re.match(result.value), (
                f"{sample_id}: {result.value!r} is not ISO YYYY-MM-DD"
            )


# ---------------------------------------------------------------------
# `bank_key` and date-order resolution.
#
# Precedence: an explicit `bank_key` from the caller > a bank brand
# detected straight from the text > same-document >12 / month-name
# inference > day-first default. `bank_key=None` (the default) must keep
# working exactly as it does today -- the coordinator that would supply it
# does not exist yet.
# ---------------------------------------------------------------------
def test_citibank_is_month_first_via_explicit_bank_key():
    # 05/08/2026 has no component over 12 -- genuinely ambiguous without
    # knowing the issuing bank. With bank_key="citibank" it must read as
    # month-first: month=05, day=08 -> 8 May.
    result = extract_transaction_date(
        "Value Date\n05/08/2026\n", ocr_used=False, bank_key="citibank"
    )
    assert result.value == "2026-05-08"


def test_citibank_bank_key_overrides_the_day_first_default():
    # Same ambiguous digits, no bank_key at all and no Citi branding in the
    # text to auto-detect -- day-first is the correct default here (day=05,
    # month=08 -> 5 August), which is exactly why bank_key exists: nothing
    # in the text itself could ever tell these two cases apart.
    result = extract_transaction_date("Value Date\n05/08/2026\n", ocr_used=False)
    assert result.value == "2026-08-05"


def test_uob_bank_key_is_day_first():
    # UOB is a plain Malaysian portal: bank_key="uob" must resolve the same
    # ambiguous digits as day-first (5 August), unlike Citibank.
    result = extract_transaction_date(
        "Value Date\n05/08/2026\n", ocr_used=False, bank_key="uob"
    )
    assert result.value == "2026-08-05"


def test_unknown_bank_key_falls_back_to_day_first():
    result = extract_transaction_date(
        "Value Date\n05/08/2026\n", ocr_used=False, bank_key="not_a_real_bank"
    )
    assert result.value == "2026-08-05"


def test_citibank_branding_in_text_is_detected_without_explicit_bank_key():
    # No bank_key passed -- exactly how the not-yet-built coordinator's
    # absence means this module is called in production today. The text
    # itself carries CitiDirect branding, which must be enough on its own
    # to resolve the ambiguous digits as month-first (8 May).
    text = "CitiDirect\nValue Date\n05/08/2026\n"
    result = extract_transaction_date(text, ocr_used=False)
    assert result.value == "2026-05-08"
    assert _detect_bank_key(text) == "citibank"


def test_detect_bank_key_skips_payment_rails():
    # DuitNow rides on top of a real bank and is never itself the issuer --
    # a receipt naming only a rail must not resolve to a bank_key.
    assert _detect_bank_key("DuitNow Transfer Successful\n") is None


def test_detect_bank_key_returns_none_for_unbranded_text():
    assert _detect_bank_key("Some generic receipt with no bank name.\n") is None


# ---------------------------------------------------------------------
# Same-document inference: a component over 12 pins the field order for
# every ambiguous numeric date in that document.
# ---------------------------------------------------------------------
def test_infer_field_order_day_first_from_over_twelve_first_component():
    # 25/01/2026 can only be day-first (25 is not a month); a second,
    # otherwise-ambiguous date elsewhere on the page must follow suit.
    text = "Reference: 25/01/2026\nValue Date\n05/08/2026\n"
    assert _infer_field_order(text, reference_year=2026) == ("d", "m", "y")
    result = extract_transaction_date(text, ocr_used=False)
    assert result.value == "2026-08-05"


def test_infer_field_order_month_first_from_over_twelve_second_component():
    # 01/25/2026 can only be month-first (25 is not a month, and it is in
    # the second slot); an otherwise-ambiguous date elsewhere must follow.
    text = "Reference: 01/25/2026\nValue Date\n05/08/2026\n"
    assert _infer_field_order(text, reference_year=2026) == ("m", "d", "y")
    result = extract_transaction_date(text, ocr_used=False)
    assert result.value == "2026-05-08"


def test_infer_field_order_is_none_when_no_component_exceeds_twelve():
    # 05/08/2026 alone is genuinely ambiguous -- no inference is possible,
    # this is exactly the Citibank problem bank_key exists to solve.
    assert _infer_field_order("05/08/2026", reference_year=2026) is None


def test_infer_field_order_ignores_contradictory_evidence():
    # One date pins day-first (25 in the first slot), another pins
    # month-first (25 in the second slot) -- a genuine mix is not usable
    # evidence for the whole document, so inference must decline to guess.
    text = "25/01/2026 and 01/25/2026"
    assert _infer_field_order(text, reference_year=2026) is None


def test_infer_field_order_from_corroborating_month_name_date():
    # "07 Aug 2026" is unambiguous. The only numeric date on the page
    # shares those exact digits, and only the day-first reading (7, 8)
    # matches it -- the month-first reading (8, 7) does not exist anywhere
    # else on the page, so this corroborates day-first.
    text = "Printed 07 Aug 2026\nSomething: 07/08/2026\n"
    assert _infer_field_order(text, reference_year=2026) == ("d", "m", "y")


# ---------------------------------------------------------------------
# New date-shaped formats.
# ---------------------------------------------------------------------
def test_dotted_iso_format_is_supported():
    # "2026.08.03" -- unambiguous regardless of field order, since the
    # leading 4-digit group can only be the year.
    result = extract_transaction_date("Value Date 2026.08.03", ocr_used=False)
    assert result.value == "2026-08-03"


def test_hyphenated_day_month_name_year_is_supported():
    # "15-Jul-2026" -- the same DD Month YYYY shape as "15 Jul 2026" but
    # hyphen-separated throughout.
    result = extract_transaction_date("Payment Date 15-Jul-2026", ocr_used=False)
    assert result.value == "2026-07-15"


def test_today_label_is_recognised():
    result = extract_transaction_date("Today : 07-Aug-2026", ocr_used=False)
    assert result.value == "2026-08-07"
    assert result.source == "label:today"


def test_weekday_prefixed_month_name_format_is_supported():
    # "Wednesday, August 5, 2026 7:02 PM" -- a leading weekday name must not
    # confuse the Month DD, YYYY matcher.
    result = extract_transaction_date(
        "Value Date: Wednesday, August 5, 2026 7:02 PM", ocr_used=False
    )
    assert result.value == "2026-08-05"


def test_debit_date_and_creation_date_labels_are_recognised():
    result = extract_transaction_date("Debit Date 07/08/2026", ocr_used=False)
    assert result.value == "2026-08-07"
    result2 = extract_transaction_date("Creation Date: 06 Aug 2026", ocr_used=False)
    assert result2.value == "2026-08-06"


# ---------------------------------------------------------------------
# Administrative / print stamps that are never the answer, even when they
# do not take the comma+bare-HH:MM print-stamp shape.
# ---------------------------------------------------------------------
def test_printed_date_time_stamp_weeks_later_never_wins_over_the_label():
    # The exact shape that broke a naive "first/earliest date wins" rule:
    # an administrative "Printed Date/Time" stamp appears first in the
    # text, three weeks after the real transaction, carries seconds (so it
    # is not comma-shaped print-stamp either), and is otherwise the only
    # other date on the page.
    text = (
        "SomePortal\n"
        "Printed Date/Time: 7 August 2026 10:57:26\n"
        "Transaction Details\n"
        "Reference No 12345\n"
        "Payment Date 15-Jul-2026\n"
        "Copyright 2026 Some Bank Berhad\n"
    )
    result = extract_transaction_date(text, ocr_used=False)
    assert result.value == "2026-07-15"
    assert result.value != "2026-08-07"
    assert result.source.startswith("label:")


def test_printed_date_time_stamp_alone_is_missing_not_the_stamp():
    # No labelled date anywhere -- the only date-shaped token on the page
    # is a "Printed Date/Time" administrative stamp, which must never win
    # by fallback even though it carries seconds (unlike a comma print
    # stamp) and would otherwise look like a perfectly genuine timestamp.
    text = "Printed Date/Time: 7 August 2026 10:57:26\nNo other dates here.\n"
    result = extract_transaction_date(text, ocr_used=False)
    assert result.value is None
    assert result.source == "missing:only_print_stamp"


def test_generated_on_stamp_never_wins_over_the_label():
    text = (
        "Payment Date 03/08/2026\n"
        "Generated on 24 August 2026 at 09:00 AM by SYSTEM\n"
    )
    result = extract_transaction_date(text, ocr_used=False)
    assert result.value == "2026-08-03"
    assert result.value != "2026-08-24"


def test_date_of_export_stamp_never_wins_over_the_label():
    text = (
        "Value Date\n"
        "03/08/2026\n"
        "Date of Export: 24/08/2026\n"
    )
    result = extract_transaction_date(text, ocr_used=False)
    assert result.value == "2026-08-03"


# ---------------------------------------------------------------------
# Window-truncation digit-run guard: a fixed-length label window must
# never be cut through the middle of a digit run and silently produce a
# shorter, differently-meaning date.
# ---------------------------------------------------------------------
def test_bare_account_number_with_no_date_yields_missing():
    # A long, separator-free digit run must never be read as a date
    # fragment -- there is no date anywhere on this page.
    text = "Account No 070000000000001 Reference AOBIFT07082026075791\n"
    result = extract_transaction_date(text, ocr_used=False)
    assert result.value is None
    assert result.found is False


def test_extend_window_does_not_cut_through_a_digit_run():
    # A naive fixed-length slice at 8 chars would land exactly after
    # "07/08/20" -- a perfectly well-formed, but wrong, 2-digit-year date
    # hiding inside a truncated "07/08/2026".
    text = "07/08/2026 rest of the receipt"
    assert text[:8] == "07/08/20"
    extended = _extend_window(text, 0, 8)
    assert extended.startswith("07/08/2026")


def test_extend_window_is_a_no_op_when_the_cut_lands_on_a_non_digit():
    text = "07/08/2026 rest of the receipt"
    assert _extend_window(text, 0, 11) == "07/08/2026 "


def test_label_window_does_not_truncate_a_real_date_into_a_fake_short_year():
    # Regression pin for the exact failure mode found in the field: a fixed
    # window sliced right after "20" of "2026" used to parse a genuine 2026
    # date as a 2-digit year and pivot it to 2020. Pad the label block out
    # so the naive `_LABEL_WINDOW`-length cut lands exactly two digits into
    # the year -- reproducing the bug's precise alignment regardless of
    # where `_LABEL_WINDOW` is currently set.
    label = "Value Date"
    tail = "07/08/2026 11:49:07\n"
    # Position, within `tail`, right after "07/08/20" (2 digits into the
    # year) -- this is where a non-extending slice would cut.
    cut_within_tail = tail.index("20", tail.index("/2")) + 2
    filler_len = _LABEL_WINDOW - cut_within_tail
    assert filler_len > 0
    # "." rather than a word character: keeps a regex word boundary intact
    # right before the date so `_NUMERIC_RE` can still match it at all.
    text = label + ("." * filler_len) + tail
    naive_cut = text[len(label):len(label) + _LABEL_WINDOW]
    assert naive_cut.endswith("07/08/20")  # confirms the alignment is real

    result = extract_transaction_date(text, ocr_used=False)
    assert result.value == "2026-08-07"
    assert result.value != "2020-08-07"
