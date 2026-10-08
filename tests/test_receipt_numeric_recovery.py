"""Sanitized regressions from visually checked local receipt failures."""
import pytest

from app.ibg.amount import extract_amount, extract_fee, extract_total_debit
from app.ibg.transaction_date import extract_transaction_date


@pytest.mark.parametrize("text,expected", [
    ("Value Date  Amount  RHB BANK\n07 2026  53,00 MYR\nAug", "53.00"),
    ("Cheque Deposit Receipt\nTotal Amount: 0,00", "0.00"),
    ("Transaction Amount\n1,234,56", "1234.56"),
    ("Amount: MYR 10,60\nService Charge: 0,10\nTotal Debit: 10,70", "10.60"),
])
def test_ocr_decimal_comma_keeps_the_transaction_amount(text, expected):
    assert extract_amount(text, ocr_used=True).value == expected


def test_fee_and_total_remain_separate_after_numeric_repair():
    text = "Amount: MYR 10,60\nService Charge: 0,10\nTotal Debit: 10,70"
    assert extract_fee(text).value == "0.10"
    assert extract_total_debit(text).value == "10.70"


@pytest.mark.parametrize("text", ["Account No: 53,00", "Reference No: 10,60", "Date: 06,08"])
def test_comma_numbers_without_money_evidence_are_not_amounts(text):
    assert extract_amount(text).value is None


def test_comma_repair_is_disabled_for_authoritative_digital_text():
    assert extract_amount("Amount: 53,00", ocr_used=False).value is None


def test_mobile_transaction_timestamp_is_not_a_browser_print_stamp():
    text = "DuitNow Transfer\nSuccessful\nReference ID\nBANK123456\n07 Aug 2026, 08:57 AM\nAmount\nRM 1855.00"
    assert extract_transaction_date(text, ocr_used=False).value == "2026-08-07"


def test_browser_footer_with_ampm_time_still_cannot_supply_transaction_date():
    text = "Reference ID\nBANK123456\nPrinted on: 07 Aug 2026, 08:57 AM\nhttps://example.com"
    assert extract_transaction_date(text, ocr_used=False).value is None


def test_ocr_zero_repair_is_restricted_to_labeled_dates():
    assert extract_transaction_date("Date/Time + Q6-08-2026 17:22:40", ocr_used=True).value == "2026-08-06"
    assert extract_transaction_date("Date/Time + Q6-08-2026 17:22:40", ocr_used=False).value is None
    assert extract_transaction_date("Customer Reference: Q6-08-2026", ocr_used=True).value is None
