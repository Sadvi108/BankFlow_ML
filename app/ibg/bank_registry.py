"""Registry of Malaysian banking entities.

Static reference data only -- no extraction logic. `bank_name.py` consumes this
rather than hand-rolling its own map, so there is exactly one place to add a
bank.

Three things make an entity resolvable, in descending order of reliability:

  1. `bic`    -- the SWIFT/BIC code. Unambiguous when present, and often the
                 only clean token left in a degraded scan.
  2. `aliases`-- portal branding and abbreviations as they actually appear on
                 receipts ("Maybank2u", "M2E", "CIMB Clicks", "PBeBank").
  3. `name`   -- the canonical display name.

`parent` is set on Islamic subsidiaries and marks them as distinct legal
entities that share a brand with a conventional parent. RHB Islamic Bank
(RHBAMYKL) is NOT RHB Bank (RHBBMYKL) -- they are separate banks with separate
BICs, and one of this repo's own sample receipts routes to the Islamic entity.
Collapsing them loses real information about where the money went.

BIC accuracy note: codes marked `bic=None` are entities whose BIC this registry
does not assert. Malaysia's digital banks were licensed in 2022-2024 and their
codes are not reliably established here. An invented BIC would be exactly the
confidently-wrong data this project exists to eliminate -- so they resolve by
name only, and `bic_verified=False` marks them for someone to confirm against
the bank's own documentation before anything trusts a BIC lookup for them.
"""
from typing import Dict, List, NamedTuple, Optional, Tuple

# Component order for ambiguous all-numeric dates like "08/05/2026".
# Malaysian portals are day-first without exception. Citibank's CitiDirect is
# not: it prints US month-first, so a real receipt reading "Value Date
# 08/05/2026" means 5 August, not 8 May. Parsing that day-first puts the
# transaction three months in the past, silently.
DAY_FIRST: Tuple[str, str, str] = ("d", "m", "y")
MONTH_FIRST: Tuple[str, str, str] = ("m", "d", "y")


class BankEntity(NamedTuple):
    key: str                    # stable identifier, never displayed
    name: str                   # canonical display name
    bic: Optional[str]          # SWIFT/BIC, None when not asserted
    aliases: List[str]          # as they appear on real receipts
    tier: str                   # retail | islamic | digital | foreign | development | rail
    parent: Optional[str]       # key of the conventional parent, Islamic subs only
    bic_verified: bool = True   # False = BIC not asserted, needs confirmation
    date_order: Tuple[str, str, str] = DAY_FIRST  # for ambiguous numeric dates


# --- Tier 1: conventional retail banks -------------------------------------
_RETAIL = [
    BankEntity("maybank", "Maybank", "MBBEMYKL",
               ["MAYBANK", "MAYBANK2U", "M2U", "M2E", "MAYBANK2E", "MBB", "MYB",
                "MALAYAN BANKING", "MAYBANK2U BIZ", "M2U BIZ",
                "MAYBANK2E-ADMIN"], "retail", None),
    BankEntity("cimb", "CIMB", "CIBBMYKL",
               ["CIMB", "CIMB BANK", "CIMB CLICKS", "CIMBCLICKS", "BIZCHANNEL",
                "BIZCHANNEL@CIMB", "MYBUSINESSCARE", "OCTO"], "retail", None),
    BankEntity("public_bank", "Public Bank", "PBBEMYKL",
               ["PUBLIC BANK", "PBEBANK", "PBE BANK", "PBB", "PB ENTERPRISE",
                "PBENTERPRISE"], "retail", None),
    BankEntity("rhb", "RHB", "RHBBMYKL",
               ["RHB", "RHB BANK", "RHBNOW", "RHB NOW", "REFLEX", "RHBGROUP",
                "RHBCAMS"], "retail", None),
    BankEntity("hong_leong", "Hong Leong Bank", "HLBBMYKL",
               ["HONG LEONG", "HONGLEONG", "HLB", "HL BANK", "HLBCONNECT",
                "CONNECTFIRST", "CONNECT FIRST"], "retail", None),
    BankEntity("ambank", "AmBank", "ARBKMYKL",
               ["AMBANK", "AMONLINE", "AM BANK", "AMACCESS", "AMBANK GROUP"],
               "retail", None),
    BankEntity("alliance", "Alliance Bank", "MFBBMYKL",
               ["ALLIANCE BANK", "ALLIANCE", "ALLIANCEONLINE", "BIZSMART",
                "ALLIANCEBIZSMART"], "retail", None),
    BankEntity("affin", "Affin Bank", "PHBMMYKL",
               ["AFFIN BANK", "AFFIN", "AFFINONLINE", "AFFINALWAYS"], "retail", None),
    BankEntity("bank_islam", "Bank Islam", "BIMBMYKL",
               ["BANK ISLAM", "BIMB", "GO BY BANK ISLAM"], "retail", None),
    BankEntity("bank_muamalat", "Bank Muamalat", "BMMBMYKL",
               ["BANK MUAMALAT", "MUAMALAT", "IMUAMALAT"], "retail", None),
    BankEntity("bank_rakyat", "Bank Rakyat", "BKRMMYKL",
               ["BANK RAKYAT", "BANK KERJASAMA RAKYAT", "IRAKYAT"], "retail", None),
    BankEntity("bsn", "BSN", "BSNAMYK1",
               ["BSN", "BANK SIMPANAN NASIONAL", "MYBSN"], "retail", None),
    BankEntity("agrobank", "Agrobank", "AGOBMYKL",
               ["AGROBANK", "BANK PERTANIAN", "AGRONET"], "retail", None),
    BankEntity("al_rajhi", "Al Rajhi Bank", "ARAJMYKL",
               ["AL RAJHI", "ALRAJHI", "AL-RAJHI"], "retail", None),
    BankEntity("mbsb", "MBSB Bank", "AFBQMYKL",
               ["MBSB", "MBSB BANK", "ASIAN FINANCE"], "retail", None),
    BankEntity("kfh", "Kuwait Finance House", "KFHOMYKL",
               ["KUWAIT FINANCE HOUSE", "KFH"], "retail", None),
    BankEntity("hsbc", "HSBC", "HBMBMYKL",
               ["HSBC", "HSBC BANK", "HSBC MALAYSIA"], "retail", None),
    BankEntity("standard_chartered", "Standard Chartered", "SCBLMYKX",
               ["STANDARD CHARTERED", "STANCHART", "SCB", "SC MALAYSIA",
                "STRAIGHT2BANK"], "retail", None),
    BankEntity("ocbc", "OCBC", "OCBCMYKL",
               ["OCBC", "OCBC BANK", "VELOCITY", "OCBC VELOCITY"], "retail", None),
    BankEntity("uob", "UOB", "UOVBMYKL",
               ["UOB", "UNITED OVERSEAS BANK", "UOB MALAYSIA", "BIBPLUS"],
               "retail", None),
    # CitiDirect prints US month-first dates -- see DAY_FIRST/MONTH_FIRST above.
    BankEntity("citibank", "Citibank", "CITIMYKL",
               ["CITIBANK", "CITI", "CITIDIRECT"], "retail", None,
               True, MONTH_FIRST),
    BankEntity("bnp_paribas", "BNP Paribas", "BNPAMYKL",
               ["BNP PARIBAS", "BNPP"], "retail", None),
]

# --- Tier 2: Islamic subsidiaries (distinct entities, own BIC) --------------
_ISLAMIC = [
    BankEntity("maybank_islamic", "Maybank Islamic", "MBISMYKL",
               ["MAYBANK ISLAMIC"], "islamic", "maybank"),
    BankEntity("cimb_islamic", "CIMB Islamic", "CTBBMYKL",
               ["CIMB ISLAMIC"], "islamic", "cimb"),
    BankEntity("public_islamic", "Public Islamic Bank", "PIBEMYKL",
               ["PUBLIC ISLAMIC"], "islamic", "public_bank"),
    BankEntity("rhb_islamic", "RHB Islamic Bank", "RHBAMYKL",
               ["RHB ISLAMIC"], "islamic", "rhb"),
    BankEntity("hong_leong_islamic", "Hong Leong Islamic Bank", "HLIBMYKL",
               ["HONG LEONG ISLAMIC", "HLISB"], "islamic", "hong_leong"),
    BankEntity("ambank_islamic", "AmBank Islamic", "AISLMYKL",
               ["AMBANK ISLAMIC", "AMISLAMIC"], "islamic", "ambank"),
    BankEntity("alliance_islamic", "Alliance Islamic Bank", "ALSRMYKL",
               ["ALLIANCE ISLAMIC"], "islamic", "alliance"),
    BankEntity("affin_islamic", "Affin Islamic Bank", "AIBBMYKL",
               ["AFFIN ISLAMIC"], "islamic", "affin"),
    BankEntity("hsbc_amanah", "HSBC Amanah", "HMABMYKL",
               ["HSBC AMANAH", "AMANAH"], "islamic", "hsbc"),
    BankEntity("scb_saadiq", "Standard Chartered Saadiq", None,
               ["SAADIQ", "STANDARD CHARTERED SAADIQ"], "islamic",
               "standard_chartered", False),
    BankEntity("ocbc_alamin", "OCBC Al-Amin", "OABBMYKL",
               ["OCBC AL-AMIN", "AL-AMIN", "AL AMIN"], "islamic", "ocbc"),
]

# --- Tier 3: digital banks (licensed 2022-2024) ----------------------------
# BICs deliberately not asserted -- see module docstring.
_DIGITAL = [
    BankEntity("gxbank", "GXBank", None, ["GXBANK", "GX BANK", "GXS"],
               "digital", None, False),
    BankEntity("aeon_bank", "AEON Bank", None, ["AEON BANK"],
               "digital", None, False),
    BankEntity("boost_bank", "Boost Bank", None, ["BOOST BANK"],
               "digital", None, False),
    BankEntity("kaf_digital", "KAF Digital Bank", None,
               ["KAF DIGITAL", "KAF BANK"], "digital", None, False),
    BankEntity("ryt_bank", "Ryt Bank", None, ["RYT BANK"],
               "digital", None, False),
]

# --- Tier 4: foreign / corporate banks operating in Malaysia ---------------
_FOREIGN = [
    BankEntity("bank_of_china", "Bank of China (Malaysia)", "BKCHMYKL",
               ["BANK OF CHINA"], "foreign", None),
    BankEntity("icbc", "ICBC (Malaysia)", "ICBKMYKL",
               ["ICBC", "INDUSTRIAL AND COMMERCIAL BANK OF CHINA"], "foreign", None),
    BankEntity("ccb", "China Construction Bank (Malaysia)", "PCBCMYKL",
               ["CHINA CONSTRUCTION BANK", "CCB"], "foreign", None),
    BankEntity("mufg", "MUFG Bank (Malaysia)", "BOTKMYKX",
               ["MUFG", "BANK OF TOKYO", "MITSUBISHI UFJ", "GCMS PLUS",
                "GCMS", "COMSUITE"], "foreign", None),
    BankEntity("smbc", "Sumitomo Mitsui (Malaysia)", "SMBCMYKL",
               ["SUMITOMO MITSUI", "SMBC"], "foreign", None),
    BankEntity("mizuho", "Mizuho Bank (Malaysia)", "MHCBMYKA",
               ["MIZUHO"], "foreign", None),
    BankEntity("deutsche", "Deutsche Bank (Malaysia)", "DEUTMYKL",
               ["DEUTSCHE BANK"], "foreign", None),
    BankEntity("jpmorgan", "J.P. Morgan (Malaysia)", "CHASMYKX",
               ["J.P. MORGAN", "JP MORGAN", "JPMORGAN", "CHASE"], "foreign", None),
    BankEntity("bangkok_bank", "Bangkok Bank (Malaysia)", "BKKBMYKL",
               ["BANGKOK BANK"], "foreign", None),
]

# --- Tier 5: development / co-operative ------------------------------------
_DEVELOPMENT = [
    BankEntity("coopbank_pertama", "Co-opbank Pertama", None,
               ["CO-OPBANK PERTAMA", "COOPBANK PERTAMA", "KOPERASI BANK PERSATUAN"],
               "development", None, False),
    BankEntity("bank_pembangunan", "Bank Pembangunan Malaysia", "BPMBMYKL",
               ["BANK PEMBANGUNAN", "BPMB"], "development", None),
    BankEntity("exim_bank", "EXIM Bank Malaysia", "EXMBMYKL",
               ["EXIM BANK", "EXPORT-IMPORT BANK OF MALAYSIA"], "development", None),
]

# --- Tier 6: payment rails and e-wallets (NOT banks) -----------------------
# Kept separate on purpose. The live engine currently returns "DuitNow" as a
# bank name, which is wrong -- DuitNow is a rail that rides on top of a bank.
# Matching one of these must never satisfy "which bank issued this receipt".
_RAILS = [
    BankEntity("duitnow", "DuitNow", None, ["DUITNOW", "DUIT NOW"], "rail", None, False),
    BankEntity("jompay", "JomPAY", None, ["JOMPAY", "JOM PAY"], "rail", None, False),
    BankEntity("fpx", "FPX", None, ["FPX", "FINANCIAL PROCESS EXCHANGE"], "rail", None, False),
    BankEntity("rentas", "RENTAS", None, ["RENTAS"], "rail", None, False),
    BankEntity("ibg", "IBG", None, ["IBG", "INTERBANK GIRO"], "rail", None, False),
    BankEntity("tng", "Touch n Go", None,
               ["TOUCH N GO", "TOUCH 'N GO", "TNG EWALLET", "TNGD"], "rail", None, False),
    BankEntity("grabpay", "GrabPay", None, ["GRABPAY", "GRAB PAY"], "rail", None, False),
    BankEntity("boost_ewallet", "Boost eWallet", None, ["BOOST EWALLET"], "rail", None, False),
    BankEntity("shopeepay", "ShopeePay", None, ["SHOPEEPAY", "SHOPEE PAY"], "rail", None, False),
]


ALL_ENTITIES: List[BankEntity] = (
    _RETAIL + _ISLAMIC + _DIGITAL + _FOREIGN + _DEVELOPMENT + _RAILS
)

BANKS_ONLY: List[BankEntity] = [e for e in ALL_ENTITIES if e.tier != "rail"]

_BY_KEY: Dict[str, BankEntity] = {e.key: e for e in ALL_ENTITIES}
_BY_BIC: Dict[str, BankEntity] = {e.bic: e for e in ALL_ENTITIES if e.bic}


def by_key(key: str) -> Optional[BankEntity]:
    return _BY_KEY.get(key)


def lookup_by_bic(bic: str) -> Optional[BankEntity]:
    """Resolve an 8- or 11-character SWIFT/BIC to an entity.

    An 11-character BIC is an 8-character institution code plus a 3-character
    branch suffix, so fall back to the first 8 on a miss.
    """
    if not bic:
        return None
    code = bic.strip().upper().replace(" ", "")
    hit = _BY_BIC.get(code)
    if hit is None and len(code) > 8:
        hit = _BY_BIC.get(code[:8])
    return hit


def lookup_by_alias(token: str) -> Optional[BankEntity]:
    """Exact match of a single token/phrase against canonical names and aliases."""
    if not token:
        return None
    probe = token.strip().upper()
    for entity in ALL_ENTITIES:
        if probe == entity.name.upper() or probe in entity.aliases:
            return entity
    return None


def is_rail(entity: Optional[BankEntity]) -> bool:
    """True for payment rails and e-wallets, which are never the issuing bank."""
    return entity is not None and entity.tier == "rail"


def date_order_for(bank_key: Optional[str]) -> Tuple[str, str, str]:
    """Component order to use when parsing an ambiguous all-numeric date.

    Falls back to day-first for an unknown or unresolved bank, which is correct
    for every Malaysian portal in this registry except Citibank.
    """
    entity = _BY_KEY.get(bank_key) if bank_key else None
    return entity.date_order if entity else DAY_FIRST


def unverified_bics() -> List[str]:
    """Entities whose BIC this registry does not assert. For the coverage report."""
    return [e.name for e in ALL_ENTITIES if not e.bic_verified]
