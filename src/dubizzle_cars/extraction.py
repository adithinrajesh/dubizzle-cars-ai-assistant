"""Deliberately narrow rules. Missing matches and conflicting values stay unknown."""

import re
from decimal import Decimal

from .models import Metadata

NUMBER = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?"
PRICE = re.compile(
    rf"\b(?:sale price|cash price|asking price|price)\s*[:=\-]?\s*(?:AED|DHS?)\s*({NUMBER})(?![\w,.])"
    rf"|\b(?:AED|DHS?)\s*({NUMBER})(?![\w,.])\s+in cash\b",
    re.I,
)
MILEAGE = re.compile(
    rf"\b(?:mileage|odometer|kilomet(?:er|re)s)\s*[:=\-]?\s*(?:just\s+)?({NUMBER})\s*(?:km|kms|kilomet(?:er|re)s)\b",
    re.I,
)
TITLE_MILEAGE = re.compile(rf"(?<![\w.,])({NUMBER})\s*(?:km|kms)\b", re.I)
REGIONS = {
    "GCC": r"\bgcc(?:\s+spec(?:s|ifications)?)?\b",
    "American": r"\b(?:american|us|usa)\s+spec(?:s|ifications)\b",
    "Japanese": r"\bjapanese\s+spec(?:s|ifications)\b",
    "European": r"\beuropean\s+spec(?:s|ifications)\b",
    "Korean": r"\bkorean\s+spec(?:s|ifications)\b",
    "Canadian": r"\bcanadian\s+spec(?:s|ifications)\b",
}
UNCERTAIN = re.compile(
    r"\b(?:or|not|non|isn't|isn’t|maybe|possibly|approximately|approx|about|around|up to|from|starting|between|over|under)\b|[?~<>]|\d\s*[-–/]\s*\d|\d[\d,. ]*\s+to\s+\d|(?<!\w)-\d",
    re.I,
)
FINANCE = re.compile(
    r"\b(?:monthly|month|finance|financing|installment|instalment|deposit|down[ -]?payment|per month)\b|/\s*mo\b",
    re.I,
)


def _unique(values):
    values = set(values)
    return next(iter(values)) if len(values) == 1 else None


def _clauses(text: str):
    # Decimal points do not split amounts. Bullets and sentence ends split offers.
    return re.split(r"[\n;|•!]|(?<=[a-zA-Z])\.\s+", text)


def extract_metadata(title: str | None, description: str | None) -> Metadata:
    prices, distances, regions, warranty = [], [], [], []
    invalid_price = invalid_mileage = invalid_region = False
    for text, is_title in ((title or "", True), (description or "", False)):
        for clause in _clauses(text):
            if UNCERTAIN.search(clause):
                if re.search(r"\b(?:mileage|odometer)\b", clause, re.I):
                    invalid_mileage = True
                if re.search(r"\bprice\b", clause, re.I) and not FINANCE.search(clause):
                    invalid_price = True
            for match in PRICE.finditer(clause):
                if FINANCE.search(clause):
                    continue
                if re.match(r"\s+(?:\d|[km]\b|million\b|thousand\b)", clause[match.end() :], re.I):
                    invalid_price = True
                    continue
                if UNCERTAIN.search(clause):
                    invalid_price = True
                    continue
                amount = Decimal(next(g for g in match.groups() if g).replace(",", ""))
                if amount > 0:
                    prices.append(amount)
            pattern = TITLE_MILEAGE if is_title else MILEAGE
            for match in pattern.finditer(clause):
                if UNCERTAIN.search(clause) or re.search(
                    r"\bwarranty\b|service interval", clause, re.I
                ):
                    invalid_mileage = True
                    continue
                value = Decimal(match[1].replace(",", ""))
                if value == value.to_integral_value():
                    distances.append(int(value))
                else:
                    invalid_mileage = True
            for region, pattern in REGIONS.items():
                if region == "GCC" and not is_title:
                    pattern = r"\bgcc\s+spec(?:s|ifications)\b|\bregional specs\s*[:\-]\s*gcc\b"
                if re.search(pattern, clause, re.I):
                    if UNCERTAIN.search(clause):
                        invalid_region = True
                    else:
                        regions.append(region)
            if re.search(r"\bwarranty\b", clause, re.I):
                warranty.append(clause.strip())
    return Metadata(
        sale_price_aed=None if invalid_price else _unique(prices),
        mileage_km=None if invalid_mileage else _unique(distances),
        warranty_mention="\n".join(dict.fromkeys(warranty)) or None,
        regional_specs=None if invalid_region else _unique(regions),
    )
