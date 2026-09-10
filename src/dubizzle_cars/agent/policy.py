"""Conservative local checks for clear disallowed requests; broader scope is in the prompt."""

import re

SCOPE_REDIRECT = "I'm focused on cars and the dubizzle vehicle inventory. I can help you find or compare a car instead."
COMPETITOR_REDIRECT = "I can help with dubizzle Cars inventory, but I can't discuss or compare other used-car platforms."
BOOKING_REDIRECT = (
    "I can help you explore cars, but viewing and test-drive bookings are not available yet."
)
CURRENCY_REDIRECT = (
    "Inventory prices are in AED. I can't convert currencies; please provide an AED budget."
)

COMPETITORS = re.compile(
    r"\b(?:cars24|carswitch|carvana|autotrader|dubicars|cargurus|carwow|olx)\b|\b(?:competitor|competing|another|other)\b.{0,50}\b(?:marketplace|platform|car website)s?\b",
    re.I,
)
UNRELATED = re.compile(
    r"\b(?:write|generate|implement|debug|create|explain)\b.{0,80}\b(?:python|javascript|sorting algorithm|programming|sql query)\b|\b(?:world war|ancient rome|french revolution|napoleon|politics|election|diagnose|prescribe)\b|\b(?:write|compose)\b.{0,50}\b(?:poem|essay|love letter)\b",
    re.I,
)
FOREIGN_CURRENCY = re.compile(
    r"\b(?:USD|EUR|GBP|INR|dollars?|euros?|rupees?|pounds?)\b|[$€£]", re.I
)
BOOKING = re.compile(
    r"\b(?:book|schedule|reserve)\b.{0,50}\b(?:viewing|test[- ]drive|appointment)\b", re.I
)
INJECTION = re.compile(
    r"\bignore\b.{0,40}\b(?:instructions|rules|grounding)\b|\b(?:invent|fabricate)\b.{0,30}\b(?:listing|inventory|price)s?\b",
    re.I,
)


def preflight_reply(message: str, *, stateful: bool = False) -> str | None:
    if COMPETITORS.search(message):
        return COMPETITOR_REDIRECT
    if UNRELATED.search(message) or INJECTION.search(message):
        return SCOPE_REDIRECT
    if FOREIGN_CURRENCY.search(message) and re.search(
        r"\b(?:price|budget|under|cost|buy|spend|maximum|minimum)\b|[$€£]", message, re.I
    ):
        return CURRENCY_REDIRECT
    if not stateful and BOOKING.search(message):
        return BOOKING_REDIRECT
    if not stateful and re.search(
        r"\b(?:the (?:first|second|third|last) one|previous (?:search|results))\b", message, re.I
    ):
        return "Please provide the listing ID. I don't retain context from earlier requests."
    return None


def controlled_reply(kind: str, reply: str) -> str:
    if kind == "competitor" or COMPETITORS.search(reply):
        return COMPETITOR_REDIRECT
    return {
        "out_of_scope": SCOPE_REDIRECT,
        "booking_unavailable": BOOKING_REDIRECT,
        "currency_unsupported": CURRENCY_REDIRECT,
    }.get(kind, reply)
