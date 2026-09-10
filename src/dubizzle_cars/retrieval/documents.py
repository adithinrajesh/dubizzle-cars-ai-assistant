"""Canonical source-only documents. Bump the version whenever construction changes."""

from ..models import Car

DOCUMENT_VERSION = "1"


def build_semantic_document(car: Car) -> str:
    heading = " ".join(str(v) for v in (car.year, car.make, car.model) if v is not None)
    sections = [heading]
    if car.trim:
        sections.append(f"Trim: {car.trim}")
    sections.extend(text for text in (car.title, car.description) if text)
    return "\n\n".join(section for section in sections if section)
