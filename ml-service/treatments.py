"""
Treatment guidance from a curated, versioned, reviewable table.

Replaces the fine-tuned Flan-T5 recommender. That model was trained on 7
examples for 100 optimizer steps and its documented loss curve ends at 3.2368 --
perplexity ~25 for a model whose only job was verbatim recall -- so it never
learned them. The brittle quality heuristics downstream always fired and a
hardcoded dictionary supplied the text anyway. A table is strictly better here:
exact, versioned, auditable, 293 MB smaller, and incapable of inventing a
chemical or a dose.

THE SAFETY INVARIANT
    A dose, pre-harvest interval or application rate is emitted ONLY for an
    entry whose `status == "reviewed"`. Everything else surfaces the active
    ingredient name plus a pointer to CIBRC, never a number.

    This is enforced here, in one place, rather than trusted to callers. See
    _sanitise_chemical(). The point is that a plausible-looking dose for an
    agrochemical is the highest real-world-harm output this codebase can
    produce, so it must be impossible to emit one by accident.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_DATA_PATH = Path(__file__).with_name("treatments.json")
_DOSE_FIELDS = ("dose", "phi_days", "rate", "application_rate", "dilution", "interval_days")
_REVIEWED = "reviewed"

_cache: Optional[dict] = None


def _load() -> dict:
    global _cache
    if _cache is None:
        _cache = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
        n_dis = len(_cache.get("diseases", {}))
        chem = sum(len(d.get("chemical", [])) for d in _cache["diseases"].values())
        reviewed = sum(1 for d in _cache["diseases"].values()
                       for c in d.get("chemical", []) if c.get("status") == _REVIEWED)
        logger.info("treatments.json rev %s: %d conditions, %d chemical entries, %d reviewed",
                    _cache.get("revised"), n_dis, chem, reviewed)
        if chem and not reviewed:
            logger.warning(
                "No chemical entry is marked 'reviewed', so no dose or PHI will be "
                "served. Populate them from %s and set status='reviewed' with a "
                "checked_on date.",
                _cache["sources"][0]["url"] if _cache.get("sources") else "CIBRC")
    return _cache


def _source_url(source_id: str | None) -> str | None:
    if not source_id:
        return None
    for s in _load().get("sources", []):
        if s.get("id") == source_id:
            return s.get("url")
    return None


def _sanitise_chemical(entry: dict) -> dict:
    """Strip every numeric guidance field unless the entry has been reviewed.

    Defence in depth: even if someone later types a dose into treatments.json
    without setting status='reviewed', it does not reach a farmer.
    """
    out: dict[str, Any] = {
        "active_ingredient": entry.get("active_ingredient"),
        "type": entry.get("type"),
        "role": entry.get("role"),
        "ppe": entry.get("ppe", []),
        "resistance_note": entry.get("resistance_note"),
        "status": entry.get("status", "needs_verification"),
        "verify_at": _source_url(entry.get("verify_at")),
    }

    if entry.get("status") == _REVIEWED:
        for f in _DOSE_FIELDS:
            if entry.get(f) is not None:
                out[f] = entry[f]
        out["checked_on"] = entry.get("checked_on")
        out["dose_disclaimer"] = (
            "Verified against the cited source on the date shown. The product "
            "label remains authoritative — read it before mixing.")
    else:
        out["dose"] = None
        out["dose_withheld"] = (
            "No dose or pre-harvest interval is shown because this entry has not "
            "been verified against the CIBRC register. Get the rate from the "
            "product label and confirm it is registered for potato in your "
            "district. Do not guess.")
        dropped = [f for f in _DOSE_FIELDS if entry.get(f) is not None]
        if dropped:
            logger.warning("Withheld unreviewed numeric fields %s for '%s'",
                           dropped, entry.get("active_ingredient"))
    return out


# The trained model's class name for late blight is "Phytopthora" -- the Kaggle
# dataset's directory is spelled without the second 'h', and the label set is
# derived from those directory names. treatments.json uses the correct spelling.
#
# Without this map the lookup missed, and the platform diagnosed LATE BLIGHT
# correctly and then showed no treatment at all -- for the single most destructive
# potato disease there is, the one where a few days of delay costs the crop.
# Normalising here rather than renaming either side keeps the model's labels
# reproducible against the published dataset and the treatment table spelled
# correctly for the agronomist who reviews it.
_CLASS_ALIASES = {
    "phytopthora": "Phytophthora",     # dataset spelling -> correct spelling
    "phytophtora": "Phytophthora",     # another common transposition
    "late_blight": "Phytophthora",
    "late blight": "Phytophthora",
    "early_blight": "Fungi",
    "healthy": "Healthy",
}


def _canonical(disease: str) -> str:
    """Map a model class name onto a treatment-table key."""
    if not disease:
        return ""
    raw = str(disease).strip()
    data = _load()
    if raw in data["diseases"]:
        return raw
    alias = _CLASS_ALIASES.get(raw.lower())
    if alias and alias in data["diseases"]:
        return alias
    # Case-insensitive last resort, so 'fungi' finds 'Fungi'.
    for key in data["diseases"]:
        if key.lower() == raw.lower():
            return key
    return raw


def get_treatment(disease: str) -> dict:
    """Full structured guidance for one class name. Never raises."""
    data = _load()
    d = data["diseases"].get(_canonical(disease))

    if d is None:
        return {
            "disease": disease,
            "known": False,
            "message": ("No curated guidance for this condition. Consult a "
                        "licensed agronomist or your State Agricultural "
                        "University extension officer."),
            "disclaimer": data["global_disclaimer"],
            "table_revision": data["revised"],
        }

    return {
        "disease": disease,
        "known": True,
        "is_disease": d.get("is_disease", True),
        "common_name": d.get("common_name"),
        "pathogen": d.get("pathogen"),
        "severity": d.get("severity"),
        "summary": d.get("summary"),
        # Cultural practice carries no dose and no legal exposure, so it is
        # always safe to show -- and for several of these conditions
        # (bacterial, viral, nematode) it is the only thing that actually works.
        "cultural_practices": d.get("cultural", []),
        "chemical_options": [_sanitise_chemical(c) for c in d.get("chemical", [])],
        "do_not": d.get("do_not", []),
        "forecasting": d.get("forecasting"),
        "escalate_if": d.get("escalate_if"),
        "disclaimer": data["global_disclaimer"],
        "jurisdiction": data.get("jurisdiction"),
        "table_revision": data["revised"],
        "sources": [{"name": s["name"], "url": s["url"]} for s in data.get("sources", [])],
    }


def get_treatment_text(disease: str) -> str:
    """Flat prose, for callers that want one string (the old API shape)."""
    t = get_treatment(disease)
    if not t["known"]:
        return t["message"]

    parts: list[str] = []
    if t.get("common_name"):
        head = t["common_name"]
        if t.get("pathogen") and t.get("is_disease"):
            head += f" ({t['pathogen']})"
        parts.append(head + ".")
    if t.get("summary"):
        parts.append(t["summary"])

    if t["cultural_practices"]:
        label = "What to do" if not t.get("is_disease") else "Management"
        parts.append(f"\n{label}:")
        parts += [f"- {c}" for c in t["cultural_practices"]]

    if t["chemical_options"]:
        parts.append("\nChemical options (verify before use):")
        for c in t["chemical_options"]:
            line = f"- {c['active_ingredient']}"
            if c.get("role"):
                line += f" — {c['role']}"
            if c.get("status") == _REVIEWED and c.get("dose"):
                line += f". Dose: {c['dose']}"
                if c.get("phi_days"):
                    line += f". Pre-harvest interval: {c['phi_days']} days"
            else:
                line += ". Dose not shown: unverified against the CIBRC register."
            parts.append(line)
            if c.get("resistance_note"):
                parts.append(f"    Note: {c['resistance_note']}")

    if t.get("do_not"):
        parts.append("\nDo not:")
        parts += [f"- {x}" for x in t["do_not"]]

    if t.get("forecasting"):
        parts.append(f"\n{t['forecasting']}")
    if t.get("escalate_if"):
        parts.append(f"\nSeek expert help if: {t['escalate_if']}")

    parts.append(f"\n{t['disclaimer']}")
    return "\n".join(parts)


def audit() -> dict:
    """Coverage report — which entries still need verification."""
    data = _load()
    rows = []
    for name, d in data["diseases"].items():
        chems = d.get("chemical", [])
        rows.append({
            "condition": name,
            "cultural_practices": len(d.get("cultural", [])),
            "chemical_entries": len(chems),
            "reviewed": sum(1 for c in chems if c.get("status") == _REVIEWED),
            "needs_verification": sum(1 for c in chems if c.get("status") != _REVIEWED),
        })
    return {
        "revision": data["revised"],
        "jurisdiction": data.get("jurisdiction"),
        "conditions": rows,
        "total_chemical_entries": sum(r["chemical_entries"] for r in rows),
        "total_reviewed": sum(r["reviewed"] for r in rows),
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--audit":
        print(json.dumps(audit(), indent=2))
    else:
        for name in _load()["diseases"]:
            print("=" * 72)
            print(get_treatment_text(name))
            print()
