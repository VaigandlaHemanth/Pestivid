"""
Late-blight risk forecasting from weather.

The single highest value-per-line feature available to this project: it is a
*potato* app whose flagship class is Phytophthora, and late blight is the one
potato disease where weather predicts the outbreak days before symptoms appear.
Warning a farmer before the lesion exists is worth more than classifying it
after.

WHAT IS IMPLEMENTED

  Smith Period (Smith, 1956) -- the classical, well-documented criterion still
  used operationally by national advisory services:

      a Smith Period occurs when, on two consecutive days,
        - minimum temperature >= 10 C, and
        - at least 11 hours with relative humidity >= 90%

  It is deliberately chosen over a bespoke rule because it is published,
  auditable, and independently verifiable.

WHAT IS NOT IMPLEMENTED, AND WHY

  INDO-BLIGHTCAST (ICAR-CPRI Shimla) is the India-specific calibration, validated
  separately across the Indo-Gangetic plains, the plateau and the hills. Its
  exact coefficients are not encoded here because I could not verify them
  against the primary source, and inventing thresholds for a spray decision is
  the same class of error as inventing a pesticide dose. `advisory()` therefore
  reports Smith-Period risk and points at ICAR-CPRI for local calibration.

  If you obtain the published INDO-BLIGHTCAST thresholds, add them as a second
  scorer and mark it `calibrated_for="IN"`. The interface is already shaped for
  it.

References
  Smith, L.P. (1956), potato blight forecasting criteria.
  ICAR-CPRI Shimla — https://cpri.icar.gov.in/
  Spray-window guidance (rain-fastness, wind, Delta T) is standard extension
  practice; the thresholds below are widely published label guidance, not
  invented, but confirm them against your product label.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

# Late blight is favoured by cool, wet weather. Outside this band the pathogen
# is limited regardless of humidity.
TEMP_MIN_C = 10.0
TEMP_SPORULATION_MAX_C = 24.0
RH_THRESHOLD = 90.0
RH_HOURS_REQUIRED = 11
CONSECUTIVE_DAYS_REQUIRED = 2


@dataclass
class DayWeather:
    """One day of observations or forecast."""
    date: str
    temp_min_c: float
    temp_max_c: float
    hours_rh_above_90: int
    rainfall_mm: float = 0.0
    wind_kmh: Optional[float] = None
    temp_c: Optional[float] = None          # for Delta T / spray window
    rh_percent: Optional[float] = None

    def is_smith_day(self) -> bool:
        return (self.temp_min_c >= TEMP_MIN_C
                and self.hours_rh_above_90 >= RH_HOURS_REQUIRED)


@dataclass
class RiskAssessment:
    level: str                    # low | moderate | high | very_high
    smith_periods: int
    smith_days: list = field(default_factory=list)
    reasons: list = field(default_factory=list)
    action: str = ""
    calibrated_for: Optional[str] = None
    caveat: str = ""


def assess(days: Iterable[DayWeather]) -> RiskAssessment:
    """Score a run of days for late-blight risk."""
    days = list(days)
    if not days:
        return RiskAssessment(level="unknown", smith_periods=0,
                              reasons=["No weather data supplied."],
                              action="Cannot assess risk without weather data.")

    flags = [d.is_smith_day() for d in days]
    smith_days = [d.date for d, f in zip(days, flags) if f]

    # count runs of >= CONSECUTIVE_DAYS_REQUIRED
    periods, run = 0, 0
    for f in flags:
        run = run + 1 if f else 0
        if run == CONSECUTIVE_DAYS_REQUIRED:
            periods += 1
        elif run > CONSECUTIVE_DAYS_REQUIRED:
            pass                     # one continuous spell counts once
    reasons: list[str] = []

    if periods:
        reasons.append(
            f"{periods} Smith Period(s): {CONSECUTIVE_DAYS_REQUIRED}+ consecutive days "
            f"with min temp >= {TEMP_MIN_C:g} C and >= {RH_HOURS_REQUIRED} h at "
            f"RH >= {RH_THRESHOLD:g}%.")
    elif smith_days:
        reasons.append(f"{len(smith_days)} qualifying day(s) but no consecutive pair yet.")
    else:
        reasons.append("No day met both the temperature and humidity criteria.")

    wet = sum(1 for d in days if d.rainfall_mm >= 1.0)
    if wet:
        reasons.append(f"{wet} day(s) with measurable rain — free moisture aids infection.")

    in_band = [d for d in days if d.temp_min_c >= TEMP_MIN_C
               and d.temp_max_c <= TEMP_SPORULATION_MAX_C]
    if in_band:
        reasons.append(f"{len(in_band)} day(s) fully inside the {TEMP_MIN_C:g}-"
                       f"{TEMP_SPORULATION_MAX_C:g} C sporulation band.")
    if all(d.temp_max_c > TEMP_SPORULATION_MAX_C for d in days):
        reasons.append(f"Every day exceeded {TEMP_SPORULATION_MAX_C:g} C — hot weather "
                       "suppresses sporulation, which lowers risk.")

    if periods >= 2:
        level, action = "very_high", (
            "Repeated Smith Periods. Assume infection pressure is present. If a "
            "protectant programme is not already running, start one now, and "
            "scout the lower canopy daily.")
    elif periods == 1:
        level, action = "high", (
            "A Smith Period has occurred. Apply or renew a protectant fungicide "
            "before the next wet spell, and scout daily. A protectant works "
            "before infection, not after.")
    elif len(smith_days) >= 2:
        level, action = "moderate", (
            "Conditions are becoming favourable. Check the forecast and be ready "
            "to spray before the next rain, rather than after it.")
    else:
        level, action = "low", (
            "Conditions are currently unfavourable. Keep scouting — this is a "
            "weather assessment, not an inspection of your field.")

    return RiskAssessment(
        level=level, smith_periods=periods, smith_days=smith_days,
        reasons=reasons, action=action, calibrated_for=None,
        caveat=("Smith Period criteria (Smith 1956), not the India-specific "
                "INDO-BLIGHTCAST calibration. For the Indo-Gangetic plains, the "
                "plateau or the hills, confirm local thresholds with ICAR-CPRI "
                "Shimla or your State Agricultural University before acting."))


# ── spray window ────────────────────────────────────────────────────────────
# Telling a farmer WHICH fungicide is half the advice; WHEN decides whether it
# works or washes off. These are widely published label thresholds -- confirm
# against your own product label, which is authoritative.
RAINFAST_HOURS = 2
WIND_MIN_KMH, WIND_MAX_KMH = 3.0, 15.0
DELTA_T_MIN, DELTA_T_MAX = 2.0, 8.0


def wet_bulb_depression(temp_c: float, rh_percent: float) -> float:
    """Delta T approximation (Stull 2011 wet-bulb estimate).

    Delta T is the spread between dry- and wet-bulb temperature. Too low and the
    spray sits and runs; too high and fine droplets evaporate before landing.
    """
    import math
    rh = max(min(rh_percent, 100.0), 1.0)
    tw = (temp_c * math.atan(0.151977 * (rh + 8.313659) ** 0.5)
          + math.atan(temp_c + rh) - math.atan(rh - 1.676331)
          + 0.00391838 * rh ** 1.5 * math.atan(0.023101 * rh)
          - 4.686035)
    return round(temp_c - tw, 2)


def spray_window(day: DayWeather, rain_expected_within_hours: Optional[float] = None) -> dict:
    """Is now a sensible time to spray? Returns blockers and cautions."""
    blockers, cautions, checks = [], [], []

    if rain_expected_within_hours is not None:
        if rain_expected_within_hours < RAINFAST_HOURS:
            blockers.append(
                f"Rain expected in {rain_expected_within_hours:g} h. Most products need "
                f"~{RAINFAST_HOURS} h to become rainfast — spraying now likely washes off. "
                "Check your label for its rainfastness period.")
        else:
            checks.append(f"Rain not expected for {rain_expected_within_hours:g} h "
                          f"(>= {RAINFAST_HOURS} h rainfast window).")

    if day.wind_kmh is not None:
        if day.wind_kmh < WIND_MIN_KMH:
            cautions.append(
                f"Wind {day.wind_kmh:g} km/h is very low. Dead-calm conditions often mean "
                "a temperature inversion, which lets fine droplets drift a long way. "
                "Prefer a light steady breeze.")
        elif day.wind_kmh > WIND_MAX_KMH:
            blockers.append(f"Wind {day.wind_kmh:g} km/h exceeds "
                            f"{WIND_MAX_KMH:g} km/h — unacceptable drift risk.")
        else:
            checks.append(f"Wind {day.wind_kmh:g} km/h is in the "
                          f"{WIND_MIN_KMH:g}-{WIND_MAX_KMH:g} km/h band.")

    delta_t = None
    if day.temp_c is not None and day.rh_percent is not None:
        delta_t = wet_bulb_depression(day.temp_c, day.rh_percent)
        if delta_t < DELTA_T_MIN:
            cautions.append(f"Delta T {delta_t} is below {DELTA_T_MIN:g}: very slow drying, "
                            "so spray may run off rather than deposit.")
        elif delta_t > DELTA_T_MAX:
            blockers.append(f"Delta T {delta_t} exceeds {DELTA_T_MAX:g}: fine droplets will "
                            "evaporate before reaching the leaf. Spray early morning or evening.")
        else:
            checks.append(f"Delta T {delta_t} is in the "
                          f"{DELTA_T_MIN:g}-{DELTA_T_MAX:g} band.")

    return {
        "suitable": not blockers,
        "blockers": blockers,
        "cautions": cautions,
        "checks_passed": checks,
        "delta_t": delta_t,
        "note": ("Timing guidance only. The product label is authoritative for "
                 "rainfastness, wind limits and re-entry, and it is the legal "
                 "document. Read it."),
    }


def advisory(days: Iterable[DayWeather],
             rain_expected_within_hours: Optional[float] = None) -> dict:
    """Combined risk + spray-window advisory, ready to serialise."""
    days = list(days)
    risk = assess(days)
    window = spray_window(days[-1], rain_expected_within_hours) if days else None
    return {
        "risk_level": risk.level,
        "smith_periods": risk.smith_periods,
        "smith_days": risk.smith_days,
        "reasons": risk.reasons,
        "action": risk.action,
        "spray_window": window,
        "model": "Smith Period (Smith 1956)",
        "calibrated_for": risk.calibrated_for,
        "caveat": risk.caveat,
        "days_assessed": len(days),
    }
