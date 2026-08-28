"""
Local-time helpers.

Every observation in this project is on Munich's wall clock: pollen counts are
converted from epoch with ``tz="Europe/Berlin"`` and then stripped to naive
timestamps (see ``pollen.py``), and Open-Meteo is queried with
``timezone=Europe/Berlin``. The history index is therefore naive local time.

The pipeline runs on UTC machines, so ``date.today()`` and
``pd.Timestamp.now()`` answer on the wrong clock — one to two hours behind the
data, depending on daylight saving. Anchoring windows with those silently
discards the most recent weather window, and rolls the date over a day early on
the late-evening run. Use these helpers instead wherever "now" or "today" is
compared against the data.

``forecast.json``'s ``generated`` field deliberately stays UTC: it is an
absolute instant marked with a ``Z`` suffix, not a Munich calendar position.
"""

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd


LOCAL_TZ = ZoneInfo("Europe/Berlin")


def local_today() -> date:
    """Today's calendar date in Munich."""
    return datetime.now(LOCAL_TZ).date()


def local_now() -> pd.Timestamp:
    """
    Munich wall-clock time as a naive Timestamp.

    Naive on purpose: it is compared against the history/weather index, which is
    naive local time.
    """
    return pd.Timestamp(datetime.now(LOCAL_TZ).replace(tzinfo=None))
