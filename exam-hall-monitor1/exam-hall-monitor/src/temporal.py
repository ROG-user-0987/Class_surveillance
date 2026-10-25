"""Temporal logic: turn a noisy per-frame signal into clean events.

EventTracker follows ONE signal (e.g. phone confidence, or a posture score)
for every student (track_id) and decides when an "event" starts and ends:

    1. Smoothing: an exponential moving average (EMA) per track_id,
           ema = alpha * value + (1 - alpha) * previous_ema
       starting from 0, so a single bright frame cannot start an event.
    2. Hysteresis: an event STARTS when the EMA rises above `on_threshold`
       and only ENDS when it falls below `off_threshold` (off < on). The gap
       between the two stops events flickering on/off around one threshold.
    3. Gap tolerance: up to `max_gap_frames` frames in a row that are low
       (EMA below off) or missing (track not seen / not measurable) do not
       end an event. One more ends it.
    4. Minimum duration: an event is reported ("fires") once, when it has
       lasted `min_duration_s` (from its start to its latest active frame).

Every event remembers its start time, duration, and the mean / max of the
RAW (unsmoothed) signal over its active frames, plus when the max happened.

Keys are usually a track_id, but any hashable key works, e.g. a pair of
track_ids for a cue that involves two students.

This file has no dependencies beyond the standard library.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Event:
    """One ongoing event for one track_id (or other key)."""

    track_id: Hashable
    start: float                 # timestamp when the EMA first rose above on_threshold
    last_active: float           # latest timestamp the EMA was at/above off_threshold
    count: int = 0               # active frames so far
    total: float = 0.0           # sum of raw values over active frames (for the mean)
    max_value: float = 0.0       # highest raw value seen
    peak_time: float = 0.0       # when max_value was seen
    peak_details: dict[str, Any] = field(default_factory=dict)  # caller's extra info at the peak
    gap_frames: int = 0          # low/missing frames in a row right now
    fired: bool = False          # already reported once

    @property
    def duration(self) -> float:
        """Seconds from start to the latest active frame."""
        return self.last_active - self.start

    @property
    def mean_value(self) -> float:
        """Mean raw value over the active frames."""
        return self.total / self.count if self.count else 0.0


class EventTracker:
    """EMA + hysteresis + gap tolerance + minimum duration, per track_id."""

    def __init__(
        self,
        *,
        ema_alpha: float,
        on_threshold: float,
        off_threshold: float,
        max_gap_frames: int,
        min_duration_s: float,
    ) -> None:
        if not off_threshold < on_threshold:
            raise ValueError(f"off_threshold ({off_threshold}) must be below on_threshold ({on_threshold})")
        self.alpha = ema_alpha
        self.on = on_threshold
        self.off = off_threshold
        self.max_gap = max_gap_frames
        self.min_duration = min_duration_s

        self.ema: dict[Hashable, float] = {}      # smoothed signal per track_id
        self.missing: dict[Hashable, int] = {}    # frames in a row each track had no value
        self.events: dict[Hashable, Event] = {}   # ongoing events per track_id
        self.ended: list[Event] = []              # events that ended since pop_ended()

    def update(
        self,
        timestamp: float,
        values: dict[Hashable, float | None],
        details: dict[Hashable, dict[str, Any]] | None = None,
    ) -> list[Event]:
        """Feed ONE processed frame.

        Args:
            timestamp: seconds from the start of the video.
            values: track_id -> raw signal for this frame. None means the
                student is visible but the signal can't be measured (e.g.
                keypoints hidden). Tracks NOT in `values` count as missing.
                Both count towards the gap.
            details: optional track_id -> extra info (e.g. which wrist);
                kept for the frame where the raw value peaks.

        Returns:
            Events that reached min_duration_s on this frame. Each event is
            returned once; set `event.fired = False` to have it re-checked
            on its next active frame (e.g. when a cooldown blocked it).
        """
        details = details or {}
        fired: list[Event] = []

        for track_id in set(self.ema) | set(self.events) | set(values):
            value = values.get(track_id)
            if value is None:
                self._missing_frame(track_id)
                continue
            self.missing[track_id] = 0

            ema = self.alpha * value + (1 - self.alpha) * self.ema.get(track_id, 0.0)
            self.ema[track_id] = ema

            event = self.events.get(track_id)
            if event is None:
                if ema <= self.on:
                    continue  # nothing going on for this student
                event = self.events[track_id] = Event(track_id, start=timestamp, last_active=timestamp)

            if ema >= self.off:
                # Active frame: extend the event and update its statistics.
                event.last_active = timestamp
                event.gap_frames = 0
                event.count += 1
                event.total += value
                if event.count == 1 or value > event.max_value:
                    event.max_value = value
                    event.peak_time = timestamp
                    event.peak_details = dict(details.get(track_id, {}))
            else:
                # Low frame: tolerated up to max_gap_frames in a row.
                event.gap_frames += 1
                if event.gap_frames > self.max_gap:
                    self.ended.append(self.events.pop(track_id))
                    continue

            if not event.fired and event.duration >= self.min_duration:
                event.fired = True
                fired.append(event)

        return fired

    def _missing_frame(self, track_id: Hashable) -> None:
        """The track had no value this frame (not seen, or not measurable)."""
        self.missing[track_id] = self.missing.get(track_id, 0) + 1
        event = self.events.get(track_id)
        if event is not None:
            event.gap_frames += 1
            if event.gap_frames > self.max_gap:
                self.ended.append(self.events.pop(track_id))
        elif self.missing[track_id] > self.max_gap:
            # Gone for a while with no event: forget it to keep memory small.
            self.ema.pop(track_id, None)
            self.missing.pop(track_id, None)

    def ongoing(self) -> list[Event]:
        """Events that have lasted at least min_duration_s and haven't ended."""
        return [e for e in self.events.values() if e.duration >= self.min_duration]

    def pop_ended(self) -> list[Event]:
        """Events that ended since the last call (each returned once)."""
        ended, self.ended = self.ended, []
        return ended

    def end_all(self) -> list[Event]:
        """End every ongoing event (e.g. at the end of the video) and return them."""
        ended = list(self.events.values())
        self.events.clear()
        return ended
