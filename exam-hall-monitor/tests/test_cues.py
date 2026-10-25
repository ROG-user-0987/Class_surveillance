"""Tests for src/cues.py and the Phase 3 logic in src/risk.py (grey state,
combined alerts). Run: python tests/test_cues.py  (or python -m pytest tests)"""

from __future__ import annotations

from helpers import make_cfg, make_record, run_all, run_scorer

from src.cues import (
    hands_hidden_signal,
    head_direction,
    low_visibility_signal,
    passing_pairs,
    peek_target,
)

CFG = make_cfg()


def test_head_direction_from_ears() -> None:
    assert head_direction(make_record(ear_turn=0.0), CFG)[1] < 0.01        # facing the camera
    side, strength = head_direction(make_record(ear_turn=0.8), CFG)
    assert side == 1 and abs(strength - 0.8) < 1e-3                        # turned to image right
    assert head_direction(make_record(ear_turn=-0.8), CFG)[0] == -1


def test_peek_hits_neighbour_only_on_the_turned_side() -> None:
    me = make_record(1, x=200, ear_turn=0.8)            # turned to image right
    right_neighbour = make_record(2, x=380)             # box starts 380 - 275(nose) = ~105 px away
    left_neighbour = make_record(3, x=20)
    signal, target = peek_target(me, [me, right_neighbour, left_neighbour], CFG)
    assert signal == 1.0 and target == 2
    far = make_record(4, x=700)                          # beyond 2 shoulder widths (200 px)
    assert peek_target(me, [me, far], CFG) == (0.0, None)
    facing = make_record(1, x=200, ear_turn=0.1)
    assert peek_target(facing, [facing, right_neighbour], CFG) == (0.0, None)


def test_passing_pairs_close_wrists() -> None:
    a = make_record(1, x=100)                            # right wrist at x = 215
    near = make_record(2, x=150)                         # left wrist at x = 185: 30 px apart
    far = make_record(3, x=600)
    signals = passing_pairs([a, near, far], CFG)
    assert signals[(1, 2)] == 1.0 and signals[(1, 3)] == 0.0
    hidden = make_record(4, x=160, wrists=False)
    assert (1, 4) not in passing_pairs([a, hidden], CFG)  # no visible wrist = not measurable


def test_hands_hidden_needs_head_down_and_hidden_wrists() -> None:
    assert hands_hidden_signal(make_record(pitch=-0.1, wrists=False), CFG)[0] == 1.0
    assert hands_hidden_signal(make_record(pitch=-0.1, wrists=True), CFG)[0] == 0.0
    assert hands_hidden_signal(make_record(pitch=-0.5, wrists=False), CFG)[0] == 0.0
    # With a baseline of -0.5, -0.2 is 0.3 lower than normal -> head down.
    assert hands_hidden_signal(make_record(pitch=-0.2, wrists=False), CFG, pitch_baseline=-0.5)[0] == 1.0


def test_low_visibility_counts_keypoints() -> None:
    assert low_visibility_signal(make_record(), CFG)[0] == 0.0
    signal, info = low_visibility_signal(make_record(visible={0, 5, 6}), CFG)
    assert signal == 1.0 and info["visible_keypoints"] == 3


def test_neighbour_peek_alert_names_the_neighbour() -> None:
    cfg = make_cfg(neighbour_peek=True, evidence_alerts=True)
    frames = [[make_record(1, x=200, ear_turn=0.8, t=i / 10), make_record(7, x=380, t=i / 10)]
              for i in range(40)]
    alerts = [a for a in run_scorer(cfg, frames) if a.behaviour == "neighbour_peek"]
    assert len(alerts) == 1 and alerts[0].tier == "orange"
    assert "student #7" in alerts[0].reason


def test_grey_pauses_posture_alerts_but_not_phone() -> None:
    cfg = make_cfg(low_visibility=True)
    # Only nose + shoulders visible (3 < 6): head turn measurable, but "cannot assess".
    frames = [[make_record(1, t=i / 10, turn=0.3, visible={0, 5, 6}, phone_conf=0.8)] for i in range(50)]
    alerts = run_scorer(cfg, frames)
    behaviours = sorted(a.behaviour for a in alerts)
    assert behaviours == ["low_visibility", "phone"], behaviours
    grey = next(a for a in alerts if a.tier == "grey")
    assert grey.timestamp < 0.5  # logged with the start of the grey period


def test_two_orange_cues_become_one_combined_alert() -> None:
    frames = [[make_record(1, t=i / 10, turn=0.3, lean_dx=-120)] for i in range(40)]  # turned AND leaning
    separate = run_scorer(make_cfg(), frames)
    assert sorted(a.behaviour for a in separate) == ["head_turn", "lean"]
    combined = run_scorer(make_cfg(cue_fusion=True), frames)
    assert len(combined) == 1 and combined[0].behaviour == "combined" and combined[0].tier == "orange"
    assert "Head turned" in combined[0].reason and "Leaning" in combined[0].reason


def test_phone_alerts_are_never_held_by_fusion() -> None:
    frames = [[make_record(1, t=i / 10, phone_conf=0.8)] for i in range(20)]
    alerts = run_scorer(make_cfg(cue_fusion=True), frames)
    assert [a.behaviour for a in alerts] == ["phone"] and alerts[0].tier == "red"


if __name__ == "__main__":
    run_all(globals())
