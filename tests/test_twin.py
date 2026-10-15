"""Digital twin: network, inventory and generators."""
from datetime import date

import numpy as np

from railsentinel.twin import generators, inventory
from railsentinel.twin.network import load_network
from railsentinel.twin.seasons import season_of
from railsentinel.twin.world import WorldParams


def test_network_uses_real_stations_and_synthetic_trains():
    net = load_network()
    assert len(net.stations) == 70
    assert net.stations["PUNE"].city == "Pune"
    assert all(t.startswith("RS") for t in net.trains)  # never collides with real train numbers
    assert set(net.resolve_place("Delhi")) == {"NDLS", "NZM", "DEE"}
    assert net.resolve_place("bombay") == net.resolve_place("Mumbai")


def test_pune_delhi_has_direct_trains_with_plausible_distance():
    net = load_network()
    segs = net.direct_segments(net.resolve_place("Pune"), net.resolve_place("Delhi"))
    assert len(segs) >= 5
    for seg in segs:
        assert 1300 < seg.distance_km < 1800
        assert seg.duration_min > 0


def test_inventory_snapshot_is_deterministic_and_consistent():
    net = load_network()
    seg = net.direct_segments(["PUNE"], ["NZM"])[0]
    w = WorldParams()
    a = inventory.snapshot(net, seg, "3A", date(2026, 11, 20), date(2026, 11, 1), w)
    b = inventory.snapshot(net, seg, "3A", date(2026, 11, 20), date(2026, 11, 1), w)
    assert a == b
    assert a["status"] in {"AVAILABLE", "RAC", "WL"}
    assert inventory.snapshot(net, seg, "3A", date(2027, 6, 1), date(2026, 11, 1), w)["status"] == "NOT_OPEN"


def test_flexi_multiplier_caps_at_one_and_a_half():
    assert inventory.flexi_multiplier(0.05) == 1.0
    assert inventory.flexi_multiplier(0.25) == 1.2
    assert inventory.flexi_multiplier(0.99) == 1.5


def test_cancellation_shift_lowers_confirmations():
    net = load_network()
    base = generators.confirmation_samples(net, 1500, np.random.default_rng(1), WorldParams())
    shifted = generators.confirmation_samples(net, 1500, np.random.default_rng(1), WorldParams(cancel_mult=0.45))
    assert (shifted.final_status == "CNF").mean() < (base.final_status == "CNF").mean() - 0.1


def test_fog_increases_delays_and_stealth_bots_look_human():
    net = load_network()
    clear = generators.delay_samples(net, 1500, np.random.default_rng(2), WorldParams())
    fog = generators.delay_samples(net, 1500, np.random.default_rng(2),
                                   WorldParams(weather_shift=0.4, weather_effect_mult=2.2))
    assert fog.delay_min.mean() > 1.5 * clear.delay_min.mean()
    loud = generators.fraud_sessions(3000, np.random.default_rng(3), WorldParams(bot_rate=0.2))
    stealth = generators.fraud_sessions(3000, np.random.default_rng(3), WorldParams(bot_rate=0.2, stealth_bots=True))
    assert stealth[stealth.label == "bot"].form_fill_seconds.median() > 5 * loud[loud.label == "bot"].form_fill_seconds.median()


def test_seasons():
    assert season_of(date(2026, 10, 25)) == "festival"
    assert season_of(date(2026, 5, 20)) == "summer_vacation"
    assert season_of(date(2026, 3, 3)) == "normal"
