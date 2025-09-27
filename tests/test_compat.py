import cantera as ct
import pytest

from hp_pox.compat import reconcile_feed_with_mechanism


def test_butane_isomers_lump_to_propane():
    gas = ct.Solution("data/gri30.yaml")
    feed = {"CH4": 0.9, "IC4H10": 0.06, "NC4H10": 0.04}
    reconciled = reconcile_feed_with_mechanism(gas, feed, policy="lump_to_propane")
    assert pytest.approx(0.9, rel=1e-12) == reconciled["CH4"]
    assert pytest.approx(0.1, rel=1e-12) == reconciled["C3H8"]
    assert set(reconciled) == {"CH4", "C3H8"}


def test_valid_feed_pass_through():
    gas = ct.Solution("data/gri30.yaml")
    feed = {"CH4": 0.7, "C2H6": 0.3}
    reconciled = reconcile_feed_with_mechanism(gas, feed)
    assert pytest.approx(feed["CH4"], rel=1e-12) == reconciled["CH4"]
    assert pytest.approx(feed["C2H6"], rel=1e-12) == reconciled["C2H6"]


def test_drop_and_renormalize_policy():
    gas = ct.Solution("data/gri30.yaml")
    feed = {"CH4": 0.7, "IC4H10": 0.3}
    reconciled = reconcile_feed_with_mechanism(gas, feed, policy="drop_and_renorm")
    assert pytest.approx(1.0, rel=1e-12) == reconciled["CH4"]
    assert "IC4H10" not in reconciled
