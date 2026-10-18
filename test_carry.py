"""Synthetic accounting, funding-timing and collateral checks."""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from carry import simulate


HOUR = 3_600_000
DAY = 24 * HOUR


def timestamp(value):
    return int(np.datetime64(value, "ms").astype(np.int64))


def market_sample(end="2024-01-03T01:00"):
    times = np.arange(timestamp("2024-01-01T00:00"), timestamp(end) + HOUR, HOUR)
    return {"time": times, **{key: np.full(len(times), 100.) for key in
                             ("spot_open", "perp_open", "mark_open", "mark_high")}}


def funding_sample(history_rate=0., events=()):
    start = timestamp("2024-01-01T00:00")
    history = [(int(time), history_rate, 100.)
               for time in np.arange(start - 7 * DAY, start, 8 * HOUR)]
    rows = sorted(history + list(events))
    return {"time": np.array([row[0] for row in rows], dtype=np.int64),
            "rate": np.array([row[1] for row in rows], dtype=float),
            "mark": np.array([row[2] for row in rows], dtype=float)}


class CarryTests(unittest.TestCase):
    def test_entry_quantity_and_flat_price_round_trip_fees(self):
        result = simulate(market_sample(), funding_sample())
        quantity = 10_000 / (100 * 1.5 + .0012 * 100 + .0007 * 100)
        assert_allclose(result["q"][1:-1], quantity)
        assert_allclose(result["q"][[0, -1]], 0.)
        assert_allclose(result["costs"][[1, -1]], quantity * .19)
        assert_allclose(result["costs"].sum(), 2 * quantity * .19)
        assert_allclose(result["equity"][-1], 10_000 - 2 * quantity * .19)
        assert_allclose(result["delta_q"][[1, -1]], [quantity, -quantity])

    def test_settled_funding_uses_held_quantity_and_settlement_mark(self):
        start = timestamp("2024-01-01T00:00")
        events = [(start + 8 * HOUR, .001, 110.),
                  (start + 16 * HOUR + 1, -.0005, 90.)]
        result = simulate(market_sample(), funding_sample(events=events))
        quantity = result["q"][1]
        expected = quantity * (110 * .001 - 90 * .0005)
        assert_allclose(result["funding_pnl"].sum(), expected)
        assert_allclose(result["equity"][-1], 10_000 + expected - result["costs"].sum())

    def test_hedge_matches_btc_quantity_rather_than_dollar_notional(self):
        market = market_sample()
        market["perp_open"][:] = market["mark_open"][:] = market["mark_high"][:] = 120
        for field in ("spot_open", "perp_open", "mark_open", "mark_high"):
            market[field][8:] += 10
        result = simulate(market, funding_sample())
        quantity = result["q"][1]
        assert_allclose(quantity, 10_000 / (150 + .0012 * 100 + .0007 * 120))
        assert_allclose(result["spot_pnl"].sum(), 10 * quantity)
        assert_allclose(result["perp_pnl"].sum(), -10 * quantity)
        assert_allclose(result["spot_pnl"] + result["perp_pnl"], 0.)

    def test_month_start_funding_does_not_reward_a_later_entry(self):
        start = timestamp("2024-01-01T00:00")
        for jitter in (0, 1, 321):
            with self.subTest(jitter=jitter):
                result = simulate(market_sample(), funding_sample(events=[(start + jitter, .1, 100.)]))
                assert_allclose(result["funding_pnl"], 0.)
                self.assertGreater(result["q"][1], 0)

    def test_funding_milliseconds_are_booked_in_the_next_hour_snapshot(self):
        start = timestamp("2024-01-01T00:00")
        for jitter in (0, 1, 999):
            with self.subTest(jitter=jitter):
                result = simulate(market_sample(), funding_sample(
                    events=[(start + 8 * HOUR + jitter, .001, 100.)]))
                assert_allclose(result["funding_pnl"][:9], 0.)
                assert_allclose(result["funding_pnl"][9], result["q"][1] * .1)
                assert_allclose(result["funding_pnl"][10:], 0.)

    def test_screen_excludes_current_and_future_funding(self):
        start = timestamp("2024-01-01T00:00")
        events = [(start, .5, 100.), (start + 8 * HOUR, .5, 100.)]
        result = simulate(market_sample(), funding_sample(history_rate=-.001, events=events),
                          policy="screen")
        assert_allclose(result["q"], 0.)
        assert_allclose(result["equity"], 10_000.)
        assert_allclose(result["costs"], 0.)

    def test_screen_choice_is_unchanged_by_unseen_future_rates(self):
        start = timestamp("2024-01-01T00:00")
        outputs = [simulate(market_sample(), funding_sample(history_rate=.001,
                            events=[(start + 8 * HOUR, rate, 100.)]), policy="screen")
                   for rate in (-.01, .01)]
        assert_array_equal(outputs[0]["q"], outputs[1]["q"])
        self.assertGreater(outputs[0]["q"][1], 0)
        self.assertNotEqual(outputs[0]["equity"][-1], outputs[1]["equity"][-1])

    def test_all_incremental_cash_flows_reconcile_to_economic_equity(self):
        market = market_sample()
        spot = 100 + np.arange(len(market["time"])) * .15
        market["spot_open"] = spot
        market["perp_open"] = spot + .2 * np.sin(np.arange(len(spot)))
        market["mark_open"] = market["perp_open"] + .05
        market["mark_high"] = market["mark_open"] + .1
        start = market["time"][0]
        funding = funding_sample(events=[(start + 8 * HOUR, .001, 102.),
                                         (start + 24 * HOUR + 2, -.0003, 104.)])
        result = simulate(market, funding)
        flows = result["spot_pnl"] + result["perp_pnl"] + result["funding_pnl"] - result["costs"]
        assert_allclose(result["equity"], 10_000 + np.cumsum(flows))
        assert_allclose(np.diff(result["q"], prepend=0), result["delta_q"])

    def test_neutral_economic_pnl_can_coexist_with_a_margin_breach(self):
        market = market_sample()
        for field in ("spot_open", "perp_open", "mark_open", "mark_high"):
            market[field][8:] = 200.
        result = simulate(market, funding_sample())
        assert_allclose(result["spot_pnl"] + result["perp_pnl"], 0.)
        assert_allclose(result["equity"][9], 10_000 - result["costs"][1])
        self.assertLess(result["margin_equity"][9], 0)
        self.assertTrue(np.any(result["margin_flag"]))
        assert_allclose(result["q"][1:-1], result["q"][1])

    def test_mark_prices_affect_collateral_without_creating_trade_price_pnl(self):
        market = market_sample()
        market["mark_open"][8:12] = 130.
        market["mark_high"][8:12] = 160.
        result = simulate(market, funding_sample())
        quantity = result["q"][1]
        assert_allclose(result["spot_pnl"], 0.)
        assert_allclose(result["perp_pnl"], 0.)
        assert_allclose(result["margin_equity"][9], quantity * (50 - 30))
        assert_allclose(result["collateral_ratio"][9], 20 / 130)
        self.assertTrue(np.any(result["margin_flag"]))

    def test_positive_funding_does_not_hide_within_hour_margin_stress(self):
        market = market_sample()
        market["mark_high"][8] = 160.
        funding = funding_sample(events=[(market["time"][8] + 1, .5, 100.)])
        result = simulate(market, funding)
        self.assertGreater(result["margin_equity"][9], result["margin_equity"][7])
        self.assertTrue(np.any(result["margin_flag"]))

    def test_negative_funding_is_included_in_within_hour_margin_stress(self):
        market = market_sample()
        market["mark_high"][8] = 140.
        without = simulate(market, funding_sample())
        with_negative = simulate(market, funding_sample(
            events=[(market["time"][8] + 1, -.1, 100.)]))
        self.assertFalse(np.any(without["margin_flag"]))
        self.assertTrue(np.any(with_negative["margin_flag"]))

    def test_settlement_mark_can_raise_the_stress_peak_above_hourly_high(self):
        market = market_sample()
        funding = funding_sample(events=[(market["time"][8] + 1, 0., 200.)])
        result = simulate(market, funding)
        assert_allclose(result["funding_pnl"], 0.)
        assert_allclose(result["perp_pnl"], 0.)
        self.assertTrue(np.any(result["margin_flag"]))

    def test_buffered_reduction_is_frozen_then_executed_one_hour_later(self):
        market = market_sample()
        for field in ("spot_open", "perp_open", "mark_open", "mark_high"):
            market[field][23:] = 130.
            market[field][25:] = 135.
        result = simulate(market, funding_sample(history_rate=.001), policy="buffered")
        old = result["q"][1]
        equity_at_signal = 10_000 - result["costs"][1]
        fee_per_btc = (.0012 + .0007) * 130
        target = (equity_at_signal - fee_per_btc * old) / (1.5 * 130 - fee_per_btc)
        assert_allclose(result["q"][24], old)
        assert_allclose(result["delta_q"][24], 0.)
        assert_allclose(result["q"][25], target)
        self.assertLess(target, old)
        assert_allclose(result["costs"][25], (old - target) * (.0012 + .0007) * 135)
        flows = result["spot_pnl"] + result["perp_pnl"] + result["funding_pnl"] - result["costs"]
        assert_allclose(result["equity"], 10_000 + np.cumsum(flows))

    def test_buffered_policy_does_not_resize_a_healthy_account(self):
        result = simulate(market_sample(), funding_sample(history_rate=.001), policy="buffered")
        assert_allclose(result["q"][1:-1], result["q"][1])
        assert_allclose(result["delta_q"][2:-1], 0.)
        self.assertFalse(np.any(result["margin_flag"]))

    def test_final_month_boundary_closes_without_reopening(self):
        result = simulate(market_sample("2024-02-01T01:00"), funding_sample())
        quantity = result["q"][1]
        assert_allclose(result["q"][-1], 0.)
        assert_allclose(result["delta_q"][-1], -quantity)
        assert_allclose(result["costs"][-1], quantity * .19)
        assert_allclose(result["costs"].sum(), 2 * quantity * .19)

    def test_monthly_roll_receives_old_funding_and_charges_both_closing_and_opening(self):
        market = market_sample("2024-02-02T01:00")
        settle = timestamp("2024-02-01T00:00")
        result = simulate(market, funding_sample(events=[(settle + 1, .01, 100.)]),
                          execution="full_reset")
        roll = int(np.flatnonzero(market["time"] == settle + HOUR)[0])
        old = result["q"][1]
        funding = old * 100 * .01
        capital_after_close = 10_000 + funding - 2 * old * .19
        new = capital_after_close / 150.19
        assert_allclose(result["funding_pnl"][roll], funding)
        assert_allclose(result["q"][roll], new)
        assert_allclose(result["costs"][roll], (old + new) * .19)
        self.assertTrue(np.all((result["time"][np.abs(result["delta_q"]) > 1e-10] // HOUR) % 24 == 1))

    def test_net_spot_roll_charges_only_reduced_spot_but_both_perpetual_orders(self):
        market = market_sample("2024-02-02T01:00")
        result = simulate(market, funding_sample(), execution="net_spot")
        roll = int(np.flatnonzero(market["time"] == timestamp("2024-02-01T01:00"))[0])
        old = result["q"][1]
        pre_roll_equity = 10_000 - old * .19
        new = (pre_roll_equity - old * .19) / (150 - .12 + .07)
        self.assertLess(new, old)
        assert_allclose(result["q"][roll], new)
        assert_allclose(result["spot_traded_q"][roll], old - new)
        assert_allclose(result["perp_traded_q"][roll], old + new)
        assert_allclose(result["spot_costs"][roll], .12 * (old - new))
        assert_allclose(result["perp_costs"][roll], .07 * (old + new))
        assert_allclose(result["perp_equity"][roll], .5 * 100 * new)

    def test_net_spot_roll_self_finances_an_increase_from_prior_income(self):
        market = market_sample("2024-02-02T01:00")
        settle = timestamp("2024-02-01T00:00")
        result = simulate(market, funding_sample(events=[(settle + 1, .01, 100.)]),
                          execution="net_spot")
        roll = int(np.flatnonzero(market["time"] == settle + HOUR)[0])
        old = result["q"][1]
        pre_roll_equity = 10_000 - old * .19 + old
        new = (pre_roll_equity + old * (.12 - .07)) / 150.19
        self.assertGreater(new, old)
        assert_allclose(result["q"][roll], new)
        assert_allclose(result["spot_costs"][roll], .12 * (new - old))
        assert_allclose(result["perp_costs"][roll], .07 * (new + old))
        assert_allclose(result["equity"][roll], pre_roll_equity - result["costs"][roll])
        assert_allclose(result["perp_equity"][roll], .5 * 100 * new)

    def test_same_quantity_roll_has_no_spot_fee_and_still_pays_perpetual_fees(self):
        market = market_sample("2024-02-02T01:00")
        settle = timestamp("2024-02-01T00:00")
        result = simulate(market, funding_sample(events=[(settle + 1, .0014, 100.)]),
                          execution="net_spot")
        roll = int(np.flatnonzero(market["time"] == settle + HOUR)[0])
        quantity = result["q"][1]
        assert_allclose(result["q"][roll], quantity)
        assert_allclose(result["spot_traded_q"][roll], 0., atol=1e-10)
        assert_allclose(result["spot_costs"][roll], 0., atol=1e-10)
        assert_allclose(result["perp_traded_q"][roll], 2 * quantity)
        assert_allclose(result["perp_costs"][roll], .14 * quantity)
        assert_allclose(result["costs"][[1, -1]], .19 * quantity)

    def test_execution_modes_have_equal_economics_when_costs_are_zero(self):
        market = market_sample("2024-02-02T01:00")
        market["spot_open"] += .02 * np.arange(len(market["time"]))
        market["perp_open"] = market["spot_open"] + .1 * np.sin(np.arange(len(market["time"])))
        market["mark_open"] = market["perp_open"].copy()
        market["mark_high"] = market["mark_open"].copy()
        funding = funding_sample(events=[(timestamp("2024-02-01T00:00"), .01, 115.)])
        outputs = [simulate(market, funding, execution=mode, spot_bps=0, perp_bps=0,
                            slippage_bps=0) for mode in ("net_spot", "full_reset")]
        for field in ("q", "equity", "spot_pnl", "perp_pnl", "funding_pnl", "costs"):
            assert_allclose(outputs[0][field], outputs[1][field], atol=1e-9)

    def test_episode_allocation_and_leg_costs_reconcile_without_external_capital(self):
        market = market_sample("2024-02-02T01:00")
        settle = timestamp("2024-02-01T00:00")
        funding = funding_sample(events=[(settle + 1, .01, 100.)])
        roll = int(np.flatnonzero(market["time"] == settle + HOUR)[0])
        for execution in ("net_spot", "full_reset"):
            with self.subTest(execution=execution):
                result = simulate(market, funding, execution=execution)
                assert_allclose(result["costs"], result["spot_costs"] + result["perp_costs"])
                episodes = result["episodes"]
                self.assertEqual(len(episodes), 2)
                self.assertAlmostEqual(episodes[0]["capital_start"], 10_000)
                self.assertAlmostEqual(episodes[0]["capital_end"], episodes[1]["capital_start"])
                for episode in episodes:
                    pnl = sum(episode[key] for key in ("spot_pnl", "perp_pnl", "funding_pnl"))
                    self.assertAlmostEqual(episode["capital_end"],
                                           episode["capital_start"] + pnl - episode["costs"])
                    self.assertAlmostEqual(episode["net_pnl"], pnl - episode["costs"])
                for key in ("spot_pnl", "perp_pnl", "funding_pnl", "costs"):
                    self.assertAlmostEqual(sum(episode[key] for episode in episodes), result[key].sum())
                self.assertAlmostEqual(sum(episode["net_pnl"] for episode in episodes),
                                       result["equity"][-1] - 10_000)
                if execution == "net_spot":
                    self.assertAlmostEqual(episodes[0]["costs"], result["costs"][1])
                    self.assertAlmostEqual(episodes[1]["costs"],
                                           result["costs"][roll] + result["costs"][-1])

    def test_resize_restores_collateral_target_when_mark_differs_from_trade_price(self):
        market = market_sample()
        market["mark_open"][23:] = 130.
        market["mark_high"][23:] = 130.
        result = simulate(market, funding_sample(history_rate=.001), policy="buffered")
        old = result["q"][1]
        mark_account = .5 * 100 * old + old * (100 - 130)
        reduction_proceeds = 100 - (.0012 + .0007) * 100 + 130 - 100
        target = (mark_account + old * reduction_proceeds) / (50 + reduction_proceeds)
        assert_allclose(result["q"][24], old)
        assert_allclose(result["q"][25], target)
        assert_allclose(result["margin_equity"][25], .5 * 100 * target)
        assert_allclose(result["collateral_ratio"][25], .5 * 100 / 130)
        assert_allclose(result["spot_pnl"], 0.)
        assert_allclose(result["perp_pnl"], 0.)
        assert_allclose(result["equity"], 10_000 - np.cumsum(result["costs"]))

    def test_more_initial_margin_reduces_position_size_for_the_same_total_capital(self):
        low = simulate(market_sample(), funding_sample(), margin=.25)
        high = simulate(market_sample(), funding_sample(), margin=1.)
        assert_allclose(low["q"][1], 10_000 / 125.19)
        assert_allclose(high["q"][1], 10_000 / 200.19)
        self.assertGreater(low["q"][1], high["q"][1])

    def test_hold_retains_quantity_and_income_without_monthly_orders(self):
        market = market_sample("2024-03-02T01:00")
        funding = funding_sample(events=[(timestamp("2024-02-01T00:00"), .1, 100.)])
        result = simulate(market, funding, policy="risk_only", execution="hold")
        quantity = result["q"][1]
        assert_allclose(result["q"][1:-1], quantity)
        assert_allclose(result["perp_entry_price"][1:-1], 100.)
        self.assertEqual(len(result["trades"]), 2)
        assert_allclose(result["costs"].sum(), 2 * quantity * .19)
        roll = np.flatnonzero(market["time"] == timestamp("2024-02-01T01:00"))[0]
        assert_allclose(result["costs"][roll], 0.)
        self.assertGreater(result["wallet_cash"][roll], result["wallet_cash"][1])

    def test_unrealized_short_profit_does_not_become_wallet_cash(self):
        market = market_sample("2024-02-02T01:00")
        for field in ("spot_open", "perp_open", "mark_open", "mark_high"):
            market[field][8:] = 80.
        result = simulate(market, funding_sample(), policy="risk_only", execution="hold")
        quantity = result["q"][1]
        assert_allclose(result["wallet_cash"][1:-1], result["wallet_cash"][1])
        assert_allclose(result["perp_unrealized"][8:-1], 20 * quantity)
        assert_allclose(result["perp_realized_pnl"][:-1], 0.)
        assert_allclose(result["perp_realized_pnl"][-1], 20 * quantity)
        assert_allclose(result["q"][1:-1], quantity)
        assert_allclose(result["wallet_cash"][-1], result["equity"][-1])

    def test_partial_close_realizes_only_closed_units_and_keeps_entry_price(self):
        market = market_sample()
        for field in ("spot_open", "perp_open", "mark_open", "mark_high"):
            market[field][23:] = 130.
        market["mark_open"][23:] = market["mark_high"][23:] = 140.
        result = simulate(market, funding_sample(), policy="risk_only", execution="hold")
        old, remaining = result["q"][24:26]
        closed = old - remaining
        self.assertGreater(closed, 0.)
        assert_allclose(result["perp_realized_pnl"][25], -30 * closed)
        assert_allclose(result["perp_entry_price"][25], 100.)
        assert_allclose(result["perp_unrealized"][25], -30 * remaining)
        assert_allclose(result["wallet_cash"][25], result["wallet_cash"][24]
                        + closed * 130 - 30 * closed - result["costs"][25])
        assert_allclose(result["margin_equity"][25], .5 * 130 * remaining)
        trade = next(x for x in result["trades"] if x["reason"] == "collateral_reduction")
        assert_allclose(trade["entry_price_before"], trade["entry_price_after"])
        assert_allclose(trade["realized_perp_pnl"], -30 * closed)

    def test_hold_checks_collateral_on_month_start_without_resetting_the_perp(self):
        market = market_sample("2024-02-02T01:00")
        changed = market["time"] >= timestamp("2024-01-31T23:00")
        for field in ("spot_open", "perp_open", "mark_open", "mark_high"):
            market[field][changed] = 130.
        result = simulate(market, funding_sample(), policy="risk_only", execution="hold")
        decision = np.flatnonzero(market["time"] == timestamp("2024-02-01T00:00"))[0]
        old, remaining = result["q"][decision:decision + 2]
        self.assertLess(remaining, old)
        assert_allclose(result["delta_q"][decision], 0.)
        assert_allclose(result["perp_traded_q"][decision + 1], old - remaining)
        assert_allclose(result["perp_entry_price"][decision + 1], 100.)
        self.assertEqual(result["trades"][1]["reason"], "collateral_reduction")

    def test_hold_screen_exit_overrides_reduction_and_reentry_uses_settled_cash(self):
        market = market_sample("2024-03-02T01:00")
        changed = market["time"] >= timestamp("2024-01-31T23:00")
        for field in ("spot_open", "perp_open", "mark_open", "mark_high"):
            market[field][changed] = 130.
        events = [(int(t), .001, 130.) for t in np.arange(
            timestamp("2024-02-23T00:00"), timestamp("2024-03-01T00:00"), 8 * HOUR)]
        result = simulate(market, funding_sample(history_rate=.001, events=events),
                          policy="buffered", execution="hold")
        feb = np.flatnonzero(market["time"] == timestamp("2024-02-01T01:00"))[0]
        march = np.flatnonzero(market["time"] == timestamp("2024-03-01T01:00"))[0]
        assert_allclose(result["q"][feb:march], 0.)
        assert_allclose(result["funding_pnl"][feb + 1:march], 0.)
        assert_allclose(result["q"][march], result["wallet_cash"][march - 1] / (195 + .247))
        self.assertEqual(result["trades"][1]["reason"], "screen_exit")
        self.assertEqual([x["reason"] for x in result["trades"]],
                         ["month_open", "screen_exit", "month_open", "final_close"])

    def test_wallet_realization_and_monthly_segments_reconcile_for_all_variants(self):
        market = market_sample("2024-02-02T01:00")
        index = np.arange(len(market["time"]))
        market["spot_open"] += .04 * index
        market["perp_open"] = market["spot_open"] + .2 * np.sin(index)
        market["mark_open"] = market["perp_open"] + .1
        market["mark_high"] = market["mark_open"] + .1
        events = [(timestamp("2024-01-15T08:00"), .001, 115.),
                  (timestamp("2024-02-01T00:00"), -.0001, 130.)]
        funding = funding_sample(history_rate=.001, events=events)
        for execution in ("net_spot", "full_reset", "hold"):
            for policy in ("buffered", "risk_only"):
                with self.subTest(execution=execution, policy=policy):
                    result = simulate(market, funding, policy=policy, execution=execution)
                    wallet_flow = (result["funding_pnl"] + result["perp_realized_pnl"]
                                   - result["delta_q"] * market["spot_open"] - result["costs"])
                    assert_allclose(result["wallet_cash"], 10_000 + np.cumsum(wallet_flow))
                    assert_allclose(result["perp_equity"], result["wallet_cash"] + result["perp_unrealized"])
                    assert_allclose(np.cumsum(result["perp_pnl"]),
                                    np.cumsum(result["perp_realized_pnl"]) + result["perp_unrealized"], atol=1e-8)
                    assert_allclose(result["margin_equity"], result["wallet_cash"]
                                    + result["q"] * (result["perp_entry_price"] - market["mark_open"]))
                    economic_flow = result["spot_pnl"] + result["perp_pnl"] + result["funding_pnl"] - result["costs"]
                    assert_allclose(result["equity"], 10_000 + np.cumsum(economic_flow))
                    for episode in result["episodes"]:
                        pnl = sum(episode[k] for k in ("spot_pnl", "perp_pnl", "funding_pnl")) - episode["costs"]
                        self.assertAlmostEqual(episode["net_pnl"], pnl)
                    self.assertAlmostEqual(sum(x["net_pnl"] for x in result["episodes"]),
                                           result["equity"][-1] - 10_000)

    def test_risk_only_keeps_risk_control_when_trailing_funding_is_negative(self):
        market = market_sample()
        for field in ("spot_open", "perp_open", "mark_open", "mark_high"):
            market[field][23:] = 130.
        funding = funding_sample(history_rate=-.001)
        for execution in ("net_spot", "hold"):
            result = simulate(market, funding, policy="risk_only", execution=execution)
            self.assertGreater(result["q"][1], 0.)
            self.assertLess(result["q"][25], result["q"][24])

    def test_invalid_market_inputs_and_parameters_fail_early(self):
        for arguments in ({"policy": "unknown"}, {"execution": "unknown"},
                          {"initial": 0}, {"margin": 0},
                          {"spot_bps": -1}, {"maintenance": -1}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                simulate(market_sample(), funding_sample(), **arguments)
        for invalid in (0., -1., np.nan, np.inf):
            market = market_sample()
            market["spot_open"][5] = invalid
            with self.subTest(price=invalid), self.assertRaises(ValueError):
                simulate(market, funding_sample())
        market = market_sample()
        market["time"][1] = market["time"][0]
        with self.assertRaises(ValueError):
            simulate(market, funding_sample())


if __name__ == "__main__":
    unittest.main()
