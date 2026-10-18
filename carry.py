"""Matched BTC quantities, settled funding, and separate derivative collateral."""

import numpy as np

HOUR = 3_600_000
DAY = 24 * HOUR


def _inputs(market, funding):
    market = {k: np.asarray(v) for k, v in market.items()}
    funding = {k: np.asarray(v) for k, v in funding.items()}
    required = ("time", "spot_open", "perp_open", "mark_open", "mark_high")
    if any(k not in market for k in required) or any(k not in funding for k in ("time", "rate", "mark")):
        raise ValueError("Missing market or funding field.")
    for data in (market, funding):
        t = data["time"]
        if t.ndim != 1 or not np.isfinite(t).all() or np.any(t != t.astype(np.int64)):
            raise ValueError("Timestamps must be integer milliseconds.")
        data["time"] = t.astype(np.int64)
        if np.any(np.diff(t) <= 0):
            raise ValueError("Timestamps must be sorted and unique.")
    if len(market["time"]) < 2 or np.any(np.diff(market["time"]) != HOUR):
        raise ValueError("Market data must have a complete hourly grid.")
    if np.any(market["time"] % HOUR):
        raise ValueError("Market rows must be UTC hour opens.")
    for data, fields in ((market, required[1:]), (funding, ("rate", "mark"))):
        for key in fields:
            value = np.asarray(data[key], dtype=float)
            if value.shape != data["time"].shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid {key} data.")
            if key != "rate" and np.any(value <= 0):
                raise ValueError("All prices must be positive.")
            data[key] = value
    if np.any(market["mark_high"] < market["mark_open"]):
        raise ValueError("Hourly mark high cannot be below its open.")
    return market, funding


def simulate(market, funding, policy="always", initial=10000, margin=0.5,
             spot_bps=10, perp_bps=5, slippage_bps=2, maintenance=0.01,
             execution="net_spot"):
    """Hour-open trades; an event exactly at a trade time uses the new quantity."""
    market, funding = _inputs(market, funding)
    if policy not in ("always", "screen", "buffered", "risk_only"):
        raise ValueError("Policy must be always, screen, buffered, or risk_only.")
    if execution not in ("net_spot", "full_reset", "hold"):
        raise ValueError("Execution must be net_spot, full_reset, or hold.")
    parameters = (initial, margin, spot_bps, perp_bps, slippage_bps, maintenance)
    if not np.isfinite(parameters).all() or initial <= 0 or margin <= 0:
        raise ValueError("Capital and margin must be finite and positive.")
    if min(spot_bps, perp_bps, slippage_bps, maintenance) < 0:
        raise ValueError("Costs and maintenance cannot be negative.")
    spot_cost = (spot_bps + slippage_bps) / 10000
    perp_cost = (perp_bps + slippage_bps) / 10000
    if spot_cost >= 1 or perp_cost >= 1:
        raise ValueError("Each leg's cost must be less than its traded value.")
    times, ft = market["time"], funding["time"]
    n = len(times)
    fields = ("equity", "margin_equity", "perp_equity", "mark_equity", "q",
              "spot_pnl", "perp_pnl", "funding_pnl", "costs", "delta_q", "traded_q",
              "spot_traded_q", "perp_traded_q", "spot_costs", "perp_costs",
              "wallet_cash", "perp_entry_price", "perp_unrealized", "perp_realized_pnl")
    out = {key: np.zeros(n) for key in fields}
    out.update(time=times.copy(), margin_flag=np.zeros(n, dtype=bool),
               collateral_ratio=np.full(n, np.nan), stressed_ratio=np.full(n, np.nan),
               initial=float(initial), events=[], trades=[], episodes=[], signals=[])
    calendar = times.astype("datetime64[ms]")
    days, months = calendar.astype("datetime64[D]"), calendar.astype("datetime64[M]")
    month_start = days == months.astype("datetime64[D]")
    hours = (times % DAY) // HOUR
    q, wallet, entry_price = 0.0, float(initial), 0.0
    account = wallet
    pending_entry, pending_reduction = False, None
    episode = None
    previous_s, previous_p = market["spot_open"][0], market["perp_open"][0]

    def trade(new_q, reason, perp_traded=None):
        nonlocal q, account, wallet, entry_price
        if not np.isfinite(new_q) or new_q < 0:
            raise ValueError("A trade cannot create negative or nonfinite quantity.")
        entry_before = entry_price
        change = float(new_q - q)
        spot_traded = abs(change)
        reset_perp = perp_traded is not None
        perp_traded = spot_traded if perp_traded is None else float(perp_traded)
        if spot_traded < 1e-14 and perp_traded < 1e-14:
            return
        spot_fee, perp_fee = spot_traded * spot_cost * s, perp_traded * perp_cost * p
        fees = spot_fee + perp_fee
        closed_q = q if reset_perp else max(0.0, -change)
        realized = closed_q * (entry_price - p)
        wallet += realized - change * s - fees
        if reset_perp or q == 0:
            entry_price = float(p) if new_q else 0.0
        elif change > 1e-14:
            raise ValueError("A held position cannot grow using unrealized P&L.")
        if new_q == 0:
            entry_price = 0.0
        account = wallet + new_q * (entry_price - p)
        out["perp_realized_pnl"][i] += realized
        out["costs"][i] += fees
        out["delta_q"][i] += change
        out["traded_q"][i] += spot_traded
        for key, amount in (("spot_traded_q", spot_traded), ("perp_traded_q", perp_traded),
                            ("spot_costs", spot_fee), ("perp_costs", perp_fee)):
            out[key][i] += amount
        out["trades"].append({"time": int(t), "reason": reason, "q_before": float(q),
                              "q_after": float(new_q), "delta_q": change, "spot": float(s),
                              "perp": float(p), "spot_traded_q": spot_traded,
                              "perp_traded_q": perp_traded, "spot_costs": float(spot_fee),
                              "perp_costs": float(perp_fee), "costs": float(fees),
                              "realized_perp_pnl": float(realized),
                              "entry_price_before": float(entry_before),
                              "entry_price_after": float(entry_price),
                              "wallet_cash_after": float(wallet)})
        if episode is not None:
            episode["costs"] += float(fees)
            episode["trades"] += 1
        q = float(new_q)

    def new_episode(capital):
        return {"start": int(t), "capital_start": float(capital), "entered": False,
                "entry_q": 0.0, "spot_pnl": 0.0, "perp_pnl": 0.0, "funding_pnl": 0.0,
                "costs": 0.0, "trades": 0, "margin_flag_hours": 0}

    def finish_episode():
        nonlocal episode
        if episode is not None:
            episode.update(end=int(t), capital_end=float(account + q * s))
            episode["net_pnl"] = episode["capital_end"] - episode["capital_start"]
            out["episodes"].append(episode)
            episode = None

    for i, t in enumerate(times):
        s, p, mark = (market[k][i] for k in ("spot_open", "perp_open", "mark_open"))
        out["spot_pnl"][i] = q * (s - previous_s)
        out["perp_pnl"][i] = q * (previous_p - p)
        if i:
            left = np.searchsorted(ft, times[i - 1], side="left")
            right = np.searchsorted(ft, t, side="left")
            for j in range(left, right):
                amount = q * funding["mark"][j] * funding["rate"][j]
                wallet += amount
                out["funding_pnl"][i] += amount
                out["events"].append({"time": int(ft[j]), "q": float(q),
                                      "mark": float(funding["mark"][j]),
                                      "rate": float(funding["rate"][j]), "amount": float(amount)})
        account = wallet + q * (entry_price - p)
        if episode is not None:
            for key in ("spot_pnl", "perp_pnl", "funding_pnl"):
                episode[key] += float(out[key][i])

        if hours[i] == 0:
            if month_start[i]:
                left = np.searchsorted(ft, t - 7 * DAY, side="left")
                right = np.searchsorted(ft, t, side="left")
                rate_sum = float(funding["rate"][left:right].sum())
                month_days = int(((months[i] + 1).astype("datetime64[D]") - months[i].astype("datetime64[D]")).astype(int))
                forecast = mark * rate_sum / 7 * month_days
                round_trip = 2 * (spot_cost * s + perp_cost * p)
                pending_entry = policy in ("always", "risk_only") or forecast > round_trip
                out["signals"].append({"time": int(t), "funding_observations": int(right - left),
                                       "past_rate_sum": rate_sum, "month_days": month_days,
                                       "forecast_per_btc": float(forecast),
                                       "round_trip_per_btc": float(round_trip),
                                       "enter": bool(pending_entry)})
                pending_reduction = None
            if (policy in ("buffered", "risk_only") and q > 0
                    and (not month_start[i] or execution == "hold")):
                mark_account = account + q * (p - mark)
                pending_reduction = None
                if mark_account / (q * mark) < margin / 2:
                    cost_unit = spot_cost * s + perp_cost * p
                    sale_credit = s - cost_unit + mark - p
                    denominator = margin * s + sale_credit
                    if denominator <= 0:
                        raise ValueError("Costs make the collateral resize equation invalid.")
                    target = (mark_account + q * sale_credit) / denominator
                    pending_reduction = float(np.clip(target, 0, q))

        terminal = i == n - 1
        roll = month_start[i] and hours[i] == 1
        if terminal:
            trade(0.0, "final_close")
            finish_episode()
            pending_reduction = None
        elif roll:
            if execution == "hold":
                capital = account + q * s
                finish_episode()
                episode = new_episode(capital)
                if not pending_entry:
                    trade(0.0, "screen_exit")
                elif q == 0 and wallet > 0:
                    cost_unit = spot_cost * s + perp_cost * p
                    trade(wallet / (s * (1 + margin) + cost_unit), "month_open")
                elif pending_reduction is not None:
                    trade(min(q, pending_reduction), "collateral_reduction")
            elif execution == "net_spot":
                capital = account + q * s
                finish_episode()
                episode = new_episode(capital)
                target = 0.0
                if pending_entry and capital > 0:
                    cs, cp = spot_cost * s, perp_cost * p
                    growing = (capital - q * (cp - cs)) / ((1 + margin) * s + cp + cs)
                    if growing >= q:
                        target = growing
                    else:
                        denominator = (1 + margin) * s + cp - cs
                        target = max(0.0, (capital - q * (cp + cs)) / denominator)
                trade(target, "month_roll" if q else "month_open", perp_traded=q + target)
            else:
                trade(0.0, "month_close")
                finish_episode()
                episode = new_episode(account)
                if pending_entry and account > 0:
                    cost_unit = spot_cost * s + perp_cost * p
                    target = account / (s * (1 + margin) + cost_unit)
                    trade(target, "month_open")
            episode["entry_q"] = float(q)
            episode["entered"] = bool(q > 0)
            pending_entry = False
            pending_reduction = None
        elif hours[i] == 1 and pending_reduction is not None:
            trade(min(q, pending_reduction), "collateral_reduction")
            pending_reduction = None

        mark_account = account + q * (p - mark)
        out["q"][i] = q
        out["wallet_cash"][i] = wallet
        out["perp_entry_price"][i] = entry_price
        out["perp_unrealized"][i] = q * (entry_price - p)
        out["perp_equity"][i] = account
        out["margin_equity"][i] = mark_account
        out["equity"][i] = account + q * s
        out["mark_equity"][i] = mark_account + q * s
        if q > 0:
            out["collateral_ratio"][i] = mark_account / (q * mark)
            left = np.searchsorted(ft, t, side="left")
            right = np.searchsorted(ft, t + HOUR, side="left")
            high = max(market["mark_high"][i], np.max(funding["mark"][left:right], initial=0))
            negative_funding = q * np.sum(funding["mark"][left:right] * np.minimum(funding["rate"][left:right], 0))
            stressed = mark_account - q * (high - mark) + negative_funding
            out["stressed_ratio"][i] = stressed / (q * high)
            out["margin_flag"][i] = stressed < maintenance * q * high
            if episode is not None:
                episode["margin_flag_hours"] += int(out["margin_flag"][i])
        previous_s, previous_p = s, p
    return out
