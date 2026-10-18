"""Funding carry: compare calendar rolls, persistent positions and a cost screen."""

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from carry import simulate


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".cache"
HOUR = 3_600_000
DAY = 24 * HOUR
POLICIES = {"always": "Always carry", "screen": "Cost screen", "buffered": "Screen + collateral rule"}
VARIANTS = {
    "monthly_screen": ("Monthly + screen", "net_spot", "buffered"),
    "monthly_plain": ("Monthly, no screen", "net_spot", "risk_only"),
    "hold_screen": ("Hold + screen", "hold", "buffered"),
    "hold_plain": ("Hold, no screen", "hold", "risk_only"),
}
INPUT_HASHES = {
    "hourly.csv": "cc114cd325bd655f836ff7b6d8e6c5e8da12f613dec82891de0397ac95d29de0",
    "funding.csv": "d55006b247f45fa2288a157c35dce3a0afeddcc9053cdbd232375f26d387fdc0",
}
HOLDOUT_HASHES = {
    "hourly.csv": "0135395814aa1e1a3cfe2bb9030f7785309856f3b0bfa75f8e4857805ede85a1",
    "funding.csv": "e51406c4d4ad9356ca692decbd50ac8658c0fc21396c82b64712649e82aabc52",
}
CONFIG = {
    "initial_usdt": 100_000.0, "initial_margin_ratio": 0.5,
    "spot_fee_bps": 10, "perp_fee_bps": 5, "slippage_bps_per_leg": 2,
    "maintenance_stress_ratio": 0.01, "funding_lookback_days": 7,
    "entry_time": "First day of month, 01:00 UTC; decision at 00:00",
    "resize_rule": "At daily 00:00, if derivative mark equity / short mark notional < half the initial ratio, reduce both legs at 01:00. Signal-time target: derivative mark equity / remaining spot value = initial ratio.",
    "execution": "Primary: retain matched positions; trade only entry, exit and collateral reductions. Monthly net-spot/perpetual resets remain as a comparison.",
    "cash_yield": 0, "capital_basis": "Spot purchase + separate derivative collateral + transaction costs",
    "primary_period": "2024-01-01 01:00 UTC to 2026-01-01 01:00 UTC",
    "data_selection": "Two complete years with exact settlement mark prices; chosen before outcomes, not by performance.",
}



def run_policy(market, funding, **overrides):
    settings = {"initial": CONFIG["initial_usdt"], "margin": CONFIG["initial_margin_ratio"],
                "spot_bps": CONFIG["spot_fee_bps"], "perp_bps": CONFIG["perp_fee_bps"],
                "slippage_bps": CONFIG["slippage_bps_per_leg"],
                "maintenance": CONFIG["maintenance_stress_ratio"]}
    settings.update(overrides)
    return simulate(market, funding, **settings)


def download_inputs(cache, holdout=False):
    """Fetch and validate the fixed BTCUSDT research sample; no strategy calculations."""
    import csv
    import datetime as dt
    import hashlib
    import io
    import json
    import math
    from concurrent.futures import ThreadPoolExecutor
    from decimal import Decimal
    from pathlib import Path
    import ssl
    import subprocess
    from urllib.error import URLError
    from urllib.parse import urlencode
    from urllib.request import Request, urlopen
    import zipfile

    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    start, end, funding_start = ((1767225600000, 1790816400000, 1764547200000) if holdout
                                else (1704067200000, 1767229200000, 1701388800000))
    original = {name.removesuffix(".csv"): digest for name, digest in
                (HOLDOUT_HASHES if holdout else INPUT_HASHES).items()}
    def utc(ms):
        return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).isoformat()
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    def fetch(url, filename):
        path = cache / filename
        downloaded = not path.exists()
        if downloaded:
            temporary = path.with_name(path.name + ".part")
            try:
                with urlopen(Request(url, headers={"User-Agent": "research-data-check/1.0"}), timeout=35) as response:
                    temporary.write_bytes(response.read())
            except URLError as error:
                if not isinstance(error.reason, ssl.SSLError):
                    raise
                # macOS curl uses its system CA store; TLS verification remains enabled.
                subprocess.run(["curl", "--fail", "--silent", "--show-error", "--max-time", "35",
                                url, "--output", str(temporary)], check=True, capture_output=True)
            temporary.replace(path)
        return path, {"url": url, "path": str(path), "sha256": sha(path), "bytes": path.stat().st_size,
                      "downloaded_now": downloaded, "checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    def archive(job):
        kind, month = job
        if kind == "funding":
            suffix = f"data/futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-{month}.zip"
        else:
            market = "spot" if kind == "spot" else "futures/um"
            dataset = "markPriceKlines" if kind == "mark" else "klines"
            bucket = "daily" if len(month) == 10 else "monthly"
            suffix = f"data/{market}/{bucket}/{dataset}/BTCUSDT/1h/BTCUSDT-1h-{month}.zip"
        url = "https://data.binance.vision/" + suffix
        path, metadata = fetch(url, f"{kind}_{month}.zip")
        checksum, checksum_metadata = fetch(url + ".CHECKSUM", path.name + ".CHECKSUM")
        if sha(path) != checksum.read_text().split()[0]:
            raise ValueError(f"Official checksum mismatch: {path}")
        with zipfile.ZipFile(path) as zipped:
            if zipped.testzip() is not None or len(zipped.namelist()) != 1:
                raise ValueError(f"Invalid archive: {path}")
            rows = list(csv.reader(io.StringIO(zipped.read(zipped.namelist()[0]).decode())))
        if rows and not rows[0][0].isdigit():
            rows = rows[1:]
        if not rows:
            raise ValueError(f"Empty data: {path}")
        metadata.update(type=kind, month=month, checksum=checksum_metadata, rows=len(rows))
        return kind, rows, metadata

    months = [f"{year}-{month:02}" for year in (2024, 2025) for month in range(1, 13)] + ["2026-01"]
    jobs = [(kind, month) for kind in ("spot", "perp", "mark") for month in months]
    jobs += [("funding", month) for month in ["2023-12"] + months]
    if holdout:
        months = [f"2026-{month:02}" for month in range(1, 10)]
        jobs = [(kind, month) for kind in ("spot", "perp", "mark") for month in months + ["2026-10-01"]]
        jobs += [("mark", "2026-06-29")]
        jobs += [("funding", month) for month in ["2025-12"] + months]
    channels = {kind: {} for kind in ("spot", "perp", "mark")}
    archived_funding, sources = {}, []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for kind, rows, metadata in pool.map(archive, jobs):
            sources.append(metadata)
            for row in rows:
                timestamp = int(row[0])
                if kind == "funding":
                    if funding_start <= timestamp <= end:
                        if timestamp in archived_funding:
                            raise ValueError("Duplicate archive funding event")
                        archived_funding[timestamp] = row
                    continue
                if timestamp >= 10**14 and timestamp % 1000:
                    raise ValueError("Sub-millisecond price bar boundary")
                timestamp //= 1000 if timestamp >= 10**14 else 1
                if start <= timestamp <= end:
                    values = [float(x) for x in row[1:5]]
                    if len(row) != 12 or not all(math.isfinite(v) and v > 0 for v in values):
                        raise ValueError("Invalid OHLC fields")
                    if not values[2] <= min(values[0], values[3]) <= max(values[0], values[3]) <= values[1]:
                        raise ValueError("Invalid OHLC ordering")
                    if timestamp in channels[kind]:
                        raise ValueError("Duplicate hourly observation")
                    channels[kind][timestamp] = row[1:5]
    expected = set(range(start, end + 1, 3600000))
    for kind, observations in channels.items():
        if set(observations) != expected:
            raise ValueError(f"Incomplete {kind} hours; missing={sorted(expected-set(observations))}")

    events, api_sources, cursor = [], [], funding_start
    for page in range(10):
        query = urlencode({"symbol": "BTCUSDT", "startTime": cursor, "endTime": end, "limit": 1000})
        path, metadata = fetch("https://fapi.binance.com/fapi/v1/fundingRate?" + query, f"primary_funding_api_{cursor}.json")
        payload = json.loads(path.read_text())
        if not isinstance(payload, list):
            raise ValueError(f"Unexpected API response: {path}")
        api_sources.append(metadata)
        events.extend(payload)
        if len(payload) < 1000:
            break
        next_cursor = max(int(event["fundingTime"]) for event in payload) + 1
        if next_cursor <= cursor:
            raise ValueError("Funding pagination did not advance")
        cursor = next_cursor
    else:
        raise ValueError("Funding pagination limit reached")
    timestamps = [int(event["fundingTime"]) for event in events]
    extra = set(timestamps) - set(archived_funding)
    valid_extra = (len(extra) == 1 and all(end - HOUR <= t < end for t in extra)) if holdout else not extra
    if len(events) != (913 if holdout else 2287) or len(set(timestamps)) != len(events) or set(archived_funding) - set(timestamps) or not valid_extra:
        raise ValueError("Funding events incomplete, duplicated or inconsistent with official archives")
    events.sort(key=lambda event: int(event["fundingTime"]))
    for j, event in enumerate(events):
        if int(event["fundingTime"]) in extra and (j == 0 or abs(int(event["fundingTime"]) - int(events[j-1]["fundingTime"]) - 8 * HOUR) > 1000):
            raise ValueError("Cannot infer an eight-hour terminal funding interval")
    for event in events:
        archived = archived_funding.get(int(event["fundingTime"]), [str(event["fundingTime"]), "8", event["fundingRate"]])
        if event.get("symbol") != "BTCUSDT" or Decimal(event["fundingRate"]) != Decimal(archived[2]):
            raise ValueError("Funding API/archive disagreement")
        if not event.get("markPrice") or not math.isfinite(float(event["markPrice"])) or float(event["markPrice"]) <= 0:
            raise ValueError("Missing exact settlement mark price")
        if archived[1] != "8" or event.get("rateType") != "Regular":
            raise ValueError("Funding interval or rate type changed; review before use")

    hourly, funding = cache / "hourly.csv", cache / "funding.csv"
    with hourly.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp_ms", "utc"] + [f"{kind}_{field}" for kind in channels for field in ("open", "high", "low", "close")])
        for timestamp in sorted(expected):
            writer.writerow([timestamp, utc(timestamp)] + [value for kind in channels for value in channels[kind][timestamp]])
    with funding.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["funding_time_ms", "utc", "funding_rate", "settlement_mark_price", "interval_hours", "rate_type"])
        for event in events:
            timestamp = int(event["fundingTime"])
            writer.writerow([timestamp, utc(timestamp), event["fundingRate"], event["markPrice"], archived_funding.get(timestamp, [None, "8"])[1], event["rateType"]])
    metadata = {"hourly": {"path": str(hourly), "sha256": sha(hourly), "rows": len(expected), "first_utc": utc(start), "last_utc": utc(end)},
                "funding": {"path": str(funding), "sha256": sha(funding), "rows": len(events), "first_utc": utc(timestamps[0]), "last_utc": utc(timestamps[-1])},
                "source_archives": sources, "source_api": api_sources, "original_csv_sha256": original,
                "api_only_terminal_events": sorted(extra),
                "terminal_interval_note": "API-only terminal interval inferred as 8 hours from observed slots; not supplied by API" if holdout else None,
                "official_daily_repair": "24 mark bars on 2026-06-29, absent from monthly archive" if holdout else None,
                "processing": "Decimal strings preserved; spot microseconds normalized; exact funding timestamps retained; no filling or interpolation."}
    metadata["matches_original_snapshot"] = all(metadata[kind]["sha256"] == digest for kind, digest in original.items())
    metadata["snapshot_status"] = "original snapshot reproduced" if metadata["matches_original_snapshot"] else "source snapshot changed; review new hashes before computing results"
    (cache / "normalized_metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def load_inputs(accept_revised=False, cache=CACHE, hashes=INPUT_HASHES):
    metadata = {}
    for name, reference in hashes.items():
        path = cache / name
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; download the official inputs first.")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != reference and not accept_revised:
            raise ValueError(f"{name} differs from the reference; --accept-revised-data labels a new input run.")
        metadata[name] = {"sha256": digest, "reference_match": digest == reference}
    with (cache / "hourly.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    market = {"time": np.array([int(r["timestamp_ms"]) for r in rows], dtype=np.int64)}
    for key in ("spot_open", "perp_open", "mark_open", "mark_high"):
        market[key] = np.array([float(r[key]) for r in rows])
    with (cache / "funding.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    funding = {"time": np.array([int(r["funding_time_ms"]) for r in rows], dtype=np.int64),
               "rate": np.array([float(r["funding_rate"]) for r in rows]),
               "mark": np.array([float(r["settlement_mark_price"]) for r in rows])}
    return market, funding, metadata


def stamp(milliseconds):
    return datetime.fromtimestamp(int(milliseconds) / 1000, timezone.utc).isoformat()


def metrics(ledger):
    initial = float(ledger["initial"])
    wealth = ledger["equity"]
    years = (ledger["time"][-1] - ledger["time"][1]) / DAY / 365.25
    peak = np.maximum.accumulate(np.r_[initial, wealth])[1:]
    net = float(wealth[-1] - initial)
    pnl = {key: float(ledger[key].sum()) for key in ("spot_pnl", "perp_pnl", "funding_pnl", "costs")}
    error = net - pnl["spot_pnl"] - pnl["perp_pnl"] - pnl["funding_pnl"] + pnl["costs"]
    if abs(error) > 1e-6:
        raise AssertionError(f"Cashflow reconciliation failed: {error}")
    flag = ledger["margin_flag"].astype(bool)
    months = ((ledger["time"] - HOUR).astype("datetime64[ms]").astype("datetime64[M]"))
    flag_months = sorted(str(m) for m in np.unique(months[flag]))
    return {"ending_usdt": float(wealth[-1]), "net_pnl": net,
            "total_return": float(wealth[-1] / initial - 1),
            "cagr": float((wealth[-1] / initial) ** (1 / years) - 1) if wealth[-1] > 0 else None,
            "max_drawdown": float(np.min(wealth / peak - 1)),
            **pnl, "combined_price_pnl": pnl["spot_pnl"] + pnl["perp_pnl"],
            "active_hours": int(np.count_nonzero(ledger["q"])),
            "stress_flag_hours": int(flag.sum()), "stress_flag_months": flag_months,
            "return_status": "Unconstrained paper path: collateral stress flagged" if flag.any()
                             else "No conservative collateral flag in this hourly model; not a liquidation guarantee",
            "reconciliation_error": float(error)}


def collateral_diagnostics(market, funding):
    """Ex-post capital needed for a fixed-quantity monthly hedge; never a trading input."""
    times = market["time"]
    calendar = times.astype("datetime64[ms]")
    months = calendar.astype("datetime64[M]")
    hours = (times // HOUR) % 24
    days = calendar.astype("datetime64[D]").astype(int) - months.astype("datetime64[D]").astype(int) + 1
    boundaries = np.flatnonzero((days == 1) & (hours == 1))
    diagnostics = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        start, end = int(times[a]), int(times[b])
        ft = funding["time"]
        event_mask = (ft >= start) & (ft < end)
        event_times = ft[event_mask]
        cashflows = funding["mark"][event_mask] * funding["rate"][event_mask]
        event_marks = funding["mark"][event_mask]
        cumulative = np.r_[0., np.cumsum(cashflows)]
        required, worst_time = 0., start
        for i in range(a, b):
            left, right = np.searchsorted(event_times, [times[i], times[i] + HOUR])
            high = max(market["mark_high"][i], event_marks[left:right].max(initial=0))
            negative = np.minimum(cashflows[left:right], 0).sum()
            needed = ((1 + CONFIG["maintenance_stress_ratio"]) * high - market["perp_open"][a]
                      - cumulative[left] - negative) / market["spot_open"][a]
            if needed > required:
                required, worst_time = float(needed), int(times[i])
        decision = start - HOUR
        past = (ft >= decision - 7 * DAY) & (ft < decision)
        mean_daily_rate = float(funding["rate"][past].sum() / 7)
        idx = a - 1
        cs = (CONFIG["spot_fee_bps"] + CONFIG["slippage_bps_per_leg"]) / 10000
        cp = (CONFIG["perp_fee_bps"] + CONFIG["slippage_bps_per_leg"]) / 10000
        cost_per_btc = 2 * (cs * market["spot_open"][idx] + cp * market["perp_open"][idx])
        daily_funding = market["mark_open"][idx] * mean_daily_rate
        diagnostics.append({"month": str(months[a]), "entry_time": stamp(start), "exit_time": stamp(end),
                            "minimum_initial_collateral_ratio_hindsight": required,
                            "worst_hour": stamp(worst_time), "funding_events": len(event_times),
                            "funding_per_btc": float(cashflows.sum()),
                            "trailing_daily_rate": mean_daily_rate,
                            "forecast_break_even_days": cost_per_btc / daily_funding if daily_funding > 0 else None,
                            "holding_days": (end - start) / DAY})
    return diagnostics


def save_ledgers(ledgers, cache=CACHE):
    cache.mkdir(parents=True, exist_ok=True)
    arrays = ("equity", "margin_equity", "perp_equity", "mark_equity", "q", "spot_pnl", "perp_pnl", "funding_pnl", "costs", "delta_q", "traded_q", "spot_traded_q", "perp_traded_q", "spot_costs", "perp_costs", "margin_flag", "collateral_ratio", "stressed_ratio")
    arrays += tuple(k for k in ("wallet_cash", "perp_unrealized", "perp_entry_price", "perp_realized_pnl") if k in next(iter(ledgers.values())))
    with (cache / "ledger.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["policy", "time", "utc", *arrays])
        for name, ledger in ledgers.items():
            for i, t in enumerate(ledger["time"]):
                writer.writerow([name, int(t), stamp(t), *[ledger[k][i] for k in arrays]])
    for name, ledger in ledgers.items():
        for kind in ("events", "trades", "episodes", "signals"):
            (cache / f"{name}_{kind}.json").write_text(json.dumps(ledger.get(kind, []), indent=2) + "\n")


def revision_study(market, funding, cache):
    ledgers = {key: run_policy(market, funding, execution=execution, policy=policy)
               for key, (_, execution, policy) in VARIANTS.items()}
    rows = {}
    for key, ledger in ledgers.items():
        trades = ledger["trades"]
        rows[key] = {**metrics(ledger), "trade_events": len(trades),
                     "spot_turnover_usdt": sum(t["spot_traded_q"] * t["spot"] for t in trades),
                     "perp_turnover_usdt": sum(t["perp_traded_q"] * t["perp"] for t in trades),
                     "collateral_reductions": sum(t["reason"] == "collateral_reduction" for t in trades)}
    stressed = run_policy(market, funding, execution="hold", policy="risk_only",
                          spot_bps=2 * CONFIG["spot_fee_bps"], perp_bps=2 * CONFIG["perp_fee_bps"],
                          slippage_bps=2 * CONFIG["slippage_bps_per_leg"])
    save_ledgers(ledgers, cache)
    return {"variants": rows, "double_cost_primary": metrics(stressed)}, ledgers



def execution_comparison(market, funding, net_ledgers):
    rows = {}
    for policy in POLICIES:
        original = run_policy(market, funding, policy=policy, execution="full_reset")
        savings = 0.0
        trades = original["trades"]
        for old, new in zip(trades, trades[1:]):
            if old["reason"] == "month_close" and new["reason"] == "month_open" and old["time"] == new["time"]:
                savings += 2 * ((CONFIG["spot_fee_bps"] + CONFIG["slippage_bps_per_leg"]) / 10000) * old["spot"] * min(old["q_before"], new["q_after"])
        rows[policy] = {"full_reset": metrics(original), "net_spot": metrics(net_ledgers[policy]),
                        "same_original_quantity_schedule_spot_cost_saving": float(savings),
                        "actual_net_pnl_change": float(net_ledgers[policy]["equity"][-1] - original["equity"][-1]),
                        "interpretation": "Exact fixed-quantity cost identity plus sequential-capital comparison. The latter also changes later quantity and funding; not all improvement is a direct fee saving."}
    return rows


def collateral_case(market, ledgers):
    baseline = ledgers["always"]
    flagged = np.flatnonzero(baseline["margin_flag"])
    if not len(flagged):
        return None
    i = int(flagged[np.argmin(baseline["stressed_ratio"][flagged])])
    month = str(baseline["time"][i].astype("datetime64[ms]").astype("datetime64[M]"))
    rows = {}
    for name, ledger in ledgers.items():
        rows[name] = {"combined_equity": float(ledger["equity"][i]),
                      "derivative_mark_equity": float(ledger["margin_equity"][i]),
                      "spot_value": float(ledger["q"][i] * market["spot_open"][i]),
                      "btc_quantity_each_leg": float(ledger["q"][i]),
                      "stressed_collateral_ratio": float(ledger["stressed_ratio"][i]),
                      "conservative_flag": bool(ledger["margin_flag"][i]),
                      "reductions_before_hour": [t for t in ledger["trades"] if t["reason"] == "collateral_reduction" and str(np.datetime64(t["time"], "ms").astype("datetime64[M]")) == month and t["time"] <= int(ledger["time"][i])]}
    return {"time": stamp(baseline["time"][i]), "selection": "Most severe flagged hour for the always policy in development data; explanatory case, not a trading signal.",
            "spot_open": float(market["spot_open"][i]), "perp_open": float(market["perp_open"][i]),
            "mark_open": float(market["mark_open"][i]), "mark_high": float(market["mark_high"][i]), "policies": rows}


def draw(report, ledgers, later_ledgers=None):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    fig.suptitle("Funding carry | Do calendar rolls and funding screens help?", fontsize=18, weight="bold")
    colours = {"monthly_screen": "#a4adb7", "monthly_plain": "#cc9660",
               "hold_screen": "#6c8ba6", "hold_plain": "#087c84"}
    for ax, sample, title in [(axes[0, 0], ledgers, "2024-2025 development history"),
                               (axes[0, 1], later_ledgers, "Jan-Sep 2026 | previously inspected history")]:
        if sample is None:
            ax.set_visible(False)
            continue
        for key, ledger in sample.items():
            label = VARIANTS[key][0] + (" (paper)" if ledger["margin_flag"].any() else "")
            ax.plot(ledger["time"].astype("datetime64[ms]"), ledger["equity"],
                    color=colours[key], label=label, lw=2 if key == "hold_plain" else 1.3)
        ax.axhline(CONFIG["initial_usdt"], color="#777", lw=.6)
        ax.set(title=title, ylabel="Combined equity (USDT)")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6 if sample is ledgers else 3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.legend(fontsize=8, frameon=False)
    x = np.arange(len(VARIANTS))
    windows = [("development", "2024-2025", "#527e9f")]
    if later_ledgers is not None:
        windows.append(("later_history", "Jan-Sep 2026", "#12a394"))
    for ax, metric, title in [(axes[1, 0], "costs", "Fees and slippage | both legs included"),
                               (axes[1, 1], "net_pnl", "Net profit | separate 100,000 USDT starts")]:
        for i, (window, label, colour) in enumerate(windows):
            values = [report["revision"][window]["variants"][key][metric] for key in VARIANTS]
            bars = ax.bar(x + (i - (len(windows) - 1) / 2) * .36, values, .34, color=colour, label=label)
            ax.bar_label(bars, labels=[f"{v:,.0f}" for v in values], fontsize=8, padding=3)
        ax.set_xticks(x, ["Monthly\n+ screen", "Monthly\nno screen", "Hold\n+ screen", "Hold\nno screen"])
        ax.set(title=title, ylabel="USDT")
        ax.margins(y=.18)
        ax.legend(fontsize=8, frameon=False)
    for ax in axes.flat:
        ax.grid(alpha=.15, axis="y")
    fig.supxlabel("All four variants apply a collateral rule. Fixed costs and hourly simultaneous fills; no live-trading or untouched-test claim.", fontsize=9)
    fig.savefig(ROOT / "results.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-revised-data", action="store_true")
    parser.add_argument("--holdout", action="store_true", help="Also run the previously inspected 2026 history")
    parser.add_argument("--download", action="store_true", help="Fetch and validate official inputs before running")
    args = parser.parse_args()
    if args.download or not all((CACHE / name).exists() for name in INPUT_HASHES):
        download_inputs(CACHE)
    market, funding, sources = load_inputs(args.accept_revised_data)
    ledgers = {p: run_policy(market, funding, policy=p) for p in POLICIES}
    report = {"configuration": CONFIG, "inputs": sources,
              "legacy_sections": "base, collateral_diagnostics, execution_comparison, collateral_case, margin_sensitivity, cost_sensitivity and holdout retain the original monthly net_spot experiment. Use revision for current comparisons.",
              "base": {p: metrics(x) for p, x in ledgers.items()},
              "collateral_diagnostics": collateral_diagnostics(market, funding),
              "execution_comparison": execution_comparison(market, funding, ledgers),
              "collateral_case": collateral_case(market, ledgers),
              "margin_sensitivity": [], "cost_sensitivity": []}
    for margin in (.25, .5, 1.):
        for policy in POLICIES:
            result = run_policy(market, funding, policy=policy, margin=margin)
            report["margin_sensitivity"].append({"margin": margin, "policy": policy, "metrics": metrics(result)})
    for slippage in (0, 2, 5):
        for policy in POLICIES:
            result = run_policy(market, funding, policy=policy, slippage_bps=slippage)
            report["cost_sensitivity"].append({"slippage_bps": slippage, "policy": policy, "metrics": metrics(result)})
    report["boundaries"] = {
        "funding": "Settled rate is known only afterwards. Screen uses only strictly earlier events; actual event timestamps and exact marks determine receipts.",
        "capital": "Return includes spot capital and isolated collateral; selling a futures contract does not release its notional as spending cash.",
        "risk": "Conservative hourly-high/negative-funding stress flag with hypothetical 1% maintenance. Not an exchange liquidation replay; paths with flags are unconstrained paper paths.",
        "ex_post": "Required collateral uses the completed path and is diagnostic only; no rule sees this in advance.",
        "limits": "Two years, fractional BTC sizing, simultaneous-leg hourly-open fills, fixed fees, zero cash yield, no exchange default/USDT depeg/ADL or exact liquidation model.",
    }
    hledgers = None
    if args.holdout:
        hcache = CACHE / "holdout"
        if args.download or not all((hcache / name).exists() for name in HOLDOUT_HASHES):
            download_inputs(hcache, holdout=True)
        hm, hf, hs = load_inputs(args.accept_revised_data, hcache, HOLDOUT_HASHES)
        hledgers = {p: run_policy(hm, hf, policy=p) for p in POLICIES}
        report["holdout"] = {"period": "2026-01-01 01:00 to 2026-10-01 01:00 UTC",
                             "status": "Chronological held-out history; rules fixed before first run. Not live forward trading.",
                             "inputs": hs, "base": {p: metrics(x) for p, x in hledgers.items()}}
        save_ledgers(hledgers, hcache)
    revision, rledgers = revision_study(market, funding, CACHE / "revision")
    report["revision"] = {
        "primary": "hold_plain",
        "design": "Four predeclared combinations of monthly/persistent execution and screen/no screen; all protect collateral. No parameter search.",
        "data_status": "Both windows were inspected before this revision. New results are retrospective, not untouched out-of-sample evidence.",
        "accounting": "Cash wallet and unrealized short PnL tracked separately. Partial closing realizes only closed units. No position increases while held.",
        "comparison_limits": "Persistent execution retains quantity instead of monthly resizing and checks collateral on month-start too. PnL differences also reflect changed exposure and funding receipts, not only saved fees.",
        "development": revision,
    }
    rhledgers = None
    if args.holdout:
        report["revision"]["later_history"], rhledgers = revision_study(hm, hf, CACHE / "holdout" / "revision")
        report["holdout"]["status"] = "Originally reserved for the legacy rules; already inspected before this revision. Not a new independent test."
    (ROOT / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    save_ledgers(ledgers)
    draw(report, rledgers, rhledgers)
    if args.holdout:
        print("Legacy 2026 net P&L (USDT):", {p: round(r["net_pnl"], 2) for p, r in report["holdout"]["base"].items()})
    for window in ("development", "later_history"):
        if window in report["revision"]:
            print(window, {k: round(v["net_pnl"], 2) for k, v in report["revision"][window]["variants"].items()})
    for name, row in report["base"].items():
        print(f"{POLICIES[name]:26} net {row['net_pnl']:9.2f} USDT  CAGR {row['cagr']:7.2%}  "
              f"DD {row['max_drawdown']:7.2%}  flagged months {len(row['stress_flag_months'])}")


if __name__ == "__main__":
    main()
