import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from execution.v2_paper import V2ShadowState, save_v2_shadow_state
from v2_live_binance import BinanceFutures
from v2_live_binance import configure_file_logging as configure_binance_file_logging
from v2_live_binance import execute_opens
from v2_live_binance import LIVE_SYMBOLS, SYMBOL_MAP, TESTNET_SYMBOLS
from v2_live_binance import TIER_A_LEVERAGE, TIER_B_LEVERAGE, leverage_for_allocation_tier
from v2_live_binance import live_mode_unlocked, reconcile_binance_positions
from v2_live_deriv import DEFAULT_MULTIPLIERS, DERIV_SYMBOL_MAP, _reconcile_deriv_positions
from v2_live_deriv import configure_file_logging as configure_deriv_file_logging
from v2.live_data import DERIV_SYMBOL_MAP as LIVE_DATA_DERIV_SYMBOL_MAP


class _FakeBinanceClient:
    def __init__(self, positions):
        self._positions = positions

    def position_risk(self):
        return self._positions


class _FakeBinanceMarkClient:
    def _get(self, path, params=None, signed=False):
        self.path = path
        return [
            {"symbol": "BTCUSDT", "markPrice": "76500.5"},
            {"symbol": "XAUUSDT", "markPrice": "4602.25"},
        ]


class _FakeBinanceExecutionClient:
    def __init__(self):
        self.calls = []

    def cancel_all_orders(self, symbol):
        self.calls.append(("cancel", symbol))

    def set_leverage(self, symbol, leverage):
        self.calls.append(("leverage", symbol, leverage))

    def market_order(self, symbol, side, qty):
        self.calls.append(("market", symbol, side, qty))
        return {"orderId": 123, "status": "FILLED", "avgPrice": "100.0"}

    def stop_market_order(self, symbol, side, stop_price):
        self.calls.append(("stop", symbol, side, stop_price))

    def take_profit_order(self, symbol, side, stop_price):
        self.calls.append(("tp", symbol, side, stop_price))


class V2LiveSafetyTests(unittest.TestCase):
    def test_binance_testnet_map_includes_fresh_oos_and_xau_without_live_xau(self):
        for symbol in ["LTC", "BCH", "ETC", "TRX", "AAVE", "FIL", "OP", "XAU"]:
            self.assertIn(symbol, SYMBOL_MAP)
            self.assertIn(symbol, TESTNET_SYMBOLS)
        self.assertEqual(SYMBOL_MAP["XAU"], "XAUUSDT")
        self.assertNotIn("XAU", LIVE_SYMBOLS)

    def test_deriv_live_map_includes_confirmed_crash_boom_300_multipliers(self):
        self.assertEqual(DERIV_SYMBOL_MAP["DERIV_CRASH300"], "CRASH300N")
        self.assertEqual(DERIV_SYMBOL_MAP["DERIV_BOOM300"], "BOOM300N")
        self.assertEqual(LIVE_DATA_DERIV_SYMBOL_MAP["CRASH300N"], "DERIV_CRASH300")
        self.assertEqual(LIVE_DATA_DERIV_SYMBOL_MAP["BOOM300N"], "DERIV_BOOM300")
        self.assertEqual(DEFAULT_MULTIPLIERS["DERIV_CRASH300"], 20)
        self.assertEqual(DEFAULT_MULTIPLIERS["DERIV_BOOM300"], 20)

    def test_noninteractive_live_requires_explicit_unlock(self):
        self.assertFalse(live_mode_unlocked(False, env={}))
        self.assertFalse(live_mode_unlocked(False, env={"ALLOW_REAL_MONEY": "yes"}))
        self.assertTrue(live_mode_unlocked(False, env={"ALLOW_REAL_MONEY": "YES_I_UNDERSTAND"}))
        self.assertTrue(live_mode_unlocked(True, env={}))

    def test_binance_tier_leverage_mapping_excludes_tier_c(self):
        self.assertEqual(TIER_A_LEVERAGE, 10)
        self.assertEqual(TIER_B_LEVERAGE, 5)
        self.assertEqual(leverage_for_allocation_tier("A"), 10)
        self.assertEqual(leverage_for_allocation_tier("B"), 5)
        self.assertIsNone(leverage_for_allocation_tier("C"))
        self.assertEqual(leverage_for_allocation_tier(None), 5)

    def test_binance_execute_open_sets_tier_a_leverage_before_order(self):
        client = _FakeBinanceExecutionClient()

        fills = execute_opens(
            client,  # type: ignore[arg-type]
            opens=[
                {
                    "symbol": "BTC",
                    "target_notional_usd": 250.0,
                    "allocation_tier": "A",
                    "meta": {"allocation_tier": "A"},
                }
            ],
            close_prices={"BTC": 100.0},
            lot_rules={"BTCUSDT": {"step": 0.001, "min_qty": 0.001}},
            active_map={"BTC": "BTCUSDT"},
            dry_run=False,
        )

        self.assertIn(("leverage", "BTCUSDT", 10), client.calls)
        self.assertLess(client.calls.index(("leverage", "BTCUSDT", 10)), client.calls.index(("market", "BTCUSDT", "BUY", 2.5)))
        self.assertEqual(fills["BTC"]["meta"]["exchange_leverage"], 10)

    def test_binance_execute_open_skips_tier_c(self):
        client = _FakeBinanceExecutionClient()

        fills = execute_opens(
            client,  # type: ignore[arg-type]
            opens=[
                {
                    "symbol": "BTC",
                    "target_notional_usd": 250.0,
                    "allocation_tier": "C",
                    "meta": {"allocation_tier": "C"},
                }
            ],
            close_prices={"BTC": 100.0},
            lot_rules={"BTCUSDT": {"step": 0.001, "min_qty": 0.001}},
            active_map={"BTC": "BTCUSDT"},
            dry_run=False,
        )

        self.assertEqual(fills, {})
        self.assertEqual(client.calls, [])

    def test_binance_leverage_open_position_error_continues_to_reconciliation(self):
        client = BinanceFutures("key", "secret", "https://example.test")
        response = Mock()
        response.json.return_value = {"code": -4161, "msg": "Leverage reduction is not supported"}
        exc = __import__("requests").HTTPError("bad request", response=response)
        client._post = Mock(side_effect=exc)  # type: ignore[method-assign]

        self.assertEqual(client.set_leverage("XAUUSDT", 3), {})

    def test_binance_mark_prices_parse_premium_index_rows(self):
        client = _FakeBinanceMarkClient()

        marks = BinanceFutures.mark_prices(client)  # type: ignore[arg-type]

        self.assertEqual(client.path, "/fapi/v1/premiumIndex")
        self.assertEqual(marks["BTCUSDT"], 76500.5)
        self.assertEqual(marks["XAUUSDT"], 4602.25)

    def test_live_runners_can_log_to_files_while_console_dashboard_stays_visible(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            binance_log = Path(tmpdir) / "binance.log"
            deriv_log = Path(tmpdir) / "deriv.log"

            configure_binance_file_logging(str(binance_log))
            configure_deriv_file_logging(str(deriv_log))

            self.assertTrue(binance_log.exists())
            self.assertTrue(deriv_log.exists())

    def test_binance_reconcile_flags_state_exchange_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            save_v2_shadow_state(
                state_path,
                V2ShadowState(positions={}, entry_prices={}, equity=5_000.0),
            )
            client = _FakeBinanceClient(
                [
                    {
                        "symbol": "LTCUSDT",
                        "positionAmt": "2.0",
                        "entryPrice": "50.0",
                        "markPrice": "55.0",
                        "notional": "110.0",
                    }
                ]
            )
            positions, entry_prices, meta, mismatch = reconcile_binance_positions(
                client,
                {"LTC": "LTCUSDT"},
                str(state_path),
            )

            self.assertTrue(mismatch)
            self.assertEqual(positions["LTC"], 110.0)
            self.assertEqual(entry_prices["LTC"], 50.0)
            self.assertEqual(meta["LTC"]["exchange_symbol"], "LTCUSDT")

    def test_binance_reconcile_allows_mark_to_market_notional_drift_when_qty_matches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            save_v2_shadow_state(
                state_path,
                V2ShadowState(
                    positions={"XAU": 105.0},
                    entry_prices={"XAU": 4496.05},
                    position_meta={"XAU": {"exchange_symbol": "XAUUSDT", "position_amt": 0.023}},
                ),
            )
            client = _FakeBinanceClient(
                [
                    {
                        "symbol": "XAUUSDT",
                        "positionAmt": "0.023",
                        "entryPrice": "4496.05",
                        "markPrice": "4602.62",
                        "notional": "105.86",
                    }
                ]
            )

            _, _, _, mismatch = reconcile_binance_positions(client, {"XAU": "XAUUSDT"}, str(state_path))

            self.assertFalse(mismatch)

    def test_deriv_reconcile_rebuilds_tracked_contracts_and_flags_unknowns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            save_v2_shadow_state(
                state_path,
                V2ShadowState(
                    positions={"DERIV_V75": 100.0},
                    entry_prices={"DERIV_V75": 1234.5},
                    position_meta={
                        "DERIV_V75": {
                            "deriv_contract_id": 42,
                            "deriv_multiplier": 50,
                            "deriv_stake": 2.0,
                            "deriv_direction": "up",
                            "deriv_symbol": "R_75",
                        }
                    },
                ),
            )

            positions, mismatch = _reconcile_deriv_positions(
                str(state_path),
                [{"contract_id": 42}, {"contract_id": 999}],
            )

            self.assertTrue(mismatch)
            self.assertIn("DERIV_V75", positions)
            self.assertEqual(positions["DERIV_V75"].contract_id, 42)
            self.assertEqual(positions["DERIV_V75"].entry_spot, 1234.5)


if __name__ == "__main__":
    unittest.main()
