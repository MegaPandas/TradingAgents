"""Re-run 002594.SZ with normalized EPS anchor."""
import os, logging
logging.basicConfig(level=logging.WARNING)

os.environ.setdefault("TRADINGAGENTS_LLM_PROVIDER", "deepseek")
os.environ.setdefault("TRADINGAGENTS_DEEP_THINK_LLM", "deepseek-v4-pro")
os.environ.setdefault("TRADINGAGENTS_QUICK_THINK_LLM", "deepseek-v4-flash")
os.environ.setdefault("TRADINGAGENTS_OUTPUT_LANGUAGE", "Chinese")
os.environ.setdefault("TRADINGAGENTS_RESULTS_DIR", "/home/mega/project/reports")

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

config = dict(DEFAULT_CONFIG)
config["llm_provider"] = "deepseek"
config["deep_think_llm"] = "deepseek-v4-pro"
config["quick_think_llm"] = "deepseek-v4-flash"
config["backend_url"] = "https://api.deepseek.com"
config["output_language"] = "Chinese"
config["results_dir"] = "/home/mega/project/reports"
config["max_debate_rounds"] = 2
config["max_risk_discuss_rounds"] = 2
config["checkpoint_enabled"] = False

selected = ("market", "social", "news", "fundamentals", "macro_policy", "business", "situation")

graph = TradingAgentsGraph(
    selected_analysts=selected,
    debug=False,
    config=config,
)

ticker = "002594.SZ"
trade_date = "2026-07-16"

print(f"Starting analysis for {ticker} on {trade_date}...")
try:
    final_state, signal = graph.propagate(ticker, trade_date, asset_type="stock")
    save_path = f"/home/mega/project/reports/{ticker}_20260716_v2"
    graph.save_reports(final_state, ticker, save_path=save_path)
    print(f"Done. Reports: {save_path}")
    print(f"Signal: {signal}")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()