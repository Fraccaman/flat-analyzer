"""Command-line entry point: `flatprice <command>`."""

from __future__ import annotations

import argparse
import functools
import http.server
import socketserver
import sys
from pathlib import Path

from flatprice.config import Config, load_config


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _select_searches(cfg: Config, args) -> Config:
    """Restrict cfg.searches to `--portal` / `--search` (repeatable) if given."""
    portals = {p.lower() for p in (getattr(args, "portal", None) or [])}
    labels = set(getattr(args, "search", None) or [])
    if not portals and not labels:
        return cfg
    chosen = [s for s in cfg.searches
              if (not portals or s.portal.lower() in portals) and (not labels or s.label in labels)]
    if not chosen:
        raise SystemExit("no configured search matches --portal/--search")
    return cfg.model_copy(update={"searches": chosen})


def cmd_scrape(cfg: Config, args) -> int:
    from flatprice.scrape import run_scrape

    cfg = _select_searches(cfg, args)
    summary = run_scrape(cfg)
    _log(f"run {summary['run_id']}: {summary['status']}, {summary['n_listings']} listings")
    for s in summary.get("searches", []):
        _log(f"  {s}")
    q = summary.get("quality") or {}
    if q:
        _log(f"  data quality: {q['flagged']}/{q['n']} flagged ({q['share'] * 100:.1f}%)"
             + (" -> POISONED run, listing fields not updated" if q.get("poisoned") else ""))
    if summary["status"] == "failed":
        return 1
    return 2 if q.get("poisoned") else 0


def cmd_quality(cfg: Config, args) -> int:
    """Recompute the data-quality flags and print a report."""
    from flatprice import quality
    from flatprice.analysis.clean import _to_bool
    from flatprice.dedup import dedupe
    from flatprice.store import Store

    store = Store(cfg.db_path)
    try:
        report = quality.apply(store, cfg.quality)
        frame = store.load_frame()
    finally:
        store.close()
    counts = None
    if not frame.empty:
        active = frame[frame["active"].map(_to_bool)] if not args.all else frame
        counts = quality.frame_counts(dedupe(active))
    print(quality.format_report(report, counts, cfg.quality))
    if args.show:
        flagged = frame[frame["data_flag"].notna()] if not frame.empty else frame
        cols = ["key", "rent_observed", "rent_gross", "rooms", "living_space_m2", "data_flag",
                "data_suspect", "data_corrected"]
        with __import__("pandas").option_context("display.width", 200, "display.max_rows", 500):
            print(flagged[cols].to_string(index=False))
    return 0


def cmd_seed(cfg: Config, args) -> int:
    from flatprice.store import Store

    store = Store(cfg.db_path)
    run_id = store.import_csv(args.csv)
    store.close()
    _log(f"seeded run {run_id} from {args.csv}")
    return 0


def _analyze(cfg: Config, include_inactive: bool = False):
    from flatprice.analysis.pipeline import analyze
    from flatprice.store import Store

    store = Store(cfg.db_path)
    try:
        frame = store.load_frame()
    finally:
        store.close()
    if frame.empty:
        raise SystemExit("No listings in database. Run `flatprice scrape` or `flatprice seed` first.")
    return analyze(frame, cfg, active_only=not include_inactive)


def cmd_analyze(cfg: Config, args) -> int:
    df, info = _analyze(cfg, args.all)
    cols = [c for c in ["score", "verdict", "rent_gross", "fair_rent", "rooms", "living_space_m2", "deal_pct",
                        "deal_pct_adj_eff", "p_below_fair", "model_confidence", "listing_kind",
                        "tax_delta_month", "municipality", "url"] if c in df]
    top = df.sort_values("score", ascending=False, na_position="last").head(args.top)
    with __import__("pandas").option_context("display.width", 200, "display.max_colwidth", 60):
        print(top[cols].to_string(index=False))
    if args.csv:
        df.to_csv(args.csv, index=False)
        _log(f"wrote {args.csv}")
    return 0


def cmd_build(cfg: Config, args) -> int:
    from flatprice.site.build import build_site

    df, info = _analyze(cfg, args.all)
    out = build_site(df, info, cfg, cfg.dist_dir)
    _log(f"site built in {out} ({len(df)} listings)")
    return 0


def cmd_run(cfg: Config, args) -> int:
    rc = cmd_scrape(cfg, args)
    if rc != 0:
        _log("scrape failed; not rebuilding site")
        return rc
    rc = cmd_build(cfg, args)
    from flatprice.store import Store

    store = Store(cfg.db_path)
    store.vacuum()
    store.close()
    return rc


def cmd_serve(cfg: Config, args) -> int:
    directory = Path(cfg.dist_dir).resolve()
    if not (directory / "index.html").exists():
        raise SystemExit(f"{directory}/index.html missing. Run `flatprice build` first.")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    with socketserver.TCPServer(("127.0.0.1", args.port), handler) as httpd:
        _log(f"serving {directory} at http://127.0.0.1:{args.port}/")
        httpd.serve_forever()
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="flatprice", description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_selectors(p):
        p.add_argument("--portal", action="append", metavar="NAME",
                       help="only run searches of this portal (repeatable)")
        p.add_argument("--search", action="append", metavar="LABEL",
                       help="only run the search with this label (repeatable)")

    p = sub.add_parser("scrape", help="scrape the configured searches into the database")
    add_selectors(p)
    p = sub.add_parser("seed", help="import an old homegate CSV as a seed run")
    p.add_argument("csv")
    p = sub.add_parser("analyze", help="score listings and print the top ones")
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--csv", help="write the full scored table to this CSV")
    p.add_argument("--all", action="store_true", help="include delisted flats")
    p = sub.add_parser("build", help="analyze and build the static site")
    p.add_argument("--all", action="store_true", help="include delisted flats")
    p = sub.add_parser("run", help="scrape + build (daily job)")
    p.add_argument("--all", action="store_true", help="include delisted flats")
    add_selectors(p)
    p = sub.add_parser("quality", help="recompute data-quality flags (perturbed portal data) and report")
    p.add_argument("--all", action="store_true", help="count delisted flats too")
    p.add_argument("--show", action="store_true", help="list listings whose latest data is flagged")
    p = sub.add_parser("serve", help="serve the built site locally")
    p.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    handlers = {
        "scrape": cmd_scrape, "seed": cmd_seed, "analyze": cmd_analyze,
        "build": cmd_build, "run": cmd_run, "serve": cmd_serve, "quality": cmd_quality,
    }
    sys.exit(handlers[args.command](cfg, args))


if __name__ == "__main__":
    main()
