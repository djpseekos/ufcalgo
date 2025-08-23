from __future__ import annotations
import argparse, sys

from src.scraper.ufcstats_events import main as events_main
from src.scraper.ufcstats_event_fights import main as fights_main
from src.scraper.ufcstats_fight_rounds import main as rounds_main
from src.scraper.ufcstats_fighters import main as fighters_main
from src.scraper.bestfightodds import main as odds_main
from src.scraper.context_enrich import main as context_main

def main():
    ap = argparse.ArgumentParser(prog="ufcalgo", description="UFC data scraper CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("events")
    sub.add_parser("fights")
    sub.add_parser("rounds")
    sub.add_parser("fighters")
    sub.add_parser("odds")
    sub.add_parser("context")

    p_all = sub.add_parser("all")
    p_all.add_argument("--since", default=None)
    p_all.add_argument("--limit-events", type=int, default=None)

    args, rest = ap.parse_known_args()

    if args.cmd == "events":
        sys.exit(events_main(rest))
    if args.cmd == "fights":
        sys.exit(fights_main(rest))
    if args.cmd == "rounds":
        sys.exit(rounds_main(rest))
    if args.cmd == "fighters":
        sys.exit(fighters_main(rest))
    if args.cmd == "odds":
        sys.exit(odds_main(rest))
    if args.cmd == "context":
        sys.exit(context_main(rest))
    if args.cmd == "all":
        # Run a sensible default pipeline
        events_main(["--since", args.since] if args.since else [])
        fights_main(["--limit-events", str(args.limit_events)] if args.limit_events else [])
        rounds_main([])
        fighters_main([])
        context_main([])
        print("[all] pipeline complete.")
        return 0

if __name__ == "__main__":
    main()