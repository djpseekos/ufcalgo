==============================
UFC Data Scraper - Components
==============================

1. CLI (cli.py)
----------------
This is the entry point.

Lets you run different scrapers from the command line:

- events   → scrape UFC events
- fights   → scrape fights per event
- rounds   → scrape per-round fight stats
- fighters → scrape fighter profiles
- odds     → scrape betting odds from BestFightOdds
- context  → enrich with altitude, timezones, weight misses, fighter-event relationships
- all      → run the full pipeline in sequence


2. Events scraper (ufcstats_events.py)
--------------------------------------
- Pulls all past UFC events (date, city, country, venue, is_apex).
- Visits each event’s page for details (venue, corrected location).
- Saves into: data/curated/events.csv


3. Event → Fight scraper (ufcstats_event_fights.py)
---------------------------------------------------
- For each event, grabs all fights and metadata:
  • Fighter IDs/names
  • Weight class
  • Winner/loser corner
- Saves into: data/curated/fights.csv


4. Fight → Round stats scraper (ufcstats_fight_rounds.py)
---------------------------------------------------------
- For each fight, scrapes detailed round-by-round stats:
  • Strikes landed/attempted (by target + position)
  • Takedowns, control time, KDs, etc.
  • Method of finish, end round/time, judges’ scores
- Saves into: data/curated/stats_round.csv (round data)
- Updates:    fights.csv with fight-level details


5. Fighter profiles (ufcstats_fighters.py)
------------------------------------------
- Pulls fighter-specific info:
  • Name, DOB, height, reach, stance
- Saves into: data/curated/fighters.csv


6. Betting odds (bestfightodds.py)
----------------------------------
- Scrapes odds from BestFightOdds:
  • Opening & closing moneylines from sportsbooks
  • Converts to implied probabilities (raw + de-vigged)
- Saves into: data/curated/odds_snapshot.csv


7. Context enrichment (context_enrich.py)
-----------------------------------------
- Joins external info into your dataset:
  • Adds altitude & timezone from altitude_lookup.csv
  • Builds fighter_event.csv linking fighters ↔ events (including weight-miss data if available)


8. Tapology fills (tapology_fill.py)
------------------------------------
- Optional enrichment step:
  • Gets missing stance or camp/gym info from Tapology profiles
- Updates: fighters.csv


Pipeline Flow
=============
The "all" command runs things in order:

events → fights → rounds → fighters → context (+ optionally odds, tapology)

Final result: a relational dataset of events, fights, fighters,
round stats, and odds — usable for analysis, modeling, or betting research.


Author’s Intention
==================
Since the author generated all the code, the goal is likely that we:

- Understand and extend this pipeline, not just run it.
- Add features (new stats, other sources, ML prep).
- Have the whole pipeline in view, so it can be debugged, refactored,
  or extended with awareness of how the pieces connect.
