# Close snapshot

The file is a daily close series from the Yahoo Finance chart API (`interval=1d`), downloaded 2026-10-11. The Bitcoin study reads this file and does not download anything.

- `btc_daily.csv`: `BTC-USD`, 2014-09-17 through 2026-10-10 (4407 rows). The download also contained 2026-10-11, the download date. That row is dropped because the session was still incomplete. Every remaining date is the next calendar day.

`python -m examples.bitcoin` scores the last 365 calendar days. Each refit uses a trailing four-year window.
