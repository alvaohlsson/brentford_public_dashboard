# Reference xT surface

`open_xt_12x8_v1.json` is Karun Singh's public 12-by-8 reference xT grid, downloaded on 2026-10-15 from:
https://karun.in/blog/data/open_xt_12x8_v1.json

Primary explanation: https://karun.in/blog/expected-threat.html
Socceraction documents this as a league-wide 2017/18 Premier League surface:
https://socceraction.readthedocs.io/en/latest/documentation/intro.html

The file is kept byte-for-byte as downloaded. `sequence_model.py` reverses row order on load to map the published top-to-bottom grid to ascending pitch y. Pitch coordinates are centred at (0,0), standardized to 105 × 68 m and attacking toward +x before lookup. Out-of-pitch coordinates return missing values; they are not silently clipped to a valuable zone.

This is an external reference surface, not a model fitted or calibrated to the current Brentford matches. It is independent of the provider's `player_targeted_xthreat` and EPV fields. The original provider fields remain in the long-ball export for audit. Do not compare their numeric scale directly with this reference xT.

No network call is made to obtain the xT grid while using the dashboard. The bundled file makes the model reproducible and usable offline.
