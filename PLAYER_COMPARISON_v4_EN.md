# Brentford dashboard — refined v4

Extract the complete folder and launch it as before:

```bat
python -m streamlit run app.py
```

Set your existing match-data folder and click **Load / apply settings** at the
top of the sidebar. No Python files need to be merged between folders.

## Player comparisons in stages 3 and 4

With no player filters, the existing comparison stays Brentford versus opponents
in the selected matches. The same shots, entries, categories and timing windows
are used as in the uploaded refined version.

When Passer, First player on the ball or Intended receiver has any selection:

- **Red — Selected situations:** Brentford long balls matching the active player
  filters.
- **Blue — Other situations:** the base Brentford sample minus those selected
  situations. This excludes the selected situations from the comparison total.
- The chart's delta is selected minus other, in percentage points.
- Both outcome metrics, the match graphs and the new comparison exports use these
  same disjoint cohorts. Selecting a single pass for the pitch does not reduce
  these aggregate comparisons to one pass.

Example: Intended receiver = Igor Thiago compares long balls aimed at Thiago
with the remaining Brentford long balls, not with the original total including
Thiago and not with opponent long balls.

Names within one field are combined. Different fields must all match. For
example, Passer = P and Intended receiver = T selects the P-to-T situations;
the remainder contains every situation not matching that complete combination.

Both groups retain the same selected matches, category, ending, opponent-response,
pass-count and manual-pass restrictions. If a manual pass list is selected, it
also limits the comparison base. Clear that list to compare against the wider
sample. Player identities that are unknown remain in the nonmatching remainder.
The grouped Opposing players first-controller option also activates this mode;
it still splits Brentford long-ball situations.

The split happens before each metric's eligibility and known-outcome checks.
Shots and final-third entries can therefore have different assessed denominators.
An entry requires starting outside the final third; moves already inside are
excluded. Empty groups and missing outcomes remain unavailable, not zero.
Differences describe this sample; they are not adjusted or causal player effects.

## Interface changes

- The dashboard now opens directly in the coach workspace. The Deeper analytics
  workspace and switch are removed from the app entry point. Shared calculations
  and supporting modules remain available to existing code.
- Stage 2 keeps receiver-centred and ball-centred support views, but removes
  Compare the two centres fairly. Each centre uses its available cases.
- Most explanatory prose is grouped under the collapsed How we count section.
  Circular question-mark popovers explain charts and active comparisons; settings
  use native help tooltips. Counts, axis labels and essential data errors remain
  visible.
- Nearest-control tolerance defaults to 2 seconds in the settings form, shared
  dashboard defaults and pipeline fallbacks. Existing explicitly loaded settings
  are respected until reapplied. Changing this tolerance can change estimated
  category assignments and downstream rates.
- The redesigned sidebar groups long-ball, timing, tracking and advanced settings
  in collapsed sections. Load / apply settings is at the top, outside those
  sections but inside the same form, so it applies all form edits together.
  Match and player filtering remain instant; data preparation remains cached.

## Files changed

| File | Purpose |
| --- | --- |
| `app.py` | Coach-only entry point; grouped sidebar and top settings button. |
| `coach_widgets.py` | Returns selection cohorts and renders matching comparison labels and deltas. |
| `player_comparison.py` (new) | Disjoint cohort splitting, all stage 3/4 outcome summaries and per-match comparison rows. |
| `presentation_views.py` | Wires comparisons into charts, match graphs, exports and explanations; removes the support-centre cohort selector. |
| `dashboard_data.py` | Default nearest-control tolerance and fallback are now 2 seconds. |
| `tests/test_player_comparison.py` (new) | Cohort arithmetic, missing data, eligibility, time-window and graph-count parity checks. |
| `tests/test_dashboard.py` | Exercises coach views, filters, empty/full selections and top-button submission. |

The A–F classifier, outcome models, sequence logic, tracking, ball support and
pitch-rendering modules are unchanged from the supplied ZIP. The coach view
retains individual-pass inspection, both support centres and velocity arrows.
Analysis modules belonging to the old deeper workspace are retained as source
files but are no longer imported or rendered by the app entry point.

Stage 3 still uses A/B final attacking phases and the configured window from
control/regain. Stage 4 still uses C/D final opponent phases, with 10/15-second
outcome panels and a fixed 15-second final-third-entry match graph. Category
membership already uses later events, so category-specific outcome rates are
descriptive route comparisons.

## Verification

All tests under `tests` passed, including synthetic CSV/JSON/Parquet loading and
Streamlit interactions. New checks cover multiple names, intersecting roles,
non-player filters, manual pass selection, unknown identities, empty/full groups,
entry exclusions, 10/15-second windows, matched graph counts, restored opponent
comparisons and settings submission. Python syntax compilation also passed.

Actual season outcomes were not rerun: raw match inputs were not supplied.
Browser screenshots were not used for visual verification. Existing parameter
check outputs in the ZIP are retained historical results; they have not been
recomputed for the new 2-second default. This is a local replacement package,
not a hosted deployment.
