# Model Governance

## Production Boundary

The current production decision path remains the legacy `composite_risk_score_smooth` and `final_decision`.

The institutional credit framework is a parallel explanatory layer. It improves communication and credit reasoning, but it should not replace the production composite until disagreement periods are reviewed and validated.

## What Is Production

- Legacy composite score.
- Four-regime final decision.
- Existing portfolio weights and risk-overlay logic.
- Existing validation and health checks.

## What Is Research / Explanatory

- Institutional six-channel credit score.
- Spread decomposition.
- HY/IG/BBB relative value.
- Rating bucket proxy view.
- Credit compensation scorecard.
- Refinancing wall framework.
- Credit tear sheet.
- Credit strategy memo.

## Promotion Criteria

Before replacing the legacy composite:

1. Compare legacy and institutional scores historically.
2. Identify dates with large score gaps.
3. Review whether gaps correspond to better or worse forward spread/equity outcomes.
4. Validate out of sample.
5. Replace public proxies with institutional data where possible.
6. Document false positives and false negatives.

## Known Limitations

- Public data are proxies for true institutional credit datasets.
- Expected loss uses simple regime-level PD assumptions.
- Recovery rate is fixed at 40%.
- Rating bucket and refinancing wall views are frameworks until proper data are supplied.
- Credit compensation scorecard bucket assumptions are transparent but heuristic; see `docs/credit_compensation_scorecard.md`.
- Macro data have publication lags and revisions.
- This is a risk overlay and strategy framework, not a standalone alpha model.
