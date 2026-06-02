# Credit Compensation Scorecard Methodology

## Objective

The credit compensation scorecard translates spread valuation, expected loss, default-cycle pressure, lending standards, and market technicals into a portfolio-facing credit view.

The scorecard is designed to answer:

- Are investors being paid enough to own credit beta?
- Which rating buckets deserve incremental risk budget?
- What should fund the next marginal allocation shift?
- How much downside is implied by spread widening?
- What would change the portfolio view?

This is an explanatory portfolio framework, not a standalone trading model.

## Core Inputs

The scorecard uses the same public-data framework as the dashboard:

- HY and IG OAS.
- HY and IG spread percentiles.
- HY/IG and BBB/IG relative-value ratios.
- Regime-based expected loss from spread decomposition.
- SLOOS lending-standard change.
- Charge-off and delinquency trends.
- Composite risk score and final regime label.
- Historical forward spread changes when available.

Spread inputs are normalized to basis points so public series stored as percentage points and synthetic test inputs are handled consistently.

## Spread Compensation

The scorecard starts from spread decomposition:

```text
Expected Loss Spread = Default Probability x Loss Given Default
Excess Spread = HY OAS - Expected Loss Spread
Compensation Ratio = HY OAS / Expected Loss Spread
```

The current implementation uses a fixed 40% recovery assumption through the spread-decomposition module. That implies 60% loss given default.

The scorecard views spread compensation as attractive only when spread exceeds expected loss by enough margin to absorb uncertainty, liquidity risk, and downgrade/default-cycle pressure.

## Recommendation Logic

The scorecard recommendation is one of:

- **Add**: compensation is attractive, spreads are historically wide, and fundamentals/lending standards are not deteriorating.
- **Hold**: compensation is adequate but not decisive.
- **Upgrade Quality**: spreads are rich enough that lower-quality beta is not well compensated.
- **Hedge**: compensation is weak or fundamentals are worsening, but conditions do not require broad de-risking.
- **De-risk**: valuation is rich with deteriorating fundamentals, or the composite risk score is defensive.

The recommendation is intentionally rule-based and transparent. It is meant to support PM discussion, not to hide judgment inside an opaque optimizer.

## Rating-Bucket Allocation

The scorecard maps the recommendation into target weights across:

```text
IG, BBB, BB, B, CCC, Cash, Hedge
```

The base allocation differs by recommendation, then adjusts for:

- HY cheapness versus IG.
- BBB pressure versus broad IG.
- lending-standard tightening.
- charge-off or delinquency deterioration.
- rich HY valuation.

The allocation favors BB/B when compensation is attractive and conditions are stable. It cuts CCC and shifts toward IG/cash/hedges when lending standards or realized credit losses deteriorate.

## Bucket Expected Return

Each bucket receives an expected excess return estimate:

```text
Expected Excess Return =
    Spread Carry
  - Expected Default Drag
  + Expected Spread Mark-to-Market
```

Where:

- **Spread Carry** uses current IG or HY spread anchors multiplied by bucket carry factors.
- **Expected Default Drag** uses regime-level expected HY loss multiplied by bucket loss factors.
- **Expected Spread Mark-to-Market** uses spread duration, spread beta, and the expected HY spread move.

The exposed assumptions table shows each bucket's spread source, carry factor, spread beta, loss factor, spread duration, and recession-widening assumption.

## Historical Analog Blend

When forward spread columns are available, the scorecard searches prior observations with similar:

- HY spread percentile,
- spread compensation ratio,
- excess spread.

If the analog sample is adequate, the expected HY spread move blends:

```text
50% rule-based expected spread move
50% historical analog median forward HY spread change
```

If no adequate analog sample exists, the scorecard falls back to the rule-based expected spread move.

The scorecard reports whether expected returns used:

- `Rules`
- `Blended historical analogs + rules`

## Risk / Reward Metrics

The scorecard summarizes allocation quality with:

- expected return / stress loss,
- carry / stress loss,
- B/CCC tail weight,
- B/CCC stress share,
- hedge stress offset.

These metrics make clear whether the allocation is being paid for downside risk, and whether lower-quality HY buckets dominate recession loss.

## Marginal Allocation Advice

The marginal allocation table answers:

```text
Where should the next 5% of risk budget go, and what should fund it?
```

The primary add recommendation selects the best eligible bucket by stress-adjusted expected return. The funding source is usually cash, a weak expected-return bucket, or another lower-ranked exposure.

The scorecard can also add risk-control rows, such as trimming B/CCC into hedges when lower-quality stress share is elevated.

Impact columns show the estimated portfolio-level expected-return change and stress-loss change from the 5% shift.

## Spread Shock Sensitivity

The spread shock table estimates mark-to-market sensitivity under parallel spread widening:

```text
+25 bps
+50 bps
+100 bps
```

For each bucket:

```text
Shock Loss = Spread Duration x Spread Beta x Spread Shock
Weighted Shock Loss = Shock Loss x Target Weight
```

The hedge bucket has negative spread beta, so it offsets widening losses. The portfolio row sums weighted losses across buckets.

## Net Spread Beta

The net spread beta table turns the rating-bucket allocation into HY-equivalent spread exposure.

For each bucket:

```text
Weighted Spread Beta = Target Weight x Bucket Spread Beta
Weighted Duration Beta = Target Weight x Bucket Spread Beta x Bucket Spread Duration
```

The portfolio row reports:

- gross long spread beta,
- hedge spread beta,
- net spread beta,
- HY-equivalent exposure,
- estimated +100 bps parallel spread-shock loss.

This table is the cleanest way to see whether the scorecard is adding credit beta, staying neutral, or reducing spread exposure after cash and hedge buckets.

## CDX Hedge Sizing

The CDX hedge-sizing table uses current net spread beta and the scorecard recommendation to estimate additional CDX HY protection.

Recommendation targets are intentionally simple:

```text
Add             0.75x net spread beta
Hold            0.45x net spread beta
Upgrade Quality 0.30x net spread beta
Hedge           0.20x net spread beta
De-risk         0.05x net spread beta
```

If current net beta is above the target, the table estimates:

- incremental CDX HY protection as percent of NAV,
- target hedge bucket,
- post-trade net spread beta,
- +100 bps loss reduction,
- estimated annual carry cost.

This is not a full CDX pricing model. It is a portfolio-sizing bridge between the scorecard recommendation and an implementable index-hedge posture.

## Validation And Replay

The validation section checks whether historical recommendations aligned with subsequent HY and IG spread moves.

Key views:

- **Validation / backtest**: forward spread outcomes by recommendation and horizon.
- **Transition stability**: recommendation changes, episode durations, whipsaw rate, and outcomes after transitions.
- **False positive / false negative episodes**: active calls followed by unfavorable material spread moves, and Hold calls that missed material tightening or widening.
- **Stress episode replay**: named spread-stress windows, the recommendation at the start of each episode, and whether the scorecard entered stress risk-on, neutral, or defensive.

Validation confidence is sample-size aware. Thin historical samples should be read as diagnostic context, not as conclusive statistical proof.

## Audit Trail

The audit trail records the current scorecard inputs and rule checks behind the recommendation.

The Inputs tab shows the observed values used by the scorecard, including expected loss, excess spread, compensation ratio, HY percentile, composite risk score, SLOOS change, charge-off change, and delinquency change.

The Rules tab shows whether each decision rule fired:

- composite risk veto,
- very rich spread,
- cheap spread,
- SLOOS tightening,
- fundamental worsening,
- underpaid stress,
- Add compensation gate,
- Add score gate.

The audit trail is designed to make the recommendation reviewable by a PM, risk committee, or future model reviewer without reading the source code.

## PM Review Packet

The scorecard section includes two PM packet downloads:

- **Download PM review packet**: markdown packet for written review.
- **Download PM packet CSV bundle**: ZIP file with the key scorecard tables as CSVs.

The packet bundles:

- current recommendation and PM final verdict,
- trade memo,
- portfolio expression,
- rating-bucket allocation,
- risk/reward table,
- marginal allocation advice,
- net spread beta,
- CDX hedge sizing,
- validation/backtest,
- transition outcomes,
- prediction errors,
- stress replay,
- audit inputs and audit rules.

The intended PM workflow is:

1. Read the PM final verdict and current snapshot.
2. Check rating-bucket allocation and marginal allocation advice.
3. Review net spread beta and CDX hedge sizing for risk budget implementation.
4. Confirm validation, transition stability, false positives/false negatives, and stress replay.
5. Use the audit trail to verify why the rule engine produced the recommendation.
6. Download the PM packet for committee review or model documentation.

## PM Final Verdict

The final verdict condenses the scorecard into one portfolio sentence:

- current recommendation,
- primary marginal trade,
- expected excess return,
- +100 bps spread shock loss,
- key hedge or exit trigger.

It is designed for PM review, interview walkthroughs, and dashboard summaries.

## Limitations

- Public spread series are proxies and do not provide full rating-bucket OAS, duration, convexity, liquidity, or sector composition.
- Bucket duration, beta, default drag, and recession-widening assumptions are transparent but heuristic.
- Expected loss uses regime-level default probability and fixed recovery, not issuer-level hazard models.
- Historical analogs are sample-dependent and should be treated as directional when sample sizes are thin.
- The hedge bucket is a simplified representation of index protection, not a full CDX hedge model with carry, roll, upfront, and basis.
- The scorecard is an explanatory allocation framework and should be validated before being used as a production optimizer.

## Production Boundary

The scorecard should be treated as a portfolio interpretation layer. The legacy composite score and final regime decision remain the production decision path until scorecard recommendations are validated out of sample and disagreement periods are reviewed.
