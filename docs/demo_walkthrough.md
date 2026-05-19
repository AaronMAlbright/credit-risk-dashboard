# Demo Walkthrough

## 30-Second Pitch

This is a macro-credit regime framework built from public macro, rates, liquidity, credit, and cross-asset data. It decomposes credit risk into institutional channels, evaluates spread compensation after expected loss, validates regimes against forward market behavior, and maps the current environment to credit positioning: HY/IG tilt, quality bias, duration stance, liquidity, and hedge posture.

## Five-Minute Walkthrough

1. Open the Streamlit app.

   ```bash
   streamlit run streamlit_app.py
   ```

2. Start on **Credit View**.

   Lead with the headline credit brief:

   - current regime,
   - risk score,
   - credit beta stance,
   - HY tilt,
   - valuation bucket.

3. Show **Spread Compensation**.

   Explain:

   ```text
   Expected Loss = Default Probability x Loss Given Default
   Excess Spread = OAS - Expected Loss
   ```

   The point is not that this is a full structural credit model. The point is that the framework ties spread levels to default compensation rather than treating wide/tight spreads mechanically.

4. Show **Credit Relative Value**.

   Highlight:

   - HY spread percentile,
   - IG spread percentile,
   - HY/IG ratio,
   - BBB/IG ratio where available,
   - quality tilt.

5. Show **Channel Contribution Attribution**.

   Explain which channel is actually driving the current score: macro cycle, rates/liquidity, credit market, fundamentals, technicals, or cross-asset confirmation.

6. Show **Regime-Conditioned Forward Performance**.

   This is the evidence layer. Point out sample size and confidence status. Avoid overclaiming.

7. Open **Model Governance**.

   Emphasize:

   - legacy composite remains production,
   - institutional channel score is parallel/research,
   - public data are proxies,
   - production version needs TRACE, rating-bucket OAS, issuer fundamentals, CDX, defaults, and maturity schedules.

8. Open **Composite Comparison And Case Studies**.

   Explain disagreement between the legacy and institutional composites before proposing replacement.

## Strong Interview Answers

### What is the project?

It is a macro-credit regime and positioning framework. It does not try to be a black-box alpha model. It is designed to help decide whether credit beta is being compensated and what type of credit exposure is appropriate.

### What is the most important output?

The Credit View: current regime, spread compensation, relative value, channel contribution, regime evidence, and positioning.

### What would you improve with institutional data?

I would replace public proxies with bond-level and index-level data: rating bucket OAS, sector OAS, TRACE liquidity, issuer leverage and coverage, default histories, recovery histories, CDX HY/IG, primary issuance, and maturity wall data.

### What are the biggest limitations?

The expected-loss model uses simple regime-level PD assumptions. The rating bucket and refinancing wall views are frameworks until better data are provided. The institutional composite runs alongside the legacy score and should not replace it until disagreement periods are reviewed.

### Why is this credible?

Because the model separates credit risk into economic channels, ties spreads to expected loss, validates regimes against forward outcomes, and clearly distinguishes production logic from research/explanatory overlays.

## Suggested Demo Close

The main design choice was transparency over complexity. Every score should answer three questions:

- what economic channel does it measure?
- what evidence supports it?
- what portfolio action does it imply?

