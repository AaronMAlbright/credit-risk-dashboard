# Interview Pitch

## 30-Second Version

I built a macro-credit regime framework that combines public macro, rates, liquidity, spread, credit-fundamental, technical, and cross-asset data to classify the current credit environment. The framework maps regimes to portfolio actions such as HY versus IG tilt, quality bias, cash level, duration stance, and hedge posture. I also validate the regimes against forward spread moves and equity returns so the output is tied to market behavior rather than just indicator levels.

## Stronger Finance Framing

The model is organized around six channels:

- Macro cycle risk
- Rates and liquidity risk
- Credit market risk
- Credit fundamentals
- Market technicals
- Cross-asset confirmation

The key idea is that credit spreads are not enough by themselves. A wide spread can be attractive if expected losses are contained, but unattractive if default risk is rising faster than compensation. So the framework includes a spread decomposition:

```text
Expected Loss = Default Probability x Loss Given Default
Excess Spread = OAS - Expected Loss
```

That lets the dashboard distinguish nominally wide spreads from genuinely attractive compensation.

## What I Would Emphasize In A Banking Interview

- I did not want a black-box signal. I wanted the model to decompose credit risk into economic channels that a credit strategist or risk manager would recognize.
- The output is not just "Risk-On" or "Risk-Off." It maps to implementable positioning: HY/IG tilt, quality upgrade or downgrade, duration stance, liquidity buffer, and potential index hedging.
- I included validation because regime models are easy to overfit. The framework checks forward return and spread behavior by regime and flags sample-size confidence.
- I treated public ETF and index data as proxies, not perfect institutional datasets. A production version would use bond-level index data, TRACE, issuer fundamentals, and rating bucket histories.

## Good Answer To "What Would You Improve?"

The biggest improvement would be replacing public proxies with institutional data: bond-level OAS, rating bucket returns, issuer leverage and coverage, default history, TRACE liquidity, and sector-level spread curves. I would also estimate default probabilities by rating and sector rather than using regime-level assumptions. That would make the expected-loss and excess-spread decomposition much closer to how a real credit desk would evaluate compensation.

## Good Answer To "Why Is This Useful?"

It helps avoid treating all spread widening the same. Some widening is an opportunity because expected losses are stable and liquidity premia are elevated. Other widening is dangerous because fundamentals and funding conditions are deteriorating at the same time. The framework tries to separate those cases and turn them into a consistent portfolio stance.

