# Data Dictionary

This dashboard uses public market and macro data. The table below separates
observed series from proxy or synthetic history so model outputs can be
interpreted with the right level of confidence.

| Series ID | Label | Provider | Column | Frequency | Quality | Transform | Limitation |
|---|---|---|---|---|---|---|---|
| DGS10 | 10Y Treasury Yield | FRED | yield_10y | daily | observed | Forward-filled up to 5 business days. | Treasury market holidays create short gaps. |
| DGS2 | 2Y Treasury Yield | FRED | yield_2y | daily | observed | Forward-filled up to 5 business days. | Treasury market holidays create short gaps. |
| UNRATE | Unemployment Rate | FRED | unemployment | monthly | observed | Snapped to business-day spine and forward-filled up to 35 days. | Monthly release, so current daily value is stale between releases. |
| NFCI | NFCI (Liquidity) | FRED | nfci | weekly | observed | Forward-filled up to 7 days. | Weekly financial-conditions index, not a real-time tradeable price. |
| BAMLH0A0HYM2 | HY Credit Spread | FRED/ICE BofA | hy_spread | daily | observed_with_proxy_history | ICE HY OAS where available; pre-ICE period can use scaled BAA10Y fallback. | Pre-ICE history is a public proxy and should not be treated as bond-index truth. |
| SP500 | S&P 500 | yfinance/FRED | sp500 | daily | observed | Uses yfinance ^GSPC first, with FRED fallback. | FRED SP500 history starts later than yfinance. |
| VIXCLS | VIX | FRED | vix | daily | observed | Forward-filled up to 5 business days. | Holiday gaps are possible. |
| T10YIE | 10Y Breakeven | FRED | breakeven_10y | daily | observed_with_synthetic_history | Observed from 2003 onward; earlier rows can use a constant long-run assumption. | Pre-2003 values are synthetic and flagged via breakeven_imputed. |
| BAMLC0A0CM | IG OAS | FRED/ICE BofA | ig_spread | daily | observed_limited_history | Forward-filled up to 5 business days. | Public FRED availability is limited in this dataset. |
| BAMLC0A4CBBB | BBB OAS | FRED/ICE BofA | bbb_spread | daily | observed_limited_history | Forward-filled up to 5 business days. | Public FRED availability is limited in this dataset. |
| BAMLH0A0HYM2EY | HY Effective Yield | FRED/ICE BofA | hy_yield | daily | observed_limited_history | Forward-filled up to 5 business days. | Public FRED availability is limited in this dataset. |
| BAMLC0A0CMEY | IG Effective Yield | FRED/ICE BofA | ig_yield | daily | observed_limited_history | Forward-filled up to 5 business days. | Public FRED availability is limited in this dataset. |
| DRTSCILM | SLOOS (C&I Tightening) | FRED | sloos_ci | quarterly | observed | Forward-filled up to 95 days. | Survey data; not a market price and updates quarterly. |
| DRBLACBS | Business Loan Delinquency Rate | FRED | ci_loan_delinquency | quarterly | observed | Forward-filled up to 95 days. | Quarterly bank-reported delinquency rate; lagging realized credit stress. |
| CORBLACBS | Business Charge-Off Rate | FRED | business_chargeoff_rate | quarterly | observed | Forward-filled up to 95 days. | Quarterly bank-reported charge-off rate; lagging realized credit losses. |

## Governance Notes

- `observed` means the column comes directly from a public data source after
  calendar alignment and limited forward fill.
- `observed_limited_history` means the series is real, but public availability
  leaves a shorter sample than the full dashboard history.
- `observed_with_proxy_history` means recent observations are real but older
  history may be a public proxy.
- `observed_with_synthetic_history` means part of the sample uses an explicit
  assumption when observations are unavailable.
- Public proxies are acceptable for demo and research use, but production use
  should replace them with licensed index, TRACE, issuer fundamental, default,
  and rating-transition datasets.
