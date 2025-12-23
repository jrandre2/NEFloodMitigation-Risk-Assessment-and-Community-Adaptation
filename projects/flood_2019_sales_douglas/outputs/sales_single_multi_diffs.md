# Sales Differences: Multi Individuals vs Multi LLCs
Window: ±730 days | Event: 2019-03-14

## Value model (OLS, HC3)
Terms: LLC_multi (pre baseline diff), post (post vs pre for Individuals), LLC_multi×post (differential post change for LLCs vs Individuals).
| term           |   pct_effect |   pct_ci_lo |   pct_ci_hi |   p_value |   n_obs |
|:---------------|-------------:|------------:|------------:|----------:|--------:|
| LLC_multi      |      -32.495 |     -56.959 |       5.874 |     0.087 |    6969 |
| post           |      -14.908 |     -42.811 |      26.609 |     0.426 |    6969 |
| LLC_multi_post |      364.188 |     155.486 |     743.374 |     0.000 |    6969 |

## Timing model (Logit, HC3)
Outcome: post (1=post-flood sale among sold). Report odds ratio and average marginal effect (pp).
| term      |   odds_ratio |   or_ci_lo |   or_ci_hi |   ame_pct_points |   ame_ci_lo_pp |   ame_ci_hi_pp |   p_value |   n_obs |
|:----------|-------------:|-----------:|-----------:|-----------------:|---------------:|---------------:|----------:|--------:|
| LLC_multi |        0.992 |      0.886 |      1.110 |           -1.102 |         -2.206 |          0.003 |     0.891 |    6969 |