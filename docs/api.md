# API

Import `sticky_hdp_slds`. That enables float64. State is a `NamedTuple` pytree, and each Gibbs block is a pure function of `(key, state, data)`.

`fit_regime_har` is the sticky regime HAR. `sample_slds` is the blocked sticky HDP-SLDS. `filter_regime_draws` is the forecast filter.

Plots in the examples need the `plots` extra.

## Regime HAR

::: sticky_hdp_slds.inference.regime_har.fit_regime_har

::: sticky_hdp_slds.inference.regime_har.sample_regime_har

::: sticky_hdp_slds.inference.regime_har.sample_regime_windows

## Sticky HDP-SLDS

::: sticky_hdp_slds.inference.slds.sample_slds

::: sticky_hdp_slds.inference.slds.slds_sweep

## Filter

::: sticky_hdp_slds.inference.hmm.filter_regime_draws

::: sticky_hdp_slds.inference.hmm.forward_filter

::: sticky_hdp_slds.inference.hmm.backward_sample
