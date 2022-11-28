# ESTIME

ESTIME-soft and ESTIME-coherence were defined in [Consistency and Coherence from Points of Contextual Similarity](https://arxiv.org/abs/2112.11638). ESTIME as the 'number of alarms' was defined in [ESTIME: Estimation of Summary-to-Text Inconsistency by Mismatched Embeddings](https://aclanthology.org/2021.eval4nlp-1.10/).

ESTIME is a reference-free estimator of summary quality with emphasis on factual consistency. It can be used for filtering generated summaries, or for estimating improvement of a generation system.

Usage is simple: create `Estime`, and use `evaluate_claims`. When creating Estime, specify the list of names of the measures to obtain for each claim. Basic usage:

```python
>>> estimator = Estime()
>>> text = """In Kander’s telling, Mandel called him up out of the blue a decade or so ago to pitch a project. It made sense why. The two men had similar profiles: Jewish combat veterans in their early 30s. New statewide officeholders in the Midwest."""
>>> summary = """Kander and Mandel had similar profiles, and it makes sense."""
>>> estimator.evaluate_claims(text, [summary])
[[5]]
```

In this example only one summary is given to the text, and hence the list of results contains only one element [5] - the scores only for this summary. The scores list contains only single score =5, because by default the list of measures contains only one measure 'alarms'. More measures can be included, e.g.: 

```
>>> estimator = Estime(output=['alarms', 'soft', 'coherence'])
>>> estimator.evaluate_claims(text, [summary])
[[5, 0.502, -0.25]]
```

For more options, see the docstring in Estime in estime.py. 

A more straightforward version without research option 'coherence_ranges_min_max' is included in [blanc](https://pypi.org/project/blanc/) package.
