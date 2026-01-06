## Comparing evaluation measures

This is a script for conveniency in considering or repeating the observations from ["How important is Recall for Measuring Retrieval Quality?"](https://arxiv.org/abs/2512.20854).

A run of run_corr.py on one of configuration examples given in config (or on a similar setup) produces maximal (over a range of &alpha;) correlations of selected measures with the 'grade' (LLM response quality score). As input, use samples_graded.json from the [retrieval-response](https://huggingface.co/datasets/primer-ai/retrieval-response) dataset. 
