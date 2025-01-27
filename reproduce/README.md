# Reproduce

To reproduce all the results in the paper, we provide all meta-evaluation datasets, codes, and evaluation scores predicted by UniEval here.

## Meta-Evaluation Benchmarks
Experiments are conducted on four tasks as follows:

- Text Summarization: [SummEval](data/summarization/summeval.json)
- Dialogue Response Generation: [Topical_Chat](data/dialogue/topical_chat.json)
- Data-to-text: [SFRES](data/data2text/sfres.json) and [SFHOT](data/data2text/sfhot.json)
- Facutal Consistency: [QAGS-CNNDM](data/fact/qags_cnndm.json) and [QAGS-XSum](data/fact/qags_xsum.json)

Please note that the overall score in SummEval is the average score of the four dimensions, while the overall scores in other benchmarks are human-annotated scores.

## Calculate Correlations with Human Scores
To verify that the proposed evaluator is qualified, we need to calculate correlations with human scores in each benchamark.

We provide scripts to automatically get evaluation scores and correlations. For example, for summarization, run the following script:
```
./eval_summarization.sh
```
The results of the predicted scores will be stored in the `predict/summarization` folder. It will then calculate the correlations between the predicted scores and the human judgments, and the results will be printed on the screen:
```
 ********** Sample Level Correlations *********
+-------------+----------+----------+----------+
|  Dimensions | Pearson  | Spearman | Kendall  |
+-------------+----------+----------+----------+
|  coherence  | 0.533249 | 0.591811 | 0.424627 |
| consistency | 0.634377 | 0.434997 | 0.349272 |
|   fluency   | 0.597067 | 0.451053 | 0.353974 |
|  relevance  | 0.434236 | 0.465623 | 0.337676 |
|   overall   | 0.69961  | 0.658277 | 0.476311 |
+-------------+----------+----------+----------+

 ********* Summary Level Correlations *********
+-------------+----------+----------+----------+
|  Dimensions | Pearson  | Spearman | Kendall  |
+-------------+----------+----------+----------+
|  coherence  | 0.553818 | 0.575186 | 0.44249  |
| consistency | 0.648491 | 0.445596 | 0.370913 |
|   fluency   | 0.605978 | 0.449168 | 0.370628 |
|  relevance  | 0.416225 | 0.42569  | 0.324938 |
|   overall   | 0.698316 | 0.647441 | 0.496725 |
+-------------+----------+----------+----------+

 ********** System Level Correlations *********
+-------------+----------+----------+----------+
|  Dimensions | Pearson  | Spearman | Kendall  |
+-------------+----------+----------+----------+
|  coherence  | 0.810345 | 0.811765 | 0.683333 |
| consistency | 0.945761 | 0.911765 |   0.75   |
|   fluency   | 0.908509 | 0.844739 | 0.661094 |
|  relevance  | 0.900644 | 0.838235 | 0.666667 |
|   overall   | 0.967897 | 0.894118 | 0.733333 |
+-------------+----------+----------+----------+
```
Results for dialogue response generation should be:
```
 ************** Turn Level Correlations *************
+-------------------+----------+----------+----------+
|     Dimensions    | Pearson  | Spearman | Kendall  |
+-------------------+----------+----------+----------+
|    naturalness    | 0.443666 | 0.513986 | 0.373973 |
|     coherence     | 0.595143 | 0.612942 | 0.465915 |
|    engagingness   | 0.55651  | 0.604739 | 0.455941 |
|    groundedness   | 0.536209 | 0.574954 | 0.451533 |
| understandability | 0.380038 | 0.467807 | 0.360741 |
|      overall      | 0.632796 | 0.662583 | 0.487272 |
+-------------------+----------+----------+----------+
```
Results for data-to-text should look like:
```
SFRES:
 ************ Sample Level Correlations ***********
+-----------------+----------+----------+----------+
|    Dimensions   | Pearson  | Spearman | Kendall  |
+-----------------+----------+----------+----------+
|   naturalness   | 0.367252 | 0.333399 | 0.247094 |
| informativeness | 0.282079 | 0.224918 | 0.169297 |
|     overall     | 0.370815 | 0.291593 | 0.214708 |
+-----------------+----------+----------+----------+

SFHOT:
+-----------------+----------+----------+----------+
|    Dimensions   | Pearson  | Spearman | Kendall  |
+-----------------+----------+----------+----------+
|   naturalness   | 0.397428 | 0.319813 | 0.237635 |
| informativeness | 0.357353 | 0.249329 | 0.191217 |
|     overall     | 0.406425 | 0.320721 | 0.236024 |
+-----------------+----------+----------+----------+
```
Results of factual consistency detection are:
```
QAGS_Xsum:
 ********** Sample Level Correlations *********
+-------------+----------+----------+----------+
|  Dimensions | Pearson  | Spearman | Kendall  |
+-------------+----------+----------+----------+
| consistency | 0.461376 | 0.48792  | 0.399218 |
+-------------+----------+----------+----------+

QAGS_CNNDM:
 ********** Sample Level Correlations *********
+-------------+----------+----------+----------+
|  Dimensions | Pearson  | Spearman | Kendall  |
+-------------+----------+----------+----------+
| consistency | 0.681681 | 0.662255 | 0.531636 |
+-------------+----------+----------+----------+
```

## Predicted Scores
[unieval_predict](./unieval_predict) folder contains the evaluation scores of UniEval on all meta-evaluation benchmarks.

---

## Adaption for TS

As an initial inspection, we score the summarization model's predictions against Human Ratings from ASSET.
```
# prepare ASSET data for scoring (may need to change hardcoded path to ASSET dir)
python get_annotated_asset_data.py
# compute correlation statistics
./eval_simplification.sh
```
Results of summarization eval model on simplification:
```
********** Sample Level Correlations *********
+-------------+----------+----------+----------+
|  Dimensions | Pearson  | Spearman | Kendall  |
+-------------+----------+----------+----------+
|  coherence  | 0.417719 | 0.523525 | 0.364111 |
| consistency | 0.383966 | 0.474313 | 0.330855 |
|   fluency   | 0.511344 | 0.537332 | 0.387069 |
+-------------+----------+----------+----------+

 ********* Summary Level Correlations *********
+-------------+-----------+-----------+-----------+
|  Dimensions |  Pearson  |  Spearman |  Kendall  |
+-------------+-----------+-----------+-----------+
|  coherence  | -0.024328 |  -0.02419 | -0.019136 |
| consistency | -0.011192 |   -0.013  | -0.014113 |
|   fluency   |  -0.02327 | -0.015393 | -0.013951 |
+-------------+-----------+-----------+-----------+
```
Single-dimensional UinEval for consistency (QAGS) has lower correlation than the multi-dimensional model:
```
 ********** Sample Level Correlations *********
+-------------+----------+----------+----------+
|  Dimensions | Pearson  | Spearman | Kendall  |
+-------------+----------+----------+----------+
| consistency | 0.308545 | 0.400624 | 0.282446 |
+-------------+----------+----------+----------+
```
