# Q1: Ranking Clients Before a Call

**Question:** Can we rank later-period clients before dialing to capture more subscribers than random selection within the same contact budget?

## Method

We preserved the chronological order of 41,188 records: the earliest 60% for training, the next 20% for validation, and the final 20% for testing. We excluded `duration` and conservatively excluded `campaign`, leaving 18 inputs. Encoding and scaling were fitted on training data only.

Logistic regression provided a linear reference; random forests captured nonlinear relationships. Validation Lift@10% was 0.460 for logistic regression, **1.217 for a depth-5 forest**, and 1.118 for a depth-10 forest. We selected the depth-5 forest (200 trees, minimum leaf size 20), refitted on the earliest 80%, and tested on the final 20%. The same configuration was separately trained on a stratified random 80% split (seed 42) for comparison.

We selected the top 10% of clients by predicted score. **Lift@10%** is their subscription rate divided by the overall rate in the same test set; random selection has expected lift 1. **Recall@10%** is the proportion of all subscribers captured.

## Results

| Metric | Time-based (primary) | Random (comparison) |
|---|---:|---:|
| Test records / selected clients | 8,238 / 824 | 8,238 / 824 |
| Overall subscription rate | 30.83% | 11.26% |
| Subscribers captured | **487** | 414 |
| Random expected capture | 254.06 | 92.82 |
| Precision@10% | **59.10%** | 50.24% |
| Recall@10% | **19.17%** | 44.61% |
| Lift@10% | **1.917** | 4.460 |

The chronological model captured about **233 more subscribers than random selection in expectation**, a **91.7% increase** at the same contact budget. Selecting 10% of clients captured 19.17% of subscribers. Random-split lift was higher, but its overall subscription rate was lower; differences in periods, prevalence, and training composition affect this comparison.

## Stability and Limitations

Validation-half lifts were 1.076 and 1.396, suggesting time-varying improvement; these are development diagnostics, not independent confirmation. Earlier EDA accessed the full dataset, while this run selected the model before final testing. Without client IDs, performance on entirely unseen people remains unverified. Historical ranking gains do not establish additional subscriptions or profit.

**Conclusion:** Under a 10% contact budget, the model improved subscriber ranking in later-period records, with gains varying over time.




# Q3: Are there ways to predict other attributes of clients based off of logical groups / clusters?
