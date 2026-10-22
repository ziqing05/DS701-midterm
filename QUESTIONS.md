<<<<<<< HEAD
# DS701 Midterm Project - Section A3
#### Ye Ziqing, Sachiel Chuckrow, Will Stiller
Checkpoint 01 (10/15) |
[GitHub Repo](https://github.com/ziqing05/DS701-midterm)

### Dataset: Bank marketing — `bank-additional-full.csv`
## Data Summary
[Download the Dataset Here](https://archive.ics.uci.edu/static/public/222/bank+marketing.zip).

The dataset contains 41188 anonymized profiles aquired from a term-deposit campaign conducted by a Portugese bank. The data contained within includes 20 data attributes relating to each profile's personal information, interaction with current and previous campaigns, and influencing social and economic factors. The final attribute `y` corresponds to the output variable of the campaign -- did the client subscribe to a term-deposit? Additional dataset information was found inside `bank-additional-names.txt`, including attribute descriptions and datatypes. From some exploratory searching, each attribute listed in `bank-additional-full.csv` follows the table below.

| Field Name | Data Type | Description | Example Values | Notes |
|------------|-----------|-------------|----------------|-------|
| **age** | Integer | Client's age in years | 56, 57, 37, 40, 45 | Numeric and continuous.<br> Most of them are between 28-48 years old; mean:40 median:38 |
| **job** | String (categorical, nominal) | Client's type of job | `admin`, `blue-collar`, `entrepreneur`, `housemaid`, `management`, `retired`, `self-employed`, `services`, `student`, `technician`, `unemployed`, `unknown` | `admin`~25.30%;<br>`blue-collar`~22.47%;<br>`technician`~16.37%|
| **marital** | String (categorical, nominal) | Marital status | `divorced`, `married`, `single`, `unknown` | `divorced` means divorced or widowed.|
| **education** | String (categorical, ordinal) | Highest education level | `basic.4y`, `basic.6y`, `basic.9y`, `high.school`, `illiterate`, `professional.course`, `university.degree`, `unknown` | Has an arguable natural order of progression |
| **default** | String (categorical) | Has credit in default? | `no`, `yes`, `unknown` | `yes` is extremely rare in the full dataset (~0.01%). |
| **housing** | String (categorical) | Has a housing loan? | `no`, `yes`, `unknown` | ~52.4% `yes` |
| **loan** | String (categorical) | Has a personal loan? | `no`, `yes`, `unknown` | ~82.4% `no`|
| **contact** | String (categorical) | Contact communication type | `cellular`, `telephone` |  |
| **month** | String (categorical, ordinal) | Month of the last contact | `jan`, `feb`, `mar`, ..., `nov`, `dec` | Lowercase 3-letter abbreviations. |
| **day_of_week** | String (categorical, ordinal) | Weekday of the last contact | `mon`, `tue`, `wed`, `thu`, `fri` | Lowercase 3-letter abbreviations. |
| **duration** | Integer | Duration of the last contact, in seconds | 261, 149, 226, 151, 307, 198 | Numeric. |
| **campaign** | Integer | Number of contacts made to this client during this campaign | 1, 2, 3, 4, 5 |  |
| **pdays** | Integer | Days since the client was last contacted in a previous campaign | 1, 4, 5, 999 | 999 means not previously contacted. |
| **previous** | Integer | Number of contacts made to this client before this campaign | 0, 1, 2| Count, mostly 0. |
| **poutcome** | String (categorical, nominal) | Outcome of the previous marketing campaign | `failure`, `nonexistent`, `success` | Value `nonexistent` lines up with previous = 0 |
| **emp.var.rate** | Float | Employment variation rate (quarterly indicator) | 1.1 | Social/economic context attribute, consistent across all profiles |
| **cons.price.idx** | Float | Consumer price index (monthly indicator) | 93.994 | Social/economic context attribute |
| **cons.conf.idx** | Float | Consumer confidence index (monthly indicator) | -36.4 | Social/economic context attribute, negative values are normal |
| **euribor3m** | Float | Euribor 3-month interest rate (daily indicator) | 4.857 | Social/economic context attribute |
| **nr.employed** | Float | Number of employees (quarterly indicator) | 5191 | Social/economic context attribute |
| **y** | String (categorical, binary) | **Target variable.** Did the client subscribe to a term deposit? | `yes`, `no` | ~11.3% `yes` |
## Insights
`duration` this attribute highly affects the output target (e.g., if duration=0 then y="no"). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

`campaign`: Count, right-skewed with some large outliers

Majority of `previous` is 0, `poutcome` is more likely to be "failure" when `previous = 1`. `success`

Only 11.3% of clients ever say yes. A predictive model that says no all the time would be correct 88.7% of the time. This is a class imbalance that will make predictions difficult to make.

## Questions

The dataset is suited towards *predictive* questions and answers. While there is possibility for other categories, predictive remains the most prominent due to the presence of the outcome variable `y`.

**Predictive:** 
* Prior to dialing, can we rank future clients based on attributes to capture more term-subscribers than calling at random?
* Can we predict if new clients will subscribe based on collected attributes?

### Additional Questions

I've also listed some other questions that could work with the dataset.

**Exploratory:**
* Is calling effort spread evenly across clients, or concentrated on a few?

**Inferential:**
* Is the number of contacts per client consistent with a constant-rate (Poisson) process?
* Do clients who subscribed receive a different number of calls than those who didn’t?
=======
# DS701 Midterm Project - Section A3
#### Ye Ziqing, Sachiel Chuckrow, Will Stiller
Checkpoint 01 (10/15) |
[GitHub Repo](https://github.com/ziqing05/DS701-midterm)

### Dataset: Bank marketing — `bank-additional-full.csv`
## Data Summary
[Download the Dataset Here](https://archive.ics.uci.edu/static/public/222/bank+marketing.zip).

The dataset contains 41188 anonymized profiles aquired from a term-deposit campaign conducted by a Portugese bank. The data contained within includes 20 data attributes relating to each profile's personal information, interaction with current and previous campaigns, and influencing social and economic factors. The final attribute `y` corresponds to the output variable of the campaign -- did the client subscribe to a term-deposit? Additional dataset information was found inside `bank-additional-names.txt`, including attribute descriptions and datatypes. From some exploratory searching, each attribute listed in `bank-additional-full.csv` follows the table below.

| Field Name | Data Type | Description | Example Values | Notes |
|------------|-----------|-------------|----------------|-------|
| **age** | Integer | Client's age in years | 56, 57, 37, 40, 45 | Numeric and continuous.<br> Most of them are between 28-48 years old; mean:40 median:38 |
| **job** | String (categorical, nominal) | Client's type of job | `admin`, `blue-collar`, `entrepreneur`, `housemaid`, `management`, `retired`, `self-employed`, `services`, `student`, `technician`, `unemployed`, `unknown` | `admin`~25.30%;<br>`blue-collar`~22.47%;<br>`technician`~16.37%|
| **marital** | String (categorical, nominal) | Marital status | `divorced`, `married`, `single`, `unknown` | `divorced` means divorced or widowed.|
| **education** | String (categorical, ordinal) | Highest education level | `basic.4y`, `basic.6y`, `basic.9y`, `high.school`, `illiterate`, `professional.course`, `university.degree`, `unknown` | Has an arguable natural order of progression |
| **default** | String (categorical) | Has credit in default? | `no`, `yes`, `unknown` | `yes` is extremely rare in the full dataset (~0.01%). |
| **housing** | String (categorical) | Has a housing loan? | `no`, `yes`, `unknown` | ~52.4% `yes` |
| **loan** | String (categorical) | Has a personal loan? | `no`, `yes`, `unknown` | ~82.4% `no`|
| **contact** | String (categorical) | Contact communication type | `cellular`, `telephone` |  |
| **month** | String (categorical, ordinal) | Month of the last contact | `jan`, `feb`, `mar`, ..., `nov`, `dec` | Lowercase 3-letter abbreviations. |
| **day_of_week** | String (categorical, ordinal) | Weekday of the last contact | `mon`, `tue`, `wed`, `thu`, `fri` | Lowercase 3-letter abbreviations. |
| **duration** | Integer | Duration of the last contact, in seconds | 261, 149, 226, 151, 307, 198 | Numeric. |
| **campaign** | Integer | Number of contacts made to this client during this campaign | 1, 2, 3, 4, 5 |  |
| **pdays** | Integer | Days since the client was last contacted in a previous campaign | 1, 4, 5, 999 | 999 means not previously contacted. |
| **previous** | Integer | Number of contacts made to this client before this campaign | 0, 1, 2| Count, mostly 0. |
| **poutcome** | String (categorical, nominal) | Outcome of the previous marketing campaign | `failure`, `nonexistent`, `success` | Value `nonexistent` lines up with previous = 0 |
| **emp.var.rate** | Float | Employment variation rate (quarterly indicator) | 1.1 | Social/economic context attribute, consistent across all profiles |
| **cons.price.idx** | Float | Consumer price index (monthly indicator) | 93.994 | Social/economic context attribute |
| **cons.conf.idx** | Float | Consumer confidence index (monthly indicator) | -36.4 | Social/economic context attribute, negative values are normal |
| **euribor3m** | Float | Euribor 3-month interest rate (daily indicator) | 4.857 | Social/economic context attribute |
| **nr.employed** | Float | Number of employees (quarterly indicator) | 5191 | Social/economic context attribute |
| **y** | String (categorical, binary) | **Target variable.** Did the client subscribe to a term deposit? | `yes`, `no` | ~11.3% `yes` |
## Insights
`duration` this attribute highly affects the output target (e.g., if duration=0 then y="no"). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

`campaign`: Count, right-skewed with some large outliers

Majority of `previous` is 0, `poutcome` is more likely to be "failure" when `previous = 1`. `success`

Only 11.3% of clients ever say yes. A predictive model that says no all the time would be correct 88.7% of the time. This is a class imbalance that will make predictions difficult to make.

## Questions

The dataset is suited towards *predictive* questions and answers. While there is possibility for other categories, predictive remains the most prominent due to the presence of the outcome variable `y`.

**Predictive:** 
* Prior to dialing, can we rank future clients based on attributes to capture more term-subscribers than calling at random?
* Can we predict if new clients will subscribe based on collected attributes?
* Are there ways to predict other attributes of clients based off of logical groups / clusters?
### Additional Questions

I've also listed some other questions that could work with the dataset.

**Exploratory:**
* Is calling effort spread evenly across clients, or concentrated on a few?

**Inferential:**
* Is the number of contacts per client consistent with a constant-rate (Poisson) process?
* Do clients who subscribed receive a different number of calls than those who didn’t?

**Mechanistic** 
* What drives a client to subscribe, and what intersections of classes might have emergent properties?

## Validation Plan

### 1. Validation Goals

We will test whether we can:
1. Rank future clients before dialing to find more subscribers within the same contact budget.
2. Predict subscription (`y`) using information available before the call.
3. Predict another client attribute using meaningful groups or clusters. We will choose this attribute before final testing.

### 2. Time-Based Split

[UCI](https://archive.ics.uci.edu/dataset/222/bank+marketing) confirms that `bank-additional-full.csv` is ordered by date. We will keep this order to train on earlier records and test on later ones.

| Set | Share | Original rows* | Purpose |
|---|---|---|---|
| Training | ~60% | 1–24,712 | Fit preprocessing and models |
| Validation | ~20% | 24,713–32,950 | Select settings and classification threshold |
| Test | ~20% | 32,951–41,188 | Evaluate the final model |

*Rows start at 1, excluding the header. Each period keeps its original class distribution.*

After selecting settings, we will refit preprocessing and the model on the first 80%, then test on the last 20% with the threshold fixed.

### 3. Random-Split Comparison

We will keep the EDA's **80%/20% random split**, using `random_state=42` and `stratify=y`.

Using the settings and threshold selected through time-based validation, we will fit preprocessing and the model separately on each split's 80% and evaluate on its 20%. Both use the same features and metrics. **Time-based results are primary; random results are the comparison.**

The existing EDA results remain exploratory; the comparison will be rerun with the final settings. Final test results will not guide further tuning.

### 4. Preventing Data Leakage
- **Inputs:** For subscription prediction, `yes` is the positive class; exclude `y` and `duration`. Use only features available before dialing.
- **Preprocessing:** Fit imputation, encoding, scaling, and feature selection on training data only; apply them to validation and test data. Final refitting uses only the development 80%.
- **Class imbalance:** Resampling and class-weight calculations use training data only. Keep validation and test distributions unchanged.
- **Model choices:** Select features, models, hyperparameters, thresholds, cluster counts, and metrics using development data, before final testing.
- **Question 3:** Exclude the predicted attribute and its direct derivatives from clustering inputs. Learn preprocessing, clusters, and group-based prediction rules from training data, then apply them to held-out records.

### 5. Evaluating Results

**Question 1: Ranking Clients**
Rank clients by predicted subscription score and evaluate the top 10%, assuming we can contact only 10% of clients. This asks: If we can contact only a small share of clients, can the model identify those most likely to subscribe?

**Lift@10% = subscription rate among the top 10% ÷ overall subscription rate in the same evaluation set.**

For example, if the overall subscription rate is 10% and the top group's rate is 30%, lift is 3: the selected clients subscribe at three times the rate expected from random selection.
- **Lift > 1:** Better than random selection in expectation.
- **Lift = 1:** Matches random selection in expectation.
- **Lift < 1:** Worse than random selection in expectation.

Report **Recall@10%**: the share of all actual subscribers captured in the top 10%. We will round the selected count up and break score ties by original row number in ascending order.

**Question 2: Predicting Subscription**
We will focus on the `yes` class:
| Metric | What it tells us |
|---|---|
| Precision | Of clients predicted as yes, how many actually subscribed? |
| Recall | Of all actual subscribers, how many did we identify? |
| F1 | How well do precision and recall balance? |
| Confusion matrix | How many predictions were correct, false positives, or false negatives? |

Accuracy is supplementary: predicting no for everyone can achieve high accuracy while finding no subscribers.

We will compare models on the validation set, using Lift@10% for ranking and reporting yes precision, recall, and F1 for classification. The classification threshold will maximize validation-set yes F1; ties go to the higher threshold. Model settings and the threshold will be fixed before final testing.

**Question 3: Predicting Other Attributes from Groups**
**To be completed after team discussion:** specify the attribute to predict and its evaluation metrics before final testing.

### 6. Stability Check

Keeping the model trained on the earliest 60% fixed, we will split the middle 20% validation set into two consecutive halves and calculate Lift@10% for each.
This check shows whether ranking performance changes over time, using validation data only. If the two periods differ substantially, we will note this in our conclusions.

### 7. Test Set Freeze

After the team confirms this plan and commits the split code, we will record the GitHub commit ID marking the start of the freeze: **to be added**.

Our EDA and random-split experiments have already used the full dataset. From the recorded commit onward, we will stop using test data for exploration, model selection, or tuning. Final testing will take place after all choices are fixed, and we will report the results without further tuning based on test scores.

>>>>>>> 6da2a31aea9e2151fafc7912268b030beeaa8b58
