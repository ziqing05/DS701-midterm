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
| **age** | Integer | Client's age in years | 56, 57, 37, 40, 45 | Numeric and continuous. |
| **job** | String (categorical, nominal) | Client's type of job | `admin`, `blue-collar`, `entrepreneur`, `housemaid`, `management`, `retired`, `self-employed`, `services`, `student`, `technician`, `unemployed`, `unknown` | |
| **marital** | String (categorical, nominal) | Marital status | `divorced`, `married`, `single`, `unknown` | `divorced` means divorced or widowed.|
| **education** | String (categorical, ordinal) | Highest education level | `basic.4y`, `basic.6y`, `basic.9y`, `high.school`, `illiterate`, `professional.course`, `university.degree`, `unknown` | Has an arguable natural order of progression |
| **default** | String (categorical) | Has credit in default? | `no`, `yes`, `unknown` | `yes` is extremely rare in the full dataset (~11.3%). |
| **housing** | String (categorical) | Has a housing loan? | `no`, `yes`, `unknown` |  |
| **loan** | String (categorical) | Has a personal loan? | `no`, `yes`, `unknown` |  |
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
