# Fortress-Effect: Measuring Premier League Home Advantage

How much does playing at home actually change a team's chance of winning? This is a beginner machine learning project using logistic regression to quantify the "12th Man" effect across 25 seasons of the English Premier League (2000/01 to 2024/25).

**Headline Result:** Playing at home adds about **18.6 percentage points** to a team's win chance (47.0% at home vs 28.4% away), with team quality held equal.

## The Question
I was watching a recent Manchester United game at Old Trafford and noticed how powerful the home crowd felt. I wanted to test if this "home advantage" is statistically real, or if it just feels that way because good teams happen to win at home.

## The Data
* **Matches:** 9,380 matches
* **Source:** [Kaggle English Premier League Dataset](https://www.kaggle.com/)
* **Raw Baseline:** Before any ML, home teams won 45.8% of the time, away teams won 29.5%, and 24.7% were draws.

## Methodology
To prevent the model from cheating (data leakage), I had to build features carefully:
1. **Reshaping:** Split every match into two rows (a home perspective and an away perspective).
2. **Feature Engineering (No Leakage):** Created rolling averages for form (`form_5`) and team strength (`GD10`) using `shift(1)` so a match NEVER sees its own result.
3. **Time Split:** Trained the logistic regression model on matches before the 2019/20 season, and tested it on 2019/20 onwards.
4. **Counterfactual Flip:** To isolate the venue effect, I took the test data, predicted win probabilities with `is_home = 1`, and then predicted the exact same matches with `is_home = 0`.

## Results & Model Evaluation
The logistic regression model learned an `is_home` weight of 0.906, meaning playing at home multiplies the odds of winning by about 2.5x.

When evaluated on unseen test seasons, the model successfully beat a naive "lazy" baseline:
* **Accuracy:** 66.4% (vs 61.5% lazy baseline)
* **Log Loss:** 0.612 (vs 0.667 lazy baseline, lower is better)

### Team-Specific Boosts (10+ Seasons)
I also compared the raw gap between home win rates and away win rates for clubs with over 10 seasons:
* **Highest Home Win Rate:** Manchester United (67.8%)
* **Biggest Home Boost:** Fulham (+25.4 points)
* **Caution:** While Fulham shows the biggest boost, they've only played 18 seasons (671 matches). Compare this to the Big Six (like Tottenham or Arsenal), who have played 25 seasons (~938 matches). Small sample sizes create shakier metrics.

## What I Learned
* **Data Leakage:** Why you can't use match stats (like shots or fouls) to predict the outcome of that same match.
* **Time Splitting:** Why time-series data must be split chronologically rather than randomly.
* **Interpreting Weights:** Translating logistic regression log-odds into real-world percentages.

## Limitations
* Draws are counted as "not a win" in this binary classification model.
* The test seasons include the 2020/21 empty-stadium COVID era.
* The team-by-team raw gaps ignore the strength of the opponents faced.

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook analysis.ipynb
