# SiteRisk — AI Construction Delay Predictor

A supervised-learning prototype that estimates the probability that a construction work activity will be delayed.

## Inputs
Activity type, weather disruption, material delay, labour shortfall, design changes, dependency delay and planned duration.

## Features
- Train/test split with stratification
- Random Forest classification pipeline
- Accuracy, precision, recall and confusion matrix
- Interactive risk assessment form
- Downloadable synthetic activity history

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Important methodology note
The bundled dataset is synthetic and intentionally designed for demonstration. Metrics shown by the app are not evidence of real-world predictive performance. A credible deployment needs a sufficiently large, representative, time-stamped dataset; leakage checks; temporal validation; probability calibration; subgroup/error analysis; and human review.

## Deployment
Push to GitHub and deploy `app.py` using [Streamlit Community Cloud](https://share.streamlit.io/).
