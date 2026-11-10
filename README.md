
# Federated Healthcare Privacy

A research prototype demonstrating privacy-conscious collaborative machine learning across three simulated hospitals using Federated Learning and Flower.

## Project Overview

This project explores how multiple healthcare institutions can collaboratively train a machine learning model without directly sharing their local training datasets.

Each simulated hospital trains a local model. Model parameters are aggregated using Federated Averaging (FedAvg) to produce a global model.

## Architecture

- **Federated Learning:** Flower 1.42.0
- **Machine Learning:** NumPy, Pandas, Scikit-learn
- **Dashboard:** Streamlit
- **Visualization:** Matplotlib, Plotly

## Project Structure

```text
FederatedHealthcarePrivacy/
├── clients/
│   └── federated_client.py
├── dashboard/
│   └── app.py
├── server/
│   └── federated_server.py
├── results/
│   ├── baseline_metrics.json
│   ├── centralized_metrics.json
│   ├── flower_metrics.json
│   ├── flower_report.txt
│   ├── flower_global_model.npz
│   ├── flower_confusion_matrix.csv
│   ├── model_comparison.csv
│   ├── model_comparison_chart.png
│   └── accuracy_comparison_chart.png
├── .gitignore
├── requirements.txt
└── README.md
```

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Single-Hospital Baseline | 73.03% | 70.83% | 77.27% | 73.91% |
| Centralized Learning | 74.16% | 71.43% | 79.55% | 75.27% |
| Federated Learning | 77.53% | 75.00% | 81.82% | 78.26% |

Federated model evaluation:
- ROC-AUC: 82.93%
- Test samples: 89
- Simulated hospitals: 3
- Federated rounds: 5

## Dashboard

The Streamlit dashboard provides:

- Model performance overview
- Model comparison
- Confusion matrix
- Hospital network overview
- Project information

## Installation

Clone the repository:

```bash
git clone https://github.com/edustudentOS/FederatedHealthcarePrivacy.git
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run dashboard/app.py
```

## Dataset Disclaimer

The experimental dataset is derived from the scikit-learn diabetes disease-progression dataset. Its continuous target was converted into binary labels for an educational classification experiment.

It is NOT a clinical diabetes diagnosis dataset.

Hospital clients are simulated, and no real patient records are used.

## Privacy and Security Limitations

This is an academic research prototype.

Federated learning does not automatically guarantee privacy. Secure aggregation, differential privacy, production authentication, and clinical validation are not yet established by this prototype.

The model must not be used for clinical diagnosis or treatment decisions.

## Future Work

- Secure aggregation
- Differential privacy
- Realistic multi-institution datasets
- Hospital-level model validation
- Model monitoring and versioning
- Production-grade authentication and access control

## Author

Ashish Kori

## License

MIT
