import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset


# загрузка данных
reference = pd.read_csv("./data/reference_data.csv")
production = pd.read_csv("./data/prod_data.csv")

# убрать служебные поля, которых нет в reference
DROP_COLS = ["timestamp", "bad_proba", "prediction", "bad_pred"]

production = production.drop(
    columns=[c for c in DROP_COLS if c in production.columns]
)

report = Report(
    metrics=[
        DataDriftPreset(),
        DataSummaryPreset(),
    ]
)

snapshot = report.run(
    reference_data=reference,
    current_data=production
)

snapshot.save_html("evidently_report.html")
print("Evidently Data Drift report saved")
