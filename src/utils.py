import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost.utils import get_gpu_device_count
from PIL import Image


def is_gpu_available():
    try:
        gpu_count = get_gpu_device_count()
        return gpu_count > 0
    except Exception:
        return False


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def save_feature_importance(
    feature_importances, X, task, task_name: str, model_name: str
):
    feature_names = X.columns
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": feature_importances}
    ).sort_values(by="importance", ascending=False)

    # Log the CSV to ClearML
    task.upload_artifact(
        name="Feature importance", artifact_object=importance_df.to_csv(index=False)
    )
    print("---Feature importance logged to ClearML---")

    top_features = importance_df.head(50)

    plt.style.use("seaborn-v0_8-darkgrid")
    plt.figure(figsize=(20, 15))
    plt.barh(top_features["feature"], top_features["importance"], color="orange")
    plt.xlabel("Feature Importance")
    plt.title(f"Top {50} Feature Importances for {model_name} Model")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    image = Image.open(buf).convert("RGB")

    task.get_logger().report_image(
        title="Top Feature Importance",
        series="Top Feature Importance",
        iteration=0,
        image=image,
    )
    print("---Top feature importance plot logged to ClearML---")
    buf.close()
