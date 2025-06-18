import insightface


def get(model="inswapper_1024.onnx", download=True):
    """Load INSwapper model automatically."""
    return insightface.model_zoo.get_model(model, download=download, download_zip=download)
