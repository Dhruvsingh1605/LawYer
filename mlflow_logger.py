import mlflow

def log_to_mlflow(query, cases, response):
    with mlflow.start_run():
        mlflow.log_param("court", query["court"])
        mlflow.log_param("topic", query["topic"])
        mlflow.log_param("year", query.get("year"))
        mlflow.log_metric("case_count", len(cases))
        mlflow.log_text(response, "gemini_response.txt")
