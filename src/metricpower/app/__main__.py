import shiny

if __name__ == "__main__":
    shiny.run_app("metricpower.app.app:app", host="127.0.0.1", port=8000, launch_browser=True)
