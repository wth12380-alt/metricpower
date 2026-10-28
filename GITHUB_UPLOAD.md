# Putting this on GitHub

```bash
cd metricpower-py
git init
git add .
git commit -m "metricpower: Python package, Shiny app, tests and validation"
# create an empty repository named metricpower on github.com, then:
git remote add origin https://github.com/wth12380-alt/metricpower.git
git branch -M main
git push -u origin main
```

The CI workflow in
`.github/workflows/ci.yml` runs the tests and a short validation on every push.

To publish the app on shinyapps.io:

```bash
pip install rsconnect-python
rsconnect add --account <account> --name <account> --token <token> --secret <secret>
rsconnect deploy shiny . --name <account> --title metricpower --entrypoint app:app
```
