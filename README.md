# Urban Air Quality Intelligence System

An end‑to‑end platform for monitoring, analysing and predicting air quality in urban environments.  The system collects real pollution and weather data from multiple public APIs, performs exploratory and predictive analytics, detects anomalies and potential health risks and surfaces the results through an interactive Streamlit dashboard.

## Table of contents

1. [Project overview](#project-overview)
2. [Repository structure](#repository-structure)
3. [Prerequisites](#prerequisites)
4. [Environment setup](#environment-setup)
5. [API key registration](#api-key-registration)
6. [Data collection](#data-collection)
7. [Exploratory data analysis](#exploratory-data-analysis)
8. [Modelling workflow](#modelling-workflow)
9. [Running the dashboard](#running-the-dashboard)
10. [Running tests and linting](#running-tests-and-linting)

## Project overview

This project aims to provide a complete and reproducible pipeline for building a data‑driven urban air quality intelligence system.  The workflow is divided into multiple phases:

* **Phase 1 – Data Collection & Exploratory Data Analysis (EDA):**
  * Collect raw air quality measurements (PM2.5, PM10, NO₂, O₃, CO, SO₂) from the OpenWeatherMap, OpenAQ, PurpleAir and WAQI APIs.
  * Collect weather variables (temperature, humidity, wind speed, precipitation) from OpenWeatherMap and Visual Crossing.
  * Optionally collect traffic and industrial proxies via OpenStreetMap or Google Maps and demographic/health data via government sources.
  * Save raw and processed datasets to the `data/` directory.
  * Explore the data in Jupyter notebooks located under `notebooks/`.

* **Phase 2 – Pollution Pattern Discovery:**
  * Use clustering algorithms to group city areas by pollution characteristics.
  * Investigate temporal patterns (daily, weekly, seasonal) in pollutant concentrations.
  * Perform association mining to uncover relationships between weather variables and pollution.

* **Phase 3 – Predictive Modelling:**
  * Train regression models to predict future PM2.5 concentrations.
  * Build classification models to categorise Air Quality Index (AQI) levels.
  * Use the Prophet library for 7‑day ahead time series forecasting.

* **Phase 4 – Anomaly Detection & Health Impact:**
  * Detect pollution spikes and other anomalies in the data.
  * Estimate respiratory health risks based on pollutant exposure and provide personalised advisories.
  * Visualise insights via an interactive Streamlit dashboard.

Each phase is documented through notebooks and scripts, and the entire pipeline can be executed end‑to‑end.

## Repository structure

The repository is organised to separate code, data and reports while keeping everything discoverable:

```
urban‑air‑quality‑intelligence‑system/
├── src/                   # Python source code
│   ├── data/             # Data collection scripts
│   ├── features/         # Feature engineering modules
│   ├── modeling/         # Machine learning models and training
│   └── utils/            # Shared utilities (config, logging, API helpers)
│
├── data/                  # Datasets (not included in the repo)
│   ├── raw/              # Raw downloaded data (JSON/CSV)
│   ├── processed/        # Cleaned & feature‑engineered data
│   ├── warehouse/        # Aggregated/intermediate data used across phases
│   └── external/         # Additional external data sources (e.g. OSM extracts)
│
├── app/                   # Streamlit dashboard
│   └── main.py           # Entry point for the web application
│
├── models/                # Persisted trained models
├── notebooks/             # Jupyter notebooks for EDA and analysis
├── reports/               # Generated reports (figures, markdown)
├── tests/                 # Unit tests and test fixtures
├── docs/                  # Additional documentation and design diagrams
├── .github/workflows/     # Continuous integration configuration
├── .env.example           # Environment variable template (copy to `.env`)
├── pyproject.toml         # Python project & dependency configuration
├── .gitignore             # Files and directories ignored by Git
└── README.md              # Project documentation and usage guide
```

Feel free to extend this structure as your needs evolve.  The key principle is to isolate raw and generated data from source code and configuration so that the project remains reproducible.

## Prerequisites

These instructions assume you are working on **macOS with Apple Silicon** (M1/M2/M3) and the **zsh** shell.  Adjust paths accordingly if you use a different operating system.

1. **Xcode Command Line Tools** – these provide compilers and other build tools used by Python packages.  Install them by running:

   ```bash
   xcode-select --install
   ```

2. **Homebrew** – a package manager for macOS.  Install it if you haven't already:

   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

3. **Python 3.11** – the project targets Python 3.11.  The easiest way to manage multiple Python versions on macOS is with [pyenv](https://github.com/pyenv/pyenv).  Install pyenv via Homebrew and then install Python 3.11:

   ```bash
   brew install pyenv
   pyenv install 3.11.9
   pyenv global 3.11.9
   # After running the above commands, restart your terminal or run `exec "$SHELL"` to refresh the environment.
   ```

   Alternatively, you can install Python 3.11 directly via Homebrew:

   ```bash
   brew install python@3.11
   echo "export PATH=\"/opt/homebrew/opt/python@3.11/bin:$PATH\"" >> ~/.zshrc
   source ~/.zshrc
   ```

4. **Virtual environment (recommended)** – create an isolated Python environment for the project:

   ```bash
   python3.11 -m venv ~/venvs/uaqi
   source ~/venvs/uaqi/bin/activate
   ```

5. **VS Code** – install [Visual Studio Code](https://code.visualstudio.com/) and the "Python" extension.  This project provides instructions on how to open the repository in VS Code and run code interactively.

## Environment setup

1. **Clone the repository**.  Open a terminal (zsh) and navigate to the directory where you want the project to live:

   ```bash
   cd ~/Projects  # or any directory you prefer
   git clone <YOUR-FORK-OR-CLONE-URL> urban-air-quality-intelligence-system
   cd urban-air-quality-intelligence-system
   ```

2. **Activate your Python environment** if you haven’t already:

   ```bash
   source ~/venvs/uaqi/bin/activate
   ```

3. **Install project dependencies** using `pip`.  Make sure you are in the root of the repository (`urban-air-quality-intelligence-system`) so that `pyproject.toml` is detected.  On Apple Silicon some packages (especially `prophet`) may require additional system libraries:

   ```bash
   # Install OpenMP for scikit‑learn and Prophet
   brew install libomp

   # Install dependencies defined in pyproject.toml
   pip install --upgrade pip
   pip install .[dev]
   ```

   The `[dev]` suffix installs both runtime and development dependencies such as `pytest` and `flake8`.  If you wish to omit the dev tools you can run `pip install .` instead.

4. **Create a `.env` file**.  Copy the template `.env.example` into a new `.env` file in the project root and populate it with your personal API keys.  Never commit the `.env` file to version control as it contains secrets:

   ```bash
   cp .env.example .env
   # Edit the .env file with your keys
   code .env  # opens the file in VS Code for editing
   ```

5. **Open the project in VS Code**.  Launch VS Code from the terminal so it picks up your virtual environment automatically:

   ```bash
   code .
   ```

   Once VS Code opens, you should see the repository structure in the Explorer pane.  If prompted, select the Python interpreter corresponding to your virtual environment (`~/<path>/uaqi/bin/python`).  You can verify by running the **Python: Select Interpreter** command (`⇧⌘P`, then search for "Python: Select Interpreter").

## API key registration

The system uses several public APIs to retrieve air quality and weather data.  Sign up for the following services and store the keys in your `.env` file.  Detailed instructions for each service are provided below.

### OpenWeatherMap

1. Navigate to [https://openweathermap.org/appid](https://openweathermap.org/appid).
2. Create an account or log in if you already have one.
3. From your dashboard, create a new API key.  Give it an appropriate name such as `uaqi`.
4. Copy the generated key and paste it into the `OPENWEATHERMAP_API_KEY` entry in your `.env` file.

### OpenAQ

1. Visit [https://docs.openaq.org/#api-platform-keys](https://docs.openaq.org/#api-platform-keys).
2. Although the OpenAQ API does not strictly require an API key for read‑only access, registering for a free API token increases your request limits.  Click "Get Started" and follow the instructions to register.
3. Once you have your token, enter it into `OPENAQ_API_KEY` in your `.env` file (or leave it blank for anonymous access).

### PurpleAir

1. Go to the PurpleAir API portal at [https://www2.purpleair.com/](https://www2.purpleair.com/) and create an account.
2. Once logged in, click "API" in the user menu and request a read API key.  Describe your use case to receive approval.
3. After receiving your key via email or the dashboard, set `PURPLEAIR_API_KEY` in your `.env` file.

### World Air Quality Index (WAQI)

1. Open [https://aqicn.org/data-platform/token/](https://aqicn.org/data-platform/token/).
2. Register with your email address and request a token.  Tokens are issued immediately for non‑commercial use.
3. Copy the token and place it in `WAQI_API_KEY`.

### Visual Crossing Weather

1. Create an account at [https://www.visualcrossing.com/weather-api](https://www.visualcrossing.com/weather-api).
2. After registering, navigate to the API Keys section of your dashboard.
3. Generate a new API key and save it under `VISUAL_CROSSING_API_KEY`.

### Central Pollution Control Board (CPCB)

1. The CPCB provides data primarily via OpenAQ and Data.gov.in.  If you have an official token (e.g. for bulk downloads) you can store it in `CPCB_API_KEY`.  Otherwise leave it blank and rely on OpenAQ.

### Google Maps / OpenStreetMap

* **OpenStreetMap:** The scripts in this repository use the open Overpass API and do not require a key.
* **Google Maps:** If you wish to query Google Maps for traffic data, create a project at [https://console.cloud.google.com/](https://console.cloud.google.com/), enable the Maps JavaScript API and Traffic API, and generate an API key.  Save it in `GOOGLE_MAPS_API_KEY`.

Once all keys are set in your `.env` file, the scripts will automatically load them using the [`python-dotenv`](https://pypi.org/project/python-dotenv/) library.

## Data collection

Data collection scripts live under `src/data/` and can be invoked directly from the command line.  The primary entry point is `collect_air_quality.py`, which fetches pollution and weather data for a given location.  Always ensure your `.env` is configured before running these commands.

### Example: collect data for Vellore, India

```bash
# Activate the virtual environment
source ~/venvs/uaqi/bin/activate

# Navigate to the project root
cd ~/Projects/urban-air-quality-intelligence-system

# Run data collection for Vellore (latitude 12.9165, longitude 79.1325) for the past 7 days
python src/data/collect_air_quality.py \
    --city "Vellore" \
    --latitude 12.9165 \
    --longitude 79.1325 \
    --start-date "2024-08-17" \
    --end-date "2024-08-24"

# Raw JSON/CSV files will be saved under data/raw/
```

Modify the `--start-date` and `--end-date` arguments to target the desired timeframe.  The script automatically calls all configured APIs (OpenWeatherMap, OpenAQ, PurpleAir, WAQI, Visual Crossing) and concatenates the results into unified CSV files.

### Other data sources

* **Traffic & industrial proxies:** Use the script `src/data/collect_osm_data.py` (planned) to download road networks and industrial sites around your city.  This uses the Overpass API and does not require an API key.
* **Demographics and health:** Additional scripts can be created under `src/data/` to scrape or download census and health data.  You will need to update `.env` with any required tokens.

## Exploratory data analysis

Open the notebooks in the `notebooks/` directory to perform exploratory analysis and understand the collected data.  Notebooks are numbered by phase:

* **01_data_collection_and_eda.ipynb** – loads the raw data and produces summary statistics and basic plots.  Run this first to inspect the quality and range of your data.
* **02_pollution_patterns.ipynb** – performs clustering, seasonal decomposition and association mining to uncover patterns.
* **03_predictive_modeling.ipynb** – trains regression, classification and forecasting models on the processed datasets.
* **04_anomaly_and_health.ipynb** – detects anomalies and estimates health impacts.

To launch a notebook, activate your environment and start Jupyter in the project root:

```bash
source ~/venvs/uaqi/bin/activate
cd ~/Projects/urban-air-quality-intelligence-system
jupyter notebook
```

Then open the desired `.ipynb` file in your browser.  Each notebook contains markdown explanations and code cells that you can execute sequentially.  Running the cells will generate charts and tables based on the data stored in the `data/` directory.

## Modelling workflow

After exploring the data you can train the models using the scripts under `src/modeling/`.  The training scripts expect processed CSV files produced by the data collection pipeline.  Example usage:

```bash
source ~/venvs/uaqi/bin/activate
cd ~/Projects/urban-air-quality-intelligence-system

# Train regression and classification models for PM2.5 and AQI
python src/modeling/train_models.py --input-file data/processed/merged_dataset.csv --output-dir models/

# Train a Prophet model for time series forecasting
python src/modeling/train_time_series.py --input-file data/processed/merged_dataset.csv --output-dir models/

# Detect anomalies and compute health risks
python src/modeling/detect_anomalies.py --input-file data/processed/merged_dataset.csv --output-file data/processed/anomalies.csv
python src/modeling/health_risk.py --input-file data/processed/merged_dataset.csv --output-file data/processed/health_risk.csv
```

Each script will save trained models and results to the `models/` and `data/processed/` folders respectively.  Consult the docstrings in each module for further details and command‑line arguments.

## Running the dashboard

The final product is an interactive dashboard built with Streamlit.  It allows you to explore current air quality conditions, historical trends, model forecasts, anomalies and health advisories for selected locations.  To run the dashboard:

```bash
source ~/venvs/uaqi/bin/activate
cd ~/Projects/urban-air-quality-intelligence-system
streamlit run app/main.py
```

Your default web browser will open automatically at <http://localhost:8501>.  Use the sidebar to choose a city, date range and view.  The dashboard will fetch live data (using your API keys) or load processed data from disk.  Interactive charts and maps are rendered using Plotly and Folium.

## Running tests and linting

This project includes a basic test suite and configuration for continuous integration.  To run tests locally:

```bash
source ~/venvs/uaqi/bin/activate
cd ~/Projects/urban-air-quality-intelligence-system
pytest
```

Static code analysis is performed with [flake8](https://flake8.pycqa.org/en/latest/).  To run lint checks:

```bash
flake8
```

The `.github/workflows/ci.yml` file defines a GitHub Actions workflow that runs linting and tests on every push.  If you fork this repository, GitHub will automatically execute the CI pipeline in your account.

---

We hope this guide provides a clear path from project setup to final deployment.  Each phase builds upon the last, enabling you to collect, analyse and act upon urban air quality data.  Feel free to open issues or submit pull requests if you have suggestions or improvements!
