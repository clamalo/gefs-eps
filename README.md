# GEFS + EPS Super Ensemble Snowfall Prediction Tool

This tool provides a web-based interface to run a super ensemble snowfall prediction model using GEFS and EPS data. It allows users to manage geographic points, run the model, and view the outputs.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Starting the Application](#starting-the-application)
  - [Managing Points](#managing-points)
    - [Adding Points](#adding-points)
    - [Deleting Points](#deleting-points)
  - [Running the Model](#running-the-model)
  - [Viewing Outputs](#viewing-outputs)
- [Troubleshooting](#troubleshooting)
- [Credits](#credits)

## Features

- Interactive web interface built with Flask.
- Map-based point management using Leaflet.
- Run snowfall prediction models using GEFS and EPS data.
- Real-time progress tracking.
- Visualization of outputs.

## Prerequisites

- Python 3.7 or higher installed on your system.
- `pip` package manager.
- See the [Python website](https://www.python.org/downloads/) for Python installation help.

## Installation

1.) **Python**

See the [Python website](https://www.python.org/downloads/) for Python installation help. Downloading the most recent version of Python for your specific operating system is best.

2. **Open the terminal application on your computer**

3. **Clone the repository**

   ```bash
   git clone https://github.com/clamalo/gefs-eps.git
   cd gefs-eps
   ```
   
4. **Install the required packages**

   ```bash
   pip install -r requirements.txt
   ```
   
## Usage

### Starting the Application

1. **Run the Flask application**

   ```bash
   python app.py
   ```

   By default, the application runs on `http://localhost:5001`.

2. **Access the web interface**

   Open a web browser and navigate to `http://localhost:5001` to access the dashboard.

### Managing Points

#### Adding Points

1. Click on the **"Manage Points on Map"** button on the dashboard to open the map interface.

2. **Add a new point**

   - Click anywhere on the map where you want to add a point.
   - A popup will appear displaying the latitude, longitude, and elevation (if available).
   - Enter a unique name for the point in the **"Enter Point Name"** field.
   - Click the **"Save Point"** button to add the point.

   **Note:** If the elevation data cannot be fetched automatically, you can manually enter it in the **"Enter Elevation"** field.

3. **Confirm the point is added**

   - The point will appear on the map as a marker.
   - A notification will confirm that the point was added successfully.

#### Deleting Points

1. On the map interface, click on the marker of the point you wish to delete.

2. In the popup that appears, click the **"Delete Point"** button.

3. **Confirm the point is deleted**

   - The marker will be removed from the map.
   - A notification will confirm that the point was deleted successfully.

### Running the Model

1. **Return to the dashboard**

   - Click the **"Back to Home"** button on the map page to return to the main dashboard.

2. **Configure model parameters**

   - **Date (YYYYMMDD):** Enter the date for which you want to run the model. If left blank, the current date will be used.
   - **Cycle:** Enter the model cycle number (e.g., `00`). Defaults to `00` if left blank.
   - **Starting Step:** Enter the starting forecast hour. Defaults to `3` if left blank.
   - **Ending Step:** Enter the ending forecast hour. Defaults to `145` if left blank.
   - **Delta_t:** Enter the time step interval in hours. Defaults to `3` if left blank.
   - **EPS:** Select `True` to include EPS data, or `False` to exclude it. Defaults to `False`.
   - **GEFS:** Select `True` to include GEFS data, or `False` to exclude it. Defaults to `True`.

3. **Run the model**

   - Click the **"RUN"** button at the bottom of the page to start the model.
   - A notification will confirm that the model is running.
   - Progress can be monitored via the progress bar on the dashboard.

4. **Stop the model (optional)**

   - If you need to stop the model run, click the **"STOP"** button next to the progress bar.

### Viewing Outputs

1. **Access the output visualization**

   - Click on the **"View Output Results"** button on the dashboard to open the output viewer.

2. **Select a point to view**

   - Use the dropdown menu to select the point for which you want to view the output.

3. **View the results**

   - The output for the selected point will be displayed in the embedded frame below the dropdown. Scroll down the page to see all of the plotted variables, with both individual members and the ensemble mean plotted.

## Troubleshooting

- **Error fetching elevation data**

  - Ensure you have an active internet connection.
  - If automatic elevation fetching fails, manually enter the elevation when adding a point.

- **Model not running**

  - Check if all required fields are filled correctly.
  - Ensure that no other instance of the model is running.

- **Dependency issues**

  - Make sure all packages in `requirements.txt` are installed.
  - If using a virtual environment, ensure it is activated.

## Credits

This tool was developed by Clay Malott.
