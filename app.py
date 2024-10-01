# app.py
import os
import subprocess
import json
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, jsonify
from datetime import datetime
import signal
import requests  # For elevation API
from urllib.parse import unquote

app = Flask(__name__)
app.secret_key = 'your_secure_secret_key'  # Replace with a secure key

OUTPUT_DIR = os.path.join(os.getcwd(), 'output')
MODEL_SCRIPT = 'run_model.py'  # Ensure this script exists
PROGRESS_FILE = os.path.join(os.getcwd(), 'data/progress.json')

# Global variable to track the model subprocess
model_process = None


# Initialize progress.json on app start
def initialize_progress():
    progress_data = {
        "percentage": 0,
        "current": 0,
        "total": 0,
        "estimated_time_remaining": "00:00:00"
    }
    try:
        # create data directory if it doesn't exist
        if not os.path.exists('data'):
            os.makedirs('data')
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress_data, f)
    except Exception as e:
        # create progress.json file if it doesn't exist
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress_data, f)

# Reset progress when the app starts
initialize_progress()

def read_points():
    points = []
    if os.path.exists('points.txt'):
        with open('points.txt', 'r') as f:
            for line in f:
                if line.strip() == "":
                    continue  # Skip empty lines
                try:
                    name, lat, lon, elevation = line.strip().split(',')
                    points.append({
                        'name': name,
                        'lat': float(lat),
                        'lon': float(lon),
                        'elevation': float(elevation)
                    })
                except ValueError:
                    print(f"Invalid line in points.txt: {line.strip()}")
    return points

def write_points(points):
    with open('points.txt', 'w') as f:
        for point in points:
            f.write(f"{point['name']},{point['lat']},{point['lon']},{point['elevation']}\n")

def get_elevation(lat, lon):
    # Use Open-Elevation API
    url = f'https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        elevation_meters = data['results'][0]['elevation']
        elevation_feet = elevation_meters * 3.28084  # Convert meters to feet
        return round(elevation_feet, 2)
    else:
        raise Exception('Error fetching elevation data')

@app.route('/', methods=['GET', 'POST'])
def index():
    global model_process
    if request.method == 'POST':
        # Retrieve form data
        date = request.form.get('date') or datetime.now().strftime('%Y%m%d')
        cycle = request.form.get('cycle') or '00'
        starting_step = request.form.get('starting_step') or '3'
        ending_step = request.form.get('ending_step') or '145'
        delta_t = request.form.get('delta_t') or '3'
        EPS = request.form.get('EPS') or 'False'
        GEFS = request.form.get('GEFS') or 'True'

        # Validate inputs
        if not all([date, cycle, starting_step, ending_step, delta_t, EPS, GEFS]):
            flash('All fields are required.', 'error')
            return redirect(url_for('index'))

        # Convert EPS and GEFS to boolean
        eps = True if EPS == 'True' else False
        gefs = True if GEFS == 'True' else False

        # Validate numeric fields
        try:
            starting_step = int(starting_step)
            ending_step = int(ending_step)
            delta_t = int(delta_t)
        except ValueError:
            flash('Starting Step, Ending Step, and Delta_t must be integers.', 'error')
            return redirect(url_for('index'))

        # Check if a model is already running
        if model_process and model_process.poll() is None:
            flash('A model run is already in progress. Please stop it before starting a new one.', 'error')
            return redirect(url_for('index'))

        # Clear the outputs directory
        try:
            if os.path.exists(OUTPUT_DIR):
                for filename in os.listdir(OUTPUT_DIR):
                    file_path = os.path.join(OUTPUT_DIR, filename)
                    if os.path.isfile(file_path) and filename.endswith('.html'):
                        os.unlink(file_path)
            else:
                os.makedirs(OUTPUT_DIR)
        except Exception as e:
            flash(f'Error clearing output directory: {e}', 'error')
            return redirect(url_for('index'))

        # Compute steps
        steps = list(range(starting_step, ending_step +1, delta_t))
        total_steps = len(steps)
        total_hours = total_steps * delta_t

        # Reset progress
        try:
            progress_data = {
                "percentage": 0,
                "current": 0,
                "total": total_hours,
                "estimated_time_remaining": "00:00:00"
            }
            with open(PROGRESS_FILE, 'w') as f:
                json.dump(progress_data, f)
        except Exception as e:
            flash(f'Error initializing progress: {e}', 'error')
            return redirect(url_for('index'))

        # Run the model in a separate subprocess
        try:
            # Start the subprocess and store the process handle
            model_process = subprocess.Popen([
                '/usr/bin/python3', MODEL_SCRIPT,
                '--date', date,
                '--cycle', cycle,
                '--starting_step', str(starting_step),
                '--ending_step', str(ending_step),
                '--delta_t', str(delta_t),
                '--eps', str(eps),
                '--gefs', str(gefs)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            flash('Model is running. Check the <a href="{}">output visualization page</a> for results over the next few minutes as the model begins to run.'.format(url_for('outputs')), 'success')
        except Exception as e:
            flash(f'Error running model: {e}', 'error')

        return redirect(url_for('index'))

    # For GET request, render the index page
    return render_template('index.html')

@app.route('/progress')
def progress():
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                progress = json.load(f)
                percentage = progress.get('percentage', 0)
                current = progress.get('current', 0)
                total = progress.get('total', 0)
                estimated_time_remaining = progress.get('estimated_time_remaining', "00:00:00")
        except:
            percentage = 0
            current = 0
            total = 0
            estimated_time_remaining = "00:00:00"
    else:
        percentage = 0
        current = 0
        total = 0
        estimated_time_remaining = "00:00:00"
    return jsonify({
        'percentage': percentage, 
        'current': current, 
        'total': total,
        'estimated_time_remaining': estimated_time_remaining
    })

@app.route('/status')
def status():
    global model_process
    is_running = False
    if model_process:
        if model_process.poll() is None:
            is_running = True
        else:
            model_process = None  # Reset if process has finished
    return jsonify({'is_running': is_running})

@app.route('/stop', methods=['POST'])
def stop():
    global model_process
    if model_process and model_process.poll() is None:
        try:
            # Terminate the subprocess
            model_process.terminate()
            try:
                model_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                model_process.kill()
            flash('Model run has been stopped.', 'success')
        except Exception as e:
            flash(f'Error stopping the model run: {e}', 'error')
    else:
        flash('No model run is currently running.', 'error')
    model_process = None
    # Reset progress
    try:
        progress_data = {
            "percentage": 0,
            "current": 0,
            "total": 0,
            "estimated_time_remaining": "00:00:00"
        }
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress_data, f)
    except Exception as e:
        flash(f'Error resetting progress: {e}', 'error')
    return redirect(url_for('index'))

@app.route('/outputs')
def outputs():
    # List all HTML files in the output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.html')]
    points = [os.path.splitext(f)[0] for f in files]
    return render_template('output_viewer.html', points=points)

@app.route('/view/<path:point>')
def view_point(point):
    # Decode the point name
    decoded_point = unquote(point)
    filename = f"{decoded_point}.html"
    # Security check: Ensure that the file exists within the output directory
    file_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(file_path):
        return send_from_directory(OUTPUT_DIR, filename)
    else:
        return "Not Found", 404

# New route for the map page
@app.route('/map')
def map_page():
    return render_template('map.html')

# New route to get points data
@app.route('/get_points')
def get_points():
    points = read_points()
    return jsonify(points)

# New route to add a point
@app.route('/add_point', methods=['POST'])
def add_point():
    data = request.get_json()
    name = data.get('name')
    lat = data.get('lat')
    lon = data.get('lon')
    elevation = data.get('elevation')  # Optional

    if not name or lat is None or lon is None:
        return jsonify({'success': False, 'message': 'Invalid data'}), 400

    # Read existing points
    points = read_points()
    # Check for duplicate names (case-insensitive)
    if any(point['name'].lower() == name.lower() for point in points):
        return jsonify({'success': False, 'message': 'A point with this name already exists.'}), 400

    # If elevation is not provided, fetch it using the API
    if elevation is None or elevation == "":
        try:
            elevation = get_elevation(lat, lon)
        except Exception as e:
            return jsonify({'success': False, 'message': 'Error getting elevation'}), 500
    else:
        try:
            elevation = float(elevation)
        except ValueError:
            return jsonify({'success': False, 'message': 'Invalid elevation value'}), 400

    # Add new point
    points.append({
        'name': name,
        'lat': lat,
        'lon': lon,
        'elevation': elevation
    })

    # Write points back to file
    try:
        write_points(points)
    except Exception as e:
        return jsonify({'success': False, 'message': 'Error writing to points.txt'}), 500

    return jsonify({'success': True})

# New route to delete a point
@app.route('/delete_point', methods=['POST'])
def delete_point():
    data = request.get_json()
    name = data.get('name')
    if not name:
        return jsonify({'success': False, 'message': 'Invalid data'}), 400
    # Read existing points
    points = read_points()
    # Remove the point with the given name (case-insensitive)
    points = [point for point in points if point['name'].lower() != name.lower()]
    # Write points back to file
    try:
        write_points(points)
    except Exception as e:
        return jsonify({'success': False, 'message': 'Error writing to points.txt'}), 500
    return jsonify({'success': True})

if __name__ == '__main__':
    # Reset progress when app.py is rerun
    initialize_progress()
    app.run(debug=True, port=5001)