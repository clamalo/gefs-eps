# /src/plotting/plot_points.py
import numpy as np
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, time
import matplotlib.colors as mcolors
import os

def median_across_steps(list):
    return np.median(np.array(list), axis=1)

def mean_across_steps(list):
    return np.mean(np.array(list), axis=1)

def member_across_steps(list, n):
    return np.array(list)[:, n]

def add_line(fig, row, member, x_values, line, name, variable, lower_percentile=None, upper_percentile=None):
    color_dict = {'total_snow': '#0000ff', 'snow': '#0000ff', 'slr': 'blue', 
                  'tp': 'limegreen', 'total_tp': 'limegreen', 'temp': 'orange', 
                  'snow_level': '#0000ff'}
    color = 'red' if name == 'Median' else color_dict[variable]

    if member is not None and member < 50:
        color = mcolors.to_rgb(color)
        color = [max(0, c - 0.3) for c in color]  # decrease brightness by 30%
        color = mcolors.to_hex(color)

    width = 3 if name == 'Median' else 1
    hoverinfo = 'none' if name != 'Median' else 'text'
    opacity = 1 if name == 'Median' else 0.6
    mode = 'lines+markers' if name == 'Median' else 'lines'

    if name == 'Median':
        hover_texts = [datetime.strptime(date, '%Y-%m-%d %H:%M:%S').strftime('%a, %m/%d, %l%p: <b>{:.2f}</b> ({:.2f}, {:.2f})'.format(val, p25, p75))
                        for date, val, p25, p75 in zip(x_values, line, lower_percentile, upper_percentile)]
        hovertemplate = '%{text}'
    else:
        hover_texts = None
        hovertemplate = None

    fig.add_trace(
        go.Scatter(
            x=x_values, 
            y=line, 
            mode=mode, 
            hoverinfo=hoverinfo, 
            text=hover_texts,
            hovertemplate=hovertemplate,
            showlegend=False, 
            line=dict(color=color, width=width), 
            name=name, 
            opacity=opacity
        ),
        row=row, col=1
    )

def plot_exceedance_probabilities(fig, row, point_data, x_values, variable_name):
    x_values_datetime = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in x_values]

    if variable_name == 'total_snow_exceedance_probabilities':
        thresholds = [1, 3, 6, 12, 18, 24, 36, 48, 60, 100]
        thresholds = [str(threshold) for threshold in thresholds]
        colors = ['#7db9dd','#346bac','#f9ff93','#ff8700','#db1400','#9e0000','#562a29','#d4d3fd','#ad9fdd','#7e6db4']
    else:  # slr_exceedance_probabilities
        thresholds = [5, 8, 10, 12, 15, 20, 25]
        thresholds = [str(threshold) for threshold in thresholds]
        colors = ['#7db9dd','#346bac','#f9ff93','#ff8700','#db1400','#9e0000','#562a29']
    
    data = {threshold: np.array(point_data[variable_name][threshold]).flatten() for threshold in thresholds}
    
    hover_texts = []
    for i, dt in enumerate(x_values_datetime):
        hover_text = f'{dt.strftime("%a, %m/%d, %I%p")}:<br>'
        if variable_name == 'total_snow_exceedance_probabilities':
            for threshold in thresholds:
                hover_text += f'{threshold}": <b>{data[threshold][i]:.2f}%</b><br>'
        else:
            for threshold in thresholds:
                hover_text += f'SLR {threshold}: <b>{data[threshold][i]:.2f}%</b><br>'
        hover_texts.append(hover_text)
    
    for i, threshold in enumerate(thresholds):
        threshold_label = f'{threshold}"' if variable_name == 'total_snow_exceedance_probabilities' else f'SLR {threshold}'
        fig.add_trace(
            go.Bar(
                x=x_values,
                y=data[threshold],
                name=threshold_label,
                marker_color=colors[i],
                opacity=1,
                text=hover_texts,
                hoverinfo='text',
                showlegend=False,
                base=np.zeros(len(x_values)),
                textposition='none'
            ),
            row=row, col=1
        )
    
    fig.update_layout(barmode='stack')
    y_min, y_max = 0, 100

    num_items = len(thresholds)
    start_y = 0.95
    spacing = 0.08
    for i, (th, col) in enumerate(zip(thresholds, colors)):
        y_pos = start_y - i * spacing
        fig.add_shape(
            type="rect",
            xref=f"x{row} domain",
            yref=f"y{row} domain",
            x0=1.02, x1=1.06,
            y0=y_pos - 0.03, y1=y_pos + 0.03,
            fillcolor=col,
            line=dict(width=0),
            layer="above"
        )
        fig.add_annotation(
            x=1.07, y=y_pos,
            xref=f"x{row} domain",
            yref=f"y{row} domain",
            text=str(th),
            showarrow=False,
            font=dict(size=10),
            xanchor="left",
            yanchor="middle"
        )
    
    fig.update_yaxes(range=[y_min, y_max], row=row, col=1)
    fig.update_yaxes(title_text='Exceedance Probability (%)', row=row, col=1)

def plot_points(gefs, eps):
    with open(os.path.join(os.getcwd(), 'data', 'points_data_hourly.json'), 'r') as f:
        points_data_dict = json.load(f)

    for point in points_data_dict:
        if point == 'metadata':
            continue

        data = points_data_dict[point]
        variables = list(data.keys())
        for key in ['latitude', 'longitude', 'elevation']:
            if key in variables:
                variables.remove(key)
        new_order = ['total_snow', 'total_snow_exceedance_probabilities', 'snow', 'tp', 'total_tp', 'temp', 'snow_level', 'slr', 'slr_exceedance_probabilities']
        ordered_variables = [v for v in new_order if v in variables]
        for v in variables:
            if v not in ordered_variables and v != 'times':
                ordered_variables.append(v)
        variables = ordered_variables

        x_values = data['times']
        x_values_datetime = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in x_values]

        num_rows = sum(1 for var in variables if var != 'times')
        fig = make_subplots(rows=num_rows, cols=1, subplot_titles=variables, vertical_spacing=0.03)
        row_counter = 1
        for variable in variables:
            if variable == 'times':
                continue

            if 'exceedance_probabilities' in variable:
                plot_exceedance_probabilities(fig, row_counter, data, x_values, variable)
                row_counter += 1
                continue

            y_values = np.array(data[variable]).flatten()
            y_min, y_max = min(y_values), max(y_values)

            member_lines = []
            if gefs and eps:
                num_members = 80
            elif eps and not gefs:
                num_members = 50
            elif gefs and not eps:
                num_members = 30

            for m in range(num_members):
                member_line = member_across_steps(data[variable], m)
                add_line(fig, row_counter, m, x_values, member_line, None, variable)
                member_lines.append(member_line)

            member_lines_array = np.array(member_lines)
            lower_percentile = np.percentile(member_lines_array, 20, axis=0)
            upper_percentile = np.percentile(member_lines_array, 80, axis=0)

            if variable == 'slr':
                median_line = median_across_steps(data[variable])
            else:
                median_line = mean_across_steps(data[variable])
            add_line(fig, row_counter, None, x_values, median_line, 'Median', variable, lower_percentile, upper_percentile)

            added_segments = set()
            for dt in x_values_datetime:
                if dt.hour >= 18 or dt.hour < 6:
                    night_start = dt.replace(hour=18, minute=0, second=0) if dt.hour >= 18 else (dt - timedelta(days=1)).replace(hour=18, minute=0, second=0)
                    night_end = night_start + timedelta(hours=12)
                    segment = (night_start.strftime('%Y-%m-%d %H:%M:%S'), night_end.strftime('%Y-%m-%d %H:%M:%S'))
                    if segment not in added_segments:
                        fig.add_shape(
                            go.layout.Shape(
                                type="rect",
                                xref=f'x{row_counter}',
                                yref=f'y{row_counter}',
                                x0=segment[0],
                                x1=segment[1],
                                y0=y_min*0.95,
                                y1=y_max*1.05,
                                fillcolor="grey",
                                opacity=0.15,
                                layer="below",
                                line_width=0,
                            )
                        )
                        added_segments.add(segment)
            
            fig.update_yaxes(range=[y_min*0.95, y_max*1.05], row=row_counter, col=1)
            if variable == 'temp':
                fig.update_yaxes(title_text='Temperature (°F)', row=row_counter, col=1)
            elif variable == 'snow_level':
                fig.update_yaxes(title_text='Snow Level (ft)', row=row_counter, col=1)
            elif variable == 'slr':
                fig.update_yaxes(title_text='SLR', row=row_counter, col=1)
            elif variable in ['tp', 'total_tp']:
                fig.update_yaxes(title_text='Precipitation (in)', row=row_counter, col=1)
            elif variable in ['snow', 'total_snow']:
                fig.update_yaxes(title_text='Snow (in)', row=row_counter, col=1)
            
            row_counter += 1

        fig.update_xaxes(range=[x_values_datetime[0], x_values_datetime[-1]])
        unique_days = {x.date() for x in x_values_datetime}
        tickvals = []
        for day in sorted(unique_days):
            for t in [time(hour=6), time(hour=12), time(hour=18), time(hour=0)]:
                tickvals.append(datetime.combine(day, t))
        ticktext = []
        for tick in tickvals:
            if tick.time() == time(hour=0):
                ticktext.append(tick.strftime('%a %m/%d'))
            else:
                ticktext.append(tick.strftime('%I%p').lstrip('0'))
        for i in range(1, row_counter):
            fig.update_xaxes(tickangle=45, tickvals=tickvals, ticktext=ticktext, row=i, col=1)

        for i, annotation in enumerate(fig.layout.annotations):
            var_name = variables[i] if i < len(variables) and variables[i] != 'times' else ""
            if var_name == 'temp':
                annotation.text = "Temperature"
            elif var_name == 'snow_level':
                annotation.text = "Snow Level"
            elif var_name == 'slr':
                annotation.text = "Snow Liquid Ratio (SLR)"
            elif var_name == 'slr_exceedance_probabilities':
                annotation.text = "SLR Exceedance Probability"
            elif var_name == 'tp':
                annotation.text = "Precipitation"
            elif var_name == 'total_tp':
                annotation.text = "Total Precipitation"
            elif var_name == 'snow':
                annotation.text = "Snow"
            elif var_name == 'total_snow':
                annotation.text = "Total Snow"
            elif var_name == 'total_snow_exceedance_probabilities':
                annotation.text = "Total Snow Exceedance Probability"

        fig.update_layout(
            height=300 * row_counter, width=1000,
            title_text=f'{point} ({int(data["elevation"])}ft)'
        )

        # Write a partial HTML snippet for embedding (without full HTML boilerplate)
        html_content = fig.to_html(full_html=False, include_plotlyjs='cdn')
        with open(os.path.join(os.getcwd(), 'output', f'{point}.html'), 'w') as f:
            f.write(html_content)

def plot_point_with_range(point_name, points_data_dict, output_dir, gefs=True, eps=False):
    """
    Plot data for a specific point with a custom time range.
    
    Args:
        point_name: Name of the point to plot
        points_data_dict: Dictionary containing filtered point data
        output_dir: Directory to save the output HTML file
        gefs: Whether GEFS data is included
        eps: Whether EPS data is included
    """
    if point_name not in points_data_dict or point_name == 'metadata':
        raise ValueError(f"Point '{point_name}' not found in data")
    
    data = points_data_dict[point_name]
    variables = list(data.keys())
    for key in ['latitude', 'longitude', 'elevation']:
        if key in variables:
            variables.remove(key)
    new_order = ['total_snow', 'total_snow_exceedance_probabilities', 'snow', 'tp', 'total_tp', 'temp', 'snow_level', 'slr', 'slr_exceedance_probabilities']
    ordered_variables = [v for v in new_order if v in variables]
    for v in variables:
        if v not in ordered_variables and v != 'times':
            ordered_variables.append(v)
    variables = ordered_variables

    x_values = data['times']
    x_values_datetime = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in x_values]

    num_rows = sum(1 for var in variables if var != 'times')
    fig = make_subplots(rows=num_rows, cols=1, subplot_titles=variables, vertical_spacing=0.03)
    row_counter = 1
    
    for variable in variables:
        if variable == 'times':
            continue

        if 'exceedance_probabilities' in variable:
            plot_exceedance_probabilities(fig, row_counter, data, x_values, variable)
            row_counter += 1
            continue

        # Handle NaN values if they exist
        if isinstance(data[variable], list) and len(data[variable]) > 0:
            # Convert any possible NaN values to NumPy's nan to avoid issues
            y_values = np.array(data[variable], dtype=float).flatten()
            y_values = y_values[~np.isnan(y_values)]  # Filter out NaN values for min/max calculation
            if len(y_values) > 0:
                y_min, y_max = np.nanmin(y_values), np.nanmax(y_values)
            else:
                y_min, y_max = 0, 1  # Fallback if all values are NaN
        else:
            y_min, y_max = 0, 1

        member_lines = []
        if gefs and eps:
            num_members = 80
        elif eps and not gefs:
            num_members = 50
        elif gefs and not eps:
            num_members = 30

        # Handle variable dimensions for both multi-member and scalar datasets
        variable_data = np.array(data[variable])
        if len(variable_data.shape) > 1 and variable_data.shape[1] >= num_members:
            # Multi-member data (array of arrays)
            for m in range(num_members):
                member_line = member_across_steps(data[variable], m)
                add_line(fig, row_counter, m, x_values, member_line, None, variable)
                member_lines.append(member_line)

            member_lines_array = np.array(member_lines)
            lower_percentile = np.percentile(member_lines_array, 20, axis=0)
            upper_percentile = np.percentile(member_lines_array, 80, axis=0)

            if variable == 'slr':
                median_line = median_across_steps(data[variable])
            else:
                median_line = mean_across_steps(data[variable])
        else:
            # Scalar data (single array)
            median_line = np.array(data[variable]).flatten()
            lower_percentile = median_line * 0.9  # Fallback
            upper_percentile = median_line * 1.1  # Fallback
        
        add_line(fig, row_counter, None, x_values, median_line, 'Median', variable, lower_percentile, upper_percentile)

        # Add night shading
        added_segments = set()
        for dt in x_values_datetime:
            if dt.hour >= 18 or dt.hour < 6:
                night_start = dt.replace(hour=18, minute=0, second=0) if dt.hour >= 18 else (dt - timedelta(days=1)).replace(hour=18, minute=0, second=0)
                night_end = night_start + timedelta(hours=12)
                segment = (night_start.strftime('%Y-%m-%d %H:%M:%S'), night_end.strftime('%Y-%m-%d %H:%M:%S'))
                if segment not in added_segments:
                    fig.add_shape(
                        go.layout.Shape(
                            type="rect",
                            xref=f'x{row_counter}',
                            yref=f'y{row_counter}',
                            x0=segment[0],
                            x1=segment[1],
                            y0=y_min*0.95,
                            y1=y_max*1.05,
                            fillcolor="grey",
                            opacity=0.15,
                            layer="below",
                            line_width=0,
                        )
                    )
                    added_segments.add(segment)
        
        fig.update_yaxes(range=[y_min*0.95, y_max*1.05], row=row_counter, col=1)
        if variable == 'temp':
            fig.update_yaxes(title_text='Temperature (°F)', row=row_counter, col=1)
        elif variable == 'snow_level':
            fig.update_yaxes(title_text='Snow Level (ft)', row=row_counter, col=1)
        elif variable == 'slr':
            fig.update_yaxes(title_text='SLR', row=row_counter, col=1)
        elif variable in ['tp', 'total_tp']:
            fig.update_yaxes(title_text='Precipitation (in)', row=row_counter, col=1)
        elif variable in ['snow', 'total_snow']:
            fig.update_yaxes(title_text='Snow (in)', row=row_counter, col=1)
        
        row_counter += 1

    # Set x-axis range
    fig.update_xaxes(range=[x_values_datetime[0], x_values_datetime[-1]])
    
    # Add tick marks for each day
    unique_days = {x.date() for x in x_values_datetime}
    tickvals = []
    for day in sorted(unique_days):
        for t in [time(hour=6), time(hour=12), time(hour=18), time(hour=0)]:
            tickvals.append(datetime.combine(day, t))
    
    ticktext = []
    for tick in tickvals:
        if tick.time() == time(hour=0):
            ticktext.append(tick.strftime('%a %m/%d'))
        else:
            ticktext.append(tick.strftime('%I%p').lstrip('0'))
    
    for i in range(1, row_counter):
        fig.update_xaxes(tickangle=45, tickvals=tickvals, ticktext=ticktext, row=i, col=1)

    # Update subplot titles
    for i, annotation in enumerate(fig.layout.annotations):
        var_name = variables[i] if i < len(variables) and variables[i] != 'times' else ""
        if var_name == 'temp':
            annotation.text = "Temperature"
        elif var_name == 'snow_level':
            annotation.text = "Snow Level"
        elif var_name == 'slr':
            annotation.text = "Snow Liquid Ratio (SLR)"
        elif var_name == 'slr_exceedance_probabilities':
            annotation.text = "SLR Exceedance Probability"
        elif var_name == 'tp':
            annotation.text = "Precipitation"
        elif var_name == 'total_tp':
            annotation.text = "Total Precipitation"
        elif var_name == 'snow':
            annotation.text = "Snow"
        elif var_name == 'total_snow':
            annotation.text = "Total Snow"
        elif var_name == 'total_snow_exceedance_probabilities':
            annotation.text = "Total Snow Exceedance Probability"

    # Update title with selected time range
    start_time_str = x_values[0]
    end_time_str = x_values[-1]
    start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
    end_time = datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S')
    time_range_str = f"{start_time.strftime('%b %d, %H:%M')} - {end_time.strftime('%b %d, %H:%M')}"
    
    fig.update_layout(
        height=300 * row_counter, width=1000,
        title_text=f'{point_name} ({int(data["elevation"])}ft) - {time_range_str}'
    )

    # Write a partial HTML snippet for embedding
    html_content = fig.to_html(full_html=False, include_plotlyjs='cdn')
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f'{point_name}.html'), 'w') as f:
        f.write(html_content)