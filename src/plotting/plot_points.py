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
        hover_texts = [datetime.strptime(date, '%Y-%m-%d %H:%M:%S').strftime('%a, %m/%d, %l%p: <b>{:.2f}</b> ({:.2f}, {:.2f})'.format(val, p25, p75)) for date, val, p25, p75 in zip(x_values, line, lower_percentile, upper_percentile)]
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



def plot_exceedance_probabilities(fig, row, point_data, x_values):
    x_values_datetime = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in x_values]

    thresholds = [1, 3, 6, 12, 18, 24, 36, 48, 60, 100]
    thresholds = [str(threshold) for threshold in thresholds]
    colors = ['#7db9dd','#346bac','#f9ff93','#ff8700','#db1400','#9e0000','#562a29','#d4d3fd','#ad9fdd','#7e6db4']
    
    # Initialize lists to store data for each threshold
    data = {threshold: [] for threshold in thresholds}
    
    for threshold in thresholds:
        y_values = np.array(point_data['exceedance_probabilities'][threshold]).flatten()
        data[threshold] = y_values
    
    # Calculate hover texts and y values for stacked bars
    hover_texts = []

    for i, dt in enumerate(x_values_datetime):
        hover_text = f'{dt.strftime("%a, %m/%d, %I%p")}:<br>'
        for threshold in thresholds:
            hover_text += f'{threshold}": <b>{data[threshold][i]:.2f}%</b><br>'
        hover_texts.append(hover_text)
    
    for i, threshold in enumerate(thresholds):
        
        fig.add_trace(
            go.Bar(
                x=x_values,
                y=data[threshold],
                name=f'{threshold}"',
                marker_color=colors[i],
                opacity=1,
                text=hover_texts,
                hoverinfo='text',
                showlegend=True,
                base=np.zeros(len(x_values)),
                textposition='none'
            ),
            row=row, col=1
        )
    
    fig.update_layout(barmode='stack')

    # Determine the min and max y-values for the shaded areas
    y_min, y_max = 0, 100

    # Create a set to store added segments
    added_segments = set()

    # Add shaded regions for night times
    for dt in x_values_datetime:
        if dt.hour >= 18 or dt.hour < 6:  # Night time: 6 PM to 6 AM
            night_start = dt.replace(hour=18, minute=0, second=0) if dt.hour >= 18 else (dt - timedelta(days=1)).replace(hour=18, minute=0, second=0)
            night_end = night_start + timedelta(hours=12)
            segment = (night_start.strftime('%Y-%m-%d %H:%M:%S'), night_end.strftime('%Y-%m-%d %H:%M:%S'))

            # Only add the segment if it hasn't been added before
            if segment not in added_segments:
                fig.add_shape(
                    go.layout.Shape(
                        type="rect",
                        xref=f'x{row}',  # Reference the specific x-axis of the subplot
                        yref=f'y{row}',  # Reference the specific y-axis of the subplot
                        x0=segment[0],
                        x1=segment[1],
                        y0=y_min,  # Use actual data min for y0
                        y1=y_max,  # Use actual data max for y1
                        fillcolor="grey",
                        opacity=0.15,
                        layer="below",
                        line_width=0,
                    )
                )
                # Add the segment to the set of added segments
                added_segments.add(segment)

    fig.update_yaxes(range=[y_min, y_max], row=row, col=1)




    

def plot_points(gefs, eps):
    with open(os.path.join(os.getcwd(), 'data', 'points_data_hourly.json'), 'r') as f:
        points_data_dict = json.load(f)

    second_key = list(points_data_dict.keys())[1]
    variables = list(points_data_dict[second_key].keys())
    variables.remove('latitude')
    variables.remove('longitude')
    variables.remove('elevation')

    for point in points_data_dict:
        if point == 'metadata':
            continue

        fig = make_subplots(rows=len(variables)-1, cols=1, subplot_titles=variables[:-1], vertical_spacing=0.03)

        # Convert x_values to datetime objects
        x_values_datetime = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in points_data_dict[point]['times']]

        # Plot every variable except for "times" and "exceedance_probabilities"
        for variable in [var for var in variables if var not in ['times']]:
            row = variables.index(variable) + 1

            if variable == 'exceedance_probabilities':
                plot_exceedance_probabilities(fig, row, points_data_dict[point], points_data_dict[point]['times'])
                continue

            x_values = points_data_dict[point]['times']
            y_values = np.array(points_data_dict[point][variable]).flatten()
            y_min, y_max = min(y_values), max(y_values)

            # Initialize a 2D list to store member lines
            member_lines = []

            # Loop over each member
            if gefs and eps:
                num_members = 80
            elif eps and not gefs:
                num_members = 50
            elif gefs and not eps:
                num_members = 30

            for m in range(num_members):
                member_line = member_across_steps(points_data_dict[point][variable], m)
                add_line(fig, row, m, x_values, member_line, None, variable)
                member_lines.append(member_line)

            # Convert to a 2D numpy array
            member_lines_array = np.array(member_lines)

            # Calculate percentiles across members (axis=0) for each timestep
            lower_percentile = np.percentile(member_lines_array, 20, axis=0)
            upper_percentile = np.percentile(member_lines_array, 80, axis=0)

            if variable == 'slr':
                median_line = median_across_steps(points_data_dict[point][variable])
            else:
                median_line = mean_across_steps(points_data_dict[point][variable])
            add_line(fig, row, None, x_values, median_line, 'Median', variable, lower_percentile, upper_percentile)

            # Create a set to store added segments
            added_segments = set()

            # Add shaded regions for night times
            for dt in x_values_datetime:
                if dt.hour >= 18 or dt.hour < 6:  # Night time: 6 PM to 6 AM
                    night_start = dt.replace(hour=18, minute=0, second=0) if dt.hour >= 18 else (dt - timedelta(days=1)).replace(hour=18, minute=0, second=0)
                    night_end = night_start + timedelta(hours=12)
                    segment = (night_start.strftime('%Y-%m-%d %H:%M:%S'), night_end.strftime('%Y-%m-%d %H:%M:%S'))

                    # Only add the segment if it hasn't been added before
                    if segment not in added_segments:
                        fig.add_shape(
                            go.layout.Shape(
                                type="rect",
                                xref=f'x{row}',  # Reference the specific x-axis of the subplot
                                yref=f'y{row}',  # Reference the specific y-axis of the subplot
                                x0=segment[0],
                                x1=segment[1],
                                y0=y_min*0.95,  # Use actual data min for y0
                                y1=y_max*1.05,  # Use actual data max for y1
                                fillcolor="grey",
                                opacity=0.15,
                                layer="below",
                                line_width=0,
                            )
                        )
                        # Add the segment to the set of added segments
                        added_segments.add(segment)
            
            # Update y-axis range to be from the minimum to the maximum y value
            fig.update_yaxes(range=[y_min*0.95, y_max*1.05], row=row, col=1)

        # Update x-axis range to be from the first to the last x value
        fig.update_xaxes(range=[x_values_datetime[0], x_values_datetime[-1]])

        unique_days = {x.date() for x in x_values_datetime}
        # Generate tickvals for 6AM, 12PM, 6PM, 12AM for each day
        tickvals = []
        for day in sorted(unique_days):
            for t in [time(hour=6), time(hour=12), time(hour=18), time(hour=0)]:
                tickvals.append(datetime.combine(day, t))
        # Generate ticktext with the desired labels
        ticktext = []
        for tick in tickvals:
            if tick.time() == time(hour=0):  # This is a 12 AM tick
                ticktext.append(tick.strftime('%a %m/%d'))  # Include the day and date for midnight
            else:
                ticktext.append(tick.strftime('%I%p').lstrip('0'))  # Include only the time for other ticks
        # Step 4: Update the x-axes layout with the new tickvals and ticktext
        for i in range(1, len(variables)):
            fig.update_xaxes(tickangle=45, tickvals=tickvals, ticktext=ticktext, row=i, col=1)

        # Adjust the layout of the figure and write to HTML file
        fig.update_layout(
            height=300 * len(variables), width=1000,
            title_text=f'{point} ({int(points_data_dict[point]["elevation"])}ft)'
        )

        fig.write_html(os.path.join(os.getcwd(), 'output', f'{point}.html'))