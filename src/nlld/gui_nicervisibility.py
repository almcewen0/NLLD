import dash
from dash import dcc, html, Input, Output, State, callback_context
from nlld import targetvisibilitydetails
import numpy as np
import pandas as pd

# Define initial sun angle range (you can adjust these defaults)
# initial_min = 70
# initial_max = 180
# filtered_df = df[(df['sunangle_start'] >= initial_min) & (df['sunangle_start'] <= initial_max)]

catalog_name = 'NICER2_Xray_Target_List_V42t.csv.gz'
ags3 = 'AGS3_LTVIS_report_20250801400_20250951200_V01.txt.gz'
iss_orbit_file = 'ISS.OEM_J2K_EPH.txt'
start_time = '2025-085T18:00:00'
end_time = '2025-085T19:30:00'

df_nicer_vis_timeflt, target_brightearth_all_df, target_od_startend_times_all = (
    targetvisibilitydetails.visibilitytargetcat(catalog_name, ags3, iss_orbit_file, start_time, end_time,
                                                freq_bound=60, freq_brearth=240, sa_ll=70, sa_ul=180,
                                                outputFile=None))

# Define how many sources (rows) to display per page
sources_per_page = 100

# Create the Dash app layout with next/previous buttons and a hidden store for the current page
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("NICER Visibility Plot with Pagination"),
    dcc.Graph(id='visibility-plot'),
    html.Div(id='page-info', style={'margin': '10px 0', 'fontWeight': 'bold'}),
    html.Button("Previous", id='prev-btn', n_clicks=0),
    html.Button("Next", id='next-btn', n_clicks=0),
    # Store to keep track of the current page
    dcc.Store(id='current-page', data=1)
])


@app.callback(
    [Output('visibility-plot', 'figure'),
     Output('page-info', 'children'),
     Output('current-page', 'data')],
    [Input('prev-btn', 'n_clicks'),
     Input('next-btn', 'n_clicks')],
    [State('current-page', 'data')]
)
def update_page(prev_clicks, next_clicks, current_page):
    ctx = callback_context
    if not ctx.triggered:
        button_id = None
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'next-btn':
        current_page += 1
    elif button_id == 'prev-btn':
        current_page -= 1
    max_page = int(np.ceil(len(df_nicer_vis_timeflt) / sources_per_page))
    current_page = max(1, min(current_page, max_page))

    start_idx = (current_page - 1) * sources_per_page
    end_idx = current_page * sources_per_page
    filtered_nicer_vis = df_nicer_vis_timeflt.iloc[start_idx:end_idx]

    # Filter the related DataFrames so they only include the targets on this page.
    filtered_targets = filtered_nicer_vis['target_name'].unique()
    filtered_brightearth = target_brightearth_all_df[target_brightearth_all_df['srcname'].isin(filtered_targets)]
    filtered_od_startend = target_od_startend_times_all[
        target_od_startend_times_all['target_name'].isin(filtered_targets)]
    
    # Ensure that you pass all required parameters.
    fig = targetvisibilitydetails.visibilityplot_plotly(
        filtered_nicer_vis,
        filtered_brightearth,
        filtered_od_startend,
        pd.to_datetime(start_time, format='%Y-%jT%H:%M:%S', utc=True),
        pd.to_datetime(end_time, format='%Y-%jT%H:%M:%S', utc=True),
        freq_bound=60,
        outputFile='visibility-plot'
    )
    info_text = f"Page {current_page} of {max_page}"
    return fig, info_text, current_page


if __name__ == '__main__':
    app.run(debug=True)
