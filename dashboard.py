import dash
from dash import dcc, html, dash_table, Input, Output, callback
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Load datasets
warehouse_df = pd.read_csv("Warehouse_with_Coordinates.csv")
january_df = pd.read_csv("Stockists_with_Coordinates.csv")
first_best_results_df = pd.read_csv("Improved_Warehouse_Optimization_Results.csv")
second_best_results_df = pd.read_csv("Warehouse_Optimization_Comparison.csv")
best_warehouses_df = pd.read_csv("addy.csv")
second_best_warehouses_df = pd.read_csv("second_best_warehouses.csv")

# Initialize the Dash app with a professional theme
app = dash.Dash(__name__, 
                external_stylesheets=[
                    'https://codepen.io/chriddyp/pen/bWLwgP.css',
                    'https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap'
                ],
                suppress_callback_exceptions=True)

# ----------------- LOAD INTERACTIVE MAPS -----------------
first_map_file_path = "optimized_warehouse_map.html"
with open(first_map_file_path, "r") as map_file:
    first_map_html = map_file.read()

second_map_file_path = "second_best_warehouse_map.html"
with open(second_map_file_path, "r") as map_file:
    second_map_html = map_file.read()

# ----------------- VISUALIZATION FUNCTIONS -----------------
def create_cost_comparison_graph(df, cost_columns, title, colors=None):
    """Create a professional cost comparison bar graph
    
    Args:
        df: DataFrame with stockist data
        cost_columns: List of column names containing cost data
        title: Graph title
        colors: Optional list of colors for the bars
    """
    # Filter out the total row if it exists
    df_filtered = df[df['Stockist'] != 'Total'].copy() if 'Total' in df['Stockist'].values else df.copy()
    
    stockist_names = df_filtered['Stockist']
    
    fig = go.Figure()
    
    for i, col in enumerate(cost_columns):
        # Handle string formatting with commas
        values = pd.to_numeric(df_filtered[col].str.replace(',', ''))
        color = colors[i] if colors and i < len(colors) else None
        fig.add_trace(go.Bar(x=stockist_names, y=values, name=col, marker_color=color))
    
    fig.update_layout(
        title=title,
        xaxis_title="Stockists",
        yaxis_title="Delivery Cost (₹)",
        barmode='group',
        height=500,
        font_family='Roboto, sans-serif',
        title_font_size=16,
        title_x=0.5,
        xaxis_title_font_size=12,
        yaxis_title_font_size=12,
        legend_title_font_size=10,
        plot_bgcolor='rgba(0,0,0,0.05)',
        paper_bgcolor='white'
    )
    
    return fig

def create_frequency_graph():
    """Create a line graph showing delivery frequency for stockists"""
    stockist_names = january_df['Town'].values
    w1_frequencies = january_df['Frequency(W1) '].values
    w2_frequencies = january_df['Frequency(W2) '].values

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stockist_names, y=w1_frequencies, mode='lines+markers', name='W1 Frequency', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=stockist_names, y=w2_frequencies, mode='lines+markers', name='W2 Frequency', line=dict(color='orange')))

    fig.update_layout(
        title="Stockist Delivery Frequency (W1 & W2)",
        xaxis_title="Stockists",
        yaxis_title="Delivery Frequency",
        height=500,
        font_family='Roboto, sans-serif',
        title_font_size=16,
        title_x=0.5,
        plot_bgcolor='rgba(0,0,0,0.05)',
        paper_bgcolor='white'
    )
    return fig

# ----------------- LAYOUT -----------------
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Warehouse Optimization Dashboard", 
                style={
                    'textAlign': 'center', 
                    'color': '#333', 
                    'fontWeight': 'bold', 
                    'fontFamily': 'Roboto, sans-serif',
                    'marginBottom': '20px'
                })
    ], className='row'),
    
    # Main Content Area
    html.Div([
        # Dropdown Menu for Navigation
        html.Div([
            html.Label("Select Solution View:", 
                      style={'fontWeight': 'bold', 'marginRight': '10px', 'fontSize': '16px'}),
            dcc.Dropdown(
                id='solution-dropdown',
                options=[
                    {'label': 'First Best Solution', 'value': 'first-best'},
                    {'label': 'Second Best Solution', 'value': 'second-best'}
                ],
                value='first-best',
                style={
                    'width': '300px',
                    'fontFamily': 'Roboto, sans-serif',
                    'borderRadius': '5px'
                },
                clearable=False
            )
        ], style={'marginBottom': '20px', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),
        
        # Dynamic Content Area
        html.Div(id='dropdown-content')
    ], className='container')
], style={'backgroundColor': '#f4f4f4', 'padding': '20px'})

# ----------------- CALLBACKS -----------------
@app.callback(
    Output('dropdown-content', 'children'),
    Input('solution-dropdown', 'value')
)
def update_dropdown_content(selected_value):
    if selected_value == 'first-best':
        # First Best Solution Content
        return html.Div([
            # Interactive Map
            html.Div([
                html.H3("First Best Warehouse Solution - Interactive Map", 
                        style={'textAlign': 'center', 'color': '#333'}),
                html.Iframe(srcDoc=first_map_html, width='100%', height='600', 
                            style={'border': '1px solid #ddd', 'borderRadius': '5px'})
            ], className='row', style={'marginBottom': '20px'}),
            
            # Cost Analysis Graph
            html.Div([
                html.H3("First Best Solution - Delivery Cost Analysis", 
                        style={'textAlign': 'center', 'color': '#333'}),
                dcc.Graph(figure=create_cost_comparison_graph(
                    first_best_results_df,
                    ['Original Delivery Cost (₹)', 'Optimized Delivery Cost (₹)'],
                    "Delivery Cost Comparison (Before vs After Optimization)",
                    colors=['red', 'green']
                )),
                
                # PNG Image for Cost Comparison if available
                html.Img(src="assets/Delivery_Cost_Comparison.png", 
                         style={'width': '80%', 'display': 'block', 'margin': 'auto'})
            ], style={'padding': '20px', 'border': '2px solid #e0e0e0', 'borderRadius': '8px', 'marginBottom': '20px'}),
            
            # Frequency Graph in First Tab
            html.Div([
                html.H3("Stockist Delivery Frequency", 
                        style={'textAlign': 'center', 'color': '#333'}),
                dcc.Graph(figure=create_frequency_graph()),
                
                # PNG Image for Frequency Comparison if available
                html.Img(src="assets/Stockist_Delivery_Frequency.png", 
                         style={'width': '80%', 'display': 'block', 'margin': 'auto'})
            ], style={'padding': '20px', 'border': '2px solid #e0e0e0', 'borderRadius': '8px', 'marginBottom': '20px'}),
            
            # Detailed Results Table
            html.Div([
                html.H3("First Best Solution - Detailed Cost Analysis", 
                        style={'textAlign': 'center', 'color': '#333'}),
                dash_table.DataTable(
                    id='first-best-table',
                    columns=[{"name": col, "id": col} for col in first_best_results_df.columns],
                    data=first_best_results_df.to_dict('records'),
                    style_table={'overflowX': 'auto', 'border': '1px solid #ddd'},
                    style_header={
                        'backgroundColor': '#36A2EB', 
                        'color': 'white', 
                        'fontWeight': 'bold',
                        'textAlign': 'center'
                    },
                    style_cell={
                        'textAlign': 'center', 
                        'padding': '10px',
                        'fontFamily': 'Roboto, sans-serif'
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(248, 248, 248)'
                        }
                    ]
                )
            ], className='row')
        ])
    
    else:  # Second Best Solution Content
        return html.Div([
            # Interactive Map
            html.Div([
                html.H3("Second Best Warehouse Solution - Interactive Map", 
                        style={'textAlign': 'center', 'color': '#333'}),
                html.Iframe(srcDoc=second_map_html, width='100%', height='600', 
                            style={'border': '1px solid #ddd', 'borderRadius': '5px'})
            ], className='row', style={'marginBottom': '20px'}),
            
            # Cost Analysis Graph
            html.Div([
                html.H3("Second Best Solution - Delivery Cost Analysis", 
                        style={'textAlign': 'center', 'color': '#333'}),
                dcc.Graph(figure=create_cost_comparison_graph(
                    second_best_results_df,
                    ['Original Cost (₹)', 'Best Cost (₹)', 'Second Best Cost (₹)'],
                    "Delivery Cost Comparison (Original vs Best vs Second Best)",
                    colors=['red', 'green', 'purple']
                )),
                
                # PNG Image for Second Best Cost Comparison if available
                html.Img(src="assets/secondbest.png", 
                         style={'width': '80%', 'display': 'block', 'margin': 'auto'})
            ], style={'padding': '20px', 'border': '2px solid #e0e0e0', 'borderRadius': '8px', 'marginBottom': '20px'}),
            
            
           # Detailed Results Table
            html.Div([
                html.H3("Second Best Solution - Detailed Cost Analysis", 
                        style={'textAlign': 'center', 'color': '#333'}),
                dash_table.DataTable(
                    id='second-best-table',
                    columns=[{"name": col, "id": col} for col in second_best_results_df.columns],
                    data=second_best_results_df.to_dict('records'),
                    style_table={'overflowX': 'auto', 'border': '1px solid #ddd'},
                    style_header={
                        'backgroundColor': '#36A2EB', 
                        'color': 'white', 
                        'fontWeight': 'bold',
                        'textAlign': 'center'
                    },
                    style_cell={
                        'textAlign': 'center', 
                        'padding': '10px',
                        'fontFamily': 'Roboto, sans-serif'
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(248, 248, 248)'
                        }
                    ]
                )
            ], className='row')
        ])

# Run the dashboard
if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_hot_reload=True)