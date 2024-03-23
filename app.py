# For data manipulation, visualization, app
import dash
from dash import Dash, dcc, html, callback,Input, Output,dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import os 
import numpy as np

# For modeling
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go


# Aggregated dataset by City
base_path = os.path.dirname(__file__)
file_name = 'combined.csv'
total_path = base_path + '//Data//' 
df1 = pd.read_csv(total_path + file_name)
df1['Graffiti_Percent'] = df1['Graffiti_Count']/df1['Total_Images']
df1['Longitude'] = df1['Longitude'] * -1
df1 = df1.rename(columns={'Estimate!!Households!!Median income (dollars)': 'Median Household Income'})
df1['size'] = 1
df1['opacity'] = 1

# Loading non aggregated dataset
raw_df = pd.read_csv('./Data/Image raw.csv')
raw_df[['Latitude', 'Longitude']] = raw_df['coordinates'].str.split(', ', expand= True)
raw_df['Latitude'] = raw_df['Latitude'].str.replace('(', '').astype(float)
raw_df['Longitude'] = raw_df['Longitude'].str.replace(')', '').astype(float)
raw_df['size'] = 1.5
raw_df['graffiti'] = raw_df['graffiti'].astype(str)

# Setting mapbox key
f = open("./mapbox_key.txt", "r")
px.set_mapbox_access_token(f.read())


def filter_dataframe(input_df, var1, var2, var3):

    f1_list, f2_list,f3_list  = [], [], []
    if var1== "all_values":
        f1_list = input_df['City'].drop_duplicates()
    else:
        f1_list = [var1]
    if var2== "all_values":
        f2_list = input_df['State[c]'].drop_duplicates()
    else:
        f2_list = [var2]
    if var3== "all_values":
        f3_list = input_df['Median Household Income'].drop_duplicates()
    else:
        f3_list = [var3]
    # Applying filters to dataframe
    input_df = input_df[(input_df['City'].isin(f1_list)) &
                              (input_df['State[c]'].isin(f2_list)) &
                               (input_df['Median Household Income'].isin(f3_list))]
    return input_df

def draw_Text(input_text):

    return html.Div([
            dbc.Card(
                dbc.CardBody([
                        html.Div([
                            html.H2(input_text),
                        ], style={'textAlign': 'center'}) 
                ])
            ),
        ])

def draw_Image(input_figure, height = '30vh'):

    return html.Div([
            dbc.Card(
                dbc.CardBody([
                    dcc.Graph(figure=input_figure.update_layout(
                            template='plotly_dark',
                            plot_bgcolor= 'rgba(0, 0, 0, 0)',
                            paper_bgcolor= 'rgba(0, 0, 0, 0)',
                        ), style={'height': height}
                    ) 
                ])
            ),  
        ])


# Building and Initializing the app
dash_app = Dash(external_stylesheets=[dbc.themes.SLATE])
app = dash_app.server

# Defining component styles
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "18rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
    "display":"inline-block"
}

CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    "display":"inline-block",
    "width": "100%"
}
FILTER_STYLE = {"width": "30%"}

# Defining components
sidebar = html.Div(children = [
            html.H2("Description", className="display-4"),
            html.Hr(),
            html.P(
                "This project compares the rate of graffiti occurence in the top US cities with socio-economic data from the 2020 US census", className="lead"
            ),
            html.H3("Model"
            ),
            html.P(
                "This project uses a CNN to identify graffiti from google streetview images then joins this with census data.", className="lead"
            ),

            html.H3("Code"
            ),
            html.P(
                "The complete code for this project is available on github.", className="lead"
            ),
            html.A(
                href="https://github.com/pinstripezebra/graffiti_dashboard",
                children=[
                    html.Img(
                        alt="Link to Github",
                        src="github_logo.png",
                    )
                ],
                style = {'color':'black'}
            )

        ], style=SIDEBAR_STYLE
    )

filters = html.Div([
            dbc.Row([
                html.Div(children= [
                html.H1('Graffiti Occurence in the Top U.S. Cities'),
                dcc.Markdown('A comprehensive tool for examining graffiti occurence rate in US cities'),

                html.Label('City'),
                dcc.Dropdown(
                    id = 'City-Filter',
                    options = [{"label": i, "value": i} for i in df1['City'].drop_duplicates()] + 
                                [{"label": "Select All", "value": "all_values"}],
                    value = "all_values"),

                html.Label('State'),
                dcc.Dropdown(
                    id = '2020 State-Filter',
                    options = [{"label": i, "value": i} for i in df1['State[c]'].drop_duplicates()] + 
                                [{"label": "Select All", "value": "all_values"}],
                    value = "all_values"),

                html.Label('Households Median income (dollars)'),
                dcc.Dropdown(
                    id = 'Median income (dollars)-Filter',
                    options = [{"label": i, "value": i} for i in df1['Median Household Income'].drop_duplicates()] + 
                                [{"label": "Select All", "value": "all_values"}],
                    value = "all_values")])
             ])
], style = FILTER_STYLE)

sources = html.Div([
                html.H3('Data Sources:'),
                html.Div([
                    html.Div(children = [
                        html.Div([
                            dcc.Markdown("""Data Description: This dataset contains 20,0000 streetview images from the
                                         top 50 cities in the United States combined with census data on population and income.""")
                        ]),
                        html.Div([
                            html.A("Dataset available on Kaggle", 
                                   href='https://www.kaggle.com/datasets/pinstripezebra/graffiti-classification', target="_blank")
                        ], style={'display': 'inline-block'})
                    ]),
                ])
             ])

dash_app.layout = html.Div(children = [
    sidebar,
     html.Div([
        filters,
        html.Div([
            dbc.Card(
                dbc.CardBody([
                    dbc.Row(id = 'kpi-Row'), 
                    html.Br(),
                    # Manually defining map row so we can use inputs in callback
                    dbc.Row([
                        dbc.Col([
                            # Adding in back bottom
                            dbc.Button('🡠', id='back-button', outline=True, size="sm",
                                className='mt-2 ml-2 col-1', 
                                style={'display': 'none'}),
                            dcc.Graph(figure = px.scatter_mapbox(df1, 
                                    lat="Latitude", lon="Longitude", color="Graffiti_Percent", size="Total_Images",
                                    hover_data=["City", 
                                            "Median Household Income", 
                                            "State"],
                                    zoom = 4).update_layout(
                                        template='plotly_dark',
                                        plot_bgcolor= 'rgba(0, 0, 0, 0)',
                                        paper_bgcolor= 'rgba(0, 0, 0, 0)',
                                        ),
                                id = "map_fig",
                                style={'height': '50vh'})
                        ], 
                        width={"size": 8, "offset": 0}),
                        dbc.Col([ dcc.Graph(id = 'top_cities')],
                                width={"size": 3, "offset": 0})
                    ]),
                    html.Br(),
                    dbc.Row(dbc.Row([
                        dbc.Col([
                            dcc.Graph(id = "income_fig2")
                        ], width={"size": 7, "offset": 0})
                    ])),
                    sources     
                ]), color = 'dark'
            )
        ])
    ],style = CONTENT_STYLE)
])


# callback for top row
@callback(
    Output(component_id='income_fig2', component_property='figure'),
    [Input('City-Filter', 'value'),
     Input('2020 State-Filter', 'value'),
     Input('Median income (dollars)-Filter', 'value'),
     Input('income_fig2', 'clickData'), # Click data from figure
     Input('map_fig', 'clickData'),# Click data from map figure
     Input('back-button', 'n_clicks')] # Click data from back button
)
def update_output_div(city, population, income, clicks, map_clicks, back_click):

    #Making copy of DF and filtering
    filtered_df = df1
    filtered_df = filter_dataframe(filtered_df, city,population, income)

    ctx = dash.callback_context
    context = ctx.triggered[0]["prop_id"]
    # If a point has been selected by the user
    if (clicks is not None or map_clicks is not None) and context != 'back-button.n_clicks':
        selected_city = ""
        if map_clicks is not None:
            selected_city = map_clicks['points'][0]['customdata'][0]
        else:
            selected_city = clicks['points'][0]['customdata'][0]
        non_selected_df = filtered_df[filtered_df['City'] != selected_city]
        selected_df = filtered_df[filtered_df['City'] == selected_city]

        income_fig = go.Figure()
        # Non selected trace
        income_fig.add_trace(
            go.Scatter(
                mode='markers',
                x=non_selected_df['Median Household Income'],
                y=non_selected_df["Graffiti_Count"],
                marker_color = non_selected_df['Median Household Income'],
                marker = dict(opacity = 0.3,
                              size = 20),
                showlegend=False
            )
        )
        # selected trace
        income_fig.add_trace(
            go.Scatter(
                mode='markers',
                x=selected_df['Median Household Income'],
                y=selected_df["Graffiti_Count"],
                marker_color = selected_df['Median Household Income'],
                marker = dict(opacity = 1,
                              size = 20),
                showlegend=False
            )
        )
        income_fig.update_layout(
            hoverlabel=dict(
                bgcolor="white",
                font_size=16,
                font_family="Rockwell"
            )
        )
        income_fig.update_layout(template='plotly_dark')
        income_fig.update_layout(
            title="Medium Household Income vs Graffiti Occurence",
            xaxis_title="Graffiti_Count",
            yaxis_title="Median Household Income",
            legend_title="Legend Title"
        )
        return income_fig
    else:
        income_fig = px.scatter(filtered_df, x='Median Household Income', y="Graffiti_Count", 
                            color="Median Household Income", 
                            size = "size",
                            trendline="lowess",
                            title = "Medium Household Income vs Graffiti Occurence",
                            hover_data=["City", 
                                        "Median Household Income", 
                                        "Graffiti_Count"])
        income_fig.update_layout(
            hoverlabel=dict(
                bgcolor="white",
                font_size=16,
                font_family="Rockwell"
            )
        )
        income_fig.update_layout(template='plotly_dark')
        return income_fig


# callback for kpi row
@callback(
    Output(component_id='kpi-Row', component_property='children'),
    [Input('City-Filter', 'value'),
     Input('2020 State-Filter', 'value'),
     Input('Median income (dollars)-Filter', 'value'),
     Input('income_fig2', 'clickData'),
     Input('map_fig', 'clickData'),
     Input('back-button', 'n_clicks')]
)
def update_kpi(city, population, income, income_select, map_select,back):

    # Copying and filtering dataframe
    filtered_df = df1
    filtered_df = filter_dataframe(filtered_df, city, population, income)

    ctx = dash.callback_context
    context = ctx.triggered[0]["prop_id"]

    # If a point has been selected by the user
    if (income_select is not None or map_select is not None) and context != 'back-button.n_clicks':
        selected_city = ""
        if map_select is not None:
            selected_city = map_select['points'][0]['customdata'][0]
        else:
            selected_city = income_select['points'][0]['customdata'][0]
        filtered_df = filtered_df[filtered_df['City'] == selected_city]
    else:
        filtered_df = filtered_df

    return dbc.Row([
                        dbc.Col([
                                draw_Text("Images: " + str(sum(filtered_df['Total_Images'])))
                        ], width=3),
                        dbc.Col([
                            draw_Text("Graffiti Images: " + str(sum(filtered_df['Graffiti_Count'])))
                        ], width=3),
                        dbc.Col([
                            draw_Text("Non-Graffiti Images: " + str(sum(filtered_df['Non_Graffiti_Count'])))
                        ], width=3),
                    ])


# callback for map row
@callback(
    [Output(component_id='map_fig', component_property='figure'),
     Output('back-button', 'style'),
     Output(component_id='top_cities', component_property='figure')],
    [Input('City-Filter', 'value'),
     Input('2020 State-Filter', 'value'),
     Input('Median income (dollars)-Filter', 'value'),
     Input('map_fig', 'clickData'), # Click data from map figure
     Input('income_fig2', 'clickData'), # Click data from scatter figure
     Input('back-button', 'n_clicks')] # Button for returning
)
def update_output_div(city, population, income, map_click, scatter_click, back_click):

    # Checking which input was fired
    ctx = dash.callback_context
    #Making copy of DF and filtering
    filtered_df = df1
    df2 = raw_df
    filtered_df = filter_dataframe(filtered_df, city,population, income)

    #Checking which input was fired for graph drilldown
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    map_fig = ""
    
    #If map or scatter has been triggered
    if trigger_id == 'map_fig' or scatter_click is not None:
        selected_city = ""
        if scatter_click is not None:
            selected_city = scatter_click['points'][0]['customdata'][0]
        else:
            selected_city = map_click['points'][0]['customdata'][0]
        map_fig = px.scatter_mapbox(df2[df2['city'] == selected_city], 
                            lat="Latitude", lon="Longitude", color="graffiti", 
                            hover_data=["city", "coordinates"],
                            size = "size",
                            zoom = 12).update_layout(
                                        template='plotly_dark',
                                        plot_bgcolor= 'rgba(0, 0, 0, 0)',
                                        paper_bgcolor= 'rgba(0, 0, 0, 0)',
                                        )
        map_fig.update_layout(
            hoverlabel=dict(
                bgcolor="white",
                font_size=16,
                font_family="Rockwell"
            )
        )
        top_fig  = px.bar(filtered_df[filtered_df['City'] == selected_city], 
                          x="Graffiti_Count", y="City", 
                          color = "Graffiti_Count",orientation='h',
                          title="Top Cities By Graffiti Count").update_layout(
                                        template='plotly_dark',
                                        plot_bgcolor= 'rgba(0, 0, 0, 0)',
                                        paper_bgcolor= 'rgba(0, 0, 0, 0)',
                                        )
        return map_fig, {'display':'block'}, top_fig
    
    else:
        map_fig = px.scatter_mapbox(filtered_df, 
                            lat="Latitude", lon="Longitude", color="Graffiti_Percent", size="Total_Images",
                            hover_data=["City", 
                                        "Median Household Income", 
                                        "State"],
                            zoom = 4).update_layout(
                                        template='plotly_dark',
                                        plot_bgcolor= 'rgba(0, 0, 0, 0)',
                                        paper_bgcolor= 'rgba(0, 0, 0, 0)',
                                        )
        map_fig.update_layout(
            hoverlabel=dict(
                bgcolor="white",
                font_size=16,
                font_family="Rockwell"
            )
        )
        top_fig  = px.bar(filtered_df.sort_values(by="Graffiti_Count"), x="Graffiti_Count", y="City", 
                          color = "Graffiti_Count",orientation='h',
                          title="Top Cities By Graffiti Count").update_layout(
                                        template='plotly_dark',
                                        plot_bgcolor= 'rgba(0, 0, 0, 0)',
                                        paper_bgcolor= 'rgba(0, 0, 0, 0)',
                                        )
        return map_fig, {'display':'none'}, top_fig
        


# Runing the app
if __name__ == '__main__':
    dash_app.run_server(debug=False)