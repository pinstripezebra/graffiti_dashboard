# For data manipulation, visualization, app
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


# loading Datasets
base_path = os.path.dirname(__file__)
file_name = 'combined.csv'
total_path = base_path + '//Data//' 
df1 = pd.read_csv(total_path + file_name)
df1['Graffiti_Percent'] = df1['Graffiti_Count']/df1['Total_Images']

# Setting mapbox key
f = open("./mapbox_key.txt", "r")
px.set_mapbox_access_token(f.read())


def filter_dataframe(input_df, var1, var2, var3):

    bp_list, sex_list,anaemia_list  = [], [], []
    # Filtering for blood pressure
    if var1== "all_values":
        bp_list = input_df['City'].drop_duplicates()
    else:
        bp_list = [var1]
    # Filtering for sex
    if var2== "all_values":
        sex_list = input_df['2020 population density'].drop_duplicates()
    else:
        sex_list = [var2]
    # Filtering for Anaemia
    if var3== "all_values":
        anaemia_list = input_df['Estimate!!Households!!Median income (dollars)'].drop_duplicates()
    else:
        anaemia_list = [var3]
    # Applying filters to dataframe
    input_df = input_df[(input_df['City'].isin(bp_list)) &
                              (input_df['2020 population density'].isin(sex_list)) &
                               (input_df['Estimate!!Households!!Median income (dollars)'].isin(anaemia_list))]
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
                "This project uses a CNN to identify graffiti from google streetview images and then joins this with census data.", className="lead"
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
                html.H1('Graffiti Occurence'),
                dcc.Markdown('A comprehensive tool for examining graffiti occurence rate in US cities'),

                html.Label('City'),
                dcc.Dropdown(
                    id = 'City-Filter',
                    options = [{"label": i, "value": i} for i in df1['City'].drop_duplicates()] + 
                                [{"label": "Select All", "value": "all_values"}],
                    value = "all_values"),

                html.Label('2020 population density'),
                dcc.Dropdown(
                    id = '2020 population density-Filter',
                    options = [{"label": i, "value": i} for i in df1['2020 population density'].drop_duplicates()] + 
                                [{"label": "Select All", "value": "all_values"}],
                    value = "all_values"),

                html.Label('Households Median income (dollars)'),
                dcc.Dropdown(
                    id = 'Median income (dollars)-Filter',
                    options = [{"label": i, "value": i} for i in df1['Estimate!!Households!!Median income (dollars)'].drop_duplicates()] + 
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
                    dbc.Row(id = 'Map-Row'),
                    html.Br(),
                    dbc.Row(id = 'EDA-Row'),
                    sources     
                ]), color = 'dark'
            )
        ])
    ],style = CONTENT_STYLE)
])

# callback for top row
@callback(
    Output(component_id='EDA-Row', component_property='children'),
    [Input('City-Filter', 'value'),
     Input('2020 population density-Filter', 'value'),
     Input('Median income (dollars)-Filter', 'value')]
)
def update_output_div(city, population, income):

    #Making copy of DF and filtering
    filtered_df = df1
    filtered_df = filter_dataframe(filtered_df, city,population, income)

    #Creating figures
    factor_fig = px.bar(filtered_df, x= 'City', y="Graffiti_Count", color = 'State', 
                              title = "City and State vs Graffiti")
    age_fig = px.scatter(filtered_df, x='City', y="Graffiti_Count", 
                         color="Estimate!!Households!!Median income (dollars)", 
                         title = "City and Income vs Graffiti")
    time_fig = px.scatter(filtered_df, x = 'City', y = 'Graffiti_Count', color = '2020',
                              title = 'Graffiti versus population')

    return dbc.Row([
                dbc.Col([
                    draw_Image(factor_fig)
                ], width={"size": 3, "offset": 0}),
                dbc.Col([
                    draw_Image(age_fig)
                ],width={"size": 3}),
                dbc.Col([
                    draw_Image(time_fig)
                ], width={"size": 3}),
            ])


# callback for kpi row
@callback(
    Output(component_id='kpi-Row', component_property='children'),
    [Input('City-Filter', 'value'),
     Input('2020 population density-Filter', 'value'),
     Input('Median income (dollars)-Filter', 'value')]
)
def update_kpi(city, population, income):

    # Copying and filtering dataframe
    filtered_df = df1
    filtered_df = filter_dataframe(filtered_df, city, population, income)

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
    Output(component_id='Map-Row', component_property='children'),
    [Input('City-Filter', 'value'),
     Input('2020 population density-Filter', 'value'),
     Input('Median income (dollars)-Filter', 'value')]
)
def update_output_div(city, population, income):

    #Making copy of DF and filtering
    filtered_df = df1
    filtered_df = filter_dataframe(filtered_df, city,population, income)

    #Creating figures
    map_fig = px.scatter_mapbox(filtered_df, 
                            lat="Latitude", lon="Longitude", color="Graffiti_Percent", size="Graffiti_Percent",
                            hover_data=["City", 
                                        "Estimate!!Households!!Median income (dollars)", 
                                        "State"],
                            zoom = 3)

    return dbc.Row([
                dbc.Col([
                    draw_Image(map_fig, height = '50vh')
                ], 
                width={"size": 8, "offset": 0})
            ])


# Runing the app
if __name__ == '__main__':
    dash_app.run_server(debug=False)