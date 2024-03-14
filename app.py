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
file_name = 'heart_failure_clinical_records_dataset,predictions.csv'
feature_name = 'feature_importance.csv'
total_path = base_path + '//Data//' 
df1 = pd.read_csv(total_path + file_name)
feature_importance = pd.read_csv(total_path + feature_name).sort_values(by = ['Importance'], 
                                                                                             ascending=False)

def filter_dataframe(input_df, var1, var2, var3):

    bp_list, sex_list,anaemia_list  = [], [], []
    # Filtering for blood pressure
    if var1== "all_values":
        bp_list = input_df['high_blood_pressure'].drop_duplicates()
    else:
        bp_list = [var1]
    # Filtering for sex
    if var2== "all_values":
        sex_list = input_df['sex'].drop_duplicates()
    else:
        sex_list = [var2]
    # Filtering for Anaemia
    if var3== "all_values":
        anaemia_list = input_df['anaemia'].drop_duplicates()
    else:
        anaemia_list = [var3]
    # Applying filters to dataframe
    input_df = input_df[(input_df['high_blood_pressure'].isin(bp_list)) &
                              (input_df['sex'].isin(sex_list)) &
                               (input_df['anaemia'].isin(anaemia_list))]
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

def draw_Image(input_figure):

    return html.Div([
            dbc.Card(
                dbc.CardBody([
                    dcc.Graph(figure=input_figure.update_layout(
                            template='plotly_dark',
                            plot_bgcolor= 'rgba(0, 0, 0, 0)',
                            paper_bgcolor= 'rgba(0, 0, 0, 0)',
                        )
                    ) 
                ])
            ),  
        ])

# Returning model performance
cmatrix = confusion_matrix(df1['DEATH_EVENT'], df1['Prediction'])

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
                "Tutorial project detailing how to develop a basic front end application exploring the factors influencing heart failure", className="lead"
            ),
            html.H3("Model"
            ),
            html.P(
                "This project uses a Random Forest Classifier to predict heart failure based on 12 independent variables.", className="lead"
            ),

            html.H3("Code"
            ),
            html.P(
                "The complete code for this project is available on github.", className="lead"
            ),
            html.A(
                href="https://github.com/pinstripezebra/Dash-Tutorial",
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
                html.H1('Heart Failure Prediction'),
                dcc.Markdown('A comprehensive tool for examining factors impacting heart failure'),

                html.Label('Blood Pressure'),
                dcc.Dropdown(
                    id = 'BP-Filter',
                    options = [{"label": i, "value": i} for i in df1['high_blood_pressure'].drop_duplicates()] + 
                                [{"label": "Select All", "value": "all_values"}],
                    value = "all_values"),

                html.Label('Sex'),
                dcc.Dropdown(
                    id = 'Sex-Filter',
                    options = [{"label": i, "value": i} for i in df1['sex'].drop_duplicates()] + 
                                [{"label": "Select All", "value": "all_values"}],
                    value = "all_values"),

                html.Label('Anaemia'),
                dcc.Dropdown(
                    id = 'Anaemia-Filter',
                    options = [{"label": i, "value": i} for i in df1['anaemia'].drop_duplicates()] + 
                                [{"label": "Select All", "value": "all_values"}],
                    value = "all_values")])
             ])
], style = FILTER_STYLE)

sources = html.Div([
                html.H3('Data Sources:'),
                html.Div([
                    html.Div(children = [
                        html.Div([
                            dcc.Markdown("""Data Description: This dataset contains 12 features that 
                                         can be used to predict mortality by heart failure with each row representing 
                                         a separate patient, the response variable is DEATH_EVENT.""")
                        ]),
                        html.Div([
                            html.A("Dataset available on Kaggle", 
                                   href='https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data?select=heart_failure_clinical_records_dataset.csv', target="_blank")
                        ], style={'display': 'inline-block'})
                    ]),

                html.H3('Citation'),
                dcc.Markdown(
                """Davide Chicco, Giuseppe Jurman: Machine learning can predict survival 
                of patients with heart failure from serum creatinine and ejection fraction alone. 
                BMC Medical Informatics and Decision Making 20, 16 (2020)""")
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
                    dbc.Row(id = 'EDA-Row'),
                    html.Br(),
                    dbc.Row(id = 'ML-Row'), 
                    sources     
                ]), color = 'dark'
            )
        ])
    ],style = CONTENT_STYLE)
])

# callback for top row
@callback(
    Output(component_id='EDA-Row', component_property='children'),
    [Input('BP-Filter', 'value'),
     Input('Sex-Filter', 'value'),
     Input('Anaemia-Filter', 'value')]
)
def update_output_div(bp, sex, anaemia):

    #Making copy of DF and filtering
    filtered_df = df1
    filtered_df = filter_dataframe(filtered_df, bp, sex, anaemia)

    #Creating figures
    factor_fig = px.histogram(filtered_df, x= 'age', facet_col="diabetes", color = 'DEATH_EVENT', 
                              title = "Age and Diabetes vs. Death")
    age_fig = px.scatter(filtered_df, x="ejection_fraction", y="serum_creatinine", facet_col="high_blood_pressure",
                         color = "DEATH_EVENT", 
                         title = "Ejection Fraction and Creatinine vs. Death")
    time_fig = px.scatter(filtered_df, x = 'time', y = 'platelets', color = 'DEATH_EVENT',
                              title = 'Time and Platelets vs Death')

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


# callback for second row
@callback(
    Output(component_id='ML-Row', component_property='children'),
    Input('Sex-Filter', 'value')
)
def update_model(value):

    # Making copy of df
    confusion = cmatrix
    #x_copy = X_cols
    f_importance = feature_importance

    # Aggregating confusion dataframe and plotting
    confusion_fig = px.imshow(confusion, 
                              labels=dict(x="Predicted Value", 
                                y="True Value", color="Prediction"),
                                aspect="auto",
                                text_auto=True,
                                title = "Confusion Matrix - Predicted vs Actual Values")
    
    # Graphing feature importance
    feature_fig =  px.bar(f_importance, x='Feature Name', y='Importance',
                          title = 'Feature Importance')

    return dbc.Row([
                dbc.Col([
                    draw_Image(feature_fig)
                ], width={"size": 5}),
                dbc.Col([
                    draw_Image(confusion_fig)
                ],width={"size": 3})
            ])

# callback for kpi row
@callback(
    Output(component_id='kpi-Row', component_property='children'),
    [Input('BP-Filter', 'value'),
     Input('Sex-Filter', 'value'),
     Input('Anaemia-Filter', 'value')]
)
def update_kpi(bp, sex, anaemia):

    # Copying and filtering dataframe
    filtered_df = df1
    filtered_df = filter_dataframe(filtered_df, bp, sex, anaemia)

    observation_count = filtered_df.shape[0]
    death_count = filtered_df[filtered_df['DEATH_EVENT']==1].shape[0]
    no_death_count = filtered_df[filtered_df['DEATH_EVENT']==0].shape[0]
    
    return dbc.Row([
                        dbc.Col([
                                draw_Text("Observations: " + str(observation_count))
                        ], width=3),
                        dbc.Col([
                            draw_Text("Death Count: " + str(death_count))
                        ], width=3),
                        dbc.Col([
                            draw_Text("Survival Count: " + str(no_death_count))
                        ], width=3),
                    ])

# Runing the app
if __name__ == '__main__':
    dash_app.run_server(debug=False)