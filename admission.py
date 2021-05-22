#Import all the libraries to use
from dash_core_components.Markdown import Markdown
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from plotly.graph_objs import layout
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#Import the dash components
import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
#--------------------------------------------------------------------#

MATHJAX_CDN = '''
https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/
MathJax.js?config=TeX-MML-AM_CHTML'''

bc='/black-dashboard.css'

#Create the app
app=dash.Dash(__name__, external_stylesheets=[bc],
external_scripts = [
                    {'type': 'text/javascript',
                     'id': 'MathJax-script',
                     'src': MATHJAX_CDN,
                     },
                    ])
server=app.server

app.title='Graduate Admissions - Ricardo Leal'

#--------------------------------------------------------------------#

#Read the data and do the data cleaning
data=pd.read_csv('Admission_Predict.csv')
students=pd.DataFrame(data)
students=students.drop(['Serial No.','GRE Score'],axis=1)
#To analize the data
students_by_university=students.groupby(by='University Rating').mean()
students_by_university=students_by_university.reset_index().round(2)

#--------------------------------------------------------------------#

#Train the regression model.
chances=students['Chance of Admit']
sections=students.drop('Chance of Admit',axis=1)

X_train, X_test, Y_train, Y_test=train_test_split(sections, chances, 
                                  test_size=0.2, random_state=42)

reg= LinearRegression()
reg.fit(X_train,Y_train)
Y_predict=reg.predict(X_test)
reg_score=(reg.score(X_test,Y_test))*100


#--------------------------------------------------------------------#

#Create all the dash style
app.layout= html.Div([

    #Put the header
    dbc.Row(dbc.Col([html.H1('Chance of admission in a graduate course', 
            style={'text-aling':'center'}),html.Br()], xl={'size':20,'offset':3}, 
            lg={'size':9,'offset':3}, md={'size':6, 'offset':3},
            sm={'size':4,'offset':1}, xs={'size':3, 'offset':0}
            )
    ),
    #First row
    dbc.Row([
        #First column.
        dbc.Col([
            dbc.Card([
            dbc.CardBody([
            html.H3('Check your probability of admission!'),
            dcc.Markdown(''' Choose some values that fit with your profile.
                The model has an accuracity of 81.92%.
            '''),

            html.Div([
                #TOEFL
                dcc.Markdown('**TOEFL Score**.'),
                dbc.Input(placeholder='Type a number between 0 and 120.',
                value=90,
                type='number',min=0,max=120,step=1, id='toefl-input'),
                html.Br(),
                #Uni
                dcc.Markdown('**University Rating.**'),
                dbc.Input(placeholder='Type a number between 1 and 5.',
                value=3,
                type='number',min=1,max=5,step=1, id='uni-input'),
                html.Br(),
                #SOP
                dcc.Markdown('**Statement of Purpose.**'),
                dbc.Input(placeholder='Type a number between 1 and 5.',
                value=4.5,
                type='number',min=1,max=5,step=0.5, id='sop-input'),
                html.Br(),
                #LOR
                dcc.Markdown('**Letter of Recommendation.**'),
                dbc.Input(placeholder='Type a number between 1 and 5.',
                value=4.5,
                type='number',min=1,max=5,step=0.5, id='lor-input'),
                html.Br(),
                #CGPA
                dcc.Markdown('**CGPA.**'),
                dbc.Input(placeholder='Type a number between 1 and 10.',
                value=8.75,
                type='number',min=1,max=10,step=0.25, id='cgpa-input'),
                html.Br(),
                #Research
                dcc.Markdown('**Research experience.**'),
                dbc.Input(placeholder='Type just 0 for no experience or 1 if you had experience.',
                value=1,
                type='number',min=0,max=1,step=1, id='res-input'),
                html.Br(),

                #Print the result
                html.H3(id='result-admission')
            ], id='styled-numeric-input')

            ])
            ])   
        ],width={'size':4, 'order':1, 'offset':1}),

        #Second Column for the regression plot
        dbc.Col([
            dbc.Card([
            dbc.CardBody([
            html.H3('Select the section for the regression plot.'),
            #This will let us select the different values in our Dataset to plot
            #against Chance of Admit
            dcc.Dropdown(id='section-data', placeholder='Choose the variable to analize.',
                options=[
                    {'label':'TOEFL Score', 'value':'TOEFL Score'},
                    {'label':'University Rating', 'value':'University Rating'},
                    {'label':'Statement of Purpose', 'value':'SOP'},
                    {'label':'Letter of Recomendation', 'value':'LOR'},
                    {'label':'CGPA', 'value':'CGPA'},
                    {'label':'Research experience', 'value':'Research'}
                ],
                value='TOEFL Score',
                multi=False,
            ),
            html.Hr(),
            #Now, put the plot
            dcc.Graph(id='regression_plot', figure={}),
        
            ])
            ]) 
        ],width={'size':4,'order':2,'offset':1})
        ],no_gutters=True
    ),

    #Now, some info about the model.
    dbc.Row(dbc.Col([html.Hr(),
            html.H1('More info about the model and the data.', 
            style={'text-aling':'center'}),html.Br()], xl={'size':15,'offset':3}, 
            lg={'size':9,'offset':3}, md={'size':6, 'offset':3},
            sm={'size':4,'offset':1}, xs={'size':3, 'offset':0}
            )
    ),
    #Now, write the info.
    dbc.Row([
        #The first row is to put text.
        dbc.Col([
            dbc.Card([
            dbc.CardBody([
        html.Br(),
        dcc.Markdown('''### Graduate admission using Machine Learning.'''),
        #html.Br(),
        dcc.Markdown('''The graduate admission can be really tough for most of the students. 
        As a student that want to form part of a graduate school, i wonder myself,
        what are my chance to get into one of this schools?'''),
        #html.Br(),
        dcc.Markdown('''In the webpage Kaggle, i found this incredible dataset that collect data 
        from undergraduate students, such as the University Rating 
        (the University place where study their undergraduate), their CGPA, 
        their TOEFL Score and others. The most important thing that this 
        dataset has is the *Chance of Admit* from every student based on 
        their qualifications.'''),
        #html.Br(),
        dcc.Markdown('''With all this data, it's possible to use Multiple Linear Regression 
        (since we had several variables) to create a model to predict the 
        chance of admission from new students that aren't in the dataset.'''),
        #html.Br(),
        dcc.Markdown('''The average data from the different university are:'''),
        dbc.Table.from_dataframe(students_by_university,
        bordered=True, dark=True, hover=True, responsive=True, striped=True),
        #html.Br()
            ])
            ]) 
        ],width={'size':7,'order':2,'offset':2})
    ]),
    #Now, do the heatmap
    dbc.Row([
        #The Checklist
        dbc.Col(
            dbc.Card([
            dbc.CardBody([
            dcc.Checklist(
            id='sections_heat',
            options=[
                    {'label':'TOEFL Score', 'value':'TOEFL Score'},
                    {'label':'University Rating', 'value':'University Rating'},
                    {'label':'Statement of Purpose', 'value':'SOP'},
                    {'label':'Letter of Recomendation', 'value':'LOR'},
                    {'label':'CGPA', 'value':'CGPA'},
                    {'label':'Research experience', 'value':'Research'}
                    ],
            value=students.columns.tolist(),
            labelStyle={'display': 'block'}
            ),
            ])
            ]) 
        , width={'size':1,'order':1,'offset':3}),
        dbc.Col(
            dbc.Card([
            dbc.CardBody([
            dcc.Graph(id='heatmap', figure={}),
            ])
            ]),  
                width={'size':'auto','order':2,'offset':0}, 
                md={'size':'auto','order':2,'offset':0},
                sm={'size':'auto','order':2,'offset':1},
                xs={'size':'auto','order':2,'offset':1})
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
            dbc.CardBody([
        dcc.Markdown('''The most importat variables for **Chance of Admit** are *CGPA*, *TOEFL Score*
        and *University Rating*.'''),
        dcc.Markdown('''Right now, the *GRE Score* are useless for our analysis
        due to COVID-19, most universities doesn't require *GRE Score* 
        for graduate admissions, so i don't include it.''')
            ])
            ])
        ],width={'size':7,'order':2,'offset':2})
    ]),
        #Now, some info about the model.
    dbc.Row(dbc.Col([html.Hr(),
            html.H1('More info about me.', 
            style={'text-aling':'center'}),html.Br()], xl={'size':15,'offset':3}, 
            lg={'size':9,'offset':3}, md={'size':6, 'offset':3},
            sm={'size':4,'offset':1}, xs={'size':3, 'offset':0}
            )
    ),
    dbc.Row([
        #The first row is to put text.
        dbc.Col([
            dbc.Card([
            dbc.CardBody([
        html.Br(),
        dcc.Markdown('''### Hi! My name is Ricardo M. Leal Lopez.'''),

        dcc.Markdown('''Thank you so much for using my web application, I hope that
        you enjoy it. '''),

        dcc.Markdown('''I'm a computational physicist that works with Python, Julia, Matlab and Fortran
        to simulate several situations, such as heat conduction, the behavior of a wave and others.
        Right now i'm on my path to study a graduate course.'''),


        dcc.Markdown('''If you want to check more projects or want to now more of my habilities/jobs,
        you can check it by [clicking here](https://ricardoleal20.github.io/Blog/).
        ''',dangerously_allow_html=True),
        html.Br()
            ])
            ])
        ],width={'size':7,'order':2,'offset':2})
    ]),

])
#--------------------------------------------------------------------#

#Use Plotly with Dash Components
@app.callback(
Output(component_id='heatmap',component_property='figure'),
[Input(component_id='sections_heat',component_property='value')]
)
def heatmap_plot(sect):
    global students
    #Theme
    theme='plotly_dark'
    fig=px.imshow(students[sect].corr())
    fig.update_layout(template=theme)
    return fig

@app.callback(
    Output(component_id='regression_plot',component_property='figure'),
    [Input(component_id='section-data',component_property='value')]
)
def reg_plot(option_section):
    global students
    #Create the predict graph
    m,b=np.polyfit(students[option_section],students['Chance of Admit'],1)
    vmin=students[option_section].min()
    vmax=students[option_section].max()
    vlen=len(students[option_section])
    x=np.linspace(vmin,vmax,vlen)
    x_rev=x[::-1]
    y=m*x+b
    y_upper=y+0.03
    y_lower=y-0.03
    y_lower=y_lower[::-1]
    k=0
    j=0
    x_plot=[]; y_plot=[ ]
    for i in range(vlen*2):
        if (i<vlen):
            x_plot.append(x[i])
            y_plot.append(y_upper[i])
        else:
            x_plot.append(x_rev[j])
            y_plot.append(y_lower[j])
            j+=1


    #Theme
    theme='plotly_dark'
    #layouts
    layout=go.Layout(
        yaxis=dict(title_text='Chance of Admit',showgrid=True),
        xaxis=dict(title_text=option_section, showgrid=True),
        template=theme,
    )
    fig= go.Figure(layout=layout)

    fig.add_trace(go.Scatter(
        x=students[option_section],
        y=students['Chance of Admit'],
        mode='markers',
        name=option_section,
        showlegend=False,
    ))

    fig.add_trace(go.Scatter(
        x=x_plot,
        y=y_plot,
        fill='toself',
        fillcolor='rgba(231,107,243,0.2)',
        line_color='rgba(255,255,255,0)',
        showlegend=False,
        mode='lines'
    ))

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        showlegend=False,
        name='Regression line',
        mode='lines'
    ))

    return fig

#--------------------------------------------------------------------#
@app.callback(
    Output(component_id='result-admission',component_property='children'),
[Input(component_id='toefl-input', component_property='value'),
Input(component_id='uni-input', component_property='value'),
Input(component_id='sop-input', component_property='value'),
Input(component_id='lor-input', component_property='value'),
Input(component_id='cgpa-input', component_property='value'),
Input(component_id='res-input', component_property='value')],
)
def admission(toefl, uni, sop, lor, cgpa, res):
    #Test values
    test_student=np.array([toefl,uni,sop,lor,cgpa,res])
    test_student=test_student.reshape(1,-1)


    predict=(reg.predict(test_student)[0]*100)
    return 'Your chance of admit is {0:.2f}%'.format(predict)

#--------------------------------------------------------------------#

#Run all the code
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)