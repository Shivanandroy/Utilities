# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 03:07:21 2018

@author: Shivanand.Roy
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 14:33:46 2018

@author: Shivanand.Roy
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 15:12:33 2018

@author: Shivanand.Roy
"""

import dash_html_components as html
import dash_core_components as dcc
import dash
import plotly.graph_objs as go

import plotly
import dash_table_experiments as dte
from dash.dependencies import Input, Output, State

import pandas as pd
import numpy as np

import json
import datetime
import operator
import os

import base64
import io

import os

from statsmodels.tsa.seasonal import seasonal_decompose
dir=r'C:\Program Files\mingw-w64\x86_64-7.2.0-posix-seh-rt_v5-rev0\mingw64\bin'
os.environ['PATH'] = dir + ';' + os.environ['PATH']


app = dash.Dash()

app.scripts.config.serve_locally = True
app.config['suppress_callback_exceptions'] = True


colors = {
        'background': '#111111',
        'text': '#7FDBFF'
        }


app.layout = html.Div([
        html.Br(),
        html.H1("Time Series Analysis", style={'textAlign':'left','backgroundColor':'black','color':'white'}),
        html.Div([
        html.Br(),
        html.Br(),
        html.Br(),
        html.H6("Upload File"),
        
    dcc.Upload(
        id='upload-data', max_size=-1,
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '40px',
            'lineHeight': '40px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center'
            #'margin': '10px'
        },
        multiple=False),
    
    html.Br(),
    html.H6("Start Year"),
    dcc.Dropdown(id='select_year',
                 options=[{'label': i, 'value':i} for i in pd.Series(pd.date_range(start='1950', end='2100', freq='AS')).dt.year],
        multi = False,
        placeholder='2015, 2016, 2017...'),
    
    html.Br(),
    html.H6("Frequency"),
    dcc.Dropdown(id='select_frequency',
                 options=[{'label': 'Daily', 'value':'D'},
                          {'label': 'Weekly', 'value':'W'},
                          {'label': 'Monthly', 'value':'MS'},
                          {'label': 'Yearly', 'value':'YS'}],
        multi = False,
        placeholder='Daily, Weekly, Monthly...'),
                 
    html.Br(),             
    html.H6("Number of Data Points"),
    dcc.Dropdown(id='select_datapoints',
                 options=[{'label': i, 'value':i} for i in range(1,5001,1)],
        multi = False,
        placeholder='36, 48, 60...'),
    html.Br(),
    html.Br()
    ], style={'width':'15%', 'fontsize':'11', 'display':'inline-block', 'vertical-align':'bottom'}),
                 
    html.Div([
            dcc.Graph(id='time_series_plot')
    
        ], style={'width':'80%', 'float':'right', 'display':'inline-block'}),
    
    html.Div([
            html.Div(dte.DataTable(rows=[{}], id='time_series_data'))
            ],style={'display':'none'}),
    
    html.Div([
            html.Div(dte.DataTable(rows=[{}], id='forecasted_data'))
            ], style={'display':'none'}),
    
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.H1("Trend, Seasonality & Residual", style={'textAlign':'left','backgroundColor':'black','color':'white'}),
    dcc.Graph(id='decomposition'),
    
    html.Br(),
    html.H1("Forecasting", style={'textAlign':'left','backgroundColor':'black','color':'white'}),
    html.Div([
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.H6("Choose a Time Series Technique"),
    
    dcc.Dropdown(id="ts_algorithm",
                 options=[{'label': 'Holt-Winters', 'value': 'holt_winters'},
                          {'label': 'ARIMA', 'value': 'arima'},
                          {'label': 'LSTM', 'value': 'lstm'},
                          {'label': 'Prophet', 'value': 'prophet'}],
                multi=False,
                value='holt_winters',
                placeholder='Holt Winters, ARIMA, LSTM'
            ),
                
    html.Br(),
    html.H6("Number of Test Data Points"),
    dcc.Dropdown(id='n_test_datapoints',
                 options=[{'label': i, 'value':i} for i in range(1,5001,1)],
                 multi = False,
                 #value=6,
                 placeholder='6, 8, 10...'),
                 
    html.Br(),
    html.H6("Number of Predictions"),
    dcc.Dropdown(id='n_predictions',
                 options=[{'label': i, 'value':i} for i in range(1,5001,1)],
                 multi = False,
                 value=6,
                 placeholder='6, 8, 10...'),
                 html.Br(),
                 html.Br(),
                 html.Br()],style={'width':'15%', 'fontsize':'11', 'display':'inline-block', 'vertical-align':'bottom'}),
                 
           
    html.Div([
    dcc.Graph(id='forecasting_plot')],style={'width':'80%', 'float':'right', 'display':'inline-block'})       
    
    
    ])

# Functions-->> Holt-Winters Method:
    
def initial_trend(series, slen):

    sum = 0.0

    for i in range(slen):

        sum += float(series[i+slen] - series[i]) / slen

    return sum / slen





def initial_seasonal_components(series, slen):

    seasonals = {}

    season_averages = []

    n_seasons = int(len(series)/slen)

    # compute season averages

    for j in range(n_seasons):

        season_averages.append(sum(series[slen*j:slen*j+slen])/float(slen))

    # compute initial values

    for i in range(slen):

        sum_of_vals_over_avg = 0.0

        for j in range(n_seasons):

            sum_of_vals_over_avg += series[slen*j+i]-season_averages[j]

        seasonals[i] = sum_of_vals_over_avg/n_seasons

    return seasonals







def triple_exponential_smoothing(series, slen, alpha, beta, gamma, n_preds):

    result = []

    seasonals = initial_seasonal_components(series, slen)

    for i in range(len(series)+n_preds):

        if i == 0: # initial values

            smooth = series[0]

            trend = initial_trend(series, slen)

            result.append(series[0])

            continue

        if i >= len(series): # we are forecasting

            m = i - len(series) + 1

            result.append((smooth + m*trend) + seasonals[i%slen])

        else:

            val = series[i]

            last_smooth, smooth = smooth, alpha*(val-seasonals[i%slen]) + (1-alpha)*(smooth+trend)

            trend = beta * (smooth-last_smooth) + (1-beta)*trend

            seasonals[i%slen] = gamma*(val-smooth) + (1-gamma)*seasonals[i%slen]

            result.append(smooth+trend+seasonals[i%slen])

    return result



def RMSE(truth, pred):
    
    return np.sqrt(np.mean((truth-pred)**2))

def MAPE(truth, pred):
    
    return np.mean(np.abs((truth-pred))/truth)*100


def best_HOLT_WINTERS_parameters(ts, max_alpha, max_beta, max_gamma, slen, n_preds, test_ts):
    
    alpha = np.arange(0, max_alpha,0.05)
    beta = np.arange(0, max_beta,0.05)
    gamma = np.arange(0, max_gamma,0.05)
    
    results_df = pd.DataFrame()
    
    best_alpha = None
    best_beta = None
    best_gamma = None
    best_RMSE = np.inf
    best_MAPE = None
    
    if (len(test_ts)== n_preds):
        for a in alpha:
            for b in beta:
                for g in gamma:
                    results = triple_exponential_smoothing(ts, slen, a, b, g, n_preds)

                    predictions = results[-n_preds:]
                    
                    rmse = RMSE(test_ts, predictions)
                    mape = MAPE(test_ts, predictions)
                    
                    print('alpha {} | beta {} | gamma {} | RMSE: {} | MAPE: {}'.format(a, b, g, rmse, mape))
                    results_df = results_df.append({'alpha':a, 'beta':b, 'gamma':g, 'RMSE': rmse, 'MAPE':mape}, ignore_index=True)
                    
                    if rmse < best_RMSE:
                        best_RMSE = rmse
                        best_alpha = a
                        best_beta = b
                        best_gamma = g
                        best_MAPE = mape
    
    
       
    return best_alpha, best_beta, best_gamma, best_RMSE, best_MAPE
                
    
    

# file upload function
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')) )
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))

    except Exception as e:
        print(e)
        return None

    return df


@app.callback(dash.dependencies.Output('time_series_data', 'rows'),
              [dash.dependencies.Input('upload-data', 'contents'),
               dash.dependencies.Input('upload-data', 'filename'),
               dash.dependencies.Input('select_year', 'value'),
               dash.dependencies.Input('select_frequency', 'value'),
               dash.dependencies.Input('select_datapoints','value')
               ])
def update_output(contents, filename, start_year, frequency, datapoints):
    
    if contents is not None:
        df = parse_contents(contents, filename)
        date = pd.Series(pd.date_range(start=str(start_year), freq=frequency, periods=datapoints))
        df['zzz'] = pd.to_datetime(date)
        return df.to_dict('records')
    else:
        return [{}]



@app.callback(Output('time_series_plot', 'figure'),
              [Input('time_series_data','rows')])
def update_ts_plot(tablerows):
    
        
    ts = pd.DataFrame(tablerows)
    ts['zzz'] = pd.to_datetime(ts.zzz)
     
    graph = go.Scatter(
            x =ts['zzz'],
            y = ts.iloc[:,0],
            mode = 'lines+markers'
            )
    
    return {
            'data':[graph],
            'layout':go.Layout(title='Time Series Plot : '+ ts.columns[0]+'<br> Min:'+str(ts[ts.columns[0]].min())+' | Mean:' + str(int(np.mean(ts[ts.columns[0]])))+' | Max:'+str(ts[ts.columns[0]].max())+'</br>',
                               height=600)
            
            }
    

@app.callback(Output('decomposition', 'figure'),
              [Input('time_series_data','rows')
              ])
def update_decomposition(tablerows):
    
          
    ts = pd.DataFrame(tablerows)
    ts['zzz'] = pd.to_datetime(ts.zzz)
    
    decomposition = seasonal_decompose(ts.set_index('zzz'))
    
    resid = go.Scatter(
            x =decomposition.resid.reset_index()['zzz'],
            y = decomposition.resid.iloc[:,0],
            mode = 'lines+markers',
            name='Residual'
            )
    season = go.Scatter(
            x =decomposition.seasonal.reset_index()['zzz'],
            y = decomposition.seasonal.iloc[:,0],
            mode = 'lines+markers',
            name='Seasonality'
            )
    trend = go.Scatter(
            x =decomposition.trend.reset_index()['zzz'],
            y = decomposition.trend.iloc[:,0],
            mode = 'lines+markers',
            name='Trend'
            )
    
      
      
    return {
            'data':[trend, season, resid],
            'layout':go.Layout(title='Trend, Seasonality & Residual',
                               height=600)
            
            }






@app.callback(dash.dependencies.Output('forecasted_data', 'rows'),
              [dash.dependencies.Input('time_series_data','rows'),
               dash.dependencies.Input('n_test_datapoints','value'),
               dash.dependencies.Input('select_year', 'value'),
               dash.dependencies.Input('select_frequency', 'value'),
               dash.dependencies.Input('select_datapoints','value'),
               dash.dependencies.Input('ts_algorithm','value')
               
               ])
def update_forecast(tablerows, testsize, start_year, frequency, datapoints, algorithm):
    
        
    if algorithm == 'holt_winters':
                
        ts = pd.DataFrame(tablerows)
        training_ts = ts.iloc[:-testsize,0]
        test_ts = ts.iloc[-testsize:,0]
        alpha, beta, gamma, rmse, mape = best_HOLT_WINTERS_parameters(training_ts,1,1,1,12,testsize,test_ts)
        
        result_ts = triple_exponential_smoothing(ts.iloc[:,0], 12, alpha, beta, gamma, 50)
            
        date = pd.Series(pd.date_range(start=str(start_year), freq=frequency, periods=datapoints+50))
        result_ts = pd.DataFrame({'forecast_points':result_ts,'zzz':date})
        
        result_ts['alpha']=alpha
        result_ts['beta'] = beta
        result_ts['gamma'] = gamma
        result_ts['rmse'] = rmse
        result_ts['mape'] = mape
        
        return result_ts.to_dict('records')
    
@app.callback(dash.dependencies.Output('forecasting_plot', 'figure'),
              [dash.dependencies.Input('time_series_data','rows'),
               dash.dependencies.Input('forecasted_data','rows'),
               dash.dependencies.Input('n_predictions','value')
               ])
def update_forecast_plot(tsrows, forecastedrows,n_predictions):
    
    ts = pd.DataFrame(tsrows)
    result_ts = pd.DataFrame(forecastedrows)
    
    forecast = go.Scatter(
            x =result_ts['zzz'].iloc[:-(50-n_predictions)],
            y = result_ts['forecast_points'].iloc[:-(50-n_predictions)],
            mode = 'lines+markers',
            name='Forecast'
            )
    
    actual = go.Scatter(
            x =ts['zzz'],
            y = ts.iloc[:,0],
            mode = 'lines+markers',
            name='Original'
            )
    
    return {
            'data':[forecast, actual],
            'layout':go.Layout(title='Time Series Forecasting : <b>Holt-Winters</b>'+'<br> Best Parameters: ('+ str(np.round(result_ts.alpha[0],2)) +', '+ str(np.round(result_ts.beta[0],2)) +', '+ str(np.round(result_ts.gamma[0],2)) +')'+' | '+'RMSE: '+str(np.round(result_ts.rmse[0],2))+' | '+'MAPE: '+str(np.round(result_ts.mape[0],1))+'%',
                               height=600)
            
            }








app.css.append_css({
    "external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"
})
    

if __name__ == '__main__':
    app.run_server(debug=True)