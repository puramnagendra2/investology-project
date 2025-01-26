import io
from django.shortcuts import render, redirect
from django.http import HttpResponse
import yfinance as yf
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from io import BytesIO
import plotly.graph_objects as go
import csv
from matplotlib.backends.backend_agg import FigureCanvasAgg
import logging
import plotly.express as px
from plotly.subplots import make_subplots

from .prediction import fetch_and_preprocess, create_sequences, build_model, predict_stock_prices


def searchPage(request):
    csv_file = 'majorApp/static/CSV/companies.csv'  # Path to your CSV file
    options = []
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            options.append({'value': row['SYMBOL'], 'text': row['NAME OF COMPANY']})

    return render(request, 'majorApp/searchPage.html', {'options': options})


def overview(request):
    if request.method == 'POST':
        c_name = request.POST.get('c_name')
        cname = request.session['cname'] = c_name
        if len(c_name) == 0:
            alert_message = True
            return render(request, 'majorApp/searchPage.html', {'alert_message': alert_message})
        else:
            c_name += ".NS"
            name = yf.Ticker(c_name)
            mutual_fund_holders = name.mutualfund_holders
            fund_names = mutual_fund_holders['Holder']
            fund_percentages = mutual_fund_holders['pctHeld']
            if fund_names is not None and fund_percentages is not None:
                pie = go.Figure(data=[go.Pie(labels=fund_names, values=fund_percentages)])
                pie.update_layout(title=f'Mutual Fund Holders Distribution for {cname}', width=1200)
                pie_chart = pie.to_html(full_html=False)
            else:
                pie = go.Figure(data=[go.Pie(labels=['No Data'], values=[100])])
                pie.update_layout(title=f'No Data of Mutual Fund Holders Distribution for {cname}', width=1200)
                pie_chart = pie.to_html(full_html=False)
            context = {'summary': get_summary(c_name).replace(';', ','),
                       'c_name': c_name,
                       'Total_Employees': name.info.get('fullTimeEmployees'),
                       'industry': name.info.get('industry'),
                       'sector': name.info.get('sector'),
                       'website': name.info.get('website'),
                       'pie_chart': pie_chart}
            return render(request, 'majorApp/overview.html', context)
    return render(request, 'majorApp/searchPage.html')


def navigation(request):
    if request.method == 'POST':
        cname = request.session.get('cname')
        action = request.POST.get('task')
        # Overview
        if action == 'overview':
            cname += ".NS"
            name = yf.Ticker(cname)
            # Mutualfund Shareholding
            mutual_fund_holders = name.mutualfund_holders
            fund_names = mutual_fund_holders['Holder']
            fund_percentages = mutual_fund_holders['pctHeld']
            print('Hello', fund_names)
            if fund_names is not None and fund_percentages is not None:
                pie = go.Figure(data=[go.Pie(labels=fund_names, values=fund_percentages)])
                pie.update_layout(title=f'Mutual Fund Holders Distribution for {cname}', width=1200)
                pie_chart = pie.to_html(full_html=False)
            else:
                pie = go.Figure(data=[go.Pie(labels=['No Data'], values=[100])])
                pie.update_layout(title=f'No Data of Mutual Fund Holders Distribution for {cname}', width=1200)
                pie_chart = pie.to_html(full_html=False)
            context = {'summary': get_summary(cname).replace(';', ','),
                       'c_name': cname,
                       'Total_Employees': name.info.get('fullTimeEmployees'),
                       'industry': name.info.get('industry'),
                       'sector': name.info.get('sector'),
                       'website': name.info.get('website'),
                       'pie_chart': pie_chart}
            return render(request, 'majorApp/overview.html', context)

        # Financials
        elif action == 'financials':
            cname += ".NS"
            stock = yf.Ticker(cname)
            balance_sheet = stock.balance_sheet
            financials = stock.financials

            # Gross Profit
            if not financials.empty:
                total_revenue = financials.loc['Total Revenue'][0] if 'Total Revenue' in financials.index else None
                cost_of_revenue = financials.loc['Cost Of Revenue'][
                    0] if 'Cost Of Revenue' in financials.index else None

                if all([total_revenue, cost_of_revenue]):
                    gross_profit = total_revenue - cost_of_revenue
                else:
                    gross_profit = 'NA'
            else:
                gross_profit = 'NA'

            # EBITDA
            if not financials.empty:
                operating_income = financials.loc['Operating Income'][
                    0] if 'Operating Income' in financials.index else None
                depreciation_and_amortization = financials.loc['Depreciation & Amortization'][
                    0] if 'Depreciation & Amortization' in financials.index else None

                if all([operating_income, depreciation_and_amortization]):
                    ebitda = operating_income + depreciation_and_amortization
                else:
                    ebitda = 'NA'
            else:
                ebitda = 'NA'

            # Plotting Total Revenue for Financial Years
            income_stmt = stock.income_stmt
            total_revenue = income_stmt.loc['Total Revenue']
            df_total_rev = total_revenue.reset_index()
            df_total_rev.columns = ['Date', 'Total Revenue']
            df_total_rev['Date'] = pd.to_datetime(df_total_rev['Date'])
            df_total_rev = df_total_rev.sort_values(by='Date').tail(4)
            df_total_rev['Total Revenue (INR Crores)'] = df_total_rev['Total Revenue'] / 1e7
            df_total_rev['Financial Year'] = (df_total_rev['Date'].dt.year - 1).astype(str) + '-' + df_total_rev[
                                                                                                        'Date'].dt.year.astype(
                str).str[2:]
            net_income = income_stmt.loc['Net Income']
            df_net_inc = net_income.reset_index()
            df_net_inc.columns = ['Date', 'Net Income']
            df_net_inc['Date'] = pd.to_datetime(df_net_inc['Date'])
            df_net_inc = df_net_inc.sort_values(by='Date').tail(4)
            df_net_inc['Net Income (INR Crores)'] = df_net_inc['Net Income'] / 1e7
            df_net_inc['Financial Year'] = (df_net_inc['Date'].dt.year - 1).astype(str) + '-' + df_net_inc[
                                                                                                    'Date'].dt.year.astype(
                str).str[2:]
            plots = make_subplots(rows=1, cols=2, shared_yaxes=True,
                                subplot_titles=[f'Total Revenue ({cname})', f'Net Income ({cname})'])
            plots.add_trace(go.Bar(x=df_total_rev['Financial Year'], y=df_total_rev['Total Revenue (INR Crores)'],
                                 marker_color='#2ca02c', name='Total Revenue (INR Crores)'), row=1, col=1)
            plots.add_trace(go.Bar(x=df_net_inc['Financial Year'], y=df_net_inc['Net Income (INR Crores)'],
                                 marker_color='skyblue', name='Net Income (INR Crores)'), row=1, col=2)
            plots.update_layout(
                title=f'{cname} Financial Performance',
                height=600,
                legend_title='Legend',
                title_x=0.5,
                showlegend=True,
            )
            plots.update_xaxes(title_text='Financial Year', row=1, col=1)
            plots.update_xaxes(title_text='Financial Year', row=1, col=2)
            plots.update_yaxes(title_text='Amount (INR Crores)', row=1, col=1)
            year_revenue = plots.to_html(full_html = False)

            # Quaterly Revenue
            quat_income_stmt = stock.quarterly_income_stmt
            quat_total_revenue = quat_income_stmt.loc['Total Revenue']
            df_total_rev = quat_total_revenue.reset_index()
            df_total_rev = df_total_rev.head(5)
            df_total_rev.columns = ['Date', 'Total Revenue']
            df_total_rev['Date'] = pd.to_datetime(df_total_rev['Date'])
            df_total_rev = df_total_rev.sort_values(by='Date')
            df_total_rev['Total Revenue (INR Crores)'] = df_total_rev['Total Revenue'] / 1e7
            df_total_rev['Quarter'] = df_total_rev['Date'].dt.to_period('Q')

            quat_net_income = income_stmt.loc['Net Income']
            df_net_inc = quat_net_income.reset_index()
            df_net_inc = df_net_inc.head(5)
            df_net_inc.columns = ['Date', 'Net Income']
            df_net_inc['Date'] = pd.to_datetime(df_total_rev['Date'])
            df_net_inc = df_net_inc.sort_values(by='Date')
            df_net_inc['Net Income (INR Crores)'] = df_net_inc['Net Income'] / 1e7
            df_net_inc['Quarter'] = df_net_inc['Date'].dt.to_period('Q')
            plots1 = make_subplots(rows=1, cols=2, shared_yaxes=True,
                                subplot_titles=[f'Total Revenue ({cname})', f'Net Income ({cname})'])
            plots1.add_trace(go.Bar(x=df_total_rev['Quarter'].astype(str), y=df_total_rev['Total Revenue (INR Crores)'],
                                 marker_color='#2ca02c', name='Total Revenue (INR Crores)'), row=1, col=1)
            plots1.add_trace(go.Bar(x=df_net_inc['Quarter'].astype(str), y=df_net_inc['Net Income (INR Crores)'],
                                 marker_color='skyblue', name='Net Income (INR Crores)'), row=1, col=2)
            plots1.update_layout(
                title=f'{cname} Quarterly Financial Performance',
                height=600,
                legend_title='Legend',
                title_x=0.5,
                showlegend=True,
            )
            plots1.update_xaxes(title_text='Quarter', row=1, col=1)
            plots1.update_xaxes(title_text='Quarter', row=1, col=2)
            plots1.update_yaxes(title_text='Amount (INR Crores)', row=1, col=1)

            quat_plot = plots1.to_html(full_html=False)

            # Context to send
            context = {
                'company_name': cname[:-3],
                'website': stock.info.get('website')[8:],
                'Revenue': financials.loc['Total Revenue'][0],
                'Net_Income': financials.loc['Net Income'][0],
                'Gross_Profit': gross_profit,
                'year_revenue': year_revenue,
                'quat_plot': quat_plot
            }
            return render(request, 'majorApp/financials.html', context)

        # Fundamentals
        elif action == 'fundamentals':
            cname += ".NS"
            stock = yf.Ticker(cname)
            info = stock.info
            balance_sheet = stock.balance_sheet
            financials = stock.financials

            # earningPerShare
            if not financials.empty:
                net_income = financials.loc['Net Income'][0] if 'Net Income' in financials.index else None
                shares_outstanding = stock.info.get('sharesOutstanding')
                if all([net_income, shares_outstanding]):
                    earningPerShare = net_income / shares_outstanding
                else:
                    earningPerShare = 0

            # Debt To Equity
            if not financials.empty:
                total_debt = info.get('totalDebt')
                total_equity = balance_sheet.loc["Stockholders Equity"].iloc[0]
                if all([total_debt, total_equity]):
                    debtyToEquity = total_debt / total_equity
                else:
                    debtyToEquity = 0
            # Profit Margin
            if not financials.empty:
                net_income = financials.loc['Net Income'][0] if 'Net Income' in financials.index else None
                total_revenue = financials.loc['Total Revenue'][
                    0] if 'Total Revenue' in financials.index else None

                if all([net_income, total_revenue]):
                    profit_margin = (net_income / total_revenue) * 100
                else:
                    profit_margin = 0
            else:
                profit_margin = 0

            # Operating Margin
            if not financials.empty:
                operating_income = financials.loc['Operating Income'][
                    0] if 'Operating Income' in financials.index else None
                total_revenue = financials.loc['Total Revenue'][
                    0] if 'Total Revenue' in financials.index else None

                if all([operating_income, total_revenue]):
                    operating_margin = (operating_income / total_revenue) * 100
                else:
                    operating_margin = 0
            else:
                operating_margin = 0

            # Context Data
            context = {
                'company_name': cname[:-3],
                'website': stock.info.get('website')[8:],
                'market_cap': info.get('marketCap'),
                'forward_pe': info.get('forwardPE'),
                'price_sales': info.get('priceToSalesTrailing12Months'),
                'price_book': info.get('priceToBook'),
                'profit_margin': info.get('profitMargins')*100,
                'operating_margin': info.get('operatingMargins')*100,
                'return_on_equity': info.get('returnOnEquity')*100,
                'return_on_assets': info.get('returnOnAssets')*100,
                'total_debt': info.get('totalDebt'),
                'debtToEquity': debtyToEquity,
                'EPS': earningPerShare,
                'Dividend_Yield': stock.info.get('dividendYield'),
                'P_FCF': stock.info.get('priceToFreeCashFlows'),
                'EV_EBITDA': stock.info.get('enterpriseToEbitda'),
                'EV_Revenue': stock.info.get('enterpriseToRevenue'),
            }
            return render(request, 'majorApp/fundamentals.html', context)

        # Projections
        elif action == 'projections':
            cname += ".NS"
            stock = yf.Ticker(cname)
            info = stock.info
            return redirect('majorApp:candlestick_chart')

        # Home
        elif action == 'home':
            return redirect('majorApp:searchPage')

        # Alert Message
        else:
            alert_message = True
            return render(request, 'majorApp/searchPage.html', {'alert_message': alert_message})
    return render(request, 'majorApp/navigation.html')


def get_summary(cname):
    name = yf.Ticker(cname)
    return name.info.get('longBusinessSummary')


def get_officers_info(cname):
    name = yf.Ticker(cname)
    officers = name.info.get('companyOfficers', [])
    officers_info = [{'name': officer.get('name'), 'title': officer.get('title')} for officer in officers]
    print(officers_info)
    first_officer_info = officers_info[0]
    return first_officer_info


# views.py
def candlestick_chart(request):
    # Fetch historical data for a specific stock, e.g., Tata Consultancy Services Limited (TCS)
    ticker = request.session.get('cname') + ".NS"

    stock = yf.Ticker(ticker)
    info = stock.info
    # Calculate the date 30 days ago from today
    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.Timedelta(days=60)

    # Fetch data for graph1
    data = yf.download(ticker, start=start_date, end=end_date)

    if data.empty:
        raise ValueError("Empty DataFrame returned by yfinance.")

    # Ensure the index is a DatetimeIndex
    data.index = pd.to_datetime(data.index)

    # Define the style with blue and red colors for the candles
    my_style = mpf.make_mpf_style(base_mpf_style='charles', rc={'figure.figsize': (10, 6)})

    # Custom market colors
    market_colors = mpf.make_marketcolors(up='blue', down='red', wick='i', edge='i', volume='in')

    # Apply the custom market colors to the style
    my_style.update(marketcolors=market_colors)

    # Create a Matplotlib figure
    fig, _ = mpf.plot(data, type='candle', style=my_style, ylabel='Price',
                      returnfig=True)

    # Render the figure as an image
    buffer = io.BytesIO()
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buffer)
    image_data = buffer.getvalue()

    # Extract company info
    stock = yf.Ticker(ticker)
    company_name = stock.info.get('longName', 'Unknown Company')
    website = stock.info.get('website', '')

    current_price = stock.history(period='1d')['Close'].iloc[-1]

    # Get yesterday's closing price
    yesterday_price = stock.history(period='2d')['Close'].iloc[0]

    # Get 52-week high and low
    high_52_weeks = stock.info['fiftyTwoWeekHigh']
    low_52_weeks = stock.info['fiftyTwoWeekLow']

    # Get open, high, low, and close prices for today
    today_data = stock.history(period='1d')
    open_price = today_data['Open'].iloc[0]
    high_price = today_data['High'].iloc[0]
    low_price = today_data['Low'].iloc[0]
    close_price = today_data['Close'].iloc[0]

    # Creating a Candlestick Plotly figure
    fig1 = go.Figure()
    fig1.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        increasing_line_color='blue',
        decreasing_line_color='red',
        name='Candlestick'
    ))
    fig1.update_layout(
        title=f'{ticker[:-3]} Candlestick Chart of 3 months',
        xaxis_title='Dates',
        yaxis_title='Prices',
        xaxis_rangeslider_visible=False,
        height=500

    )
    plotly_image1 = fig1.to_html(full_html=False)

    # Creating a line chart
    fig2 = go.Figure()
    line_trace = go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Line Chart'
    )
    fig2.add_trace(line_trace)
    fig2.update_layout(
        title=f'{ticker[:-3]} Line Chart for 3months',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        height=500
    )
    plotly_image2 = fig2.to_html(full_html=False)

    # Prediction Model Building
    df = stock.history(period="3mo")
    df = df.reset_index()
    dates, close_prices, scaled_close_prices, scaler = fetch_and_preprocess(ticker)
    sequence_length = 5
    X = create_sequences(scaled_close_prices, sequence_length)
    split_ratio = 0.8
    split_index = int(len(X) * split_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y = scaled_close_prices[sequence_length:]
    y_train, y_test = y[:split_index], y[split_index:]
    model = build_model(X_train, y_train, sequence_length)
    # loss = model.evaluate(X_test, y_test, verbose=0)
    predicted_prices = predict_stock_prices(model, scaled_close_prices, scaler, sequence_length)
    dates_predicted = pd.date_range(start=df['Date'].iloc[-1], periods=6)[1:]
    extended_dates = df['Date'].tolist() + [dates_predicted[0]]
    extended_prices = close_prices.tolist() + [predicted_prices.flatten()[0]]
    fig_pro = go.Figure()
    fig_pro.add_trace(go.Scatter(x=extended_dates, y=extended_prices, mode='lines', name='Original Data'))
    fig_pro.add_trace(go.Scatter(x=dates_predicted, y=predicted_prices.flatten(), mode='lines', name='Projected Data'))

    # Update layout
    fig_pro.update_layout(title=f'{ticker[:-3]} Stock Prices with Projections for 5days',
                          xaxis_title='Date',
                          yaxis_title='Stock Price (Rs)',
                          legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                          showlegend=True)

    # Convert Plotly figure to HTML for rendering in Django template
    project = fig_pro.to_html(full_html=False)

    # Context for template
    context = {
        'company_name': company_name,
        'website': website[8:],  # Removing 'https://' from the beginning
        'current_price': current_price,
        'yesterday_price': yesterday_price,
        '52_week_high': high_52_weeks,
        '52_week_low': low_52_weeks,
        'open': open_price,
        'high': high_price,
        'low': low_price,
        'close': close_price,
        'image_data': image_data,
        'plotly_image1': plotly_image1,
        'plotly_image2': plotly_image2,
        'project': project
    }
    return render(request, 'majorApp/projections.html', context=context)
