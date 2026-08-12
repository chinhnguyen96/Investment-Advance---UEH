import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import yfinance as yf ### new yfinance

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ==============================================================================
# Summary
# ==============================================================================


@st.cache_data
def getsummary(ticker):

    stock = yf.Ticker(ticker)
    info = stock.info

    summary = pd.DataFrame({
        "attribute": [
            "Current Price",
            "Previous Close",
            "Open",
            "Day High",
            "Day Low",
            "52 Week High",
            "52 Week Low",
            "Volume",
            "Market Cap",
            "PE Ratio"
        ],
        "value": [
            info.get("currentPrice"),
            info.get("previousClose"),
            info.get("open"),
            info.get("dayHigh"),
            info.get("dayLow"),
            info.get("fiftyTwoWeekHigh"),
            info.get("fiftyTwoWeekLow"),
            info.get("volume"),
            info.get("marketCap"),
            info.get("trailingPE")
        ]
    })

    return summary

### Get data yfinance
@st.cache_data
def getstockdata(ticker):

    df = yf.download(
        ticker,
        period="max",
        progress=False,
        auto_adjust=False
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df


# ==============================================================================
# TAB 1 - SUMMARY
# ==============================================================================

def tab1():

    st.title("📈 Stock Summary Dashboard")

    # =========================
    # Use ticker from Sidebar
    # =========================

    if ticker == "-":
        st.info("👈 Please select a ticker from the sidebar")
        return

    stock = yf.Ticker(ticker)

    try:
        info = stock.info
    except:
        info = {}

    company_name = info.get("longName", ticker)

    # =========================
    # Company Name
    # =========================

    st.subheader(f"🏢 {company_name}")
    st.caption(f"Ticker: {ticker}")

    # =========================
    # Report Date
    # =========================
    
    st.write("### 📅 Report Date")
    
    report_date = st.date_input(
        "Select report date",
        value=datetime.today().date(),
        max_value=datetime.today().date(),
        label_visibility="collapsed"
    )
    
    st.caption(
        f"Selected date: {report_date.strftime('%d/%m/%Y')}"
    )
    
    # =========================
    # Historical Data by Report Date
    # =========================
    
    price_data = yf.download(
        ticker,
        start=report_date - timedelta(days=10),
        end=report_date + timedelta(days=1),
        auto_adjust=False,
        progress=False
    )
    
    # Fix MultiIndex columns from yfinance
    if isinstance(price_data.columns, pd.MultiIndex):
        price_data.columns = price_data.columns.get_level_values(0)
    
    # Only keep trading data up to selected report date
    price_data = price_data[
        price_data.index.date <= report_date
    ]
    
    if not price_data.empty:
    
        # Latest available trading session on or before report_date
        current = float(price_data["Close"].iloc[-1])
        volume = int(price_data["Volume"].iloc[-1])
    
        # Previous trading session
        if len(price_data) >= 2:
            previous = float(price_data["Close"].iloc[-2])
        else:
            previous = current
    
        # Daily change
        change = current - previous
    
        if previous != 0:
            change_pct = change / previous * 100
        else:
            change_pct = 0
    
    else:
        st.warning("No trading data available for the selected date.")
        return
    
    
    # =========================
    # Current Fundamental Data
    # =========================
    
    market_cap = info.get("marketCap", 0) or 0
    week_high = info.get("fiftyTwoWeekHigh", 0) or 0
    week_low = info.get("fiftyTwoWeekLow", 0) or 0
    pe_ratio = info.get("trailingPE", 0) or 0
    eps = info.get("trailingEps", 0) or 0
    
    
    # =========================
    # KPI Cards
    # =========================
    
    c1, c2, c3, c4 = st.columns(4)
    
    c1.metric(
        "Close Price",
        f"${current:,.2f}",
        help="Closing price for the selected report date (USD)"
    )
    
    c2.metric(
        "Daily Change",
        f"${change:+,.2f}",
        f"{change_pct:+.2f}%",
        help="Change from the previous trading session"
    )
    
    c3.metric(
        "Market Cap",
        f"${market_cap / 1_000_000_000:.2f}B",
        help="Current Market Capitalization in Billion USD"
    )
    
    c4.metric(
        "Trading Volume",
        f"{volume:,.0f}",
        help="Number of shares traded on the selected report date"
    )
    
    
    c5, c6, c7, c8 = st.columns(4)
    
    c5.metric(
        "52-Week High",
        f"${week_high:,.2f}",
        help="Current 52-week highest price (USD)"
    )
    
    c6.metric(
        "52-Week Low",
        f"${week_low:,.2f}",
        help="Current 52-week lowest price (USD)"
    )
    
    c7.metric(
        "P/E Ratio",
        f"{pe_ratio:,.2f}x",
        help="Current Price-to-Earnings Ratio"
    )
    
    c8.metric(
        "Earnings Per Share",
        f"${eps:,.2f}",
        help="Current EPS (USD per share)"
    )
    

    # =========================
    # Company Profile
    # =========================

    st.subheader("🏢 Company Profile")

    profile = pd.DataFrame({
        "Information": [
            "Ticker",
            "Company",
            "Sector",
            "Industry",
            "Country",
            "Website"
        ],
        "Value": [
            ticker,
            info.get("longName", "N/A"),
            info.get("sector", "N/A"),
            info.get("industry", "N/A"),
            info.get("country", "N/A"),
            info.get("website", "N/A")
        ]
    })

    st.table(profile)

    # =========================
    # Price Chart
    # =========================

    df = yf.download(
        ticker,
        period="5y",
        auto_adjust=False,
        progress=False
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if not df.empty:

        fig = px.area(
            df,
            x=df.index,
            y="Close",
            title=f"{company_name} ({ticker}) - Historical Closing Price"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )
    else:
        st.warning("No historical price data available.")        
        
    
#==============================================================================
# Tab 2 Chart
#==============================================================================


#The code below divides the streamlit page into 5 columns. The first two columns
#have a date picker option to select start and end dates and the the other three
#have dropdown selection boxes for duration, interval, and type of plot.

def tab2():

    st.title("📊 Stock Price Chart")

    # =========================
    # Check ticker
    # =========================

    if ticker == "-":
        st.info("👈 Please select a ticker from the sidebar")
        return

    # Company name
    try:
        info = yf.Ticker(ticker).info
        company_name = info.get("longName", ticker)
    except:
        company_name = ticker

    st.subheader(f"🏢 {company_name}")
    st.caption(f"Ticker: {ticker}")

    st.divider()

    # =========================
    # FILTERS
    # =========================

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        start_date = st.date_input(
            "Start Date",
            datetime.today().date() - timedelta(days=365)
        )

    with c2:
        end_date = st.date_input(
            "End Date",
            datetime.today().date()
        )

    with c3:
        duration = st.selectbox(
            "Duration",
            ["Custom", "1mo", "3mo", "6mo", "1y", "3y", "5y", "max"],
            index=4
        )

    with c4:
        interval = st.selectbox(
            "Interval",
            ["1d", "1wk", "1mo"]
        )

    with c5:
        chart_type = st.selectbox(
            "Chart Type",
            ["Line", "Candlestick"]
        )

    # =========================
    # MOVING AVERAGE OPTIONS
    # =========================

    st.write("#### 📈 Technical Overlays")

    ma1, ma2, ma3 = st.columns(3)

    with ma1:
        show_sma20 = st.checkbox("SMA 20", value=True)

    with ma2:
        show_sma50 = st.checkbox("SMA 50", value=True)

    with ma3:
        show_sma200 = st.checkbox("SMA 200", value=False)

    # =========================
    # DOWNLOAD DATA
    # =========================

    try:

        if duration == "Custom":

            df = yf.download(
                ticker,
                start=start_date,
                end=end_date + timedelta(days=1),
                interval=interval,
                auto_adjust=False,
                progress=False
            )

        else:

            df = yf.download(
                ticker,
                period=duration,
                interval=interval,
                auto_adjust=False,
                progress=False
            )

        # Fix MultiIndex from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df.empty:
            st.warning("No stock data available for the selected period.")
            return

        # =========================
        # MOVING AVERAGES
        # =========================

        df["SMA20"] = df["Close"].rolling(20).mean()
        df["SMA50"] = df["Close"].rolling(50).mean()
        df["SMA200"] = df["Close"].rolling(200).mean()

        # =========================
        # PRICE CHART
        # =========================

        st.subheader("📈 Price Movement")

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.75, 0.25]
        )

        # -------------------------
        # Line / Candlestick
        # -------------------------

        if chart_type == "Line":

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["Close"],
                    mode="lines",
                    name="Close Price"
                ),
                row=1,
                col=1
            )

        else:

            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df["Open"],
                    high=df["High"],
                    low=df["Low"],
                    close=df["Close"],
                    name=ticker
                ),
                row=1,
                col=1
            )

        # =========================
        # SMA
        # =========================

        if show_sma20:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["SMA20"],
                    mode="lines",
                    name="SMA 20"
                ),
                row=1,
                col=1
            )

        if show_sma50:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["SMA50"],
                    mode="lines",
                    name="SMA 50"
                ),
                row=1,
                col=1
            )

        if show_sma200:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["SMA200"],
                    mode="lines",
                    name="SMA 200"
                ),
                row=1,
                col=1
            )

        # =========================
        # VOLUME
        # =========================

        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                name="Volume"
            ),
            row=2,
            col=1
        )

        # =========================
        # CHART SETTINGS
        # =========================

        fig.update_layout(
            height=700,
            title=f"{company_name} ({ticker})",
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            legend=dict(
                orientation="h",
                y=1.02,
                x=0
            )
        )

        fig.update_yaxes(
            title_text="Price (USD)",
            row=1,
            col=1
        )

        fig.update_yaxes(
            title_text="Volume",
            row=2,
            col=1
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # =========================
        # PRICE STATISTICS
        # =========================

        st.subheader("📊 Period Statistics")

        latest_close = float(df["Close"].iloc[-1])

        period_high = float(df["High"].max())
        period_low = float(df["Low"].min())

        avg_volume = float(df["Volume"].mean())

        s1, s2, s3, s4 = st.columns(4)

        s1.metric(
            "Latest Close",
            f"${latest_close:,.2f}"
        )

        s2.metric(
            "Period High",
            f"${period_high:,.2f}"
        )

        s3.metric(
            "Period Low",
            f"${period_low:,.2f}"
        )

        s4.metric(
            "Avg Volume",
            f"{avg_volume:,.0f}"
        )

    except Exception as e:

        st.error(
            f"Unable to load chart data: {e}"
        )
           
             

#==============================================================================
# Tab 3 Statistics
#==============================================================================

#The code below obtains information using get_stats_valuation and get_stats in
#Yahoo Finance. It then slices the dataframes and displays them in different 
#columns of the streamlit page under different headings.

def tab3():
     st.title("Statistics")
     st.write(ticker)
     c1, c2 = st.columns(2)
     
         
     
     with c1:
         st.header("Valuation Measures")
         #@st.cache
         def getvaluation(ticker):
                 return si.get_stats_valuation(ticker)
    
         if ticker != '-':
                valuation = getvaluation(ticker)
                valuation[1] = valuation[1].astype(str)
                valuation = valuation.rename(columns = {0: 'Attribute', 1: ''})
                valuation.set_index('Attribute', inplace=True)
                st.table(valuation)
                
        
         st.header("Financial Highlights")
         st.subheader("Fiscal Year")
         
         #@st.cache
         def getstats(ticker):
                 return si.get_stats(ticker)
         
         if ticker != '-':
                stats = getstats(ticker)
                stats['Value'] = stats['Value'].astype(str)
                stats.set_index('Attribute', inplace=True)
                st.table(stats.iloc[29:31,])
                
        
         st.subheader("Profitability")
         
         if ticker != '-':
                stats = getstats(ticker)
                stats['Value'] = stats['Value'].astype(str)
                stats.set_index('Attribute', inplace=True)
                st.table(stats.iloc[31:33,])
                
                
                
         st.subheader("Management Effectiveness")
         
         if ticker != '-':
                stats = getstats(ticker)
                stats['Value'] = stats['Value'].astype(str)
                stats.set_index('Attribute', inplace=True)
                st.table(stats.iloc[33:35,])
         
         
                
         st.subheader("Income Statement")
         
         if ticker != '-':
                stats = getstats(ticker)
                stats['Value'] = stats['Value'].astype(str)
                stats.set_index('Attribute', inplace=True)
                st.table(stats.iloc[35:43,])  
            
         
         st.subheader("Balance Sheet")
         
         if ticker != '-':
                stats = getstats(ticker)
                stats['Value'] = stats['Value'].astype(str)
                stats.set_index('Attribute', inplace=True)
                st.table(stats.iloc[43:49,])
         
         st.subheader("Cash Flow Statement")
         
         if ticker != '-':
                stats = getstats(ticker)
                stats['Value'] = stats['Value'].astype(str)
                stats.set_index('Attribute', inplace=True)
                st.table(stats.iloc[49:,])
         
        
                           
     with c2:
         st.header("Trading Information")
         
         
         st.subheader("Stock Price History")
                  
         if ticker != '-':
                stats = getstats(ticker)
                stats['Value'] = stats['Value'].astype(str)
                stats.set_index('Attribute', inplace=True)
                st.table(stats.iloc[:7,])
         
         st.subheader("Share Statistics")
                  
         if ticker != '-':
                stats = getstats(ticker)
                stats['Value'] = stats['Value'].astype(str)
                stats.set_index('Attribute', inplace=True)
                st.table(stats.iloc[7:19,])
         
         st.subheader("Dividends & Splits")
                  
         if ticker != '-':
                stats = getstats(ticker)
                stats['Value'] = stats['Value'].astype(str)
                stats.set_index('Attribute', inplace=True)
                st.table(stats.iloc[19:29,])
         
         
         
            
     

#==============================================================================
# Tab 4 Financials
#==============================================================================

#The code below obtains yearly and quartely financial statements from Yahoo Finance
#and displays them according the options selected by the users in streamlit. A
#combination of if statements is used to display according to the selected options.


def tab4():
      st.title("Financials")
      st.write(ticker)
      
      statement = st.selectbox("Show", ['Income Statement', 'Balance Sheet', 'Cash Flow'])
      period = st.selectbox("Period", ['Yearly', 'Quarterly'])
      
      @st.cache
      def getyearlyincomestatement(ticker):
            return si.get_income_statement(ticker)
      
      @st.cache
      def getquarterlyincomestatement(ticker):
            return si.get_income_statement(ticker, yearly = False)
      
      @st.cache
      def getyearlybalancesheet(ticker):
            return si.get_balance_sheet(ticker)
      
      @st.cache
      def getquarterlybalancesheet(ticker):
            return si.get_balance_sheet(ticker, yearly = False)      

      @st.cache
      def getyearlycashflow(ticker):
            return si.get_cash_flow(ticker)
      
      @st.cache
      def getquarterlycashflow(ticker):
            return si.get_cash_flow(ticker, yearly = False)
        
          
      if ticker != '-' and statement == 'Income Statement' and period == 'Yearly':
                data = getyearlyincomestatement(ticker)
                st.table(data)
            
      if ticker != '-' and statement == 'Income Statement' and period == 'Quarterly':
                data = getquarterlyincomestatement(ticker)
                st.table(data)            

      if ticker != '-' and statement == 'Balance Sheet' and period == 'Yearly':
                data = getyearlybalancesheet(ticker)
                st.table(data)            
      
      if ticker != '-' and statement == 'Balance Sheet' and period == 'Quarterly':
                data = getquarterlybalancesheet(ticker)
                st.table(data)        
      
      if ticker != '-' and statement == 'Cash Flow' and period == 'Yearly':
                data = getyearlycashflow(ticker)
                st.table(data)        
      
        
      if ticker != '-' and statement == 'Cash Flow' and period == 'Quarterly':
                data = getquarterlycashflow(ticker)
                st.table(data)      
                
                 
        
      
        
      
#==============================================================================
# Tab 5 Analysis
#==============================================================================

#In the code below, get_analysts_info is used to obtain the data. The output is
#in the form of a dictionary. .items() is used to get the items from the dictionary
#and then a for loop i used under which the dictionary items are changed into a list
# and each element of the list is then converted to a dataframe for displaying.


def tab5():
      st.title("Analysis")
      st.write("Currency in USD")
      st.write(ticker)
      
      @st.cache
      def getanalysis(ticker):
            analysis_dict = si.get_analysts_info(ticker)
            return analysis_dict.items()
 
           
      if ticker != '-':           
           for i in range(6):
            analysis = getanalysis(ticker)
            df = pd.DataFrame(list(analysis)[i][1])
            st.table(df)
            
           
#==============================================================================
# Tab 6 Monte Carlo Simulation
#==============================================================================

#The code below performs and displays the monte carlo simulation for a specified
#time horizon and number of intervals



def tab6():
     st.title("Monte Carlo Simulation")
     st.write(ticker)
     
     #Dropdown for selecting simulation and horizon
     simulations = st.selectbox("Number of Simulations (n)", [200, 500, 1000])
     time_horizon = st.selectbox("Time Horizon (t)", [30, 60, 90])
     
     #The code below takes past 30 day data using get_data. Then it gets the close
     #price column and uses .pct_change() to get the daily return. Daily volatility 
     #is then calculated as the standard deviation of the daily return.
     @st.cache
     def montecarlo(ticker, time_horizon, simulations):
     
         end_date = datetime.now().date()
         start_date = end_date - timedelta(days=30)
     
         stock_price = si.get_data(ticker, start_date, end_date)
         close_price = stock_price['close']
     
     
         daily_return = close_price.pct_change()
         daily_volatility = np.std(daily_return)
     
         #Initialize the simulation dataframe    
         simulation_df = pd.DataFrame()
     
         for i in range(simulations):        
                      
                # The list to store the next stock price
                next_price = []
    
    #    Create the next stock price
                last_price = close_price[-1]
    
                for x in range(time_horizon):
                               
                      # Generate the random percentage change around the mean (0) and std (daily_volatility)
                      future_return = np.random.normal(0, daily_volatility)

            # Generate the random future price
                      future_price = last_price * (1 + future_return)

            # Save the price and go next
                      next_price.append(future_price)
                      last_price = future_price
    
    #    Store the result of the simulation
                simulation_df[i] = next_price
                
         return simulation_df   
          
#The code below plots the monte carlo simulation using maplotlib. It also calculates
#variance at risk and displays it. the VAR is calculated using the last row of
#the montecarlo simulation. the distribution of this ending price is displaued and
#the 5th percentile of the distribution is marked


     if ticker != '-':
         mc = montecarlo(ticker, time_horizon, simulations)
                  
         end_date = datetime.now().date()
         start_date = end_date - timedelta(days=30)
         
         stock_price = si.get_data(ticker, start_date, end_date)
         close_price = stock_price['close']
         
         fig, ax = plt.subplots(figsize=(15, 10))
         

         ax.plot(mc)
         plt.title('Monte Carlo simulation for ' + str(ticker) + ' stock price in next ' + str(time_horizon) + ' days')
         plt.xlabel('Day')
         plt.ylabel('Price')
         
         
         plt.axhline(y= close_price[-1], color ='red')
         plt.legend(['Current stock price is: ' + str(np.round(close_price[-1], 2))])
         ax.get_legend().legendHandles[0].set_color('red')

         st.pyplot(fig)
         
         # Value at Risk
         st.subheader('Value at Risk (VaR)')
         ending_price = mc.iloc[-1:, :].values[0, ]
         fig1, ax = plt.subplots(figsize=(15, 10))
         ax.hist(ending_price, bins=50)
         plt.axvline(np.percentile(ending_price, 5), color='red', linestyle='--', linewidth=1)
         plt.legend(['5th Percentile of the Future Price: ' + str(np.round(np.percentile(ending_price, 5), 2))])
         plt.title('Distribution of the Ending Price')
         plt.xlabel('Price')
         plt.ylabel('Frequency')
         st.pyplot(fig1)
         
         
         future_price_95ci = np.percentile(ending_price, 5)
         # Value at Risk
         VaR = close_price[-1] - future_price_95ci
         st.write('VaR at 95% confidence interval is: ' + str(np.round(VaR, 2)) + ' USD')
         
         
     
  
#==============================================================================
# Tab 7 Your Portfolio's Trend
#==============================================================================

#The code below uses a multiselect box to allow user to select multiple tickers.
#Then a new dataframe is created with each ticker as a column. A for loop is used to
#populate each column with the close price of that ticker. Then plotly is used to 
#visualize the trend of the selected portfolio
#Reference:
#https://blog.quantinsti.com/stock-market-data-analysis-python/


def tab7():
      st.title("Your Portfolio's Trend")
      alltickers = si.tickers_sp500()
      selected_tickers = st.multiselect("Select tickers in your portfolio", options = alltickers, default = ['AAPL'])
      
      
      df = pd.DataFrame(columns=selected_tickers)
      for ticker in selected_tickers:
          df[ticker] = yf.download(ticker, period = '5Y')['Close']
                
               
      fig = px.line(df)
      st.plotly_chart(fig) 
      
        
    
    
    
#==============================================================================
# Main body
#==============================================================================

def run():

    # Danh sách ticker mẫu
    ticker_list = [
        '-',
        'AAPL',
        'MSFT',
        'GOOG',
        'META',
        'AMZN',
        'NVDA',
        'TSLA',
        'NFLX',
        'AMD',
        'BTC-USD',
        'ETH-USD',
        '^GSPC',
        '^IXIC'
    ]

    global ticker
    ticker = st.sidebar.selectbox("Select a ticker", ticker_list)
    
    # Add a radio box
    select_tab = st.sidebar.radio("Select tab", ['Summary', 'Chart', 'Statistics', 'Financials', 'Analysis', 'Monte Carlo Simulation', "Your Portfolio's Trend"])
    
    # Show the selected tab
    if select_tab == 'Summary':
        tab1()
    elif select_tab == 'Chart':
        tab2()
    elif select_tab == 'Statistics':
        tab3()
    elif select_tab == 'Financials':
        tab4()
    elif select_tab == 'Analysis':
        tab5()
    elif select_tab == 'Monte Carlo Simulation':
        tab6()
    elif select_tab == "Your Portfolio's Trend":
        tab7()
       
    
if __name__ == "__main__":
    run()    
