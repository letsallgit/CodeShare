import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from utilsV2 import load_dataframe
from utilsV2 import process_data, display_kpis,create_visuals, calculate_metrics, create_bar_chart, create_line_chart, user_created_visuals, calculate_statistics, filter_by_year, filter_by_month, filter_by_department, filter_by_vendor_type
import calendar
import time
from st_aggrid import AgGrid, GridOptionsBuilder


# Page configuration
st.set_page_config(page_title='Purchasing Dashboard', layout='wide')

st.sidebar.header('Navigation')
page = st.sidebar.radio("Choose a page", ('Home', 'Data Table', 'Visuals', 'User'))

@st.cache_data
def load_dataframe():

    df=pd.read_csv('/Users/otb/Desktop/Purchasing Project/Working Version/CsvSanitizedPurchasesInvoices22-24.csv')
    return df


with st.spinner('Loading... Please wait.'):
    time.sleep(2)  # Simulate loading time


########################### Home #############################


if page == 'Home':

    df=pd.read_csv('/Users/otb/Desktop/Purchasing Project/Working Version/CsvSanitizedPurchasesInvoices22-24.csv')

        
    load_dataframe()
    process_data(df)
 
    
    kpi_expander = st.expander("Key Metrics & Indicators", expanded=True)



   ######################################################################################################         
    import plotly.graph_objects as go

    def display_kpis_plotly(metrics):
        fig = go.Figure()

        for i, (key, value) in enumerate(metrics.items()):
            fig.add_trace(go.Indicator(
                mode = "number",
                value = value,
                title = {'text': key},
                domain = {'row': 0, 'column': i}))

        fig.update_layout(
            grid = {'rows': 1, 'columns': len(metrics), 'pattern': "independent"},
            template="plotly_white"
        )

        fig.show()
    
    
    # checkbox = st.checkbox("show side by side comparison", True, False)

    # if checkbox == True:
    #     df_2022 = df[df['2022'] == 2022]  # replace with your actual condition
    #     metrics_2022 = calculate_metrics(df_2022)
    #     col1 = st.columns(len(metrics_2022))

    #     with col1:
    #         st.markdown("<h1 style='text-align: center; color: black;'>FY 2022</h1>", unsafe_allow_html=True)
    #         display_kpis(metrics_2022)

    #     df_2023 = df[df['Year'] == 2023]  # replace with your actual condition
    #     metrics_2023 = calculate_metrics(df_2023)
    #     col2 = st.columns(len(metrics_2023))

    #     with col2:
    #         st.markdown("<h1 style='text-align: center; color: red;'>FY 2023</h1>", unsafe_allow_html=True)
    #         display_kpis(metrics_2023)

    # else: 
    #     df_2022 = df[df['Fiscal Year'] == 2022]  # replace with your actual condition
    #     metrics_2022 = calculate_metrics(df_2022)
    #     display_kpis(metrics_2022)

    #     df_2023 = df[df['Fiscal Year'] == 2023]  # replace with your actual condition
    #     metrics_2023 = calculate_metrics(df_2023)
    #     display_kpis(metrics_2023)
        
    ################ slider 
        


########################### User Geerated Charts

elif page == 'User':
    df = load_dataframe()  # Load your DataFrame
    processed_df = process_data(df)
    filtered_df = processed_df.copy()
    user_created_visuals(df)

    #Convert dates to datetime objects for the slider
    def date_slider():

        df['Invoice Date'] = pd.to_datetime(df['Invoice Date'])
        date_options = df['Invoice Date'].dt.date.sort_values().unique()

        # Select slider for date range
        st.subheader("Date Range")
        min_date, max_date = date_options[0], date_options[-1]
        selected_date_range = st.sidebar.slider("Date Range", 
                                                min_value=min_date, 
                                                max_value=max_date, 
                                                value=(min_date, max_date))

        # Filter the dataframe based on the selected date range
        df = df[(df['Invoice Date'].dt.date >= selected_date_range[0]) & (df['Invoice Date'].dt.date <= selected_date_range[1])]


    filtered_df = process_data(df)  # Define the filtered_df variable
    apply_filters_to_charts = st.checkbox("Apply filters to charts")

    if apply_filters_to_charts == True:
        create_visuals(filtered_df)
        st.write("Filtered View")
        
    else:
        create_visuals(filtered_df)


########################### Data Table #############################
    
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def user_created_visuals(df):
    # Dropdown for selecting the chart type
    chart_type = st.selectbox("Select Chart Type", ['Bar Chart', 'Pie Chart', 'Line Chart'])

    # Dropdown for selecting the data category (assuming 'Vendor Name' and 'Invoice Amount' are columns in your DataFrame)
    category = st.selectbox("Select Category for Analysis", ['Vendor Name', 'Invoice Amount', 'Vendor Type'])

    # Number input for how many top items to display
    top_n = st.number_input("Select Number of Top Items to Display", min_value=1, max_value=20, value=10)

    # Generate chart based on selections
    if st.button('Generate Chart'):
        if category == 'Vendor Name':
            data = df.groupby('Vendor Name')['Invoice Amount'].sum().nlargest(top_n)
        elif category == 'Vendor Type':
            data = df.groupby('Vendor Type')['Invoice Amount'].sum().nlargest(top_n) 
        else:
            # Implement other categories as needed
            data = None

        # Plotting
        if data is not None:
            plt.figure(figsize=(10, 6))
            if chart_type == 'Bar Chart':
                sns.barplot(x=data.values, y=data.index, palette="viridis")
                plt.xlabel('Total Expenditure', fontsize=14)
                plt.ylabel('Vendor Name', fontsize=14)
                plt.title('Bar Chart', fontsize=20)
            elif chart_type == 'Pie Chart':
                plt.pie(data, labels=data.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("viridis", len(data)))
                plt.title('Pie Chart', fontsize=20)
            elif chart_type == 'Line Chart':
                sns.lineplot(x=data.index, y=data.values, palette="viridis")
                plt.xlabel('Vendor Name', fontsize=14)
                plt.ylabel('Total Expenditure', fontsize=14)
                plt.title('Line Chart', fontsize=20)

            # Display the chart
            st.pyplot(plt)

    #interactive time-series
    # Sidebar widgets to select the date range
    # Convert Invoice Date to datetime
    df['Invoice Date'] = pd.to_datetime(df['Invoice Date'])

    # Convert dates to strings for the slider
    df['Invoice Date string'] = df['Invoice Date'].dt.strftime('%Y-%m-%d')
    date_options = df['Invoice Date string'].sort_values().unique()

    # Select slider for date range
    st.subheader("Select Date Range for Analysis")
    selected_date_range = st.select_slider("Date Range", options=date_options, value=(date_options[0], date_options[-1]))

    # Convert selected date strings back to timestamps
    start_date, end_date = pd.to_datetime(selected_date_range[0]), pd.to_datetime(selected_date_range[1])

    # Filter the dataframe based on the selected date range
    df = df[(df['Invoice Date'] >= start_date) & (df['Invoice Date'] <= end_date)]

    # Group by Invoice Date and sum the Invoice Amount
    grouped_data = df.groupby('Invoice Date')['Invoice Amount'].sum()

    import plotly.graph_objects as go

# Create a figure
fig = go.Figure()



# Display the figure
fig.show()




