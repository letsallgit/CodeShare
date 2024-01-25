import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import plotly.graph_objects as go


@st.cache_data
def load_dataframe():

    df=pd.read_csv('/Users/otb/Desktop/Purchasing Project/Working Version/CsvSanitizedPurchasesInvoices22-24.csv')
    return df

def process_data(df):

    df['Invoice Date'] = pd.to_datetime(df['Invoice Date'], format='%d-%m-%y', errors='coerce')
    df['Year'] = df['Invoice Date'].dt.year
    df['Month'] = df['Invoice Date'].dt.month
    df['Quarter'] = df['Invoice Date'].dt.quarter
    df['Month Name'] = df['Invoice Date'].dt.strftime('%B')
    # New column combining Year and Month
    df['Year-Month'] = df['Invoice Date'].dt.strftime('%Y-%B')
    # New column combining Year and Quarter
    df['Year-Quarter'] = df['Year'].astype(str) + '-Q' + df['Quarter'].astype(str)
    return df

    #Filters

def filter_by_year(df, selected_year):

    df['Invoice Date'] = pd.to_datetime(df['Invoice Date'], format='%d-%m-%y', errors='coerce')
    if selected_year != 'All':
        df = df[df['Year'] == selected_year]
    return df, selected_year

def filter_by_month(df,selected_month):
    df['Invoice Date'] = pd.to_datetime(df['Invoice Date'], format='%d-%m-%y', errors='coerce')
    if selected_month != 'All':
        df = df[df['Month'] == selected_month]
    return df, selected_month

def filter_by_vendor_type(df,selected_vendor_type):

    if 'All' not in selected_vendor_type:
        df = df[df['Vendor Type'].isin(selected_vendor_type)]
    return df, selected_vendor_type

def filter_by_department(df, selected_department):
    if 'All' not in selected_department:
       df = df[df['Department'].isin(selected_department)]
    return df, selected_department

def date_slider(df, selected_date_range):
    df['Invoice Date'] = pd.to_datetime(df['Invoice Date'], format='%d-%m-%y', errors='coerce')

    date_options = df['Invoice Date'].dt.date.sort_values().unique()

    # Select slider for date range
    st.subheader("Date Range")
    min_date, max_date = date_options[0], date_options[-1]
    selected_date_range = st.slider("Date Range", 
                                            min_value=min_date, 
                                            max_value=max_date, 
                                            value=(min_date, max_date), key="dateSlider")

    # Filter the dataframe based on the selected date range
    df = df[(df['Invoice Date'].dt.date >= selected_date_range[0]) & (df['Invoice Date'].dt.date <= selected_date_range[1])]

def calculate_statistics(df, years):
    # Add the missing argument 'years' to the function call
    stats_list = calculate_statistics(df, years)
    stats_list = []
    for year in years:
        df_year = df[df['Year'] == year]
        # Calculate statistics for the year
        stats = {
            'Year': year,
            'Mean': round(df_year['Invoice Amount'].mean(), 0),
            'Median': round(df_year['Invoice Amount'].median(), 0),
            'Mode': round(df_year['Invoice Amount'].mode()[0], 0) if not df_year['Invoice Amount'].mode().empty else None,
            'Minimum': round(df_year['Invoice Amount'].min(), 0),
            'Maximum': round(df_year['Invoice Amount'].max(), 0),
            'Sum': round(df_year['Invoice Amount'].sum(), 0),
            'Count': df_year['Invoice Amount'].count()
        }
        stats_list.append(stats)
        st.table(stats_list)
    # Calculate top 5 vendors by total invoice amount
    top_vendors = df_year.groupby('Vendor Name')['Invoice Amount'].sum().nlargest(5)

    # For each top vendor, calculate additional statistics
    for vendor in top_vendors.index:
        vendor_df = df_year[df_year['Vendor Name'] == vendor]
        vendor_stats = {
            'Year': f"Top Vendor: {vendor}",
            'Mean': round(vendor_df['Invoice Amount'].mean(), 0),
            'Median':round(vendor_df['Invoice Amount'].median(), 0),
            'Mode': round(vendor_df['Invoice Amount'].mode(), 0),
            'Minimum': round(vendor_df['Invoice Amount'].min(), 0),
            'Maximum': round(vendor_df['Invoice Amount'].max(), 0),
            'Sum': round(top_vendors[vendor], 0),
            'Count': None,
            # Add other stats if needed
        }
        stats_list.append(vendor_stats)

def calculate_metrics(df):
        metrics = {}
        metrics['total_Invoice Amount'] = df['Invoice Amount'].sum()
        metrics['average_Invoice Amount'] = df['Invoice Amount'].mean()
        metrics['num_vendors'] = df['Vendor Name'].nunique()
        metrics['num_departments'] = df['Department'].nunique()

        
        return metrics

def display_kpis(metrics):
    num_metrics = len(metrics)
    num_cols = min(4, num_metrics)  # Maximum of 4 metrics per row
    cols = st.columns(num_cols)
    for i, (key, value) in enumerate(metrics.items()):
        with cols[i % num_cols]:
            st.metric(label=key, value=f"{value:,.0f}")
    ## values as a %
    #         def display_kpis(metrics):
    # num_metrics = len(metrics)
    # num_cols = min(4, num_metrics)  # Maximum of 4 metrics per row
    # cols = st.columns(num_cols)
    # for i, (key, value) in enumerate(metrics.items()):
    #     with cols[i % num_cols]:
    #         st.metric(label=key, value=f"{value:.2%}")

def calculate_stat_metrics(df):
    metrics = {
        'Standard Deviation': df['Invoice Amount'].std(),
        'Variance': df['Invoice Amount'].var(),
        'Skewness': df['Invoice Amount'].skew(),
        'Kurtosis': df['Invoice Amount'].kurt(),
    }
    return metrics

# filtered_metrics = {k: v for k, v in metrics.items() if filter_condition(k, v)}
# def display_kpis(metrics):
#   Define the options for the multiselect box
# options = list(metrics.keys())
    
#     # Create the multiselect box
#     selected_metrics = st.multiselect('Select metrics to display', options, default=options)
    
#     # Filter the metrics dictionary based on the selected metrics
#     filtered_metrics = {k: v for k, v in metrics.items() if k in selected_metrics}
#     kpi_expander = st.expander("Key Performance Indicators", expanded=True)
    
#     with kpi_expander:
#         num_metrics = len(metrics)
#         num_cols = min(4, num_metrics)  # Maximum of 4 metrics per row
#         cols = st.columns(num_cols)
#         for i, (key, value) in enumerate(metrics.items()):
#             with cols[i % num_cols]:
#                 st.metric(label=key, value=f"{value:,.0f}")
   
    ##########################  ###################### Visuals ########################  ##########################  
            
# def pydeck_map(data):
#     layer = pdk.Layer('ScatterplotLayer', data, get_position='[longitude, latitude]', get_color='[200, 30, 0, 160]', get_radius=200)
#     view_state = pdk.ViewState(latitude=data['latitude'].mean(), longitude=data['longitude'].mean(), zoom=6)
#     st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))

#     pydeck_map(filtered_df)


def create_bar_chart(df):
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Month:O', title='Month'),
        y=alt.Y('Invoice Amount:Q', title='Invoice Amount'),
        color=alt.Color('Department:N', legend=alt.Legend(title="Department"))
    ).properties(
        width=600,
        height=400,
        title='Monthly Invoice Amount by Department'
    )
    st.altair_chart(chart, use_container_width=True)

def create_line_chart(df):
    fig = go.Figure()
    for department in df['Department'].unique():
        df_department = df[df['Department'] == department]
        fig.add_trace(go.Scatter(x=df_department['Invoice Date'], y=df_department['Invoice Amount'],
                                mode='lines', name=department))
    fig.update_layout(height=600, width=800, title_text="Time Series with Rangeslider",
                    xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)
    create_line_chart(df)

def create_stacked_bar_chart_vendorType_invoiceAmount(df):
    with st.container():
        chart_col1, _ = st.columns(2)
        with chart_col1:
            num_vendors = st.number_input('Number of Vendors', min_value=1, max_value=100, value=10, step=1)
            vendor_spending = df.groupby(['Vendor Name', 'Vendor Type'])['Invoice Amount'].sum().nlargest(num_vendors).unstack()
            top_vendors = df.groupby('Vendor Name')['Invoice Amount'].sum().nlargest(5)
            df = filter_by_year(df, selected_year) if selected_year else df
            selected_year = st.selectbox('Select Year', options=['All'] + sorted(df['Year'].unique().tolist()), key='year_specific_chart')

            # Apply filters using existing filter function
            st.subheader('Total Spending on Top {} Vendors by Category'.format(num_vendors))
            st.bar_chart(vendor_spending)

def create_visuals(df, selected_year=None, selected_month=None, selected_department=None, vendor_type=None):
    # Apply filters using existing filter functions
    load_dataframe()
    process_data(df)
    processed_df = process_data(df)
    df = processed_df.copy()
    df = filter_by_year(df, selected_year)
    df = filter_by_month(df, selected_month)
    df = filter_by_department(df, selected_department)
    df = filter_by_vendor_type(df, vendor_type)

    # Create charts
    create_bar_chart(df)
    create_line_chart(df)
    create_stacked_bar_chart_vendorType_invoiceAmount(df)

    # Chart creation logic remains the same
    with st.container():
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:

            # Trend Line Chart
            st.subheader('Monthly Expenditure Trends')

            monthly_expense = df.groupby('Year-Month')['Invoice Amount'].sum()
            st.line_chart(monthly_expense)

        with chart_col2:
            st.subheader('Quarterly Expenditure Trends')
            quarterly_expense = df.groupby('Year-Quarter')['Invoice Amount'].sum()
            st.line_chart(quarterly_expense)
    

        
    # Assuming 'vendor_total' is a DataFrame containing the total expenditure for each vendor

    total_spend = df['Invoice Amount'].sum()
    vendor_total_by_name = df.groupby('Vendor Name')['Invoice Amount'].sum().reset_index()
    vendor_total_by_name_and_type = df.groupby(['Vendor Name', 'Vendor Type'])['Invoice Amount'].sum().reset_index()
    top_vendor_by_type = df.groupby(['Vendor Type','Vendor Name'])['Invoice Amount'].sum().nlargest(10).reset_index()
    #percentage of a top 10 vendor vendor_total_by_typeof their total inoice amount stratified by type
    top_vendor_by_type['Percentage'] = (top_vendor_by_type)['Invoice Amount'] /(total_spend) * 100
    st.table(top_vendor_by_type['Percentage'])
    # top_vendors = (top_vendors_by_name, top_vendors_by_name_and_type).merge((top_vendors_by_name), (top_vendors_by_name_and_type), on = 'Vendor Name', suffixes=('', '_total'))
    
    # top_vendors = round(vendor_total_by_name.nlargest(10, 'Invoice Amount'))
    # top_vendors = round(vendor_total_by_type.nlargest(10, 'Invoice Amount'))
    # Merging with the top_vendors DataFrame to get the total expenditure for each vendor
    # Specify suffixes for overlapping columns
   
    #Total Invoice Amount
    # total_spend = df['Invoice Amount'].sum()
    # total_spend_by_type = 
    # percentage of vendor total of total invoices
    top_vendors_by_name_and_type = round(vendor_total_by_name_and_type.nlargest(10, 'Invoice Amount'))
    top_vendors_by_name = round(vendor_total_by_name.nlargest(10, 'Invoice Amount'))
    vendor_total_by_name ['Percentage'] = (vendor_total_by_name)['Invoice Amount'] /(total_spend) * 100
    vendor_total_by_name_and_type['Percentage'] = (top_vendors_by_name_and_type)['Invoice Amount'].sum()/(total_spend) * 100
    
   
    st.table(top_vendors_by_name)
    st.table(top_vendors_by_name_and_type)
    # top_vendors = top_vendors.merge((vendor_total_by_name, vendor_total_by_type), on='Vendor Name',  suffixes=('', '_total'))
    # Merge again with proper column handling to avoid duplicates
    # top_vendors = top_vendors.merge(vendor_total[['Vendor Name', 'Percentage']], on='Vendor Name')
    # Streamlit code to display the DataFrame
    # st.write("Top Vendors Table:", top_vendors)
    #top 10 vendors# Display the DataFrame
    # df = pd.DataFrame
    st.table(top_vendors)
    # st.write(df.head(0))
            
    
    load_dataframe()

    df = load_dataframe() 
    # Step 1: Group by 'Vendor Name' and 'Vendor Type' and calculate the sums
    grouped = df.groupby(['Vendor Name', 'Vendor Type'])['Invoice Amount'].sum().reset_index()

    # Step 2: Get the total invoice amount for each vendor
    vendor_totals = df.groupby('Vendor Name')['Invoice Amount'].sum().reset_index()

    # Step 3: Merge the grouped data with the vendor totals
    merged_data = pd.merge(grouped, vendor_totals, on='Vendor Name', suffixes=('', '_total'))

    # Step 4: Calculate the percentage
    merged_data['Percentage'] = (merged_data['Invoice Amount'] / merged_data['Invoice amount_total']) * 100

    # Step 5: Get the top 10 vendors by total invoice amount
    top_vendors = vendor_totals.nlargest(10, 'Invoice Amount')

    # Step 6: Filter the merged data to include only the top 10 vendors
    top_vendors_data = merged_data[merged_data['Vendor Name'].isin(top_vendors['Vendor Name'])]

    # Step 7: Display the data
    for vendor in top_vendors_data['Vendor Name'].unique():
        vendor_data = top_vendors_data[top_vendors_data['Vendor Name'] == vendor]
        total_amount = vendor_totals[vendor_totals['Vendor Name'] == vendor]['Invoice Amount'].values[0]
        print(f"Vendor Name: {vendor}\tTotal Invoice Amount: ${total_amount:,.2f}\n")
        print(vendor_data[['Vendor Type', 'Invoice Amount', 'Percentage']])
        print("\n")

    # # Updated Altair chart
    # chart = alt.Chart(top_vendors).mark_bar().encode(
    #     x=alt.X('Vendor Name:N', title='Vendor Name', sort='-y'),
    #     y=alt.Y('Invoice Amount:Q', title='Total Expenditure (in Millions)',
    #             axis=alt.Axis(format='$,.2fM'), 
    #             scale=alt.Scale(domain=[0, top_vendors['Invoice Amount'].max() / 1_000_000])),
    #     # color='Vendor Type:N',
    #     tooltip=[
    #         alt.Tooltip('Vendor Name', title='Vendor Name'),
    #         # alt.Tooltip('Vendor Type', title='Vendor Type'),
    #         alt.Tooltip('Invoice Amount:Q', title='Invoice Amount', format='$,.0f'),
    #         alt.Tooltip('Percentage:Q', title='Percentage', format='.2f')
    #     ]
    # ).transform_calculate(
    #     InvoiceAmountMillions='datum["Invoice Amount"] / 1000_000'
    # ).properties(
    #     width=800,
    #     height=500
    # )

    # with st.container():
    #     st.subheader('Total Expenditure on Top 10 Vendors by Vendor Type (FY 22-23)'),
    #     st.altair_chart(chart, use_container_width=True)


#######################   #######################



      

################# 111111  #######################
            



        # Display the chart in Streamlit
        st.pyplot(plt)

#     # Top Vendors by Total Expenditure
#     st.subheader('Top Vendors by Total Expenditure')
#     top_vendors = df.groupby('Vendor Name')['Invoice Amount'].sum().nlargest(10)
#     filter_by_year(df, selected_year=selected_year)
#     st.bar_chart(top_vendors)

#     #Vendor Market Share
        st.subheader('Vendor Share (top 10)')
        # total_spend = df['Invoice Amount'].sum()
        vendor_spend = df.groupby('Vendor Name')['Invoice Amount'].sum()
        vendor_share = (vendor_spend / total_spend) * 100
        top_vendors_by_share = vendor_share.nlargest(10)
#     #Creating a bar chart for vendor market share
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_vendors_by_share, y=top_vendors_by_share.index)
        plt.xlabel('Market Share (%)')
        plt.ylabel('Vendor Name')
        plt.title('Top Vendors by Share of Total Expenditure')
        plt.tight_layout()
        #Display the chart in Streamlit
        st.pyplot(plt)



                    # Expenditure by Category
    st.subheader('Expenditure by Category')
    expenditure_by_category = df.groupby('Vendor Type')['Invoice Amount'].sum().nlargest(10)
    st.bar_chart(expenditure_by_category)



                #     #Monthly Spending Trends with Category Breakdown
                #     st.subheader('Monthly Spending Trends')
                #     monthly_spending = df.groupby('Month')['Invoice Amount'].sum()
                #     st.line_chart(monthly_spending)
#       #Grouping by month and category
    monthly_category_spending = df.groupby(['Month', 'Vendor Type'])['Invoice Amount'].sum().unstack()
    # Line chart with category breakdown
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=monthly_category_spending)
    plt.xlabel('Month')
    plt.ylabel('Spending')
    plt.title('Monthly Spending Trends by Category')
    plt.legend(title='Category', loc='upper left')
    plt.tight_layout()
    # Display the chart in Streamlit
    st.pyplot(plt)

                    # Spending by Category and Month

                    # Creating a stacked bar chart
    # spending_by_cat_month.plot(kind='bar', stacked=True, figsize=(10,6))
    plt.title('Spending by Category and Month')
    plt.xlabel('Month')
    plt.ylabel('Spending')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Display the chart in Streamlit
    st.pyplot(plt)

                # Line chart with category breakdown
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=monthly_category_spending)
    plt.xlabel('Month')
    plt.ylabel('Spending')
    plt.title('Monthly Spending Trends by Category')
    plt.legend(title='Category', loc='upper left')
    plt.tight_layout()

    # Display the chart in Streamlit
    st.pyplot(plt)

#                   # # Spending Proportion by Department
#     # st.subheader('Spending Proportion by Department')
#     # department_spending = df.groupby('Department')['Invoice Amount'].sum()
#     # st.bar_chart(department_spending)

#     # Trend Analysis
#     st.subheader('Trend Analysis - Total Spending Over Time')
#     st.line_chart(monthly_spending)


#                       # Vendor and Category Interaction
#     st.subheader('Vendor and Category Interaction')
#     vendor_cat_interaction = df.pivot_table(index='Vendor Name', columns='Vendor Type', values='Invoice Amount', aggfunc='sum')
#     st.bar_chart(vendor_cat_interaction)

#     # # Invoice Amount vs. Frequency
#     # st.subheader('Invoice Amount vs. Frequency')
#     # st.scatter_chart(df['Invoice Amount'], df['Invoice Date'])


        # Scatter Plot of Invoice Amount vs. Number of Invoices
        #     st.subheader('Invoice Amount vs. Number of Invoices for Each Vendor')
        #     # Adding a column for vendor type to the vendor_invoice_scatter DataFrame
    # vendor_type = df.groupby('Vendor Name')['Vendor Type'].first().reset_index()
    # vendor_invoice_scatter = vendor_invoice_scatter.merge(vendor_type, on='Vendor Name')

    # # Creating a scatter plot
    # plt.figure(figsize=(10, 6))
    # sns.scatterplot(data=vendor_invoice_scatter, x='Total Amount', y='Invoice Count', hue='Vendor Type')
    # plt.xlabel('Total Invoice Amount')
    # plt.ylabel('Number of Invoices')
    # plt.title('Invoice Amount vs. Number of Invoices by Vendor Type')
    # plt.legend(title='Vendor Type')
    # plt.tight_layout()
    # # Display the chart in Streamlit
    # st.pyplot(plt)


#     # histogram
#     # Assuming 'Invoice Amount' is the column we want to plot
#     st.subheader("Histogram")
#     plt.figure(figsize=(10, 6))
#     plt.hist(df['Invoice Amount'], bins=50, color='blue', edgecolor='black')
#     plt.title('Histogram of Invoice Amounts')
#     plt.xlabel('Invoice Amount')
#     plt.ylabel('Frequency')
#     plt.show()

#     # Define the bins
#     bins = [0, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 10000, 20000, 30000, 40000, 50000, 100000, #200000, 400000, 600000, 800000, 1000000, 1500000]

#     # Use cut to create the binned column
#     # Assuming 'Invoice Amount' is the column to be binned
#     # Replace 'Invoice Amount' with the actual column name if different
#     # Add labels=False to return bin numbers instead of interval ranges if desired

#     # Apply the binning
#     df['Invoice Amount bin'] = pd.cut(df['Invoice Amount'], bins=bins, right=False)

#     # Calculate the frequency for each bin
#     frequency_table = df['Invoice Amount bin'].value_counts().sort_index()

#     # Convert the frequency table to a DataFrame for better formatting
#     frequency_df = frequency_table.reset_index()
#     frequency_df.columns = ['BIN', 'FREQUENCY']

#     # Adjust the bin labels to show the upper limit
#     frequency_df['BIN'] = frequency_df['BIN'].apply(lambda x: x.right)

#     # Display the frequency table
#     st.write(frequency_df)

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
                sns.barplot(x=data.values, y=data.index)
                plt.xlabel('Total Expenditure')
                plt.ylabel('Vendor Name')
            elif chart_type == 'Pie Chart':
                plt.pie(data, labels=data.index, autopct='%1.1f%%', startangle=140)
            elif chart_type == 'Line Chart':
                sns.lineplot(x=data.index, y=data.values)

            # Display the chart
        st.pyplot(plt)

        #interactive time-series
        # Sidebar widgets to select the date range
        # Convert Invoice Date to datetime
        #df['Invoice Date'] = pd.to_datetime(df['Invoice Date'])

        # Convert dates to strings for the slider
        df['Invoice Date string'] = df['Invoice Date'].dt.strftime('%Y-%m-%d')
        date_options = df['Invoice Date string'].sort_values().unique()

        # Select slider for date range
        st.subheader("Select Date Range for Analysis")
        selected_date_range = st.select_slider("Date Range", options=date_options, value=(date_options[0], date_options[-1]))

        # Convert selected date strings back to timestamps
        #start_date, end_date = pd.to_datetime(selected_date_range[0]), pd.to_datetime(selected_date_range[1])

        # Filter the dataframe based on the selected date range
        #df = df[(df['Invoice Date'] >= start_date) & (df['Invoice Date'] <= end_date)]

        # Group by Invoice Date and sum the Invoice Amount
        #grouped_data = df.groupby('Invoice Date')['Invoice Amount'].sum()



##################  OLD FILTER OR VISUAL CODE
        


def user_created_visuals2(df):
    # Set Seaborn style and palette
    sns.set_style("whitegrid")
    sns.set_palette("pastel")


    df=pd.read_csv('/Users/otb/Desktop/Purchasing Project/Working Version/CsvSanitizedPurchasesInvoices22-24.csv')


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
                sns.barplot(x=data.values, y=data.index)
                plt.xlabel('Total Expenditure', fontsize=14)
                plt.ylabel('Vendor Name', fontsize=14)
                plt.title('Bar Chart', fontsize=20)
            elif chart_type == 'Pie Chart':
                plt.pie(data, labels=data.index, autopct='%1.1f%%', startangle=140)
                plt.title('Pie Chart', fontsize=20)
            elif chart_type == 'Line Chart':
                sns.lineplot(x=data.index, y=data.values)
                plt.xlabel('Vendor Name', fontsize=14)
                plt.ylabel('Total Expenditure', fontsize=14)
                plt.title('Line Chart', fontsize=20)

            # Display the chart
            st.pyplot(plt)

    """    ######dfddddddddddddd
    # Stacked Bar Chart for Vendor Analysis
    load_dataframe()
    process_data(df)
    #Calculating the sum of invoice amounts for top 10 vendors by vendor type
    vendor_spending = df.groupby(['Vendor Name', 'Vendor Type'])['Invoice Amount'].sum().reset_index()
    top_vendors = vendor_spending.nlargest(11, 'Invoice Amount')
    # Altair chart with y-axis in millions
    chart = alt.Chart(top_vendors).mark_bar().encode(
        x=alt.X('Vendor Name:N', title='Vendor Name', sort='-y'),
        y=alt.Y('Invoice Amount:Q', title='Total Expenditure (in Millions)', 
                axis=alt.Axis(format='$,.2fM'),  # Format y-axis as currency in millions
                scale=alt.Scale(domain=[0, top_vendors['Invoice Amount'].max() / 1_000_000])
            ),
            color='Vendor Type:N',
            tooltip=[
                alt.Tooltip('Vendor Name', title='Vendor Name'),
                alt.Tooltip('Vendor Type', title='Vendor Type'),
                alt.Tooltip('Invoice Amount:Q', title='Invoice Amount', format='$,.0f')
            ]
        ).transform_calculate(
            # Divide the Invoice Amount by 1 million for display
            InvoiceAmountMillions='datum["Invoice Amount"] / 1000_000'
        ).properties(
            width=800,
            height=500,

        )
    """