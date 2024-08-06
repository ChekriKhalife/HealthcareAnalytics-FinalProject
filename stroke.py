import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

import io
import base64
from pmdarima import auto_arima


# Load your data
@st.cache_data
def load_data():
    df = pd.read_csv("https://github.com/ChekriKhalife/HealthcareAnalytics-FinalProject/raw/main/Stroke_data.csv")
    
    # Print the first few values of the 'year' column
    print(df['year'].head())
    
    # Convert 'year' column to datetime, then extract year
    df['year'] = pd.to_datetime(df['year'], errors='coerce').dt.year
    
    return df

df = load_data()

# Set page config
st.set_page_config(page_title="Stroke Analysis Dashboard", layout="wide")

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stSelectbox [data-baseweb="select"] {
        background-color: #ffffff;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: transform 0.3s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .metric-card h3 {
        font-size: 18px;
        margin-bottom: 10px;
        color: #2c3e50;
    }
    .metric-card p {
        font-size: 24px;
        font-weight: bold;
        color: #3498db;
    }
    .filter-container {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .filter-container h3 {
        color: #2c3e50;
        font-size: 20px;
        margin-bottom: 15px;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def create_metric_card(title, value):
    return f"""
    <div class="metric-card">
        <h3>{title}</h3>
        <p>{value}</p>
    </div>
    """

def filter_data_for_section(section, df, user_selections):
    filtered = df.copy()
    
    if 'countries' in user_selections and user_selections['countries']:
        filtered = filtered[filtered['location'].isin(user_selections['countries'])]
    
    if 'measures' in user_selections and user_selections['measures']:
        filtered = filtered[filtered['measure'].isin(user_selections['measures'])]
    
    if 'year' in user_selections and user_selections['year'] is not None:
        filtered = filtered[filtered['year'] == user_selections['year']]
    
    if 'age' in user_selections and user_selections['age'] is not None:
        filtered = filtered[filtered['age'] == user_selections['age']]
    
    if 'sex' in user_selections and user_selections['sex'] is not None and user_selections['sex'] != "All":
        filtered = filtered[filtered['sex'] == user_selections['sex']]
    
    if 'rei' in user_selections and user_selections['rei']:
        filtered = filtered[filtered['rei'].isin(user_selections['rei'])]
    
    return filtered

def filter_data(df, countries, year_range, measures, sex, age_groups, reis):
    return df[
        (df['location'].isin(countries)) &
        (df['year'].between(year_range[0], year_range[1])) &
        (df['measure'].isin(measures)) &
        (df['sex'].isin(sex)) &
        (df['age'].isin(age_groups)) &
        (df['rei'].isin(reis))
    ]

# Function to perform the forecast using multiple fallback methods
def forecast_data(data, periods, risk_factors, metric):
    try:
        # Prepare the data
        target = data.groupby('year')[metric].mean()
        
        # Determine the number of seasonal periods based on the data's length
        # Set a minimum of 2 for seasonal_periods
        seasonal_periods = max(2, len(target) // 2)  # Adjust this logic based on your data
        
        if len(target) > seasonal_periods:
            # Only apply seasonal modeling if the data length supports it
            model = ExponentialSmoothing(target, trend='add', seasonal='add', seasonal_periods=seasonal_periods)
        else:
            # If insufficient data, fall back to a non-seasonal model
            model = ExponentialSmoothing(target, trend='add', seasonal=None)
        
        fitted_model = model.fit()
        forecast = fitted_model.forecast(periods)
        
        # If forecast is successful, return it
        if not np.isnan(forecast).any():
            return forecast
        
        # If Holt-Winters fails, try simple exponential smoothing
        model = ExponentialSmoothing(target, trend=None, seasonal=None)
        fitted_model = model.fit()
        forecast = fitted_model.forecast(periods)
        
        # If simple exponential smoothing is successful, return it
        if not np.isnan(forecast).any():
            return forecast
        
        # If all else fails, use moving average
        window = min(len(target), 3)  # Use up to 3 years for moving average
        ma = target.rolling(window=window).mean()
        last_ma = ma.iloc[-1]
        last_year = pd.to_datetime(str(data['year'].max()), format='%Y')
        forecast = pd.Series([last_ma] * periods, index=pd.date_range(start=last_year + pd.DateOffset(years=1), periods=periods, freq='Y'))
        
        return forecast
    
    except Exception as e:
        st.error(f"Error in forecasting: {e}")
        # If everything fails, return a flat line based on the last known value
        last_value = data[metric].iloc[-1]
        last_year = pd.to_datetime(str(data['year'].max()), format='%Y')
        forecast = pd.Series([last_value] * periods, index=pd.date_range(start=last_year + pd.DateOffset(years=1), periods=periods, freq='Y'))
        return forecast

# Sidebar
st.sidebar.title("Stroke Analysis Dashboard")

# Sidebar navigation
selected_section = st.sidebar.radio("Select Section", 
    ["Introduction", "Geographical Analysis", "Temporal Trends", "Demographic Analysis", "Risk Factor Analysis", 
     "Forecasting", "Statistical Tests"])

# Main content
if selected_section == "Introduction":
    st.title("Introduction to Stroke Analysis Dashboard")
    
    # What is Stroke
    st.header("What is Stroke")
    st.markdown("""
    A stroke occurs when the blood supply to part of the brain is interrupted or reduced, preventing brain tissue from getting oxygen and nutrients. 
    Strokes are a leading cause of death and disability worldwide, and they can be classified into two main types: ischemic (caused by a blockage) and hemorrhagic (caused by bleeding).
    """)

    # Symptoms and Risk Factors
    st.subheader("Symptoms")
    st.markdown("""
    Common symptoms of stroke include sudden numbness or weakness in the face, arm, or leg, especially on one side of the body; confusion; trouble speaking or understanding speech; vision problems in one or both eyes; difficulty walking; dizziness; and loss of balance or coordination.
    """)

    st.subheader("Risk Factors")
    st.markdown("""
    Major risk factors for stroke include high blood pressure, smoking, diabetes, high cholesterol, obesity, and physical inactivity. Age, family history, and certain genetic factors also play a significant role.
    """)

    # Countries Overview
    st.header("Countries Overview")
    st.markdown("""
    This analysis focuses on countries from diverse regions, including the Middle East and North Africa, to explore stroke burden variations and commonalities. 
    These countries were selected based on the diversity in healthcare infrastructure, and the varying incidence rates of stroke.
    Understanding the differences and similarities in stroke risk factors and healthcare responses in these regions helps provide insights that can inform global strategies for prevention and treatment.
    """)

    # Data Overview
    st.header("Data Overview")
    st.markdown(f"""
    - *Number of Rows:* {df.shape[0]}
    - *Number of Columns:* {df.shape[1] - 1}  # Adjusted to exclude the unnamed index column
    """)

    # Remove unnamed index column for display
    display_df = df.drop(columns=df.columns[0])

    st.subheader("Sample of the Dataset")
    st.write(display_df.head())

    st.subheader("Source Description")
    st.markdown("""
    The data is sourced from the [Institute for Health Metrics and Evaluation (IHME)](http://www.healthdata.org/gbd), 
    a global research center at the University of Washington that provides comprehensive estimates on the impact of diseases, injuries, and risk factors.
    The dataset includes variables such as location, year, measure, sex, age, and various risk factors associated with stroke.
    """)

    # Did You Know?
    st.header("Did You Know?")
    st.markdown("""
    - *80% of Strokes are Preventable:* Lifestyle changes and risk factor management can prevent the majority of strokes.
    - *Stroke is a Leading Cause of Death:* In many countries, stroke is among the top causes of death, emphasizing the need for effective prevention strategies.
    - *Youth are at Risk Too:* Stroke can affect younger individuals; about 10-15% of all strokes occur in people under 45.
    - *Gender Disparities Exist:* Women are more likely to have strokes and die from them compared to men, largely due to longer life expectancy and differences in risk factors.
    - *Economic Impact is Significant:* The cost of treating strokes and the resulting disability has a substantial economic impact on healthcare systems worldwide.
    """)

    # Link to download the full CSV file
    st.subheader("Download Full Dataset")
    
    @st.cache_data
    def convert_df(df):
        return df.to_csv().encode('utf-8')

    csv_full = convert_df(df)
    st.download_button(
        label="Download Full CSV",
        data=csv_full,
        file_name="full_stroke_data.csv",
        mime="text/csv",
        
    )

elif selected_section == "Geographical Analysis":
    st.title("Stroke Geographical Analysis")
    
    # Filters in the middle
    st.header("Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        countries = df['location'].unique()
        selected_countries = st.multiselect("Select Countries", countries, default=countries[:3], key="georgraphical_countries")
    
    with col2:
        years = df['year'].dropna().unique().astype(int)
        year_range = st.slider("Select Year Range", int(years.min()), int(years.max()), 
                               (int(years.min()), int(years.max())), key="geographical_years")
    
    with col3:
        measures = df['measure'].unique()
        selected_measures = st.multiselect("Select Measures", measures, default=measures, key="geographical_measures")
    
    # Filter data
    filtered_df = df[
        (df['location'].isin(selected_countries)) &
        (df['year'].between(year_range[0], year_range[1])) &
        (df['measure'].isin(selected_measures))
    ]
    
    # Key Metrics
    st.header("Key Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_countries = len(filtered_df['location'].unique())
        st.metric("Countries Analyzed", total_countries)
    
    with col2:
        avg_burden = filtered_df['Rate'].mean()
        st.metric("Average Burden Rate", f"{avg_burden:.2f}")
    
    with col3:
        max_burden = filtered_df['Rate'].max()
        st.metric("Max Burden Rate", f"{max_burden:.2f}")
    
    # Global Distribution Map
    st.header("Global Stroke Burden Distribution")
    choropleth_df = filtered_df.groupby(['location', 'year'])['Rate'].mean().reset_index()
    fig_map = px.choropleth(choropleth_df, 
                            locations='location', 
                            locationmode='country names',
                            color='Rate',
                            animation_frame='year',
                            color_continuous_scale=px.colors.sequential.Plasma,
                            range_color=(choropleth_df['Rate'].min(), choropleth_df['Rate'].max()),
                            title="Stroke Burden by Country Over Time")
    fig_map.update_layout(height=500)
    st.plotly_chart(fig_map, use_container_width=True)
    
    # Comprehensive Stroke Burden Analysis
    st.header("Comprehensive Stroke Burden Analysis")

    # Calculate average burden rate, current burden rate, and top risk factor for each country
    country_analysis = filtered_df.groupby('location').agg({
        'Rate': ['mean', lambda x: x.iloc[-1]],  # Average and most recent burden rate
        'rei': lambda x: x.value_counts().index[0],  # Most common risk factor
        'year': 'max'  # Most recent year
    }).reset_index()

    country_analysis.columns = ['Country', 'Average Burden', 'Current Burden', 'Top Risk Factor', 'Latest Year']
    country_analysis = country_analysis.sort_values('Average Burden', ascending=False)

    # Reset index and add rank
    country_analysis = country_analysis.reset_index(drop=True)
    country_analysis['Rank'] = country_analysis.index + 1

    # Reorder columns
    country_analysis = country_analysis[['Rank', 'Country', 'Average Burden', 'Current Burden', 'Top Risk Factor', 'Latest Year']]

    # Function to highlight rows
    def highlight_rows(row):
        if row.name < len(country_analysis) // 2:
            return ['background-color: lightyellow']*len(row)
        else:
            return ['background-color: lightblue']*len(row)

    # Apply styling
    styled_df = country_analysis.style.apply(highlight_rows, axis=1).format({
        'Average Burden': '{:.2f}',
        'Current Burden': '{:.2f}',
        'Latest Year': '{:.0f}'
    })

    # Display the table
    st.dataframe(styled_df, height=700)

    st.markdown("""
    <div style='font-size: 0.9em; margin-top: 10px;'>
    <p><strong>Table Explanation:</strong></p>
    <ul>
        <li>This table shows all selected countries ranked by average stroke burden rate.</li>
        <li>Countries highlighted in <span style='background-color: lightyellow;'>light yellow</span> are those with higher burden rates.</li>
        <li>Countries highlighted in <span style='background-color: lightblue;'>light blue</span> are those with lower burden rates.</li>
        <li>Rank: The relative position of each country based on its average burden rate.</li>
        <li>Average Burden: The mean stroke burden rate for each country over the selected period.</li>
        <li>Current Burden: The most recent stroke burden rate for each country in the selected period.</li>
        <li>Top Risk Factor: The most common risk factor for stroke in each country.</li>
        <li>Latest Year: The most recent year for which data is available in the selected range.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # Key Insights
    st.header("Key Insights")
    highest_burden = country_analysis.iloc[0]
    lowest_burden = country_analysis.iloc[-1]

    st.write(f"1. {highest_burden['Country']} has the highest average stroke burden rate of {highest_burden['Average Burden']:.2f}, with a current burden rate of {highest_burden['Current Burden']:.2f} as of {highest_burden['Latest Year']:.0f}. The top risk factor is {highest_burden['Top Risk Factor']}.")
    st.write(f"2. {lowest_burden['Country']} has the lowest average stroke burden rate of {lowest_burden['Average Burden']:.2f}, with a current burden rate of {lowest_burden['Current Burden']:.2f} as of {lowest_burden['Latest Year']:.0f}. The top risk factor is {lowest_burden['Top Risk Factor']}.")
    st.write(f"3. The average burden rate across all selected countries is {country_analysis['Average Burden'].mean():.2f}.")

    # Download button for filtered data
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download filtered data as CSV",
        data=csv,
        file_name="filtered_stroke_data.csv",
        mime="text/csv",
        key="geographical_download_button"
    )

elif selected_section == "Temporal Trends":
    st.title("Temporal Trends in Stroke Burden")
    
    # Custom CSS for improved layout
    st.markdown("""
    <style>
    .stSelectbox, .stMultiSelect {
        background-color: #f0f2f5;
        padding: 10px;
        border-radius: 5px;
    }
    .chart-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Filters
    st.subheader("Data Filters")
    col1, col2 = st.columns(2)
    with col1:
        countries = df['location'].unique()
        selected_countries = st.multiselect("Select Countries", countries, default=countries[:3], key="temporal_countries")
        
        measures = df['measure'].unique()
        selected_measures = st.multiselect("Select Measures", measures, default=measures[:2], key="temporal_measures")
    
    with col2:
        years = df['year'].dropna().unique().astype(int)
        year_range = st.slider("Select Year Range", int(years.min()), int(years.max()), 
                               (int(years.min()), int(years.max())), key="temporal_years")
        
        risk_factors = df['rei'].unique()
        selected_risk_factor = st.selectbox("Select Risk Factor", risk_factors, key="temporal_risk_factor")
    
    # Filter data
    filtered_df = df[
        (df['location'].isin(selected_countries)) &
        (df['year'].between(year_range[0], year_range[1])) &
        (df['measure'].isin(selected_measures)) &
        (df['rei'] == selected_risk_factor)
    ]
    
    if filtered_df.empty:
        st.warning("No data available for the selected filters. Please adjust your selection.")
    else:
        # Prepare data for visualization
        temporal_df = filtered_df.groupby(['year', 'measure', 'location'])['Rate'].mean().reset_index()
        
        # Stroke Burden Rate Trends
        st.subheader("Stroke Burden Rate Trends")
        fig = px.line(temporal_df, x='year', y='Rate', color='location', facet_col='measure',
                      title=f'Stroke Burden Rate Trends by Country and Measure ({year_range[0]}-{year_range[1]})',
                      labels={'Rate': 'Burden Rate', 'year': 'Year', 'location': 'Country'},
                      color_discrete_sequence=px.colors.qualitative.Safe)
        
        fig.update_layout(
            legend_title_text='Country',
            height=500,
            font=dict(size=12),
            plot_bgcolor='rgba(240, 242, 245, 0.8)',
            hoverlabel=dict(bgcolor="white", font_size=12),
            title=dict(font=dict(size=16))
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', title_text="Burden Rate")
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Year-over-Year Change Analysis
        st.subheader("Annual Change in Stroke Burden")
        yoy_change = temporal_df.copy()
        yoy_change['YoY_Change'] = yoy_change.groupby(['measure', 'location'])['Rate'].pct_change()
        
        fig_yoy = px.bar(yoy_change, x='year', y='YoY_Change', color='location', facet_col='measure',
                         title=f'Annual Percentage Change in Stroke Burden by Country and Measure ({year_range[0]}-{year_range[1]})',
                         labels={'YoY_Change': 'Annual Change (%)', 'year': 'Year', 'location': 'Country'},
                         color_discrete_sequence=px.colors.qualitative.Safe)
        
        fig_yoy.update_layout(
            legend_title_text='Country',
            height=500,
            font=dict(size=12),
            plot_bgcolor='rgba(240, 242, 245, 0.8)',
            hoverlabel=dict(bgcolor="white", font_size=12),
            title=dict(font=dict(size=16))
        )
        fig_yoy.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
        fig_yoy.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', title_text="Annual Change (%)")
        fig_yoy.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        
        st.plotly_chart(fig_yoy, use_container_width=True)
        
        # Key Insights
        st.subheader("Key Insights on Annual Changes")
        for measure in selected_measures:
            measure_data = yoy_change[yoy_change['measure'] == measure]
            max_increase = measure_data.loc[measure_data['YoY_Change'].idxmax()]
            max_decrease = measure_data.loc[measure_data['YoY_Change'].idxmin()]
            
            st.markdown(f"**{measure}:**")
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"🔼 Largest Annual Increase: **{max_increase['YoY_Change']:.2%}**\n"
                        f"Country: {max_increase['location']}\n"
                        f"Year: {max_increase['year']}")
            with col2:
                st.info(f"🔽 Largest Annual Decrease: **{max_decrease['YoY_Change']:.2%}**\n"
                        f"Country: {max_decrease['location']}\n"
                        f"Year: {max_decrease['year']}")
        
        # Long-term Trend Analysis
        st.subheader("Long-term Trend Analysis")
        trend_results = []
        for country in selected_countries:
            for measure in selected_measures:
                country_measure_data = temporal_df[(temporal_df['location'] == country) & (temporal_df['measure'] == measure)]
                if not country_measure_data.empty:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(country_measure_data['year'], country_measure_data['Rate'])
                    trend_direction = "Increasing" if slope > 0 else "Decreasing"
                    significance = "Significant" if p_value < 0.05 else "Not Significant"
                    trend_results.append({
                        "Country": country,
                        "Measure": measure,
                        "Overall Trend": trend_direction,
                        "Annual Change Rate": f"{slope:.4f}",
                        "Statistical Significance": significance
                    })
        
        trend_df = pd.DataFrame(trend_results)
        st.dataframe(trend_df.style.applymap(lambda x: 'color: #007bff' if x == 'Increasing' else 'color: #dc3545', subset=['Overall Trend'])
                              .applymap(lambda x: 'background-color: #ffc107' if x == 'Significant' else '', subset=['Statistical Significance']))
        
        # Download button for filtered data
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Analysis Data",
            data=csv,
            file_name="temporal_trends_analysis.csv",
            mime="text/csv",
            key="temporal_trends_download_button"
        )

elif selected_section == "Demographic Analysis":
    st.title("Demographic Analysis of Stroke Burden")

    # Custom CSS for improved layout and design
    st.markdown("""
    <style>
    .filter-container, .chart-container {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .stSelectbox > div > div > div, .stMultiSelect > div > div > div {
        background-color: #f0f2f5;
    }
    .chart-title {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Filters in the middle
    st.markdown("<div class='filter-container'>", unsafe_allow_html=True)
    st.subheader("Data Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        countries = df['location'].unique()
        selected_countries = st.multiselect("Select Countries", countries, default=countries[:3], key="demo_countries")
        
        measures = df['measure'].unique()
        selected_measure = st.selectbox("Select Measure", measures, key="demo_measure")
    
    with col2:
        years = df['year'].dropna().unique().astype(int)
        selected_year = st.selectbox("Select Year", years, index=len(years)-1, key="demo_year")
        
        sexes = df['sex'].unique()
        selected_sex = st.multiselect("Select Sex", sexes, default=sexes, key="demo_sex")
    
    with col3:
        age_groups = df['age'].unique()
        selected_age_groups = st.multiselect("Select Age Groups", age_groups, default=age_groups, key="demo_age")
        
        risk_factors = df['rei'].unique()
        selected_risk_factor = st.selectbox("Select Risk Factor", risk_factors, key="demo_risk_factor")
    
    st.markdown("</div>", unsafe_allow_html=True)

    # Filter data
    filtered_df = df[
        (df['location'].isin(selected_countries)) &
        (df['year'] == selected_year) &
        (df['measure'] == selected_measure) &
        (df['sex'].isin(selected_sex)) &
        (df['age'].isin(selected_age_groups)) &
        (df['rei'] == selected_risk_factor)
    ]
    
    if filtered_df.empty:
        st.warning("No data available for the selected filters. Please adjust your selection.")
    else:
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.header(f"Demographic Analysis for {selected_year}")
        
        # 1. Age Distribution
        st.markdown("<p class='chart-title'>Stroke Burden by Age Group</p>", unsafe_allow_html=True)
        age_dist = filtered_df.groupby('age')['Rate'].mean().reset_index()
        fig_age = px.bar(age_dist, x='age', y='Rate', color='age',
                         title='Stroke Burden by Age Group',
                         labels={'Rate': 'Burden Rate', 'age': 'Age Group'},
                         color_discrete_sequence=px.colors.sequential.Plasma)
        
        fig_age.update_layout(
            font=dict(size=12),
            plot_bgcolor='rgba(240, 242, 245, 0.8)',
            hoverlabel=dict(bgcolor="white", font_size=12),
            title=dict(font=dict(size=16))
        )
        fig_age.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
        fig_age.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', title_text="Burden Rate")
        st.plotly_chart(fig_age, use_container_width=True)
        
        # 2. Gender Distribution
        st.markdown("<p class='chart-title'>Stroke Burden by Gender</p>", unsafe_allow_html=True)
        gender_dist = filtered_df.groupby('sex')['Rate'].mean().reset_index()
        fig_gender = px.bar(gender_dist, x='sex', y='Rate', color='sex',
                            title='Stroke Burden by Gender',
                            labels={'Rate': 'Burden Rate', 'sex': 'Gender'},
                            color_discrete_sequence=px.colors.sequential.Plasma)
        
        fig_gender.update_layout(
            font=dict(size=12),
            plot_bgcolor='rgba(240, 242, 245, 0.8)',
            hoverlabel=dict(bgcolor="white", font_size=12),
            title=dict(font=dict(size=16))
        )
        fig_gender.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
        fig_gender.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', title_text="Burden Rate")
        st.plotly_chart(fig_gender, use_container_width=True)
        
        # 3. Risk Factor Distribution
        st.markdown("<p class='chart-title'>Stroke Burden by Risk Factor</p>", unsafe_allow_html=True)
        risk_dist = filtered_df.groupby('rei')['Rate'].mean().reset_index()
        fig_risk = px.pie(risk_dist, values='Rate', names='rei',
                          title='Distribution of Stroke Burden by Risk Factor',
                          color_discrete_sequence=px.colors.sequential.Plasma)
        
        fig_risk.update_layout(
            font=dict(size=12),
            plot_bgcolor='rgba(240, 242, 245, 0.8)',
            hoverlabel=dict(bgcolor="white", font_size=12),
            title=dict(font=dict(size=16))
        )
        st.plotly_chart(fig_risk, use_container_width=True)
        
        # Key Insights
        st.header("Key Insights")
        max_age_burden = age_dist.iloc[age_dist['Rate'].idxmax()]
        st.markdown(f"- **Highest Stroke Burden by Age Group:** {max_age_burden['age']} ({max_age_burden['Rate']:.2f})")
        max_gender_burden = gender_dist.iloc[gender_dist['Rate'].idxmax()]
        st.markdown(f"- **Gender with Highest Stroke Burden:** {max_gender_burden['sex']} ({max_gender_burden['Rate']:.2f})")
        
        # Download button for filtered data
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download filtered data as CSV",
            data=csv,
            file_name="demographic_analysis.csv",
            mime="text/csv",
            key="demographic_download_button"
        )
        st.markdown("</div>", unsafe_allow_html=True)

elif selected_section == "Risk Factor Analysis":
    st.title("Analysis of Stroke Risk Factors")

    # Filters for risk factors
    st.subheader("Data Filters")
    col1, col2, col3 = st.columns(3)
    with col1:
        countries = df['location'].unique()
        selected_countries = st.multiselect("Select Countries", countries, default=countries[:3], key="risk_countries")
    
    with col2:
        years = df['year'].dropna().unique().astype(int)
        year_range = st.slider("Select Year Range", int(years.min()), int(years.max()), 
                               (int(years.min()), int(years.max())), key="risk_years")
    
    with col3:
        risk_factors = df['rei'].unique()
        selected_risk_factors = st.multiselect("Select Risk Factors", risk_factors, default=risk_factors, key="risk_factors")

    # Filter data
    filtered_df = df[
        (df['location'].isin(selected_countries)) &
        (df['year'].between(year_range[0], year_range[1])) &
        (df['rei'].isin(selected_risk_factors))
    ]
    
    if filtered_df.empty:
        st.warning("No data available for the selected filters. Please adjust your selection.")
    else:
        st.header("Risk Factors Overview")
        
        # Distribution of Risk Factors
        risk_factor_counts = filtered_df['rei'].value_counts().reset_index()
        risk_factor_counts.columns = ['Risk Factor', 'Count']
        fig_risk = px.bar(risk_factor_counts, x='Risk Factor', y='Count',
                          title='Distribution of Stroke Risk Factors',
                          labels={'Count': 'Number of Records', 'Risk Factor': 'Risk Factor'},
                          color_discrete_sequence=px.colors.sequential.Plasma)
        fig_risk.update_layout(
            font=dict(size=12),
            plot_bgcolor='rgba(240, 242, 245, 0.8)',
            hoverlabel=dict(bgcolor="white", font_size=12),
            title=dict(font=dict(size=16))
        )
        fig_risk.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
        fig_risk.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', title_text="Number of Records")
        st.plotly_chart(fig_risk, use_container_width=True)
        
        # Trends in Risk Factors over Time
        st.header("Trends in Risk Factors over Time")
        risk_trends = filtered_df.groupby(['year', 'rei'])['Rate'].mean().reset_index()
        fig_trends = px.line(risk_trends, x='year', y='Rate', color='rei',
                             title='Trends in Stroke Burden by Risk Factor',
                             labels={'Rate': 'Burden Rate', 'year': 'Year', 'rei': 'Risk Factor'},
                             color_discrete_sequence=px.colors.sequential.Plasma)
        fig_trends.update_layout(
            font=dict(size=12),
            plot_bgcolor='rgba(240, 242, 245, 0.8)',
            hoverlabel=dict(bgcolor="white", font_size=12),
            title=dict(font=dict(size=16))
        )
        fig_trends.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
        fig_trends.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', title_text="Burden Rate")
        st.plotly_chart(fig_trends, use_container_width=True)
        
        # Key Insights
        st.header("Key Insights")
        for risk_factor in selected_risk_factors:
            risk_data = risk_trends[risk_trends['rei'] == risk_factor]
            latest_data = risk_data.iloc[-1]
            st.markdown(f"- **{risk_factor}:** The latest data in {latest_data['year']} shows a burden rate of {latest_data['Rate']:.2f}.")
        
        # Download button for filtered data
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download filtered data as CSV",
            data=csv,
            file_name="risk_factors_analysis.csv",
            mime="text/csv",
            key="risk_factors_download_button"
        )

elif selected_section == "Forecasting":
    st.title("Forecasting Future Trends in Stroke Burden")

    st.subheader("Data Filters")
    col1, col2 = st.columns(2)
    with col1:
        countries = df['location'].unique()
        selected_country = st.selectbox("Select Country", countries, key="forecast_country")
    
    with col2:
        metrics = df['measure'].unique()
        selected_metric = st.selectbox("Select Metric", metrics, key="forecast_metric")

    st.subheader("Risk Factor Selection")
    selected_risk_factors = st.multiselect("Select Risk Factors", df['rei'].unique(), key="forecast_risk_factors")

    st.subheader("Forecast Settings")
    forecast_years = st.slider("Select Number of Years to Forecast", 1, 10, 5, key="forecast_years")

    # Filter data for forecasting
    forecast_data_df = df[
        (df['location'] == selected_country) &
        (df['measure'] == selected_metric) &
        (df['rei'].isin(selected_risk_factors))
    ]

    if forecast_data_df.empty:
        st.warning("No data available for the selected filters. Please adjust your selection.")
    else:
        st.header("Forecast Results")

        # Perform forecasting
        forecasted_values = forecast_data(forecast_data_df, forecast_years, selected_risk_factors, 'Rate')

        # Plot forecast results
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=forecasted_values.index, y=forecasted_values.values, mode='lines', name='Forecasted Rate'))
        fig_forecast.update_layout(
            title=f"Forecasted Stroke Burden Rate in {selected_country} for {selected_metric}",
            xaxis_title="Year",
            yaxis_title="Burden Rate",
            plot_bgcolor='rgba(240, 242, 245, 0.8)',
            hoverlabel=dict(bgcolor="white", font_size=12),
            font=dict(size=12),
            showlegend=True
        )
        st.plotly_chart(fig_forecast, use_container_width=True)

        # Provide actionable insights based on forecast
        st.subheader("Actionable Insights")
        st.write("Based on the forecast, here are some key insights:")
        st.write(f"1. The forecast indicates a {forecasted_values.pct_change().iloc[-1]:.2%} change in stroke burden rate by the end of the forecast period.")
        st.write("2. The selected risk factors have the potential to significantly influence future trends in stroke burden.")
        st.write("3. It is recommended to focus on preventive measures targeting the top risk factors to mitigate the projected increase.")

        # Download button for forecast data
        csv = forecasted_values.to_csv().encode('utf-8')
        st.download_button(
            label="Download forecast data as CSV",
            data=csv,
            file_name="forecasted_stroke_data.csv",
            mime="text/csv",
            key="forecast_download_button"
        )

# Closing section: Call to action
st.sidebar.markdown("""
---
## Call to Action
This dashboard provides valuable insights into the burden of stroke and its associated risk factors. 
It is crucial to translate these findings into effective public health strategies and interventions. 
Act now to reduce the global impact of stroke.
""")
