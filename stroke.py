import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.statespace.sarimax import SARIMAX
import io
import base64

# Load your data
df=pd.read_csv("C:\\Users\\hp\\Desktop\\healthcare\\Stroke_data.csv")
df['year'] = pd.to_datetime(df['year'], errors='coerce').dt.year  # Ensure year is an integer


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
    
    # Check if the filtered dataset is empty
    if filtered.empty:
        st.warning("No data available for the selected filters. Please adjust your selection.")
        st.write("Selected filters:", user_selections)
        return None
    
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
# Function to perform the forecast
def forecast_data(data, periods, risk_factors, metric):
    # Prepare the data
    data_pivoted = data.pivot(index='year', columns='rei', values=metric)
    data_pivoted = data_pivoted.fillna(method='ffill').fillna(0)
    
    # Ensure all selected risk factors are in the data
    for rf in risk_factors:
        if rf not in data_pivoted.columns:
            data_pivoted[rf] = 0
    
    # Select only the chosen risk factors
    exog = data_pivoted[risk_factors]
    
    # Create a Series for the target variable
    target = data.groupby('year')[metric].mean()
    
    # Ensure the index of exog matches the target
    exog = exog.reindex(target.index)
    
    # Fit the SARIMAX model
    try:
        model = SARIMAX(target, exog=exog, order=(1,1,1), seasonal_order=(1,1,1,12))
        results = model.fit()
        
        # Forecast
        forecast = results.forecast(steps=periods, exog=exog.iloc[-periods:])
        return forecast
    except Exception as e:
        st.error(f"Error in forecasting: {str(e)}")
        return None
# Sidebar
st.sidebar.title("Stroke Analysis Dashboard")

# Sidebar navigation
selected_section = st.sidebar.radio("Select Section", 
    ["Introduction", "Geographical Analysis", "Temporal Trends", "Demographic Analysis", "Risk Factor Analysis", 
     "Forecasting", "Statistical Tests"])

# Main content
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
    key="geographical_download_button"  # Added unique key
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
        age_data = filtered_df.groupby('age')['Rate'].mean().reset_index()
        fig_age = px.bar(
            age_data,
            x='age',
            y='Rate', 
            labels={'Rate': 'Average Burden Rate', 'age': 'Age Group'},
            color='Rate',
            color_continuous_scale=px.colors.sequential.Tealgrn
        )
        fig_age.update_layout(
            height=400,
            xaxis_title="Age Group",
            yaxis_title="Average Burden Rate",
            plot_bgcolor="#f0f2f5",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_age, use_container_width=True)
        
        # 2. Gender Comparison
        st.markdown("<p class='chart-title'>Gender Comparison</p>", unsafe_allow_html=True)
        gender_data = filtered_df.groupby('sex')['Rate'].mean().reset_index()
        fig_gender = px.pie(
            gender_data,
            names='sex',
            values='Rate',
            labels={'Rate': 'Average Burden Rate', 'sex': 'Gender'},
            color='sex',
            color_discrete_map={'Male': '#4C78A8', 'Female': '#F58518'}
        )
        fig_gender.update_traces(textposition='inside', textinfo='percent+label')
        fig_gender.update_layout(
            height=400,
            showlegend=True,
            legend_title_text='Gender',
            plot_bgcolor="#f0f2f5",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_gender, use_container_width=True)
        
        # 3. Country Comparison
        st.markdown("<p class='chart-title'>Country Comparison</p>", unsafe_allow_html=True)
        country_data = filtered_df.groupby('location')['Rate'].mean().reset_index()
        fig_country = px.bar(
            country_data,
            x='location',
            y='Rate', 
            labels={'Rate': 'Average Burden Rate', 'location': 'Country'},
            color='Rate',
            color_continuous_scale=px.colors.sequential.Sunset
        )
        fig_country.update_layout(
            height=400,
            xaxis_title="Country",
            yaxis_title="Average Burden Rate",
            plot_bgcolor="#f0f2f5",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_country, use_container_width=True)
        
        # 4. Age-Gender Heatmap
        st.markdown("<p class='chart-title'>Age-Gender Distribution</p>", unsafe_allow_html=True)
        heatmap_gender = filtered_df.groupby(['age', 'sex'])['Rate'].mean().reset_index()
        heatmap_pivot_gender = heatmap_gender.pivot(index='age', columns='sex', values='Rate')
        fig_heatmap_gender = px.imshow(
            heatmap_pivot_gender, 
            labels=dict(x="Gender", y="Age Group", color="Burden Rate"),
            color_continuous_scale=px.colors.sequential.YlOrBr,
            aspect='auto'  # Ensures the aspect ratio is optimized
        )
        fig_heatmap_gender.update_layout(
            height=500,
            title_text="Age-Gender Distribution of Stroke Burden",
            plot_bgcolor="#f0f2f5",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_heatmap_gender, use_container_width=True)
        
        # 5. Age-Country Heatmap
        st.markdown("<p class='chart-title'>Age-Country Distribution</p>", unsafe_allow_html=True)
        heatmap_country = filtered_df.groupby(['age', 'location'])['Rate'].mean().reset_index()
        heatmap_pivot_country = heatmap_country.pivot(index='age', columns='location', values='Rate')
        fig_heatmap_country = px.imshow(
            heatmap_pivot_country, 
            labels=dict(x="Country", y="Age Group", color="Burden Rate"),
            color_continuous_scale=px.colors.sequential.Bluered,
            aspect='auto'
        )
        fig_heatmap_country.update_layout(
            height=500,
            title_text="Age-Country Distribution of Stroke Burden",
            plot_bgcolor="#f0f2f5",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_heatmap_country, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Key Insights
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.header("Key Demographic Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Most affected age group
            most_affected_age = age_data.loc[age_data['Rate'].idxmax()]
            st.info(f"🔍 Most Affected Age Group\n\n"
                    f"**Age Group:** {most_affected_age['age']}\n"
                    f"**Average Burden Rate:** {most_affected_age['Rate']:.2f}")

            # Gender comparison
            if len(gender_data) > 1:
                more_affected_gender = gender_data.loc[gender_data['Rate'].idxmax()]
                less_affected_gender = gender_data.loc[gender_data['Rate'].idxmin()]
                gender_diff_pct = (more_affected_gender['Rate'] - less_affected_gender['Rate']) / less_affected_gender['Rate'] * 100
                st.info(f"⚖️ Gender Comparison\n\n"
                        f"**{more_affected_gender['sex']}** are {gender_diff_pct:.2f}% more affected than **{less_affected_gender['sex']}**\n"
                        f"**{more_affected_gender['sex']} Rate:** {more_affected_gender['Rate']:.2f}\n"
                        f"**{less_affected_gender['sex']} Rate:** {less_affected_gender['Rate']:.2f}")
            else:
                st.info("⚖️ Gender Comparison\n\nNot available (only one gender selected)")

        with col2:
            # Country comparison
            most_affected_country = country_data.loc[country_data['Rate'].idxmax()]
            least_affected_country = country_data.loc[country_data['Rate'].idxmin()]
            st.info(f"🌍 Country Comparison\n\n"
                    f"**Highest burden:** {most_affected_country['location']} ({most_affected_country['Rate']:.2f})\n"
                    f"**Lowest burden:** {least_affected_country['location']} ({least_affected_country['Rate']:.2f})")

            # Overall statistics
            overall_avg = filtered_df['Rate'].mean()
            overall_max = filtered_df['Rate'].max()
            overall_min = filtered_df['Rate'].min()
            st.info(f"📊 Overall Statistics\n\n"
                    f"**Average Rate:** {overall_avg:.2f}\n"
                    f"**Highest Rate:** {overall_max:.2f}\n"
                    f"**Lowest Rate:** {overall_min:.2f}")

        st.markdown("</div>", unsafe_allow_html=True)

        # Download button for filtered data
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Demographic Analysis Data",
            data=csv,
            file_name=f"demographic_analysis_{selected_year}.csv",
            mime="text/csv",
            key="demographic_analysis_download_button"
        )
elif selected_section == "Risk Factor Analysis":
    st.title("Risk Factor Analysis for Stroke Burden")
    
    # Filters for this section
    countries = df['location'].unique()
    selected_countries = st.multiselect("Select Countries", countries, default=countries[:3])
    
    years = df['year'].dropna().unique().astype(int)
    year_range = st.slider("Select Year Range", int(years.min()), int(years.max()), (int(years.min()), int(years.max())))
    
    measures = df['measure'].unique()
    selected_measure = st.selectbox("Select Measure", measures)
    
    # Filter data
    filtered_df = filter_data(df, selected_countries, year_range, [selected_measure], df['sex'].unique(), df['age'].unique(), df['rei'].unique())
    
    # Ensure correct aggregation: Use the mean for Rate
    risk_df = filtered_df.pivot_table(values='Rate', index='location', columns='rei', aggfunc='mean')
    
    # Visualize with a heatmap
    fig = px.imshow(risk_df, 
                    labels=dict(x="Risk Factor", y="Country", color="Burden Rate"),
                    title=f'Heatmap of Risk Factors Impact on {selected_measure}',
                    color_continuous_scale=px.colors.sequential.YlOrRd)
    
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Radar chart for mean impact of risk factors
    risk_mean = risk_df.mean().reset_index()
    risk_mean.columns = ['Risk Factor', 'Mean Impact']
    
    fig = go.Figure(data=go.Scatterpolar(
      r=risk_mean['Mean Impact'],
      theta=risk_mean['Risk Factor'],
      fill='toself',
      line=dict(color='rgb(31, 119, 180)')
    ))

    fig.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=True,
          range=[0, risk_mean['Mean Impact'].max()]
        )),
      showlegend=False,
      title=f'Average Impact of Risk Factors on {selected_measure}'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add insights
    st.subheader("Key Insights")
    top_risk_factor = risk_mean.loc[risk_mean['Mean Impact'].idxmax()]
    st.write(f"1. The risk factor with the highest average impact on {selected_measure} is '{top_risk_factor['Risk Factor']}' with a mean impact of {top_risk_factor['Mean Impact']:.2f}.")
    
    country_top_risks = risk_df.idxmax(axis=1)
    most_common_risk = country_top_risks.value_counts().index[0]
    st.write(f"2. The most common top risk factor across countries for {selected_measure} is '{most_common_risk}'.")
    
    risk_variability = risk_df.std() / risk_df.mean()
    most_variable_risk = risk_variability.idxmax()
    st.write(f"3. The risk factor with the most variability across countries for {selected_measure} is '{most_variable_risk}'.")

elif selected_section == "Statistical Tests":
    st.title("Statistical Tests and Advanced Analysis")
    
    st.write("This section provides more in-depth statistical analysis of the stroke burden data.")
    
    # Filters for this section
    countries = df['location'].unique()
    selected_countries = st.multiselect("Select Countries", countries, default=countries[:3])
    
    years = df['year'].dropna().unique().astype(int)
    year_range = st.slider("Select Year Range", int(years.min()), int(years.max()), (int(years.min()), int(years.max())))
    
    measures = df['measure'].unique()
    selected_measure = st.selectbox("Select Measure", measures)
    
    # Filter data
    filtered_df = filter_data(df, selected_countries, year_range, [selected_measure], df['sex'].unique(), df['age'].unique(), df['rei'].unique())
    
    # T-test for gender differences
    male_data = filtered_df[filtered_df['sex'] == 'Male']['Rate']
    female_data = filtered_df[filtered_df['sex'] == 'Female']['Rate']
    t_stat, p_value = stats.ttest_ind(male_data, female_data)
    
    st.subheader("1. T-test for Gender Differences")
    st.write(f"T-statistic: {t_stat:.4f}")
    st.write(f"P-value: {p_value:.4f}")
    if p_value < 0.05:
        st.write("There is a statistically significant difference in stroke burden between males and females.")
    else:
        st.write("There is no statistically significant difference in stroke burden between males and females.")
    
    # ANOVA for age group differences
    age_groups = filtered_df['age'].unique()
    age_data = [filtered_df[filtered_df['age'] == age]['Rate'] for age in age_groups]
    f_stat, p_value = stats.f_oneway(*age_data)
    
    st.subheader("2. ANOVA for Age Group Differences")
    st.write(f"F-statistic: {f_stat:.4f}")
    st.write(f"P-value: {p_value:.4f}")
    if p_value < 0.05:
        st.write("There are statistically significant differences in stroke burden among age groups.")
    else:
        st.write("There are no statistically significant differences in stroke burden among age groups.")
    
    # Correlation matrix of risk factors
    st.subheader("3. Correlation Matrix of Risk Factors")
    risk_corr = filtered_df.pivot_table(values='Rate', index='location', columns='rei', aggfunc='mean').corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(risk_corr, annot=True, cmap='coolwarm', ax=ax)
    plt.title("Correlation Matrix of Risk Factors")
    st.pyplot(fig)
    
    # Top correlated risk factors
    top_corr = risk_corr.unstack().sort_values(ascending=False).drop_duplicates()
    top_corr = top_corr[top_corr != 1].head(5)
    
    st.write("Top 5 correlated risk factors:")
    for (factor1, factor2), corr in top_corr.items():
        st.write(f"- {factor1} and {factor2}: {corr:.4f}")

# Footer
st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #0E1117;
    color: white;
    text-align: center;
    padding: 10px;
    font-size: 14px;
}
</style>
<div class="footer">
    Stroke Analysis Dashboard | Created with Streamlit | Data source: IHME
</div>
""", unsafe_allow_html=True)

# Add a download button for the filtered data
@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')

csv = convert_df(filtered_df)

st.download_button(
    label="Download filtered data as CSV",
    data=csv,
    file_name="filtered_stroke_data.csv",
    mime="text/csv",
)

# Add a section for user feedback
st.header("Feedback")
feedback = st.text_area("Please provide any feedback or suggestions for improving this dashboard:")
if st.button("Submit Feedback"):
    # In a real application, you would save this feedback to a database or file
    st.success("Thank you for your feedback!")

# Add a section for additional resources
st.header("Additional Resources")
st.markdown("""
- [World Health Organization - Stroke](https://www.who.int/health-topics/stroke)
- [American Stroke Association](https://www.stroke.org/)
- [National Stroke Association](https://www.stroke.org/)
- [Global Burden of Disease Study](http://www.healthdata.org/gbd)
""")

# Add a disclaimer
st.markdown("---")
st.markdown("""
Disclaimer: This dashboard is for informational purposes only. The data presented here should not be used for medical diagnosis or treatment. Always consult with a qualified healthcare provider for medical advice.
""")

# Main function to run the app
def main():
    st.sidebar.markdown("---")
    st.sidebar.info("Created by [Your Name/Organization]")
    st.sidebar.text("Version 1.0")

if __name__ == "__main__":
    main()
