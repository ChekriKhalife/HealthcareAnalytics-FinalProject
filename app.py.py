import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Load your data
df = pd.read_csv("C:\\Users\\Chekri\\Desktop\\Healthcare analytics\Stroke_data.csv")

# Convert the 'year' column to datetime and extract only the year
df['year'] = pd.to_datetime(df['year'], errors='coerce').dt.year

# Set page config
st.set_page_config(page_title="Stroke Analysis Dashboard", layout="wide")

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
    }
    .Widget>label {
        color: #31333F;
        font-weight: bold;
    }
    .stSelectbox [data-baseweb="select"] {
        background-color: #ffffff;
    }
    .stAlert {
        background-color: #f0f2f6;
        color: #31333F;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 20px;
    }
    .css-1aumxhk {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

def create_card(title, content):
    st.markdown(f"""
    <div style="
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;">
        <h3 style="color: #31333F;">{title}</h3>
        {content}
    </div>
    """, unsafe_allow_html=True)

def filter_data(df, countries, year_range, measure, sex, age_groups):
    return df[
        (df['location'].isin(countries)) &
        (df['year'].between(year_range[0], year_range[1])) &
        (df['measure'] == measure) &
        (df['sex'].isin(sex)) &
        (df['age'].isin(age_groups))
    ]

# Sidebar
st.sidebar.title("Stroke Analysis Dashboard")

# Sidebar filters
selected_section = st.sidebar.radio("Select Section", 
    ["Overview", "Temporal Trends", "Demographic Analysis", "Risk Factor Analysis", 
     "Comparative Analysis", "Age-specific Analysis", "Forecasting", "Risk Factor Impact Simulator", "Statistical Tests"])

countries = df['location'].unique()
selected_countries = st.sidebar.multiselect("Select Countries", countries, default=countries[:3])

years = df['year'].unique()
year_range = st.sidebar.slider("Select Year Range", int(years.min()), int(years.max()), (int(years.min()), int(years.max())))

measures = df['measure'].unique()
selected_measure = st.sidebar.selectbox("Select Measure", measures)

sexes = df['sex'].unique()
selected_sex = st.sidebar.multiselect("Select Sex", sexes, default=sexes)

age_groups = df['age'].unique()
selected_age_groups = st.sidebar.multiselect("Select Age Groups", age_groups, default=age_groups[:5])

# Filter data based on sidebar selections
filtered_df = filter_data(df, selected_countries, year_range, selected_measure, selected_sex, selected_age_groups)

# Main content
if selected_section == "Overview":
    st.title("Stroke Analysis Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        create_card("Total Deaths", f"{filtered_df[filtered_df['measure'] == 'Deaths']['Rate'].sum():,.0f}")
    
    with col2:
        create_card("Total YLDs", f"{filtered_df[filtered_df['measure'] == 'YLDs']['Rate'].sum():,.0f}")
    
    with col3:
        create_card("Countries Analyzed", f"{len(selected_countries)}")
    
    # Create a choropleth map
    fig = px.choropleth(filtered_df, 
                        locations='location', 
                        color='Rate',
                        hover_name='location', 
                        animation_frame='year',
                        color_continuous_scale=px.colors.sequential.Plasma)
    
    fig.update_layout(title_text='Stroke Burden by Country', title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)
    
    # Add insights
    st.subheader("Key Insights")
    country_rates = filtered_df.groupby('location')['Rate'].mean().sort_values(ascending=False)
    st.write(f"1. {country_rates.index[0]} has the highest average stroke burden rate of {country_rates.iloc[0]:.2f}.")
    st.write(f"2. {country_rates.index[-1]} has the lowest average stroke burden rate of {country_rates.iloc[-1]:.2f}.")
    
    yearly_rates = filtered_df.groupby('year')['Rate'].mean()
    trend = "increasing" if yearly_rates.iloc[-1] > yearly_rates.iloc[0] else "decreasing"
    st.write(f"3. The overall trend in stroke burden is {trend} over the selected time period.")

elif selected_section == "Temporal Trends":
    st.title("Temporal Trends in Stroke Burden")
    
    temporal_df = filtered_df.groupby(['year', 'measure'])['Rate'].sum().reset_index()
    
    fig = px.line(temporal_df, x='year', y='Rate', color='measure',
                  title='Stroke Burden Over Time',
                  labels={'Rate': 'Burden Rate', 'year': 'Year'},
                  color_discrete_sequence=px.colors.qualitative.Set1)
    
    fig.update_layout(legend_title_text='Measure')
    st.plotly_chart(fig, use_container_width=True)
    
    # Add year-over-year change
    yoy_change = temporal_df.set_index('year').groupby('measure')['Rate'].pct_change().reset_index()
    fig_yoy = px.bar(yoy_change, x='year', y='Rate', color='measure',
                     title='Year-over-Year Change in Stroke Burden',
                     labels={'Rate': 'YoY Change (%)', 'year': 'Year'},
                     color_discrete_sequence=px.colors.qualitative.Set2)
    
    fig_yoy.update_layout(legend_title_text='Measure')
    st.plotly_chart(fig_yoy, use_container_width=True)
    
    # Add insights
    st.subheader("Key Insights")
    max_increase = yoy_change.loc[yoy_change['Rate'].idxmax()]
    max_decrease = yoy_change.loc[yoy_change['Rate'].idxmin()]
    st.write(f"1. The largest increase in stroke burden was {max_increase['Rate']:.2%} for {max_increase['measure']} in {max_increase['year']}.")
    st.write(f"2. The largest decrease in stroke burden was {max_decrease['Rate']:.2%} for {max_decrease['measure']} in {max_decrease['year']}.")
    
    trend = temporal_df.groupby('measure').apply(lambda x: stats.linregress(x['year'], x['Rate']).slope)
    for measure, slope in trend.items():
        trend_direction = "increasing" if slope > 0 else "decreasing"
        st.write(f"3. The overall trend for {measure} is {trend_direction} with a slope of {slope:.4f} per year.")

elif selected_section == "Demographic Analysis":
    st.title("Demographic Analysis of Stroke Burden")
    
    demo_df = filtered_df.groupby(['age', 'sex'])['Rate'].sum().reset_index()
    
    fig = px.bar(demo_df, x='age', y='Rate', color='sex', barmode='group',
                 title='Stroke Burden by Age and Sex',
                 labels={'Rate': 'Burden Rate', 'age': 'Age Group', 'sex': 'Sex'},
                 color_discrete_sequence=px.colors.qualitative.Set2)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Pie chart
    country_df = filtered_df.groupby('location')['Rate'].sum().reset_index()
    fig = px.pie(country_df, values='Rate', names='location',
                 title='Distribution of Stroke Burden by Country',
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add insights
    st.subheader("Key Insights")
    most_affected_age = demo_df.loc[demo_df['Rate'].idxmax()]
    st.write(f"1. The most affected demographic group is {most_affected_age['sex']} in the {most_affected_age['age']} age group.")
    
    gender_diff = demo_df.groupby('sex')['Rate'].sum()
    more_affected_gender = gender_diff.idxmax()
    gender_diff_pct = (gender_diff[more_affected_gender] - gender_diff[gender_diff.idxmin()]) / gender_diff[gender_diff.idxmin()] * 100
    st.write(f"2. {more_affected_gender} are {gender_diff_pct:.2f}% more affected by stroke burden overall.")
    
    country_burden = country_df.set_index('location')['Rate']
    st.write(f"3. {country_burden.idxmax()} has the highest total stroke burden, while {country_burden.idxmin()} has the lowest.")

elif selected_section == "Risk Factor Analysis":
    st.title("Risk Factor Analysis for Stroke Burden")
    
    risk_df = filtered_df.pivot_table(values='Rate', index='location', columns='rei', aggfunc='sum')
    
    fig = px.imshow(risk_df, aspect="auto",
                    title='Heatmap of Risk Factors Impact on Stroke Burden',
                    color_continuous_scale=px.colors.sequential.YlOrRd)
    
    fig.update_layout(xaxis_title="Risk Factors", yaxis_title="Countries")
    st.plotly_chart(fig, use_container_width=True)
    
    # Radar chart
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
      title='Average Impact of Risk Factors on Stroke Burden'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add insights
    st.subheader("Key Insights")
    top_risk_factor = risk_mean.loc[risk_mean['Mean Impact'].idxmax()]
    st.write(f"1. The risk factor with the highest average impact is '{top_risk_factor['Risk Factor']}' with a mean impact of {top_risk_factor['Mean Impact']:.2f}.")
    
    country_top_risks = risk_df.idxmax(axis=1)
    most_common_risk = country_top_risks.value_counts().index[0]
    st.write(f"2. The most common top risk factor across countries is '{most_common_risk}'.")
    
    risk_variability = risk_df.std() / risk_df.mean()
    most_variable_risk = risk_variability.idxmax()
    st.write(f"3. The risk factor with the most variability across countries is '{most_variable_risk}'.")

elif selected_section == "Comparative Analysis":
    st.title("Comparative Analysis of Stroke Burden")
    
    comp_df = filtered_df.groupby(['location', 'measure'])['Rate'].sum().unstack()
    
    fig = px.scatter(comp_df, x='Deaths', y='YLDs',
                     hover_name=comp_df.index, 
                     text=comp_df.index,
                     title='Deaths vs YLDs by Country',
                     labels={'Deaths': 'Death Rate', 'YLDs': 'YLDs Rate'},
                     color_discrete_sequence=px.colors.qualitative.Bold)
    
    fig.update_traces(textposition='top center')
    fig.update_layout(xaxis_title="Death Rate", yaxis_title="YLDs Rate")
    st.plotly_chart(fig, use_container_width=True)
    
    # Add correlation analysis
    correlation = comp_df['Deaths'].corr(comp_df['YLDs'])
    st.write(f"Correlation between Deaths and YLDs: {correlation:.2f}")
    
    # Add insights
    st.subheader("Key Insights")
    highest_death_rate = comp_df.loc[comp_df['Deaths'].idxmax()]
    highest_yld_rate = comp_df.loc[comp_df['YLDs'].idxmax()]
    st.write(f"1. {highest_death_rate.name} has the highest death rate at {highest_death_rate['Deaths']:.2f}.")
    st.write(f"2. {highest_yld_rate.name} has the highest YLDs rate at {highest_yld_rate['YLDs']:.2f}.")
    
    if correlation > 0.7:
        st.write("3. There is a strong positive correlation between death rates and YLDs rates across countries.")
    elif correlation > 0.3:
        st.write("3. There is a moderate positive correlation between death rates and YLDs rates across countries.")
    else:
        st.write("3. There is a weak correlation between death rates and YLDs rates across countries.")

elif selected_section == "Age-specific Analysis":
    st.title("Age-specific Analysis of Stroke Burden")
    
    age_df = filtered_df.groupby(['age', 'sex', 'location'])['Rate'].mean().reset_index()
    
    fig = px.line(age_df, x='age', y='Rate', color='sex', facet_col='location', facet_col_wrap=3,
                  title='Age-specific Stroke Burden by Country and Sex',labels={'Rate': 'Average Burden Rate', 'age': 'Age Group'},
                  color_discrete_sequence=px.colors.qualitative.Set1)
    
    fig.update_layout(height=800)
    st.plotly_chart(fig, use_container_width=True)
    
    # Add insights
    st.subheader("Key Insights")
    peak_age = age_df.loc[age_df['Rate'].idxmax()]
    st.write(f"1. The highest stroke burden is observed in the {peak_age['age']} age group for {peak_age['sex']} in {peak_age['location']}.")
    
    age_diff = age_df.groupby('age')['Rate'].mean()
    largest_gap = (age_diff - age_diff.shift()).abs().idxmax()
    st.write(f"2. The largest increase in stroke burden occurs between the {largest_gap} and the previous age group.")
    
    gender_gap = age_df.groupby(['age', 'sex'])['Rate'].mean().unstack()
    max_gender_diff = (gender_gap['Male'] - gender_gap['Female']).abs().idxmax()
    st.write(f"3. The largest gender disparity in stroke burden is observed in the {max_gender_diff} age group.")

elif selected_section == "Forecasting":
    st.title("Stroke Burden Forecasting")
    
    forecast_country = st.selectbox("Select a country for forecasting", selected_countries)
    forecast_df = filtered_df[(filtered_df['location'] == forecast_country) & (filtered_df['measure'] == 'Deaths')]
    forecast_df = forecast_df.groupby('year')['Rate'].sum().reset_index()
    forecast_df['year'] = pd.to_datetime(forecast_df['year'], format='%Y')
    forecast_df = forecast_df.set_index('year')
    
    # ARIMA model
    model = ARIMA(forecast_df['Rate'], order=(1,1,1))
    results = model.fit()
    
    # Forecast
    forecast_years = st.slider("Select number of years to forecast", 1, 10, 5)
    forecast = results.forecast(steps=forecast_years)
    forecast_index = pd.date_range(start=forecast_df.index[-1], periods=forecast_years+1, freq='Y')[1:]
    forecast_df_future = pd.DataFrame(forecast, index=forecast_index, columns=['Forecast'])
    
    # Confidence intervals
    forecast_df_future['Lower CI'] = results.get_forecast(forecast_years).conf_int()['lower Rate']
    forecast_df_future['Upper CI'] = results.get_forecast(forecast_years).conf_int()['upper Rate']
    
    # Plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Rate'],
                             mode='lines', name='Historical', line=dict(color='blue')))
    
    fig.add_trace(go.Scatter(x=forecast_df_future.index, y=forecast_df_future['Forecast'],
                             mode='lines', name='Forecast', line=dict(color='red')))
    
    fig.add_trace(go.Scatter(x=forecast_df_future.index, y=forecast_df_future['Upper CI'],
                             fill=None, mode='lines', line_color='rgba(255,0,0,0.2)', name='Upper CI'))
    
    fig.add_trace(go.Scatter(x=forecast_df_future.index, y=forecast_df_future['Lower CI'],
                             fill='tonexty', mode='lines', line_color='rgba(255,0,0,0.2)', name='Lower CI'))
    
    fig.update_layout(title=f'Stroke Burden Forecast for {forecast_country}',
                      xaxis_title='Year', yaxis_title='Burden Rate',
                      legend_title='Legend')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add insights
    st.subheader("Key Insights")
    forecast_change = (forecast_df_future['Forecast'].iloc[-1] - forecast_df['Rate'].iloc[-1]) / forecast_df['Rate'].iloc[-1] * 100
    trend = "increase" if forecast_change > 0 else "decrease"
    st.write(f"1. The forecast predicts a {abs(forecast_change):.2f}% {trend} in stroke burden for {forecast_country} over the next {forecast_years} years.")
    
    uncertainty = (forecast_df_future['Upper CI'] - forecast_df_future['Lower CI']).mean() / forecast_df_future['Forecast'].mean() * 100
    st.write(f"2. The average uncertainty in the forecast is {uncertainty:.2f}% of the predicted value.")
    
    if forecast_df_future['Forecast'].is_monotonic_increasing:
        st.write(f"3. The forecast suggests a consistently increasing trend in stroke burden for {forecast_country}.")
    elif forecast_df_future['Forecast'].is_monotonic_decreasing:
        st.write(f"3. The forecast suggests a consistently decreasing trend in stroke burden for {forecast_country}.")
    else:
        st.write(f"3. The forecast suggests a fluctuating trend in stroke burden for {forecast_country}.")

elif selected_section == "Risk Factor Impact Simulator":
    st.title("Risk Factor Impact Simulator")
    
    risk_factors = df['rei'].unique()
    
    col1, col2 = st.columns(2)
    
    with col1:
        risk_weights = {}
        for factor in risk_factors:
            risk_weights[factor] = st.slider(f"Impact of {factor}", 0.0, 2.0, 1.0, 0.1)
    
    with col2:
        baseline = filtered_df[filtered_df['year'] == filtered_df['year'].max()]['Rate'].sum()
        simulated = baseline * np.mean(list(risk_weights.values()))
        
        create_card("Baseline Stroke Burden", f"{baseline:,.0f}")
        create_card("Simulated Stroke Burden", f"{simulated:,.0f}")
        
        percent_change = ((simulated - baseline) / baseline) * 100
        change_color = "green" if percent_change < 0 else "red"
        create_card("Percent Change", f"<span style='color:{change_color};'>{percent_change:.2f}%</span>")
    
    # Radar chart of risk factor adjustments
    fig = go.Figure(data=go.Scatterpolar(
      r=list(risk_weights.values()),
      theta=list(risk_weights.keys()),
      fill='toself'
    ))

    fig.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=True,
          range=[0, 2]
        )),
      showlegend=False,
      title='Risk Factor Adjustments'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add insights
    st.subheader("Key Insights")
    most_adjusted = max(risk_weights, key=risk_weights.get)
    least_adjusted = min(risk_weights, key=risk_weights.get)
    st.write(f"1. The most significantly adjusted risk factor is '{most_adjusted}' with a weight of {risk_weights[most_adjusted]}.")
    st.write(f"2. The least adjusted risk factor is '{least_adjusted}' with a weight of {risk_weights[least_adjusted]}.")
    st.write(f"3. The simulated scenario results in a {abs(percent_change):.2f}% {'decrease' if percent_change < 0 else 'increase'} in overall stroke burden.")

elif selected_section == "Statistical Tests":
    st.title("Statistical Tests and Advanced Analysis")
    
    st.write("This section provides more in-depth statistical analysis of the stroke burden data.")
    
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
    risk_corr = filtered_df.pivot_table(values='Rate', index='location', columns='rei', aggfunc='sum').corr()
    
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
    
    # Time series decomposition
    st.subheader("4. Time Series Decomposition")
    country_for_decomposition = st.selectbox("Select a country for time series decomposition", selected_countries)
    ts_data = filtered_df[(filtered_df['location'] == country_for_decomposition) & (filtered_df['measure'] == 'Deaths')]
    ts_data = ts_data.groupby('year')['Rate'].sum()
    
    from statsmodels.tsa.seasonal import seasonal_decompose
    decomposition = seasonal_decompose(ts_data, model='additive', period=1)
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))
    decomposition.observed.plot(ax=ax1)
    ax1.set_title('Observed')
    decomposition.trend.plot(ax=ax2)
    ax2.set_title('Trend')
    decomposition.seasonal.plot(ax=ax3)
    ax3.set_title('Seasonal')
    decomposition.resid.plot(ax=ax4)
    ax4.set_title('Residual')
    plt.tight_layout()
    st.pyplot(fig)
    
    st.write("This decomposition helps visualize the trend, seasonal patterns, and residual noise in the time series data.")

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