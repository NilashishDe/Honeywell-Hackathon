import streamlit as st
import pandas as pd
import joblib
import spacy
import re
import plotly.express as px
from streamlit_extras.stylable_container import stylable_container

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Enhanced CSS with modern styling and gradients
css='''
/* ===== Background Gradient ===== */
.stApp {
    background: linear-gradient(135deg, #2c2c2c, #000000);
    color: white !important;
    font-family: 'Poppins', sans-serif;
}

/* ===== Input Textbox ===== */
.stTextInput > div > div > input {
    background-color: rgba(255, 255, 255, 0.15);
    border: 1px solid rgba(255, 255, 255, 0.25);
    color: white !important;
    border-radius: 12px;
    padding: 10px;
}
.stTextInput > div > div > input:focus {
    border: 1px solid #00c6ff;
    box-shadow: 0 0 10px rgba(0, 198, 255, 0.6);
}

/* ===== Buttons with Gradient ===== */
.stButton > button {
    background: linear-gradient(135deg, #00c6ff, #0072ff);
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 12px 24px;
    font-size: 16px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease-in-out;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0072ff, #00c6ff);
    transform: scale(1.05);
    box-shadow: 0 4px 12px rgba(0,0,0,0.25);
}
div[data-testid="stPlotlyChart"],
div[data-testid="stAltairChart"],
div[data-testid="stVegaLiteChart"],
div[data-testid="stPyplotChart"],
div[data-testid="stDeckGlChart"] {
  background: linear-gradient(180deg, #1f1f1f, #141414) !important; /* pleasant dark card */
  border-radius: 14px !important;      /* space from neighbors */
  border: 1px solid rgba(255,255,255,0.06) !important;
  box-shadow: 0 8px 24px rgba(0,0,0,0.28) !important;
}

/* Smoothen the inner plot container corners so it matches the card */
div[data-testid="stPlotlyChart"] > div,
div[data-testid="stAltairChart"] > div,
div[data-testid="stVegaLiteChart"] > div,
div[data-testid="stPyplotChart"] > div,
div[data-testid="stDeckGlChart"] > div {
  border-radius: 10px !important;
  overflow: hidden !important;
}

/* Subtle hover lift */
div[data-testid*="Chart"]:hover {
  transform: translateY(-2px);
  transition: transform .2s ease, box-shadow .2s ease;
  box-shadow: 0 10px 28px rgba(0,0,0,0.34) !important;
}
'''

st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
# --- Load Assets ---
# Set the page configuration for the web app
st.set_page_config(page_title="AeroPredict NLP", layout="wide")

# Use Streamlit's caching to load data and models only once, improving performance
@st.cache_data
def load_data():
    """Loads the cleaned flight data, caching it for performance."""
    df = pd.read_csv('cleaned_flight_data.csv')
    return df

@st.cache_resource
def load_models():
    """Loads the NLP and ML models, caching them for performance."""
    nlp = spacy.load("en_core_web_sm")
    model = joblib.load('flight_delay_model.joblib')
    return nlp, model

try:
    # Load all necessary files when the app starts
    df = load_data()
    nlp, model = load_models()

    # Create lists of known entities for the NLP parser
    KNOWN_DESTINATIONS = df['To'].unique()
    KNOWN_AIRCRAFT_MODELS = df['Aircraft Model'].unique()
    KNOWN_DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # --- Initialize Session State ---
    # This is crucial to "remember" the last prediction's details
    if 'last_prediction_entities' not in st.session_state:
        st.session_state.last_prediction_entities = None

    # --- NLP Parsing Function ---
    def parse_prompt_with_nlp(prompt, is_booking_query=False):
        """Extracts flight details from a natural language prompt."""
        prompt = prompt.lower()
        entities = {"From": None, "To": None, "day_of_week": None, "time_of_day": None}
        
        # For standard prediction, we need different entities
        if not is_booking_query:
            entities = {"To": None, "Aircraft Model": None, "day_of_week": None, "scheduled_hour": None}

        # --- Entity Extraction Logic ---
        # Simple keyword matching for origin and destination
        all_cities = pd.concat([df['From'], df['To']]).unique()
        for city in all_cities:
            city_name = city.split('(')[0].strip().lower()
            if f"from {city_name}" in prompt:
                entities["From"] = city
            if f"to {city_name}" in prompt:
                entities["To"] = city

        for day in KNOWN_DAYS:
            if day.lower() in prompt:
                entities["day_of_week"] = day
                break
        
        if is_booking_query:
            if "morning" in prompt: entities["time_of_day"] = "Morning"
            elif "afternoon" in prompt: entities["time_of_day"] = "Afternoon"
            elif "evening" in prompt: entities["time_of_day"] = "Evening"
        else:
            for craft_model in KNOWN_AIRCRAFT_MODELS:
                if craft_model.lower() in prompt:
                    entities["Aircraft Model"] = craft_model
                    break
            match = re.search(r'(\d{1,2})\s*(am|pm)?', prompt)
            if match:
                hour = int(match.group(1))
                meridiem = match.group(2)
                if meridiem == 'pm' and hour < 12: hour += 12
                entities["scheduled_hour"] = hour
        
        return entities

    # --- Streamlit UI ---
    st.title('✈️ AeroPredict: Conversational Flight Delay Forecaster')
    st.write("Ask about your flight or for insights. Example: **'What is the best time to take off to avoid delays?'**")

    prompt = st.text_input("Enter your flight query here:", "")

    # --- Combined Logic for Quick Insights and Prediction ---
    prompt_lower = prompt.lower()
    is_analytical_query = False

    if ("highest delay" in prompt_lower or "worst time" in prompt_lower) and "time" in prompt_lower:
        is_analytical_query = True
        avg_delay_per_hour = df.groupby('scheduled_hour')['delay_minutes'].mean()
        hour_with_highest_delay = avg_delay_per_hour.idxmax()
        max_delay_value = avg_delay_per_hour.max()
        st.error(f"The **worst time** to fly is **{hour_with_highest_delay}:00 ({hour_with_highest_delay % 12 or 12} {'AM' if hour_with_highest_delay < 12 else 'PM'})**, with the highest average delay of **{max_delay_value:.0f} minutes**.")

    elif ("lowest delay" in prompt_lower or "best time" in prompt_lower) and "time" in prompt_lower:
        is_analytical_query = True
        avg_delay_per_hour = df.groupby('scheduled_hour')['delay_minutes'].mean()
        hour_with_lowest_delay = avg_delay_per_hour.idxmin()
        min_delay_value = avg_delay_per_hour.min()
        st.success(f"The **best time** to fly is **{hour_with_lowest_delay}:00 ({hour_with_lowest_delay % 12 or 12} {'AM' if hour_with_lowest_delay < 12 else 'PM'})**, with the lowest average delay of just **{min_delay_value:.0f} minutes**.")

    if not is_analytical_query:
        if st.button('Predict from Prompt'):
            if prompt:
                entities = parse_prompt_with_nlp(prompt, is_booking_query=False)
                st.session_state.last_prediction_entities = entities
            else:
                st.warning("Please enter a query.")
                st.session_state.last_prediction_entities = None

    if st.session_state.last_prediction_entities:
        entities = st.session_state.last_prediction_entities
        st.subheader("🔍 What I Understood:")
        if all(entities.values()):
            st.success(f"**Destination:** {entities['To']} | **Aircraft Model:** {entities['Aircraft Model']} | **Day:** {entities['day_of_week']} | **Hour:** {entities['scheduled_hour']}")
            input_data = pd.DataFrame({
                'From': ['Mumbai (BOM)'], 'To': [entities['To']], 'Aircraft Model': [entities['Aircraft Model']],
                'day_of_week': [entities['day_of_week']], 'scheduled_hour': [entities['scheduled_hour']]
            })
            delay_probability = model.predict_proba(input_data)[0][1]
            st.subheader("📈 Prediction Result")
            if delay_probability > 0.5:
                st.error(f"High risk of delay! Model predicts a {delay_probability:.0%} probability of significant delay.")
            else:
                st.success(f"Likely on time! Model predicts only a {delay_probability:.0%} probability of significant delay.")
            with st.expander("🎲 Explore 'What-If' Scenarios"):
                with st.form(key='scenario_form'):
                    scenario_day = st.selectbox("Change Day", options=KNOWN_DAYS, index=KNOWN_DAYS.index(entities['day_of_week']))
                    scenario_hour = st.slider("Change Hour", 0, 23, value=entities['scheduled_hour'])
                    scenario_aircraft = st.selectbox("Change Aircraft Model", options=sorted(df['Aircraft Model'].unique()), index=sorted(df['Aircraft Model'].unique()).index(entities['Aircraft Model']))
                    submitted = st.form_submit_button("Run Scenario")
                    if submitted:
                        scenario_input_data = pd.DataFrame({
                            'From': ['Mumbai (BOM)'], 'To': [entities['To']], 'Aircraft Model': [scenario_aircraft],
                            'day_of_week': [scenario_day], 'scheduled_hour': [scenario_hour]
                        })
                        scenario_prob = model.predict_proba(scenario_input_data)[0][1]
                        st.markdown(f"##### Scenario Result: Flying on a **{scenario_day}** at **{scenario_hour}:00** with a **{scenario_aircraft}**")
                        if scenario_prob > 0.5: st.error(f"The delay risk changes to **{scenario_prob:.0%}**.")
                        else: st.success(f"The delay risk changes to **{scenario_prob:.0%}**.")
        else:
             st.error("I couldn't understand all the details. Please include a destination, aircraft model, day, and time.")
             st.write("Details Found:", {k: v for k, v in entities.items() if v is not None})

    st.markdown("---")
    
    st.subheader("🔬 Advanced Airport Analytics")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Flight Recommender", "Aircraft Reliability", "Route Scorecard", "Busiest & Best Times", "Top Delaying Flights", "Data Insights"])

    # --- Flight Recommender Feature ---
    with tab1:
        st.markdown("#### Find the Best Flight to Book")
        st.write("Enter your travel plans below to find the most reliable flight options based on historical data.")
        booking_prompt = st.text_input("Enter your travel plans (e.g., 'I want to fly from Mumbai to Delhi on a Friday morning')", key="booking_prompt")
        
        if st.button("Find Best Flight"):
            if booking_prompt:
                booking_entities = parse_prompt_with_nlp(booking_prompt, is_booking_query=True)
                
                if not all([booking_entities["From"], booking_entities["To"], booking_entities["day_of_week"], booking_entities["time_of_day"]]):
                    st.error("Please provide a full query including departure city, arrival city, day, and time of day (morning, afternoon, or evening).")
                else:
                    st.success(f"Searching for flights from **{booking_entities['From']}** to **{booking_entities['To']}** on **{booking_entities['day_of_week']} {booking_entities['time_of_day']}**...")
                    
                    # Filter data based on query
                    time_filters = {
                        "Morning": (df['scheduled_hour'] >= 5) & (df['scheduled_hour'] < 12),
                        "Afternoon": (df['scheduled_hour'] >= 12) & (df['scheduled_hour'] < 17),
                        "Evening": (df['scheduled_hour'] >= 17) & (df['scheduled_hour'] <= 23)
                    }
                    
                    filtered_df = df[
                        (df['From'] == booking_entities["From"]) &
                        (df['To'] == booking_entities["To"]) &
                        (df['day_of_week'] == booking_entities["day_of_week"]) &
                        (time_filters[booking_entities["time_of_day"]])
                    ]

                    if filtered_df.empty:
                        st.warning("No historical data found for this specific search. Try a different day or time.")
                    else:
                        # Analyze and recommend best flight slots with more detail
                        recommendations = filtered_df.groupby(['STA', 'Aircraft Model', 'Tail Number']).agg(
                            on_time_percentage=('is_delayed', lambda x: (1 - x.mean()) * 100),
                            avg_delay=('delay_minutes', 'mean'),
                            total_flights=('is_delayed', 'count')
                        ).reset_index().sort_values(by='on_time_percentage', ascending=False)
                        
                        st.markdown("##### Recommended Flight Slots")
                        st.dataframe(recommendations)

    with tab2:
        st.markdown("#### Aircraft Model Reliability Report")
        selected_aircraft_model = st.selectbox("Select Aircraft Model", options=sorted(df['Aircraft Model'].unique()))
        if st.button("Generate Aircraft Report"):
            aircraft_df = df[df['Aircraft Model'] == selected_aircraft_model]
            if not aircraft_df.empty:
                on_time_percentage = (1 - aircraft_df['is_delayed'].mean()) * 100
                avg_delay = aircraft_df['delay_minutes'].mean()
                aircraft_df['Route'] = aircraft_df['From'] + ' → ' + aircraft_df['To']
                busiest_routes = aircraft_df['Route'].value_counts().nlargest(5).index.tolist()
                st.markdown(f"##### Performance for **{selected_aircraft_model}**")
                report_col1, report_col2 = st.columns(2)
                with report_col1:
                    st.metric(label="On-Time Percentage", value=f"{on_time_percentage:.1f}%")
                    st.metric(label="Average Delay (minutes)", value=f"{avg_delay:.0f} min")
                with report_col2:
                    st.markdown("**Most Frequent Routes:**")
                    for route in busiest_routes: st.write(f"- {route}")

    with tab3:
        st.markdown("#### On-Time Performance Route Scorecard")
        col1, col2 = st.columns(2)
        with col1: from_city = st.selectbox("Select Departure City", options=sorted(df['From'].unique()))
        with col2: to_city = st.selectbox("Select Arrival City", options=sorted(df['To'].unique()))
        if st.button("Generate Route Scorecard"):
            route_df = df[(df['From'] == from_city) & (df['To'] == to_city)]
            if not route_df.empty:
                on_time_percentage = (1 - route_df['is_delayed'].mean()) * 100
                avg_delay = route_df['delay_minutes'].mean()
                daily_performance = route_df.groupby('day_of_week')['is_delayed'].mean().sort_values()
                best_day, worst_day = daily_performance.index[0], daily_performance.index[-1]
                st.markdown(f"##### Performance for **{from_city} → {to_city}**")
                score_col1, score_col2 = st.columns(2)
                with score_col1:
                    st.metric(label="On-Time Percentage", value=f"{on_time_percentage:.1f}%")
                    st.metric(label="Best Day to Fly", value=best_day)
                with score_col2:
                    st.metric(label="Average Delay (minutes)", value=f"{avg_delay:.0f} min")
                    st.metric(label="Worst Day to Fly", value=worst_day)

    with tab4:
        st.markdown("#### Busiest Time Slot & Best Time to Takeoff")
        col1, col2 = st.columns(2)
        with col1:
            flights_per_hour = df['scheduled_hour'].value_counts().sort_index()
            fig_busy = px.bar(flights_per_hour, title='Busiest Times: Flights per Hour', labels={'index': 'Hour', 'value': '# of Flights'})
            st.plotly_chart(fig_busy, use_container_width=True)
        with col2:
            avg_delay_per_hour = df.groupby('scheduled_hour')['delay_minutes'].mean().sort_index()
            fig_best = px.bar(avg_delay_per_hour, title='Best Times: Average Delay per Hour', labels={'index': 'Hour', 'value': 'Avg. Delay (min)'})
            st.plotly_chart(fig_best, use_container_width=True)

    with tab5:
        st.markdown("#### Flights with Biggest Cascading Impact")
        st.info("Isolating flights with the longest delays, as these are most likely to cause downstream disruptions.")
        top_delayed_flights = df.sort_values(by='delay_minutes', ascending=False).head(10)
        top_delayed_flights = top_delayed_flights[['Date', 'From', 'To', 'Aircraft Model', 'Tail Number', 'STA', 'delay_minutes']]
        top_delayed_flights['delay_minutes'] = top_delayed_flights['delay_minutes'].round(0).astype(int)
        st.dataframe(top_delayed_flights, use_container_width=True)

    with tab6:
        st.markdown("#### General Data Insights")
        col1, col2 = st.columns(2)
        with col1:
            status_counts = df['is_delayed'].value_counts().rename({0: 'On-Time', 1: 'Delayed'})
            fig_pie = px.pie(status_counts, values='count', names=status_counts.index, title='Overall Flight Status', color=status_counts.index, color_discrete_map={'On-Time':'green', 'Delayed':'red'})
            st.plotly_chart(fig_pie, use_container_width=True)
            delayed_flights = df[df['is_delayed'] == 1]
            aircraft_delay_counts = delayed_flights['Aircraft Model'].value_counts().nlargest(5)
            fig_aircraft = px.bar(aircraft_delay_counts, title='Top 5 Aircraft Models by Delays', labels={'index': 'Aircraft Model', 'value': '# of Delays'})
            st.plotly_chart(fig_aircraft, use_container_width=True)
        with col2:
            df['Route'] = df['From'].str.split('(').str[0].str.strip() + ' → ' + df['To'].str.split('(').str[0].str.strip()
            route_delays = df.groupby('Route')['delay_minutes'].mean().nlargest(5)
            fig_route = px.bar(route_delays, title='Top 5 Routes by Avg. Delay', labels={'index': 'Route', 'value': 'Avg. Delay (min)'})
            st.plotly_chart(fig_route, use_container_width=True)
            day_delay_perc = (df.groupby('day_of_week')['is_delayed'].mean() * 100).sort_values(ascending=False)
            fig_day = px.bar(day_delay_perc, title='% of Flights Delayed by Day', labels={'index': 'Day', 'value': '% Delayed'})
            st.plotly_chart(fig_day, use_container_width=True)

except FileNotFoundError:
    st.error("Error: The 'cleaned_flight_data.csv' or 'flight_delay_model.joblib' file was not found.")
    st.warning("Please run the 'train_model.py' script first to generate the necessary files.")

