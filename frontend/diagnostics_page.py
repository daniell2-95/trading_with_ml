from utilities import load_data, load_mlflow_runs, make_predictions, get_dates
from plotly.subplots import make_subplots
import streamlit as st
import plotly.graph_objects as go


def page_diagnostics():
    """
    Function responsible for executing diagnostics related functions and displays
    """

    runs = load_mlflow_runs()
    
    model_name = st.selectbox(
        "Select model to run diagnostics on:",
        sorted(runs.loc[~runs['tags.best_run'].isna()]['tags.best_run'], reverse = True)
    )

    # Load model features
    features = eval(runs[runs['tags.best_run'] == model_name]['tags.features'].iloc[0])

    # Load data
    spy_data = load_data('SPY')
    dates = get_dates(data = spy_data)

    # Filter data to show performance during training
    current_date = dates.iloc[st.session_state.start_date_index - st.session_state.day_number]
    plot_data = spy_data[spy_data['t'] <= current_date]

    # Remove nulls
    plot_data = plot_data.dropna()
    
    # WRITE MODEL PREDICTION FUNCTION

    # Training predictions
    preds = make_predictions(model_name = model_name, data = plot_data[features])

    # Button to deploy model
    if st.button("Deploy Model?"):
        #st.session_state.current_model = model
        st.session_state.current_model_name = model_name
        st.session_state['features'] = features
        st.session_state.model_class = runs[runs['tags.best_run'] == model_name]['tags.model_class'].iloc[0]

        # Check if there is a model already deployed on a given day
        if st.session_state.model_deployment_history:
            if st.session_state.day_number >= st.session_state.model_deployment_history[-1]["day_deployed"]:
                # Replace same day model in case user changes their mind
                if st.session_state.day_number == st.session_state.model_deployment_history[-1]["day_deployed"]:
                    st.session_state.model_deployment_history.pop()
                # Deploy new model
                st.session_state.model_deployment_history.append(
                    {"day_deployed": st.session_state.day_number, 
                     "model_name": model_name, 
                     "prediction_history": [],
                     "prediction_correctness": []})
        else:
            st.session_state.model_deployment_history.append(
                {"day_deployed": st.session_state.day_number, 
                 "model_name": model_name, 
                 "prediction_history": [],
                 "prediction_correctness": []})

    # Convert value of short predicions to -1 for easier calculation purposes
    preds[preds == 0] = -1

    # Plot returns over time during training
    fig = make_subplots(rows = 1, cols = 1)
    fig.add_trace(go.Scatter(x = plot_data["t"], 
                             y = plot_data['returns'].shift(-1).cumsum(), 
                             name = "Observed"), row = 1, col = 1)
    fig.add_trace(go.Scatter(x = plot_data["t"], 
                             y = (plot_data['returns'].shift(-1) * preds).cumsum(),
                             name = "ML Strategy"), row = 1, col = 1)
    
    st.plotly_chart(fig)