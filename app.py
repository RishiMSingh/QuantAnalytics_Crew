import streamlit as st
from Trading_Crew import financial_trading_crew  # Import the crew from Trading_Crew.py

# Streamlit application
def main():
    st.title("Financial Trading Decision Support System")

    # Sidebar for user inputs
    with st.sidebar:
        stock_selection = st.text_input("Stock Selection (e.g., TSLA)", "TSLA")
        initial_capital = st.number_input("Initial Capital ($)", min_value=100, max_value=10000000, value=1000)
        risk_tolerance = st.selectbox("Risk Tolerance", ["Low", "Medium", "High"])
        trading_strategy_preference = st.selectbox("Trading Strategy Preference", ["Day Trading", "Swing Trading", "Long-Term Investing"])
        news_impact_consideration = st.checkbox("Consider News Impact", value=True)
        investment_horizon = st.selectbox("Investment Horizon", ["Short-term", "Medium-term", "Long-term"])
        data_sources = st.multiselect("Data Sources", ["Bloomberg", "Yahoo Finance", "SEC Filings"], default=["Bloomberg", "Yahoo Finance"])
        quant_model_preference = st.selectbox("Quantitative Model Preference", ["Machine Learning", "Statistical Analysis"])
        technical_indicators = st.multiselect("Technical Indicators", ["Moving Average", "RSI", "MACD"], default=["Moving Average", "RSI", "MACD"])

    # Main container for the generated report
    report_container = st.container()

    # Button state management
    generate_button_disabled = st.session_state.get("generate_button_disabled", False)
    generate_report_button = st.button("Generate Report", disabled=generate_button_disabled)

    # If the user clicks the "Generate Report" button
    if generate_report_button:
        st.session_state["generate_button_disabled"] = True
        # Prepare the input dictionary
        financial_trading_inputs = {
            'stock_selection': stock_selection,
            'initial_capital': initial_capital,
            'risk_tolerance': risk_tolerance,
            'trading_strategy_preference': trading_strategy_preference,
            'news_impact_consideration': news_impact_consideration,
            'investment_horizon': investment_horizon,
            'data_sources': data_sources,
            'quant_model_preference': quant_model_preference,
            'technical_indicators': technical_indicators,
        }

        # Execute the crew's process with kickoff method
        result = financial_trading_crew.kickoff(financial_trading_inputs)

        # Display the generated report in the container
        with report_container:
            st.subheader("Generated Report")
            st.write(result)

        # Re-enable the button after processing
        st.session_state["generate_button_disabled"] = False

if __name__ == "__main__":
    main()