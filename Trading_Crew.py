# Warning control
import warnings
warnings.filterwarnings('ignore')
from crewai import Agent, Task, Crew
import os
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from crewai import Crew, Process
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_MODEL_NAME"] = os.getenv('OPENAI_MODEL_NAME')
os.environ["SERPER_API_KEY"] = os.getenv('SERPER_API_KEY')

search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

cio_agent = Agent(
    role="Chief Investment Officer",
    goal=
        """Develop and oversee the overall investment strategy to ensure optimal asset allocation and risk management.
        Integrate insights from all team members to make informed, high-level decisions about {stock_selection} stock purchases and sales.
        Continuously assess market conditions to adjust the portfolio strategy in response to new opportunities and risks.
        Based on the analysis from the Research Analyst, Quant Analyst, Technical Analyst, and Risk Manager, provide a final recommendation on whether to buy, hold, or sell specific stocks.""",
    backstory="A seasoned investment strategist with decades of experience in financial markets, you synthesize information from various sources, including other team members, to make data-driven, high-level decisions on stock acquisitions and disposals. Your final recommendation on buying, holding, or selling a stock will be based on a comprehensive analysis of all available data and insights.",
    verbose=True,
    allow_delegation=True,
    tools=[scrape_tool, search_tool]
)

research_analyst_agent = Agent(
    role="Research Analyst",
    goal=
        """Conduct thorough research on individual stocks and industries to provide actionable insights for investment decisions.
        Analyze financial reports, earnings, and industry trends to uncover valuable investment opportunities.
        Monitor market developments and provide timely updates to the team to inform strategic decisions.""",
    backstory="Armed with a strong analytical background, you dive into financial reports, earnings calls, and industry analyses to uncover opportunities and potential red flags, ensuring the team remains fully informed of market developments.",
    verbose=True,
    allow_delegation=True,
    tools=[scrape_tool, search_tool]
)


quant_analyst_agent = Agent(
    role="Quantitative Analyst",
    goal=
        """Develop and refine mathematical models to identify patterns in stock price movements and optimize trading strategies.
        Analyze large datasets to create predictive models that inform trading decisions.
        Continuously improve models based on backtesting and real-time performance to enhance predictive accuracy.""",
    backstory="Leveraging expertise in advanced mathematics and algorithmic trading, you translate vast amounts of data into actionable insights, providing a quantitative foundation for our investment strategies.",
    verbose=True,
    allow_delegation=True,
    tools=[scrape_tool, search_tool]
)

technical_analyst_agent = Agent(
    role="Technical Analyst",
    goal=
        """Analyze chart patterns and technical indicators to predict short-term and long-term market trends.
        Identify trading opportunities based on market trends, support and resistance levels, and other technical indicators.
        Provide precise entry and exit points for trades to maximize profitability."""
    ,
    backstory="With a deep understanding of market behavior, you utilize a variety of technical tools to predict market movements, identifying key opportunities and risks in real-time.",
    verbose=True,
    allow_delegation=True,
    tools=[scrape_tool, search_tool]
)

risk_manager_agent = Agent(
    role="Risk Manager",
    goal=
        """Identify and mitigate financial risks associated with stock investments, ensuring the portfolio is aligned with the risk appetite.
        Implement strategies to protect the portfolio from unexpected market movements.
        Continuously monitor and assess potential risks, adjusting the portfolio's exposure as needed to avoid significant losses.""",
    backstory="With an acute awareness of market risks, you continuously monitor and evaluate the portfolio, proactively managing exposure to ensure the portfolio remains resilient against potential losses.",
    verbose=True,
    allow_delegation=True,
    tools=[scrape_tool, search_tool]
)

investment_strategy_task = Task(
    description=(
        "Develop and oversee a comprehensive investment strategy for {stock_selection}. "
        "Integrate detailed insights from the Research Analyst, Quantitative Analyst, Technical Analyst, "
        "and Risk Manager agents. Ensure the strategy accounts for the user's risk tolerance ({risk_tolerance}), "
        "investment horizon ({investment_horizon}), and trading strategy preference ({trading_strategy_preference}). "
        "Evaluate all data points and provide a final, definitive recommendation to either buy, sell, or hold {stock_selection}, "
        "including an actionable plan for executing the recommended strategy."
    ),
    expected_output=(
        "A detailed investment strategy document for {stock_selection}, including a summary of all analyses, "
        "risk assessments, and quantitative models. The document should culminate in a clear and actionable "
        "buy/sell/hold recommendation, with specific instructions on how to execute the recommended strategy."
    ),
    agent=cio_agent,
)

research_analysis_task = Task(
    description=(
        "Conduct an exhaustive analysis of {stock_selection}, using data from multiple sources ({data_sources}), "
        "including Bloomberg, Yahoo Finance, and SEC filings. Analyze financial statements, earnings reports, "
        "and industry trends. Assess the impact of recent news ({news_impact_consideration}) on the stock’s performance. "
        "Evaluate the company’s competitive positioning, market opportunities, and threats. Provide a deep-dive report "
        "that outlines all relevant factors influencing {stock_selection} and concludes with a weighted analysis "
        "that leans towards a recommendation to buy, sell, or hold."
    ),
    expected_output=(
        "A comprehensive research report for {stock_selection}, summarizing financial health, industry trends, competitive analysis, "
        "and news impact. The report should include a weighted analysis that supports a specific recommendation "
        "to buy, sell, or hold, backed by data and trends."
    ),
    agent=research_analyst_agent,
)

quantitative_analysis_task = Task(
    description=(
        "Develop and apply advanced quantitative models using {quant_model_preference} techniques to analyze "
        "historical and real-time data for {stock_selection}. Focus on predicting future price movements, volatility, "
        "and potential returns. Perform scenario analysis to test the robustness of trading strategies under various "
        "market conditions, including worst-case scenarios. The analysis should culminate in a statistically-supported "
        "recommendation to either buy, sell, or hold {stock_selection}, factoring in risk tolerance ({risk_tolerance}) "
        "and investment horizon ({investment_horizon})."
    ),
    expected_output=(
        "A quantitative analysis report for {stock_selection}, including predictive models, scenario analysis, "
        "and a data-driven recommendation to buy, sell, or hold. The report should include confidence intervals "
        "and probability assessments to support the recommendation."
    ),
    agent=quant_analyst_agent,
)

technical_analysis_task = Task(
    description=(
        "Perform an in-depth technical analysis of {stock_selection} using the specified indicators ({technical_indicators}), "
        "including Moving Average, RSI, and MACD. Analyze short-term and long-term trends, identify support and resistance levels, "
        "and detect any significant chart patterns. Additionally, assess market sentiment and trading volume to provide context. "
        "The analysis should provide precise entry and exit points and conclude with a clear recommendation to buy, sell, or hold {stock_selection}, "
        "based on the technical indicators and current market conditions."
    ),
    expected_output=(
        "A detailed technical analysis report for {stock_selection}, including annotated charts, trend analysis, "
        "and indicator readings. The report should culminate in a clear and actionable recommendation to buy, sell, or hold, "
        "with suggested entry/exit points."
    ),
    agent=technical_analyst_agent,
)


risk_management_task = Task(
    description=(
        "Conduct a thorough risk assessment for {stock_selection} by analyzing potential financial, market, and operational risks. "
        "Utilize stress testing and scenario analysis to evaluate the impact of various adverse conditions, including market downturns, "
        "economic recessions, and competitive pressures. The analysis should include a detailed risk mitigation strategy, recommending "
        "hedging options, portfolio diversification, and other protective measures. The final output should conclude with a recommendation "
        "on whether the risk level justifies a buy, sell, or hold decision, considering the user's risk tolerance ({risk_tolerance}) "
        "and investment horizon ({investment_horizon})."
    ),
    expected_output=(
        "A comprehensive risk assessment report for {stock_selection}, including stress test results, scenario analysis, "
        "and risk mitigation strategies. The report should provide a clear recommendation on whether the stock’s risk profile "
        "supports a buy, sell, or hold decision, aligned with the user's risk tolerance and investment objectives."
    ),
    agent=risk_manager_agent,
)

# Define the crew with agents and tasks
financial_trading_crew = Crew(
    agents=[research_analyst_agent, 
            quant_analyst_agent, 
            technical_analyst_agent, 
            risk_manager_agent,
            cio_agent],
    
    tasks=[investment_strategy_task, 
           research_analysis_task, 
           quantitative_analysis_task, 
           technical_analysis_task, 
           risk_management_task],
    
    manager_llm=ChatOpenAI(model="gpt-3.5-turbo", 
                           temperature=0.7),
    process=Process.hierarchical,
    verbose=True
)

financial_trading_inputs = {
    'stock_selection': 'TSLA',
    'initial_capital': 1000,  # Assuming this is in dollars for better context
    'risk_tolerance': 'Low',
    'trading_strategy_preference': 'Day Trading',
    'news_impact_consideration': True,
    'investment_horizon': 'Short-term',
    'data_sources': ['Bloomberg', 'Yahoo Finance', 'SEC Filings'],
    'quant_model_preference': 'Machine Learning',
    'technical_indicators': ['Moving Average', 'RSI', 'MACD'],
}