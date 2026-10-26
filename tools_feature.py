import os 
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
import requests
from typing import Any
from dotenv import load_dotenv

# Loading API Keys : 
load_dotenv()
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")


# Live Search Tool 1
tavily_search = TavilySearch(
    max_results=3,
    topic="general",
    search_depth="advanced",
    include_raw_content=False
)
@tool
def search_tool(query: str) -> str:
    """Search the internet for current events, recent news, or real-time information."""
    return str(tavily_search.invoke({"query": query}))

# Live stock Fetch Tool 2:
@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHAVANTAGE_API_KEY}"
    r = requests.get(url)
    return r.json()

# Live Weather Fetch Tool 3:
@tool
def get_weather_data(city: str) -> str:
    """
    Fetch current weather information for a city.
    """

    url = (
        f"http://api.weatherstack.com/current?"
        f"access_key={WEATHERSTACK_API_KEY}&query={city}"
    )

    response = requests.get(url)

    data = response.json()

    if "current" not in data:
        return f"Could not fetch weather data for {city}"

    return (
        f"City: {city}\n"
        f"Temperature: {data['current']['temperature']}°C\n"
        f"Weather: {data['current']['weather_descriptions'][0]}\n"
        f"Humidity: {data['current']['humidity']}%"
    )    