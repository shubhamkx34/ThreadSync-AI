import os 
from langchain_core.tools import tool

from langchain_tavily import TavilySearch
import math
import requests
from typing import Any
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Loading API Keys : 
load_dotenv()
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")
llmModel = ChatGroq(model='openai/gpt-oss-120b')


# Live Search Tool 1
search_tool = TavilySearch(
    max_results=3,
    topic="general",
    search_depth="advanced",
    include_raw_content=False
)

# Live stock Fetch Tool 2:
@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=D71XPZPBYI82R4LA"
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