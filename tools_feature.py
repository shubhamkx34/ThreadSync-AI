import os
import asyncio
import requests
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv

load_dotenv()
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Live Search Tool 1 (via Tavily Remote MCP Server)
async def _run_tavily_mcp(query: str) -> str:
    client = MultiServerMCPClient({
        "tavily": {
            "transport": "http",
            "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}"
        }
    })
    mcp_tools = await client.get_tools()
    # Select the tavily-search tool exposed by the MCP server
    search = next((t for t in mcp_tools if "search" in t.name), mcp_tools[0])
    result = await search.ainvoke({"query": query, "max_results": 3, "search_depth": "advanced"})
    return str(result)
@tool
def search_tool(query: str) -> str:
    """Search the internet for current events, recent news, or real-time information."""
    return asyncio.run(_run_tavily_mcp(query))

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