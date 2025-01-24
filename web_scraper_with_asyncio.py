# Implementing a Web Scraper with AsyncIO

# Description: Scraping data from multiple web pages concurrently using asynchronous programming.
# Key Libraries: aiohttp, asyncio, beautifulsoup4

import aiohttp
import asyncio
from bs4 import BeautifulSoup

async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()

async def scrape(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in urls]
        htmls = await asyncio.gather(*tasks)
        for html in htmls:
            soup = BeautifulSoup(html, 'html.parser')
            print(soup.title.string)

urls = ["https://www.python.org", "https://www.djangoproject.com", "https://flask.palletsprojects.com"]
asyncio.run(scrape(urls))
