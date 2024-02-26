import requests
from bs4 import BeautifulSoup

url = "https://english.onlinekhabar.com/pm-urges-action-on-cooperative.html"
response = requests.get(url)
html_content = response.content

# Create a Beautiful Soup object
soup = BeautifulSoup(html_content, 'html.parser')

# Extract titles and text
titles = soup.find_all('h1')
texts = soup.find_all('p')

# Store titles and texts in lists
title_list = [title.text for title in titles]
text_list = [text.text for text in texts]

# Concatenate titles starting from index 1
scraped_title = " ".join(title_list)

# Concatenate all texts
scraped_text = " ".join(text_list[1:])

print(scraped_title)
print(scraped_text)
