import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_job_postings(query, num_pages=5):
    base_url = "https://www.indeed.com/jobs"
    job_listings = []

    for page in range(0, num_pages * 10, 10):
        params = {
            'q': query,
            'start': page
        }

        response = requests.get(base_url, params=params)
        soup = BeautifulSoup(response.text, 'html.parser')
        job_cards = soup.find_all('div', class_='jobsearch-SerpJobCard')

        for card in job_cards:
            title = card.h2.a.get('title')
            company = card.find('span', class_='company').text.strip()
            location = card.find('div', class_='recJobLoc').get('data-rc-loc')
            summary = card.find('div', class_='summary').text.strip()
            
            job_listings.append({
                'Title': title,
                'Company': company,
                'Location': location,
                'Summary': summary
            })

    return job_listings

job_listings = scrape_job_postings("data scientist", num_pages=2)

df = pd.DataFrame(job_listings)

print(df)
