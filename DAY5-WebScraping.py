
import os
import requests
import json
from typing import List
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from IPython.display import Markdown, display, update_display
from openai import AzureOpenAI


load_dotenv()  # This will load variables from the .env file

# Ensure environment variables are set
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")

# Check for missing environment variables
if not endpoint or not api_key:
    raise EnvironmentError("Missing AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY environment variables.")

# Initialize the AzureOpenAI client
client = AzureOpenAI(
    api_key=api_key,
    api_version="2024-07-01-preview",
    azure_endpoint=endpoint
)

completion = client.chat.completions.create(
    model="gpt-4o-mini", 
    messages=[
        {"role": "user", "content": "Hello there"}
    ]
)

# A class to represent a Webpage

# Some websites need you to use proper headers when fetching them:
headers = {
 "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

class Website:
    """
    A utility class to represent a Website that we have scraped, now with links
    """

    def __init__(self, url):
        self.url = url
        response = requests.get(url, headers=headers)
        self.body = response.content
        soup = BeautifulSoup(self.body, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        if soup.body:
            for irrelevant in soup.body(["script", "style", "img", "input"]):
                irrelevant.decompose()
            self.text = soup.body.get_text(separator="\n", strip=True)
        else:
            self.text = ""
        links = [link.get('href') for link in soup.find_all('a')]
        self.links = [link for link in links if link]

    def get_contents(self):
        return f"Webpage Title:\n{self.title}\nWebpage Contents:\n{self.text}\n\n"
    
#ed = Website('https://edwarddonner.com')
#print(ed.get_contents())
#print(ed.links)

link_system_prompt = "You are provided with a list of links found on a webpage. \
You are able to decide which of the links would be most relevant to include in a brochure about the company, \
such as links to an About page, or a Company page, or Careers/Jobs pages.\n"
link_system_prompt += "You should respond in JSON as in this example:"
link_system_prompt += """ #one shot prompt - one example
{
    "links": [
        {"type": "about page", "url": "https://full.url/goes/here/about"},
        {"type": "careers page": "url": "https://another.full.url/careers"}
    ]
}
"""

def get_links_user_prompt(website):
    user_prompt = f"Here is the list of links on the website of {website.url} - "
    user_prompt += "please decide which of these are relevant web links for a brochure about the company, respond with the full https URL in JSON format. \
Do not include Terms of Service, Privacy, email links.\n"
    user_prompt += "Links (some might be relative links):\n"
    user_prompt += "\n".join(website.links)
    return user_prompt

def get_links(url):
    website = Website(url)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": link_system_prompt},
            {"role": "user", "content": get_links_user_prompt(website)}
      ],
        response_format={"type": "json_object"}
    )
    result = response.choices[0].message.content
    return json.loads(result)

x = get_links("https://burgtranslations.com")
#print(x)

def get_all_details(url):
    result = "Landing page:\n"
    result += Website(url).get_contents()
    links = get_links(url)
    print("Found links:", links)
    for link in links["links"]:
        result += f"\n\n{link['type']}\n"
        result += Website(link["url"]).get_contents()
    return result

system_prompt = "You are a student trying to do research about which graduate school to apply to. \
    With a bachelor's degree in Data Science, concentrating in Artifiical intelligence and a minor in mathematics, you would like to find a program that expands on the artifical intelligence concentration. \
    You are able to decide which links to use to figure out if the college program is right for you    "

# Or uncomment the lines below for a more humorous brochure - this demonstrates how easy it is to incorporate 'tone':

# system_prompt = "You are an assistant that analyzes the contents of several relevant pages from a company website \
# and creates a short humorous, entertaining, jokey brochure about the company for prospective customers, investors and recruits. Respond in markdown.\
# Include details of company culture, customers and careers/jobs if you have the information."

def get_brochure_user_prompt(uni_name, url):
    user_prompt = f"You are looking at a university called: {uni_name}\n"
    user_prompt += f"Here are the contents of its landing page and other relevant pages; use this information to build a short brochure of the program in markdown.\n"
    user_prompt += get_all_details(url)
    user_prompt = user_prompt[:5_000] # Truncate if more than 5,000 characters
    return user_prompt

def create_brochure(company_name, url):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": get_brochure_user_prompt(company_name, url)}
          ],
    )
    result = response.choices[0].message.content
    display(Markdown(result))

a = create_brochure("University of Technology Sydney", "https://www.uts.edu.au/courses/graduate-diploma-in-artifical-intelligence")
#print(a)


def stream_brochure(company_name, url):
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": get_brochure_user_prompt(company_name, url)}
          ],
        stream=True
    )

    for chunk in stream:
        if chunk.choices:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                print(delta.content, end='')
    
    #for chunk in stream:
        #print(chunk.choices[0].delta.content or '', end='')


    #response = ""
    #display_handle = display(Markdown(""), display_id=True)
    #for chunk in stream:
        #response += chunk.choices[0].delta.content or ''
        #response = response.replace("```","").replace("markdown", "")
        #update_display(Markdown(response), display_id=display_handle.display_id)

stream_brochure("University of Technology Sydney", "https://www.uts.edu.au/courses/graduate-diploma-in-artifical-intelligence")