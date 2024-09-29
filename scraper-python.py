import requests
import lxml
import csv
from bs4 import BeautifulSoup

header = ['Title', 'HTML']

def scrape():
    f = open('articlesClickbait.txt','r')
    data = f.readlines()
    for url in data:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "lxml")
        title = soup.select_one('h1').text
        data = [soup, title]
        with open('data.csv','a') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(header)
            csvwriter.writerow(data)

def scrapeLinks():
    url = "https://www.npr.org/sections/news/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "lxml")
    links = soup.find_all('a')
    for link in links:
        print(link['href'])
    
if __name__== '__main__':
    scrape()