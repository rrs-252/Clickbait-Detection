import newspaper
from newspaper import Article
from newspaper import Source
from newspaper import news_pool
import pandas as pd

# The various News Sources we will like to web scrape from
gamespot = newspaper.build('https://www.gamespot.com/news/', memoize_articles=False)
bbc = newspaper.build("https://www.bbc.com/news", memoize_articles=False)

# Place the sources in a list
papers = [gamespot, bbc]

# Essentially you will be downloading 4 articles parallely per source.
# Since we have two sources, that means 8 articles are downloaded at any one time. 
# Greatly speeding up the processes.
# Once downloaded it will be stored in memory to be used in the for loop below 
# to extract the bits of data we want.
news_pool.set(papers, threads_per_source=4)

news_pool.join()

# Create our final dataframe
final_df = pd.DataFrame()

# Create a download limit per sources
# NOTE: You may not want to use a limit
limit = 100

for source in papers:
    # temporary lists to store each element we want to extract
    list_title = []
    list_text = []
    list_source =[]

    count = 0

    for article_extract in source.articles:
        article_extract.parse()

        if count > limit: # Lets have a limit, so it doesnt take too long when you're
            break         # running the code. NOTE: You may not want to use a limit

        # Appending the elements we want to extract
        list_title.append(article_extract.title)
        list_text.append(article_extract.text)
        list_source.append(article_extract.source_url)

        # Update count
        count +=1


    temp_df = pd.DataFrame({'Title': list_title, 'Text': list_text, 'Source': list_source})
    # Append to the final DataFrame
    final_df = final_df.append(temp_df, ignore_index = True)
    
# From here you can export this to csv file
final_df.to_csv('my_scraped_articles.csv')
