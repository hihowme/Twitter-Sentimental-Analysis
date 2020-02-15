# The difference of People's attitudes towards HongKong -- text sentiment analysis based on twitter and Weibo

Given the protest that is happening in HongKong, people in China and people in America have a different attitude. This project would try to use python to scrape the comment from the most top post containing 'HongKong' on 2 representative social media in China and America -- Weibo and Twitter, and then use the NLP method to compare the text sentiment from those 2 platforms. Finally, we would use `matplotlib` to get the output graph showing the difference.

## Overview

This project would go like this. The input would be "HongKong" or "HongKong protest", which would be passed to the search module, which will automatically find the top 10 posts containing that word with the most comments. The comments text will be scraped, stored, and transformed to numbers by NLP. The numbers will be analyzed to get the sentimental difference.

## Breakdown Modules

### Getting access to API with `tweepy` and `weibo`

I will create a **Twitter developer account** to use the **Twitter API**. After I have the account, I could just install the  `tweepy` package to get access to **Twitter API**.  In the case of getting data from Weibo, I would use the package `weibo` instead of `tweepy` to get access to the **Weibo API**. 
 
### Web Crawler with `request`, `urllib`, `beautifulsoup` and `selenium`

With the API, we could have the comments information for a single post, which is not enough for our analysis. Therefore, we would write a web crawler to achieve bigger data and higher efficiency. In this part, we would use `request`, `urllib`, `beautifulsoup` and `selenium` to scrape the comment of the post.

### Saving Data with MySQL in case of the anti-crawler using `pymysql`

One big problem with the big data web scraping is that many websites have the technology of anti-crawler. If the website detects that you are scraping the data, your program will not work and your data without saving will be lost too. However, if you save the data as soon as you scrape it, the program would be extremely slow. Therefore, the best way is to store the data we get in MySQL using `pymysql`.
 
### NLP Analysis Using `snowNLP` and `NLTK`(package to be determined on the progress)

Finally, we would use the NLP method to analyze the text we have. Here we would use the package `snowNLP` to analyze the text from Weibo, which is in Chinese and `NLTK` to analyze the text from twitter, which is in English. After we have done this, we would have a number representing the sentiment of the comment and we would compare the sentiment difference between those 2 groups to compare the attitude difference and make a graph with `matplotlib`. The frequency graph of words would also be presented.

## Potential Problems

- There would be a lot of details to deal with in especially **Web Crawler Part**, like how to deal with the emoji and how to "behave" like a person to avoid anti-crawler. So the details are to be determined at work.
- NLP works with the package might not work because this is a particular political issue, people on twitter may use specific words to express the negative sentiment, while the NLP may regard them as neutral.

## Test case

I plan to first scrape and analyze some test posts posted by me and my friends. Our test post case would try to mimic the real post regarding this issue to see if this works. Details remain to be determined during the work.

## Reference
- [ANTHONY SISTILLI](https://www.toptal.com/resume/anthony-sistilli) [“Twitter Data Mining: A Guide to Big Data Analytics Using Python"](https://www.toptal.com/python/twitter-data-mining-using-python).
