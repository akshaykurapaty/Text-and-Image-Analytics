# Text-and-Image-Analytics

There are two files in the repository 
1) textanalytics.py:
This files contains the python code that does the text analytics/image analytics builds predictive modelling and does all the work for us. 
2) ScrapyDataExtraction:
This folder is the scrapy project which contains all the code that crawls the web page and extracts all the relevant information from the webpage. The above folder contains code that extracts each ad description, face image posted, type of ad and the price posted on the site.

Execute the scrapy project first using scrapy crawl car -o output.csv

Followed by the execution of scrapy point the python current working directory to the path where data is extracted and eecute the python
code. This code builds the model and does the comparission for us.
