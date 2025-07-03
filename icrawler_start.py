from icrawler.builtin import GoogleImageCrawler 
import os
# basic utility used to download images through the web 
# making sure that the images are noncomercial , photos(eventhough it doesn't work)
# max size of image is 800x640

base_path = r".\dataset"
categories = ["thumbs up" , "thumbs down" , "hand gestures"]

for category in categories :
    # better than using string concat because this is smarter and handles 
    # cross platform code 
    constructed_path = os.path.join(base_path,category.replace(' ','_')) 
    
    GoogleCrawler = GoogleImageCrawler( downloader_threads=4,storage={r"root_dir" : constructed_path})
    GoogleCrawler.crawl(keyword=category ,max_size=(800,640), max_num=20 , filters={"type" : "photo" , 'license': 'noncommercial'})
