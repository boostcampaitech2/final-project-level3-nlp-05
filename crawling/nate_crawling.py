from collections import Counter
import requests

def main():
    base_url = "https://news.nate.com/today/keywordList"
    date = "2021-11-24"
    times = [str(i).rjust(2, "0") for i in range(9, 24)]

    keywords = []
    
    for time in times:
        url = base_url + "?service_dtm=" + date + "%20"+time+":00:00"
        response = requests.get(url)
        result = eval(response.text)
        
        for key in result['data'].keys():
            keywords.append(result['data'][key]['keyword_service'].replace("<br \/>", " "))  
    
    counter = Counter(keywords)
    
    for key, value in counter.items():
        if value > 1 :
            print(key, value)
        

if __name__ == "__main__":
    main()