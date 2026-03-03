import os
import requests
from bs4 import BeautifulSoup
import time

CF_DATA_DIR = "cfData"
os.makedirs(CF_DATA_DIR, exist_ok=True)

# We will fetch up to MAX_PROBLEMS to avoid taking hours
MAX_PROBLEMS = 1000

print("Fetching problem list from Codeforces API...")
try:
    response = requests.get("https://codeforces.com/api/problemset.problems")
    data = response.json()
except Exception as e:
    print("Failed to fetch from CF API:", e)
    exit(1)

if data['status'] != 'OK':
    print("Codeforces API returned an error.")
    exit(1)

problems = data['result']['problems']
print(f"Total problems available on CF: {len(problems)}")

# The Qlink structure expects all links in one file.
qlinks_path = os.path.join(CF_DATA_DIR, "Qlink.txt")

# We only process the first 1000 to keep it manageable, or less
problems_to_scrape = problems[:MAX_PROBLEMS]

cnt = 0
with open(qlinks_path, "w", encoding="utf-8") as link_file:
    for prob in problems_to_scrape:
        contestId = prob.get('contestId')
        index = prob.get('index')
        name = prob.get('name')
        
        if not contestId or not index:
            continue
            
        cnt += 1
        url = f"https://codeforces.com/problemset/problem/{contestId}/{index}"
        
        # Save link to Qlink.txt
        link_file.write(url + "\n")
        
        # Create directory structure similar to LeetCode (cfData/1/1.txt)
        prob_dir = os.path.join(CF_DATA_DIR, str(cnt))
        os.makedirs(prob_dir, exist_ok=True)
        prob_file = os.path.join(prob_dir, f"{cnt}.txt")
        
        # Instead of HTML parsing (which is slow), we know the problem name, 
        # and we can optionally fetch the body.
        # Note: Codeforces rate-limits strictly. Scraping 1000 items sequentially via requests 
        # takes ~500 seconds assuming 0.5s pause.
        
        # For the sake of the Search Engine, let's grab the HTML question text
        try:
            time.sleep(0.3)  # Anti-ban
            html_resp = requests.get(url, timeout=10)
            soup = BeautifulSoup(html_resp.text, 'html.parser')
            
            # Codeforces problem statement is inside div class="problem-statement"
            statement = soup.find('div', class_='problem-statement')
            if statement:
                # We'll just extract all text
                text_content = f"{name}\n" + statement.get_text(separator=' ', strip=True)
            else:
                text_content = name
                
            with open(prob_file, "w", encoding="utf-8") as pf:
                pf.write(text_content + "\n")
                
            if cnt % 50 == 0:
                print(f"Successfully scraped {cnt}/{MAX_PROBLEMS} problems...")
                
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")
            with open(prob_file, "w", encoding="utf-8") as pf:
                pf.write(name + "\n") # Fallback to just the name

print(f"Finished scraping {cnt} Codeforces problems!")
