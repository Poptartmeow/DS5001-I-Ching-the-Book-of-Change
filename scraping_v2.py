import pandas as pd
import requests
from bs4 import BeautifulSoup
import sys
import time
import random
import re
import os


# --- Config ---

base_url = "https://sacred-texts.com/ich/"
headers = {
    'user-agent': f'academic-research-scraper (Python/{sys.version})'
}

# Regex to identify navigation / boilerplate paragraphs.
# Matches the "Next: ... Sacred Texts | I Ching « Previous: ..." footer
# and also stray page-number markers like "p. 59".
NAV_PATTERN = re.compile(
    r'Sacred\s+Texts|Next\s*:|Previous\s*:|«|»|\bIndex\b',
    re.IGNORECASE
)
PAGE_NUM_PATTERN = re.compile(r'\bp\.\s*\d+\b', re.IGNORECASE)



# --- Helpers ---

def is_nav_paragraph(text: str) -> bool:
    """Return True if the paragraph is navigation / boilerplate, not content."""
    return bool(NAV_PATTERN.search(text))


def clean_text(text: str) -> str:
    """Strip page-number markers and normalise whitespace."""
    text = PAGE_NUM_PATTERN.sub('', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def spider(c_url: str) -> dict:
    """
    Fetch one page and return a dict with:
      - hexagram_img : URL of the hexagram image (None for appendix pages)
      - paragraphs   : list of clean paragraph strings, nav text excluded
    """
    r = requests.get(c_url, headers=headers)

    # Remove stray <i>/<I> tags that fragment words mid-sentence
    html = re.sub(r'</?[Ii]>', '', r.text)
    soup = BeautifulSoup(html, 'html.parser')

    # Hexagram image (only present on the 64 main hexagram pages)
    try:
        img_src = soup.find('img', src=lambda s: s and 'hex' in s)['src']
        hexagram_img = base_url + img_src
    except (TypeError, KeyError):
        hexagram_img = None

    # Paragraphs — filter navigation, clean whitespace
    paragraphs = []
    for p in soup.find_all('p'):
        text = p.get_text(separator=' ', strip=True)
        if not text:
            continue
        if is_nav_paragraph(text):
            continue
        paragraphs.append(clean_text(text))

    return {'hexagram_img': hexagram_img, 'paragraphs': paragraphs}



# --- Discover links from the index page ---

print("Fetching index page...")
r = requests.get(base_url, headers=headers)
index_soup = BeautifulSoup(r.text, 'html.parser')

# All links starting with 'ic', skipping the first 7 (intro + plates)
all_links = index_soup.find_all('a', href=lambda x: x and x.startswith('ic'))
all_links = all_links[7:]

all_hrefs = [a['href'] for a in all_links]
all_titles = [a.get_text(strip=True) for a in all_links]

len(all_titles) # double check 75 blocks, looks fine

# Split into hexagrams (first 64) and appendices (remainder)
hex_hrefs = all_hrefs[:64]
hex_titles = all_titles[:64]
app_hrefs = all_hrefs[64:]
app_titles = all_titles[64:]

print(f"Found {len(hex_hrefs)} hexagram pages and {len(app_hrefs)} appendix pages.")


# --- Scrape — build one row per paragraph ---

rows = []

def scrape_section(hrefs, titles, doc_type):
    """Scrape a list of pages and append paragraph-level rows."""
    for n, (href, title) in enumerate(zip(hrefs, titles)):
        doc_id = n if doc_type == 'hexagram' else 64 + n
        url = base_url + href
        print(f"  [{doc_type}] {doc_id:>3}  {title[:60]}")

        data = spider(url)

        for para_num, para_text in enumerate(data['paragraphs']):
            rows.append({
                'doc_id': doc_id,
                'doc_type': doc_type,
                'title': title,
                'url': url,
                'hexagram_img': data['hexagram_img'],
                'para_num': para_num,
                'para_text': para_text,
                'word_count': len(para_text.split()),
            })

        delay = random.uniform(5, 10)
        print(f"    ↳ {len(data['paragraphs'])} paragraphs  |  waiting {delay:.1f}s...")
        time.sleep(delay)


print("\n--- Scraping hexagrams ---")
scrape_section(hex_hrefs, hex_titles, 'hexagram')

print("\n--- Scraping appendices ---")
scrape_section(app_hrefs, app_titles, 'appendix')


# --- Save ---

df = pd.DataFrame(rows) # 3119 rows

# Inspected df, some row show word count as 0 but 'para_text' not Nan, probabely there are some invisible text got scraped
wd_0 = df[df["word_count"]==0] # 405 rows
wd_0.para_text.unique() # chech values before removing, all [''], good to remove

# get a copy with word count > 0
df_1 = df[df['word_count']>0].copy() # 2714, row number checked after removal

# reset para_num after removal
df_1['para_num']=df.groupby('doc_id').cumcount()

# check results - picke 0,30,65,74
df_1.loc[df_1['doc_id']==74] 

# Summary
print("\n--- SCRAPE SUMMARY ---")
print(f"Total paragraph rows : {len(df_1)}")
print(f"  Hexagram paragraphs: {len(df_1[df_1.doc_type == 'hexagram'])}")
print(f"  Appendix paragraphs: {len(df_1[df_1.doc_type == 'appendix'])}")
print(f"Total words          : {df_1.word_count.sum():,}")
print(f"Unique documents     : {df_1.doc_id.nunique()}")

df_1.to_csv('input/i_ching_paragraphs.csv', index=False)
print("\nSaved → output/i_ching_paragraphs.csv")
print("OHCO structure for CORPUS: doc_id | para_num | sent_num | token_num")
