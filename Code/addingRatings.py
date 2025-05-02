import requests
from bs4 import BeautifulSoup
import os
import re
import time

folder = "DSP-Final/movie_scripts"

def search_bbfc_url(title, year):
    query = title.lower().replace(" ", "%20")
    print(f"Searching: {query}")
    search_url = f"https://www.bbfc.co.uk/search?q={query}"

    response = requests.get(search_url)
    if response.status_code != 200:
        print("Failed to retrieve BBFC search results")
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    search_divs = soup.find_all('div', class_='SearchItem_Wrapper__1lhgt')

    for div in search_divs:
        # Get the combined title and year string, e.g. "Deep Fear (2023)"
        title_element = div.find('h3', class_='Type_title__142II SearchItem_Title__38hx7')
        if not title_element:
            continue

        full_title = title_element.get_text(strip=True)

        # Use regular expression to find the last set of brackets with a year inside
        year_match = re.search(r"\((\d{4})\)", full_title)

        if year_match:
            found_year = int(year_match.group(1))

            # Loose match title and exact match year
            if title.lower() in full_title.lower() and found_year == year:
                # Extract the rating from the same div
                rating_span = div.find('span', class_='Icon_Icon__9RCS8 SearchItem_Rating__2fbS8')
                if rating_span:
                    rating = rating_span.get('aria-label', '').replace("Rated ", "").strip()
                    print(f"Matched rating: {rating}")
                    return rating

    print("No matching title/year found")
    return None


def append_rating_to_script(filename, rating):
    """
    Prepend the UK age rating to the top of the script file.
    """
    try:
        with open(filename, "r+", encoding="utf-8") as file:
            content = file.read()
            file.seek(0, 0)
            file.write(f"UK Age Rating: {rating}\n\n{content}")
    except Exception as e:
        print(f"Error updating file {filename}: {e}")

def main():
    for scriptname in os.listdir(folder):
        if not scriptname.endswith(".txt"):
            continue

        try:
            stoppos = scriptname.rindex("_(")
            title = scriptname[:stoppos].replace("_", " ")
            year = int(scriptname[stoppos+2:stoppos+6])
        except Exception as e:
            print(f"Error parsing title or year for {scriptname}: {e}")
            continue

        print(f"Processing: {title} ({year})")

        rating = search_bbfc_url(title, year)
        print(f"UK Age Rating for {title}: {rating}")

        file_path = os.path.join(folder, scriptname)
        append_rating_to_script(file_path, rating)

        # Be polite to the BBFC site
        time.sleep(2)

    print("All scripts have been updated with UK age ratings.")


if __name__ == "__main__":
    main()
