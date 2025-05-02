import requests
from bs4 import BeautifulSoup
import os
import time
import re

# Step 1: Create a directory to store scripts
output_dir = "DSP-Final/movie_scripts"
os.makedirs(output_dir, exist_ok=True)

# Step 2: Base URL of the Springfield Springfield page
base_url = "https://www.springfieldspringfield.co.uk/movie_scripts.php?page=2101"
base_site = "https://www.springfieldspringfield.co.uk"

def fetch_script_links(base_url):
    """Fetch links to individual script pages."""
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all <a> tags with the specified class
    script_links = []
    for link in soup.find_all('a', class_='btn btn-dark btn-sm', href=True):
        href = link['href']
        title = link.get_text(strip=True)
        
        # Construct the full URL and store with title
        full_url = base_site + href
        script_links.append({"title": title, "url": full_url})
    
    return script_links

def fetch_script_text(script_url):
    """Fetch and parse script text from a given URL."""
    try:
        response = requests.get(script_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract the main script content
        script_content = soup.find("div", class_="scrolling-script-container")
        if script_content:
            # Use .stripped_strings to preserve line breaks
            return "\n".join(line for line in script_content.stripped_strings)
        else:
            print(f"No script content found at {script_url}")
            return None
    except Exception as e:
        print(f"Error fetching script from {script_url}: {e}")
        return None

def save_script(title, content):
    title = re.sub(r'[\\/*?:"<>|]', "_", title)
    """Save script content to a text file."""
    filename = os.path.join(output_dir, f"{title}.txt")
    print(filename)
    with open(filename, "w", encoding="utf-8") as file:
        file.write(content)

def main():
    print("Fetching script links...")
    for i in range(100):
        page_number = int(base_url.split('=')[-1])  # Split and get the last part after '='
    
        # Increment the page number
        new_page = page_number + i
        
        # Rebuild the URL with the updated page number
        new_url = base_url.split('=')[0] + "=" + str(new_page)
        script_links = fetch_script_links(new_url)
        print(f"Found {len(script_links)} script links.")

        for idx, script in enumerate(script_links):
            print(f"Fetching script {idx + 1}/{len(script_links)}: {script['title']}")
            script_text = fetch_script_text(script['url'])
            
            if script_text:
                # Save script using its title as the filename
                save_script(script['title'].replace(" ", "_").replace("/", "_"), script_text)
            
            # To prevent overwhelming the server, pause briefly
            time.sleep(2)

    print(f"All scripts have been saved to {output_dir}.")

if __name__ == "__main__":
    main()
