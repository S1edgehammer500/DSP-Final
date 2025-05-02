import os

ratings = {"U":0, "PG":0, "12":0, "12A":0, "15":0, "18":0}
folder = "DSP-Final/movie_scripts"

def count_ratings(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            lines = file.readlines()
            for line in lines:
                if line.startswith("UK Age Rating:"):
                    rating = line.strip().split("UK Age Rating:")[1].strip()
                    if rating in ratings and ratings[rating] < 250:
                        ratings[rating] += 1
                    elif ratings[rating] >= 250:
                        file.close()
                        os.remove(filepath)
                    else:
                        print(f"Invalid rating for {filepath}")
    except Exception as e:
        print(f"Error opening {filepath}: {e}")
    
def main():
    for scriptname in os.listdir(folder):
        filepath = os.path.join(folder, scriptname)
        if not scriptname.endswith(".txt"):
            continue

        count_ratings(filepath)
    print(f"Total files for each rating: {ratings}")

if __name__ == "__main__":
    main()

