import os

ratings = ['U', 'PG', '12', '12A', '15', '18']
folder = "DSP-Final/movie_scripts"

# add a mock line to an empty file so that it can be deleted
def remove_empty_files(filename):
    with open(filename, "r+", encoding="utf-8") as file:
        lines = file.readlines()
        # insert the new line at the top and push the old first line to second line
        lines.insert(0, "Jake Tovey DSP Project - Empty File" + "\n")
        file.seek(0)
        file.writelines(lines)

def clean_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            lines = file.readlines()

        valid_rating = None
        new_lines = []

        for line in lines:
            # check if the script contains a valid rating
            stripped_line = line.strip()
            if stripped_line.startswith("UK Age Rating:"):
                rating = stripped_line.strip().split("UK Age Rating:")[1].strip()
                if rating in ratings and valid_rating is None:
                    valid_rating = rating
                    new_lines.append(f"UK Age Rating: {valid_rating}\n\n")
                 # skip all rating lines
                continue
            # ignore mock line added to empty files
            elif line.strip() == "Jake Tovey DSP Project - Empty File":
                continue
            new_lines.append(line)

        # rewrite valid files and delete invalid ones
        if valid_rating:
            with open(filepath, "w", encoding="utf-8") as file:
                file.writelines(new_lines)
        else:
            os.remove(filepath)
            print(f"DELETED: {filepath}")
    except Exception as e:
        print(f"Error cleaning file {filepath}: {e}")

def main():
    # loop through movie_scripts to run the code on each script
    for scriptname in os.listdir(folder):
        filepath = os.path.join(folder, scriptname)
        if not scriptname.endswith(".txt"):
            os.remove(filepath)
            print(f"DELETED: {filepath}")
            continue

        remove_empty_files(filepath)
        clean_file(filepath)
        print(f"Finished cleaning: {scriptname}")

if __name__ == "__main__":
    main()
