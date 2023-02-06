from bs4 import BeautifulSoup

# List of file names to merge
files = ["Explorer.html", "HexapodController.html", "HexapodExplorer.html", "HexapodRobot.html"]

# Read the contents of each file into a BeautifulSoup object
soups = [BeautifulSoup(open(file), "html.parser") for file in files]

# Merge the contents of all BeautifulSoup objects into one
merged_soup = BeautifulSoup("", "html.parser")
for soup in soups:
    merged_soup.append(soup)

# Write the merged content to a new file
with open("merged_documentation.html", "w") as outfile:
    outfile.write(str(merged_soup))