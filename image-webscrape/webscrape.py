from bs4 import BeautifulSoup
import time
import requests
import os

URL_PREFIX = "https://terraria.wiki.gg{}{}"
IMAGE_URL_PREFIX = "/wiki/Category:Weapon_item_images"
SUFFIXES = ["", "?filefrom=Lightning+Aura+Rod.png#mw-category-media", "?filefrom=Wand+of+Frosting.png#mw-category-media"]
headers = {
    "cookie": "CONSENT=YES+cb.20230531-04-p0.en+FX+908",
}
FOLDER = "terraria_images"

if not os.path.exists(FOLDER):
    os.makedirs(FOLDER)

def fix_filename(filename):
    filename = filename.split('?')[0]
    return filename.replace("_", " ").replace("%27", "'").replace("%2B", "+").replace("%2F", "/")

def download_image(img_url, img_filename):
    print('Downloading', img_filename)
    response = requests.get(img_url, headers=headers, timeout=10)
    if response.status_code == 200:
        with open(os.path.join(FOLDER, img_filename), 'wb') as f:
            f.write(response.content)
    else:
        print("Failed to download", img_filename)

def process_suffix(suffix):
    print('Getting images from', URL_PREFIX.format(IMAGE_URL_PREFIX, suffix))
    request = requests.get(url=URL_PREFIX.format(IMAGE_URL_PREFIX, suffix), headers=headers, timeout=10)
    soup = BeautifulSoup(request.content, "html.parser")

    boxes = soup.find_all("li", class_="gallerybox")
    for box in boxes:
        img_el = box.find("img")
        src = img_el["src"]
        img_url = URL_PREFIX.format("", src)
        img_filename = fix_filename(src.split("/")[-1])
        download_image(img_url, img_filename)



if __name__ == "__main__":
    for suffix in SUFFIXES:
        process_suffix(suffix)
    print("All images downloaded.")