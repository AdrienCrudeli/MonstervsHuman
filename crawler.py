"I had to use this crawler cause i didnt find any dataset on the net, what was downloaded can be founded under dataset folder, i has been clean in parallel"

from icrawler.builtin import BingImageCrawler
import os

classes = {
    "human": "normal person face portrait",
    "zombie": "zombie makeup face horror",
    "vampire": "vampire costume face makeup",
    "skeleton": "skeleton costume halloween face"
}

for class_name, query in classes.items():
    for split, num in [("train", 150), ("val", 40)]:
        save_dir = f"./dataset/{split}/{class_name}"
        os.makedirs(save_dir, exist_ok=True)

        crawler = BingImageCrawler(storage={"root_dir": save_dir})
        crawler.crawl(keyword=query, max_num=num)

        print(f"✅ {split}/{class_name} — {num} images téléchargées")