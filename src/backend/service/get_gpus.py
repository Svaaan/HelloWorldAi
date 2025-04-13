import os
import time
import random
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from tqdm import tqdm

YEAR_RANGE = ['2025', '2024', '2023', '2022', '2021', '2020', '2019']
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gpu-db.json')


def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-infobars")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    driver = webdriver.Chrome(options=chrome_options)
    driver.set_window_size(1280, 800)
    return driver


def safe_sleep(min_s=2, max_s=4):
    time.sleep(random.uniform(min_s, max_s))


def scroll_to_bottom(driver, pause=1.2, max_tries=30):
    last_height = driver.execute_script("return document.body.scrollHeight")
    tries = 0
    while tries < max_tries:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
        tries += 1


def wait_for_table_rows(driver, min_rows=20, timeout=30):
    start_time = time.time()
    while time.time() - start_time < timeout:
        rows = driver.find_elements(By.CSS_SELECTOR, 'table.processors tbody tr')
        if len(rows) >= min_rows:
            return rows
        time.sleep(1)
    return driver.find_elements(By.CSS_SELECTOR, 'table.processors tbody tr')


def parse_shader_info(shader_info):
    parts = shader_info.split(' / ')
    try:
        shaders = int(parts[0]) if len(parts) > 0 else None
        tmus = int(parts[1]) if len(parts) > 1 else None
        rops = int(parts[2]) if len(parts) > 2 else None
    except ValueError:
        shaders = tmus = rops = None
    return shaders, tmus, rops


def scrape_year(driver, year, all_data):
    url = f"https://www.techpowerup.com/gpu-specs/?released={year}&mfgr=NVIDIA&sort=name"
    driver.get(url)

    input(f"Year {year}: Page loaded and CAPTCHA solved if needed? Then press Enter to continue...")

    scroll_to_bottom(driver, pause=1.2, max_tries=30)
    print(f"Waiting for at least 20 rows for year {year}...")

    rows = wait_for_table_rows(driver, min_rows=20)
    if not rows:
        print(f"Timeout reached with {len(rows)} rows. Proceeding anyway.")

    year_data = []

    for row in rows:
        cells = row.find_elements(By.TAG_NAME, 'td')
        if len(cells) < 7:
            continue

        name = cells[0].text.strip()
        gpu_chip = cells[1].text.strip()
        released = cells[2].text.strip()
        bus = cells[3].text.strip()
        memory = cells[4].text.strip()
        gpu_clock = cells[5].text.strip()
        memory_clock = cells[6].text.strip()
        shader_info = cells[7].text.strip() if len(cells) > 7 else ""

        if not name or not shader_info:
            continue

        shaders, tmus, rops = parse_shader_info(shader_info)

        gpu_entry = {
            "name": name,
            "gpu_chip": gpu_chip,
            "released": released,
            "bus": bus,
            "memory": memory,
            "gpu_clock": gpu_clock,
            "memory_clock": memory_clock,
            "shaders": shaders,
            "tmus": tmus,
            "rops": rops
        }

        year_data.append(gpu_entry)
        print(f"✅ [{year}] {name} - {shaders} shaders / {tmus} TMUs / {rops} ROPs")
        safe_sleep(0.2, 0.4)

    print(f"🧩 Year {year}: Collected {len(year_data)} GPUs.")
    all_data.extend(year_data)

    # Auto save after each year
    save_results(all_data)

    return year_data


def save_results(data):
    if not data:
        print(" No data collected. Nothing to save.")
        return
    df = pd.DataFrame(data)
    df = df.drop_duplicates(subset=["name"]).sort_values("name")
    df.to_json(OUTPUT_FILE, orient='records', indent=4)
    print(f"✅ Saved {len(df)} unique GPUs to {OUTPUT_FILE}")


def main():
    driver = setup_driver()
    all_data = []

    print(f"🔍 Starting scrape for years: {YEAR_RANGE} (NVIDIA only)")

    for year in tqdm(YEAR_RANGE, desc="Year Progress"):
        retries = 4
        for attempt in range(1, retries + 1):
            year_data = scrape_year(driver, year, all_data)
            if year_data:
                break
            print(f"Retrying year {year} (attempt {attempt + 1})...")
            safe_sleep(3, 5)

    driver.quit()
    save_results(all_data)


if __name__ == "__main__":
    main()
