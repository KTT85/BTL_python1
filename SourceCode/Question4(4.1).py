import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException, NoSuchElementException
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from fuzzywuzzy import fuzz
from unidecode import unidecode
import time
import os
import tempfile

def prepare_player_data(file_path='results.csv'):
    """Bước 1: Chuẩn bị dữ liệu từ results.csv"""
    print("\nBước 1: Đang chuẩn bị dữ liệu từ results.csv...")
    df = pd.read_csv(file_path)

    # Lọc những cầu thủ đá > 900 phút
    filtered_df = df[df['Minutes'] > 900].copy()

    # Xử lý cột Age: lấy số đầu tiên trước dấu '-'
    filtered_df['Age'] = filtered_df['Age'].astype(str).str.extract(r'(\d+)', expand=False).astype(int)

    print(f"- Đã tìm thấy {len(filtered_df)} cầu thủ có thời gian thi đấu > 900 phút")
    return filtered_df

def collect_transfer_values():
    """Bước 2: Thu thập dữ liệu từ FootballTransfers (không lấy cột Age nữa)"""
    print("\nBước 2: Đang thu thập dữ liệu từ FootballTransfers...")
    
    urls = [f"https://www.footballtransfers.com/us/values/players/most-valuable-soccer-players/playing-in-uk-premier-league/{i}" for i in range(1, 23)]
    
    options = Options()
    options.add_argument('--headless=new')
    options.add_argument('--disable-blink-features=AutomationControlled')
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    wait = WebDriverWait(driver, 20)
    all_players = []
    
    try:
        for page_num, url in enumerate(urls, 1):
            print(f"\n- Đang xử lý trang {page_num}/22...")
            driver.get(url)
            time.sleep(3)

            if page_num == 1:
                try:
                    cookie_btn = wait.until(EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler")))
                    cookie_btn.click()
                    print("- Đã chấp nhận cookie")
                    time.sleep(1)
                except Exception:
                    print("- Không tìm thấy nút cookie consent")

            try:
                table = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table.table")))
                rows = table.find_elements(By.TAG_NAME, "tr")[1:]
                
                page_data = []
                for row in rows:
                    try:
                        cols = row.find_elements(By.TAG_NAME, "td")
                        if len(cols) >= 6:
                            player_data = {
                                'Player': cols[2].text.strip(),
                                'Team': cols[4].text.strip(),
                                'Transfer_Value': cols[5].text.strip()
                            }
                            page_data.append(player_data)
                    except StaleElementReferenceException:
                        continue
                
                if page_data:
                    all_players.extend(page_data)
                    print(f"  → Đã thu thập {len(page_data)} cầu thủ từ trang {page_num}")
                    
            except Exception as e:
                print(f"- Lỗi khi xử lý trang {page_num}: {str(e)}")
                continue
    
    finally:
        driver.quit()
    
    print(f"\n- Tổng cộng đã thu thập {len(all_players)} cầu thủ")
    return pd.DataFrame(all_players)

def normalize_name(name):
    return unidecode(name.lower().strip())

def match_and_filter_data(results_df, transfer_df, similarity_threshold=75):
    print("\nBước 3: Đang so khớp dữ liệu...")
    matched_data = []
    
    for _, player in results_df.iterrows():
        best_match = None
        best_score = 0
        player_name_norm = normalize_name(player['Player'])

        for _, transfer in transfer_df.iterrows():
            transfer_name_norm = normalize_name(transfer['Player'])
            score = fuzz.token_set_ratio(player_name_norm, transfer_name_norm)
            if score > best_score:
                best_score = score
                best_match = transfer
        
        if best_score >= similarity_threshold:
            matched_data.append({
                'Player': player['Player'],
                'Team': player['Team'],
                'Minutes': player['Minutes'],
                'Position': player['Position'],
                'Age': player['Age'],
                'Transfer_Value': best_match['Transfer_Value']
            })
            print(f"  + Khớp: {player['Player']} ({best_score}%)")
    
    matched_df = pd.DataFrame(matched_data)
    print(f"- Đã tìm thấy {len(matched_df)} cặp khớp với độ tương đồng >= {similarity_threshold}%")
    return matched_df

def save_results(df, filename='transfer_values.csv'):
    """Bước 4: Lưu kết quả"""
    print(f"\nBước 4: Đang lưu kết quả vào {filename}...")
    try:
        df.to_csv(filename, index=False)
        print(f"- Đã lưu {len(df)} cầu thủ vào file {filename}")
        print("\nMẫu dữ liệu:")
        print(df.head())
    except Exception as e:
        print(f"! Lỗi khi lưu file: {str(e)}")

def main():
    results_df = prepare_player_data()
    transfer_df = collect_transfer_values()
    
    if not transfer_df.empty:
        matched_df = match_and_filter_data(results_df, transfer_df, similarity_threshold=75)
        save_results(matched_df)
    else:
        print("Không thể thu thập dữ liệu từ website!")

if __name__ == "__main__":
    main()
