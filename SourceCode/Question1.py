import os
import time
import logging
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import random
from selenium.webdriver.chrome.options import Options

class FootballStatsScraper:
    def __init__(self):
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('scraper.log'),
                logging.StreamHandler()
            ]
        )

        # Define required statistics
        self.required_stats = {
            'standard': [
                'player',
                'nationality',  # Nation
                'team',        # Team
                'position',    # Position
                'age',         # Age
                'Playing Time_games',        # matches played
                'Playing Time_games_starts', # starts
                'Playing Time_minutes',      # minutes
                'Performance_goals',         # goals
                'Performance_assists',       # assists
                'Performance_cards_yellow',  # yellow cards
                'Performance_cards_red',     # red cards
                'Expected_xg',              # expected goals (xG)
                'Expected_xg_assist',       # expected Assist Goals (xAG)
                'Progression_progressive_carries',  # PrgC
                'Progression_progressive_passes',   # PrgP
                'Progression_progressive_passes_received',  # PrgR
                'Per 90 Minutes_goals_per90',      # Gls per 90
                'Per 90 Minutes_assists_per90',    # Ast per 90
                'Per 90 Minutes_xg_per90',         # xG per 90
                'Per 90 Minutes_xg_assist_per90'   # xGA per 90
            ],
            'keeper': [
                'Performance_gk_goals_against_per90',  # GA90
                'Performance_gk_save_pct',            # Save%
                'Performance_gk_clean_sheets_pct',    # CS%
                'Penalty Kicks_gk_pens_save_pct'      # Penalty kicks Save%
            ],
            'shooting': [
                'Standard_shots_on_target_pct',     # SoT%
                'Standard_shots_on_target_per90',   # SoT/90
                'Standard_goals_per_shot',          # G/sh
                'Standard_average_shot_distance'     # Dist
            ],
            'passing': [
                'Total_passes_completed',           # Cmp
                'Total_passes_pct',                 # Cmp%
                'Total_passes_progressive_distance', # TotDist
                'Short_passes_pct_short',           # Short Cmp%
                'Medium_passes_pct_medium',         # Medium Cmp%
                'Long_passes_pct_long',             # Long Cmp%
                'assisted_shots',                   # KP
                'passes_into_final_third',          # 1/3
                'passes_into_penalty_area',         # PPA
                'crosses_into_penalty_area',        # CrsPA
                'progressive_passes'                # PrgP
            ],
            'gca': [
                'SCA_sca',       # SCA
                'SCA_sca_per90', # SCA90
                'GCA_gca',       # GCA
                'GCA_gca_per90'  # GCA90
            ],
            'defense': [
                'Tackles_tackles',      # Tkl
                'Tackles_tackles_won',  # TklW
                'Challenges_challenges',       # Att
                'Challenges_challenges_lost',  # Lost
                'Blocks_blocks',        # Blocks
                'Blocks_blocked_shots', # Sh
                'Blocks_blocked_passes', # Pass
                'interceptions'         # Int
            ],
            'possession': [
                'Touches_touches',              # Touches
                'Touches_touches_def_pen_area', # Def Pen
                'Touches_touches_def_3rd',      # Def 3rd
                'Touches_touches_mid_3rd',      # Mid 3rd
                'Touches_touches_att_3rd',      # Att 3rd
                'Touches_touches_att_pen_area', # Att Pen
                'Take-Ons_take_ons',           # Att
                'Take-Ons_take_ons_won_pct',   # Succ%
                'Take-Ons_take_ons_tackled_pct', # Tkld%
                'Carries_carries',              # Carries
                'Carries_carries_progressive_distance', # ProDist
                'Carries_progressive_carries',   # ProgC
                'Carries_carries_into_final_third', # 1/3
                'Carries_carries_into_penalty_area', # CPA
                'Carries_miscontrols',          # Mis
                'Carries_dispossessed',         # Dis
                'Receiving_passes_received',     # Rec
                'Receiving_progressive_passes_received' # PrgR
            ],
            'misc': [
                'Performance_fouls',     # Fls
                'Performance_fouled',    # Fld
                'Performance_offsides',  # Off
                'Performance_crosses',   # Crs
                'Performance_ball_recoveries', # Recov
                'Aerial Duels_aerials_won',    # Won
                'Aerial Duels_aerials_lost',   # Lost
                'Aerial Duels_aerials_won_pct' # Won%
            ]
        }
        
        # Column name mappings for final output
        self.column_mappings = {
            'player': 'Player', 
            'nationality': 'Nation',
            'team': 'Team',
            'position': 'Position',
            'age': 'Age',
            'Playing Time_games': 'MP',
            'Playing Time_games_starts': 'Starts',
            'Playing Time_minutes': 'Minutes',
            'Performance_goals': 'Goals',
            'Performance_assists': 'Assists',
            'Performance_cards_yellow': 'Yellow Cards',
            'Performance_cards_red': 'Red Cards',
            'Expected_xg': 'xG',
            'Expected_xg_assist': 'xAG',
            'Progression_progressive_carries': 'PrgC',
            'Progression_progressive_passes': 'PrgP',
            'Progression_progressive_passes_received': 'PrgR',
            'Per 90 Minutes_goals_per90': 'Gls/90',
            'Per 90 Minutes_assists_per90': 'Ast/90',
            'Per 90 Minutes_xg_per90': 'xG/90',
            'Per 90 Minutes_xg_assist_per90': 'xGA/90',
            'Performance_gk_goals_against_per90': 'GA90',
            'Performance_gk_save_pct': 'Save%',
            'Performance_gk_clean_sheets_pct': 'CS%',
            'Penalty Kicks_gk_pens_save_pct': 'PKsv%',
            'Standard_shots_on_target_pct': 'SoT%',
            'Standard_shots_on_target_per90': 'SoT/90',
            'Standard_goals_per_shot': 'G/Sh',
            'Standard_average_shot_distance': 'Dist',
            'Total_passes_completed': 'Cmp',
            'Total_passes_pct': 'Cmp%',
            'Total_passes_progressive_distance': 'TotDist',
            'Short_passes_pct_short': 'Short%',
            'Medium_passes_pct_medium': 'Med%',
            'Long_passes_pct_long': 'Long%',
            'assisted_shots': 'KP',
            'passes_into_final_third': 'Passes 1/3',
            'passes_into_penalty_area': 'PPA',
            'crosses_into_penalty_area': 'CrsPA',
            'progressive_passes': 'PrgoP',
            'SCA_sca': 'SCA',
            'SCA_sca_per90': 'SCA90',
            'GCA_gca': 'GCA',
            'GCA_gca_per90': 'GCA90',
            'Tackles_tackles': 'Tkl',
            'Tackles_tackles_won': 'TklW',
            'Challenges_challenges': 'Att',
            'Challenges_challenges_lost': 'Lost',
            'Blocks_blocks': 'Blocks',
            'Blocks_blocked_shots': 'Sh',
            'Blocks_blocked_passes': 'Pass',
            'interceptions': 'Int',
            'Touches_touches': 'Touches',
            'Touches_touches_def_pen_area': 'Def Pen',
            'Touches_touches_def_3rd': 'Def 3rd',
            'Touches_touches_mid_3rd': 'Mid 3rd',
            'Touches_touches_att_3rd': 'Att 3rd',
            'Touches_touches_att_pen_area': 'Att Pen',
            'Take-Ons_take_ons': 'TO Att',
            'Take-Ons_take_ons_won_pct': 'Succ%',
            'Take-Ons_take_ons_tackled_pct': 'Tkld%',
            'Carries_carries': 'Carries',
            'Carries_carries_progressive_distance': 'ProDist',
            'Carries_progressive_carries': 'ProgC',
            'Carries_carries_into_final_third': 'Carries 1/3',
            'Carries_carries_into_penalty_area': 'CPA',
            'Carries_miscontrols': 'Mis',
            'Carries_dispossessed': 'Dis',
            'Receiving_passes_received': 'Rec',
            'Receiving_progressive_passes_received': 'ProgR',
            'Performance_fouls': 'Fls',
            'Performance_fouled': 'Fld',
            'Performance_offsides': 'Off',
            'Performance_crosses': 'Crs',
            'Performance_ball_recoveries': 'Recov',
            'Aerial Duels_aerials_won': 'Won',
            'Aerial Duels_aerials_lost': 'Aerial Lost',
            'Aerial Duels_aerials_won_pct': 'Won%'
        }
        
        self.driver = None
        self.stats_data = {} 

    def setup_driver(self):
        """Initialize and configure Chrome WebDriver"""
        try:
            chrome_options = Options()
            chrome_options.add_argument('--headless=new')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36')
            
            # Sử dụng selenium-manager (đơn giản hơn)
            driver = webdriver.Chrome(options=chrome_options)
            
            driver.set_page_load_timeout(30)
            
            logging.info("Chrome WebDriver initialized successfully")
            return driver
        except Exception as e:
            logging.error(f"Failed to initialize WebDriver: {str(e)}")
            raise

    def get_table_data(self, driver, url, table_id):
        """Fetch table data from the specified URL"""
        try:
            logging.info(f"Fetching data from {url}")
            driver.get(url)
            time.sleep(5)
            
            # Wait for table to be present
            wait = WebDriverWait(driver, 30)
            table = wait.until(EC.presence_of_element_located((By.ID, table_id)))
            
            if not table:
                raise ValueError(f"Table with ID '{table_id}' not found")
            
            # Scroll to ensure all data is loaded
            driver.execute_script("arguments[0].scrollIntoView(true);", table)
            time.sleep(2)
            
            # Get table HTML
            table_html = table.get_attribute('outerHTML')
            soup = BeautifulSoup(table_html, 'html.parser')
            
            # Get all header rows
            headers = []
            thead = soup.find('thead')
            if thead:
                # Get column groups (top header row)
                colgroups = []
                for tr in thead.find_all('tr'):
                    if 'over_header' in tr.get('class', []):
                        for th in tr.find_all(['th', 'td']):
                            colspan = int(th.get('colspan', 1))
                            colgroups.extend([th.text.strip()] * colspan)
                
                # Get stat columns (bottom header row)
                stat_headers = []
                for tr in thead.find_all('tr'):
                    if 'over_header' not in tr.get('class', []):
                        for th in tr.find_all(['th', 'td']):
                            if 'data-stat' in th.attrs:
                                stat_headers.append(th['data-stat'])
                            else:
                                stat_headers.append(th.text.strip())
                
                # Combine headers
                if len(colgroups) == len(stat_headers):
                    headers = [f"{cg}_{sh}" if cg else sh for cg, sh in zip(colgroups, stat_headers)]
                else:
                    headers = stat_headers
            
            if not headers:
                raise ValueError("No headers found in table")
            
            # Log available headers for debugging
            logging.info(f"Available headers: {headers}")
            
            # Get data rows
            rows = []
            tbody = soup.find('tbody')
            if not tbody:
                raise ValueError("No tbody found in table")
                
            for tr in tbody.find_all('tr'):
                # Skip header rows in tbody
                if 'class' in tr.attrs and ('thead' in tr['class'] or 'spacer' in tr['class']):
                    continue
                
                row = []
                tds = tr.find_all(['td', 'th'])
                
                # Skip if no cells found
                if not tds:
                    continue
                    
                # Process each cell
                for td in tds:
                    if 'data-stat' in td.attrs:
                        # Get the text content
                        text = td.text.strip()
                        # Handle special cases
                        if td['data-stat'] == 'player':
                            # Get player name and remove any special characters
                            text = ' '.join(text.split())
                        elif td['data-stat'] == 'nationality':
                            # Keep exact nationality code (e.g., 'ENG', 'FRA')
                            parts = td.text.strip().split()
                            if parts:
                                text = parts[-1]  # Get the last part (country code)
                            else:
                                text = ""
                        elif td['data-stat'] == 'age':
                            # Get age from data-age attribute if available
                            text = td.get('data-age', td.text.strip())
                            # If data-age is not available, try to get from title attribute
                            if not text:
                                text = td.get('title', td.text.strip())
                            # If neither is available, use the text content
                            if not text:
                                text = td.text.strip()
                        elif td['data-stat'] in ['team', 'position']:
                            # Keep original formatting for these fields
                            text = td.text.strip()
                        else:
                            # For numeric fields, keep only digits, decimal points, and minus signs
                            text = ''.join(c for c in text if c.isdigit() or c in '.-')
                            text = text if text else "N/a"

                        row.append(text)
                    else:
                        row.append(td.text.strip())
                
                # Only add non-empty rows
                if row and not all(cell == '' for cell in row):
                    rows.append(row)
            
            if not rows:
                raise ValueError("No data rows found in table")
            
            # Make sure all rows have the same length as headers
            uniform_rows = []
            for row in rows:
                if len(row) < len(headers):
                    # Pad row with empty strings if needed
                    row = row + [''] * (len(headers) - len(row))
                elif len(row) > len(headers):
                    # Trim row if needed
                    row = row[:len(headers)]
                uniform_rows.append(row)
            
            # Remove duplicate rows
            unique_rows = []
            seen = set()
            for row in uniform_rows:
                row_tuple = tuple(row)
                if row_tuple not in seen:
                    seen.add(row_tuple)
                    unique_rows.append(row)
            
            logging.info(f"Successfully retrieved {len(unique_rows)} unique rows of data")
            return headers, unique_rows
            
        except Exception as e:
            logging.error(f"Error fetching data from {url}: {str(e)}")
            logging.error("Traceback:", exc_info=True)  # Print full traceback
            raise
    def process_player_stats(self, data, required_stats):
        """Process player statistics from raw data"""
        try:
            if not isinstance(data, pd.DataFrame):
                logging.error("Input data must be a DataFrame")
                return None

            # Improved minutes column detection
            minutes_cols = ['Playing Time_minutes', 'minutes', 'Min', 'Mins']
            minutes_col = None
            for col in minutes_cols:
                if col in data.columns:
                    minutes_col = col
                    break

            if minutes_col:
                data = data.rename(columns={minutes_col: 'Minutes'})
                data['Minutes'] = pd.to_numeric(
                    data['Minutes'].astype(str).str.replace(r'[^\d.-]', '', regex=True),
                    errors='coerce'
                ).fillna(0)
                data = data[data['Minutes'] > 90]
                logging.info(f"Filtered {len(data)} players with >90 minutes")
            else:
                logging.warning("Minutes data not found. Skipping filtering.")

            # Unified column mapping
            column_mapping = {
                'player': 'Player',
                'nationality': 'Nation',
                'position': 'Position',
                'age': 'Age',
                'team': 'Team',
                'squad': 'Team',
                'Playing Time_games': 'MP',
                'Playing Time_games_starts': 'Starts',
            }
            data = data.rename(columns=column_mapping)

            # Select required columns
            required_columns = ['Player', 'Team', 'Nation', 'Position', 'Age', 'MP', 'Starts', 'Minutes']
            required_columns += [col for col in required_stats if col not in required_columns]
            data = data[[col for col in required_columns if col in data.columns]]

            return data

        except Exception as e:
            logging.error(f"Error in process_player_stats: {str(e)}", exc_info=True)
            return None
    def merge_stats(self, stats_dict):
        """Merge different types of statistics into a single DataFrame"""
        try:
            if 'standard' not in stats_dict or stats_dict['standard'] is None:
                logging.error("No standard stats available to merge")
                return None

            result_df = stats_dict['standard'].copy()
            logging.info(f"Starting merge with standard stats. Shape: {result_df.shape}")

            # Basic column renaming
            result_df = result_df.rename(columns=self.column_mappings)

            # Filter minutes
            if 'Minutes' in result_df.columns:
                result_df['Minutes'] = pd.to_numeric(
                    result_df['Minutes'].astype(str).str.replace(r'[^\d.-]', '', regex=True),
                    errors='coerce'
                ).fillna(0)
                result_df = result_df[result_df['Minutes'] > 90]

            # Merge other stats with conflict resolution
            stat_types = ['keeper', 'shooting', 'passing', 'gca', 'defense', 'possession', 'misc']
            for stat_type in stat_types:
                if stat_type in stats_dict and stats_dict[stat_type] is not None:
                    try:
                        stat_df = stats_dict[stat_type].copy().rename(columns=self.column_mappings)
                        
                        # Merge with suffix
                        result_df = pd.merge(
                            result_df,
                            stat_df,
                            on=['Player', 'Team'],
                            how='left',
                            suffixes=('', f'_{stat_type}')
                        )
                        # Remove duplicated columns
                        result_df = result_df.loc[:, ~result_df.columns.str.endswith(f'_{stat_type}')]
                        
                    except Exception as e:
                        logging.error(f"Error merging {stat_type}: {str(e)}")
                        continue

            # Final cleanup
            final_columns = [v for k, v in self.column_mappings.items() if v in result_df.columns]
            result_df = result_df[final_columns].drop_duplicates().fillna("N/a").sort_values(['Player', 'Team'])
            
            logging.info(f"Final merged DataFrame shape: {result_df.shape}")
            return result_df

        except Exception as e:
            logging.error(f"Error in merge_stats: {str(e)}", exc_info=True)
            return None

    def save_results(self, df, filename="results.csv"):
        """Save the results to a CSV file"""
        try:
            # Try to close the file if it's open
            try:
                if hasattr(self, '_output_file') and not self._output_file.closed:
                    self._output_file.close()
            except:
                pass
            
            # Try different filenames if the original is locked
            base_name = filename.rsplit('.', 1)[0]
            ext = filename.rsplit('.', 1)[1]
            counter = 0
            while counter < 100:  # Try up to 100 different filenames
                try:
                    if counter == 0:
                        current_filename = filename
                    else:
                        current_filename = f"{base_name}_{counter}.{ext}"
                    df.to_csv(current_filename, index=False, encoding='utf-8-sig')
                    logging.info(f"Results saved to {current_filename}")
                    break
                except PermissionError:
                    counter += 1
                    continue
                except Exception as e:
                    raise
            
            if counter == 100:
                raise Exception("Could not find a suitable filename to save results")
            
        except Exception as e:
            logging.error(f"Failed to save results: {str(e)}")
            raise

    def collect_all_stats(self):
        """Collect statistics from all tables"""
        try:
            # Initialize WebDriver
            self.driver = self.setup_driver()
            
            # Define URLs and table IDs
            urls = {
                'standard': 'https://fbref.com/en/comps/9/stats/Premier-League-Stats',
                'keeper': 'https://fbref.com/en/comps/9/keepers/Premier-League-Stats',
                'shooting': 'https://fbref.com/en/comps/9/shooting/Premier-League-Stats',
                'passing': 'https://fbref.com/en/comps/9/passing/Premier-League-Stats',
                'gca': 'https://fbref.com/en/comps/9/gca/Premier-League-Stats',
                'defense': 'https://fbref.com/en/comps/9/defense/Premier-League-Stats',
                'possession': 'https://fbref.com/en/comps/9/possession/Premier-League-Stats',
                'misc': 'https://fbref.com/en/comps/9/misc/Premier-League-Stats'
            }
            
            table_ids = {
                'standard': 'stats_standard',
                'keeper': 'stats_keeper',
                'shooting': 'stats_shooting',
                'passing': 'stats_passing',
                'gca': 'stats_gca',
                'defense': 'stats_defense',
                'possession': 'stats_possession',
                'misc': 'stats_misc'
            }
            
            # Collect data from each table
            for stat_type, url in urls.items():
                try:
                    logging.info(f"Collecting {stat_type} data from {url}")
                    table_id = table_ids[stat_type]
                    
                    # Try to get data with retries
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            headers, rows = self.get_table_data(self.driver, url, table_id)
                            if headers and rows:
                                # Convert to DataFrame
                                df = pd.DataFrame(rows, columns=headers)
                                # Store DataFrame in stats_data
                                self.stats_data[stat_type] = df
                                logging.info(f"Successfully collected {stat_type} data")
                                break
                        except Exception as e:
                            if attempt == max_retries - 1:
                                logging.warning(f"Failed to collect {stat_type} data after {max_retries} attempts: {str(e)}")
                            else:
                                logging.warning(f"Attempt {attempt + 1} failed for {stat_type}, retrying...")
                                time.sleep(5)
                except Exception as e:
                    logging.error(f"Error collecting {stat_type} data: {str(e)}")
                    continue
            
            # Process and merge all statistics
            if not self.stats_data:
                raise ValueError("No data collected from any table")

        # Xử lý standard stats trước
            if 'standard' not in self.stats_data:
                raise ValueError("Standard stats not found")

        # Gọi process_player_stats để xử lý dữ liệu
            base_df = self.process_player_stats(
                self.stats_data['standard'], 
                self.required_stats['standard']
            )
            if base_df is None:
                raise ValueError("Failed to process standard stats")

        # Merge các loại thống kê khác
            for stat_type in ['keeper', 'shooting', 'passing', 'gca', 'defense', 'possession', 'misc']:
                if stat_type in self.stats_data:
                    try:
                    # Gọi process_player_stats cho từng loại thống kê
                        df = self.process_player_stats(
                            self.stats_data[stat_type], 
                            self.required_stats[stat_type]
                        )
                        if df is not None:
                        # Merge vào base_df
                            base_df = pd.merge(
                                base_df, 
                                df, 
                                on=['Player', 'Team'], 
                                how='left'
                            )
                    except Exception as e:
                        logging.error(f"Error merging {stat_type}: {str(e)}")
                        continue
                
            # Fill missing values with "N/a"
            base_df = base_df.fillna("N/a")
            
            # Sort by Player and Squad
            base_df = base_df.sort_values(['Player', 'Team'])  
            
            self.save_results(base_df)
            
        except Exception as e:
            logging.error(f"Error in collect_all_stats: {str(e)}")
            raise

    def run(self):
        """Run the complete scraping process"""
        try:
            self.collect_all_stats()
            final_df = self.merge_stats(self.stats_data)
            self.save_results(final_df)
            logging.info("Scraping process completed successfully")
        except Exception as e:
            logging.error(f"Error during scraping process: {str(e)}")
            raise
        finally:
            if self.driver:
                self.driver.quit()

if __name__ == "__main__":
    scraper = FootballStatsScraper()
    scraper.run() 