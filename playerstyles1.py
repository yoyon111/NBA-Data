from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
from fake_useragent import UserAgent
import time
import unicodedata
from webdriver_manager.chrome import ChromeDriverManager

def normalize_text(text):
    """Remove accents, extra spaces, and convert to lowercase for matching"""
    # Convert to string and lowercase
    text = str(text).lower()
    # Remove accents by decomposing unicode and filtering out combining characters
    text = ''.join(c for c in unicodedata.normalize('NFD', text) 
                   if unicodedata.category(c) != 'Mn')
    # Replace any whitespace with a single space and strip
    text = ' '.join(text.split())
    return text

def get_offensive_stats(url, play_type, player_name=None):
    ua = UserAgent()
    options = Options()
    options.add_argument('--headless=new')
    options.add_argument(f'user-agent={ua.random}')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--log-level=3')
    options.add_argument('--disable-dev-shm-usage')
    service = Service(ChromeDriverManager().install())
    
    try:
        print(f"Loading {play_type} stats from {url}...")
        driver = webdriver.Chrome(service=service, options=options)
        driver.get(url)
        
        wait = WebDriverWait(driver, 15)
        time.sleep(3)
        
        try:
            print(f"Looking for 'ALL' button for {play_type}...")
            all_button_selectors = [
                "//button[text()='All']",
                "//button[contains(text(), 'All')]",
                "//button[@value='All']",
                "//button[contains(@class, 'DropDown') and contains(., 'All')]",
                "//select/option[@value='-1']",
                "//button[contains(@aria-label, 'All')]"
            ]
            
            all_button = None
            for selector in all_button_selectors:
                try:
                    all_button = driver.find_element(By.XPATH, selector)
                    print(f"Found 'ALL' button with selector: {selector}")
                    break
                except:
                    continue
            
            if all_button:
                all_button.click()
                print(f"Clicked 'ALL' button for {play_type}, waiting for table to load...")
                time.sleep(5)
            else:
                print(f"Could not find 'ALL' button, will try to scrape current view for {play_type}")
        except Exception as e:
            print(f"Error clicking 'ALL' button for {play_type}: {e}")
            print("Proceeding with current view...")
        
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        table = soup.find('table', class_='Crom_table__p1iZz')
        
        if not table:
            print(f"Could not find {play_type} stats table.")
            driver.quit()
            return None
        
        headers = [th.text.strip() for th in table.find('thead').find_all('th')]
        print(f"{play_type} headers: {headers}")
        
        all_rows = []
        for tr in table.find('tbody').find_all('tr'):
            row = [td.text.strip() for td in tr.find_all('td')]
            all_rows.append(row)
        
        print(f"Scraped {len(all_rows)} total rows from {play_type}")
        
        driver.quit()
        
        df = pd.DataFrame(all_rows, columns=headers)
        
        if 'PTS' in df.columns:
            df['PTS'] = pd.to_numeric(df['PTS'], errors='coerce')
        
        desired_cols = ['TEAM', 'PLAYER', 'PTS']
        available_cols = [col for col in desired_cols if col in df.columns]
        
        if not available_cols:
            print(f"None of the expected columns ({desired_cols}) found for {play_type}!")
            return df
        
        df = df[available_cols]
        df['Play_Type'] = play_type
        
        if player_name:
            if 'PLAYER' not in df.columns:
                print(f"PLAYER column not found for {play_type}, cannot filter by name!")
                return None
            player_df = df[df['PLAYER'].str.contains(player_name, case=False, na=False)]
            if player_df.empty:
                return None
            return player_df
        
        return df
        
    except Exception as e:
        print(f"Error scraping {play_type}: {e}")
        if 'driver' in locals():
            driver.quit()
        return None

def get_defensive_stats(url, play_type):
    ua = UserAgent()
    options = Options()
    options.add_argument('--headless=new')
    options.add_argument(f'user-agent={ua.random}')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--log-level=3')
    options.add_argument('--disable-dev-shm-usage')
    service = Service(ChromeDriverManager().install())
    
    try:
        print(f"Loading {play_type} defensive stats from {url}...")
        driver = webdriver.Chrome(service=service, options=options)
        driver.get(url)
        
        time.sleep(3)
        
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        table = soup.find('table', class_='Crom_table__p1iZz')
        
        if not table:
            print(f"Could not find {play_type} defensive stats table.")
            driver.quit()
            return None
        
        headers = [th.text.strip() for th in table.find('thead').find_all('th')]
        print(f"{play_type} defensive headers: {headers}")
        
        all_rows = []
        rank = 1
        for tr in table.find('tbody').find_all('tr'):
            row = [td.text.strip() for td in tr.find_all('td')]
            # Add rank as first element
            all_rows.append([rank] + row)
            rank += 1
        
        print(f"Scraped {len(all_rows)} teams from {play_type} defense")
        
        driver.quit()
        
        # Add 'RANK' as first column
        df = pd.DataFrame(all_rows, columns=['RANK'] + headers)
        
        # PPP should be in 5th column (index 4 in original, now index 5 with RANK added)
        # But let's find it by name to be safe
        if 'PPP' in df.columns:
            df['PPP'] = pd.to_numeric(df['PPP'], errors='coerce')
        
        # Keep TEAM (column 1), PPP (column 5), and RANK
        if 'TEAM' not in df.columns or 'PPP' not in df.columns:
            print(f"Missing TEAM or PPP column for {play_type} defense!")
            return None
        
        df = df[['RANK', 'TEAM', 'PPP']]
        df['Play_Type'] = play_type
        
        return df
        
    except Exception as e:
        print(f"Error scraping {play_type} defense: {e}")
        if 'driver' in locals():
            driver.quit()
        return None

def main():
    # Player offensive stats URLs
    offensive_play_types = {
        "Pick-and-Roll": "https://www.nba.com/stats/players/ball-handler?dir=D&sort=PTS",
        "Isolation": "https://www.nba.com/stats/players/isolation",
        "Transition": "https://www.nba.com/stats/players/transition",
        "Roll Man": "https://www.nba.com/stats/players/roll-man",
        "Post-Up": "https://www.nba.com/stats/players/playtype-post-up",
        "Spot-Up": "https://www.nba.com/stats/players/spot-up",
        "Cut": "https://www.nba.com/stats/players/cut",
        "Off Screen": "https://www.nba.com/stats/players/off-screen",
        "Putbacks": "https://www.nba.com/stats/players/putbacks",
        "Hand-Off": "https://www.nba.com/stats/players/hand-off"
    }
    
    # Team defensive stats URLs
    defensive_play_types = {
        "Isolation": "https://www.nba.com/stats/teams/isolation?TypeGrouping=defensive&dir=A&sort=PPP",
        "Transition": "https://www.nba.com/stats/teams/transition?TypeGrouping=defensive&dir=A&sort=PPP",
        "Pick-and-Roll": "https://www.nba.com/stats/teams/ball-handler?TypeGrouping=defensive&dir=A&sort=PPP",
        "Roll Man": "https://www.nba.com/stats/teams/roll-man?TypeGrouping=defensive&dir=A&sort=PPP",
        "Post-Up": "https://www.nba.com/stats/teams/playtype-post-up?TypeGrouping=defensive&dir=A&sort=PPP",
        "Spot-Up": "https://www.nba.com/stats/teams/spot-up?TypeGrouping=defensive&dir=A&sort=PPP",
        "Hand-Off": "https://www.nba.com/stats/teams/hand-off?TypeGrouping=defensive&dir=A&sort=PPP",
        "Off Screen": "https://www.nba.com/stats/teams/off-screen?TypeGrouping=defensive&dir=A&sort=PPP",
        "Putbacks": "https://www.nba.com/stats/teams/putbacks?TypeGrouping=defensive&dir=A&sort=PPP"
    }
    
    # Scrape offensive stats
    all_offensive_stats = {}
    print("=== SCRAPING OFFENSIVE STATS ===")
    for play_type, url in offensive_play_types.items():
        print(f"\nScraping {play_type} offensive stats...")
        stats = get_offensive_stats(url, play_type)
        if stats is not None:
            all_offensive_stats[play_type] = stats
    
    # Scrape defensive stats
    all_defensive_stats = {}
    print("\n\n=== SCRAPING DEFENSIVE STATS ===")
    for play_type, url in defensive_play_types.items():
        print(f"\nScraping {play_type} defensive stats...")
        stats = get_defensive_stats(url, play_type)
        if stats is not None:
            all_defensive_stats[play_type] = stats
    
    if not all_offensive_stats:
        print("No offensive data was successfully scraped.")
        return
    
    while True:
        player = input("\nEnter a player name to search (or press Enter to exit): ").strip()
        if not player:
            print("Exiting program.")
            break
        
        # Get player's offensive stats
        player_data = []
        player_team = None
        
        # Normalize search term
        search_normalized = normalize_text(player)
        
        for play_type, df in all_offensive_stats.items():
            if 'PLAYER' in df.columns and 'PTS' in df.columns and 'TEAM' in df.columns:
                # Match if search term is found anywhere in the normalized player name
                def matches_player(name):
                    name_normalized = normalize_text(name)
                    return search_normalized in name_normalized
                
                player_stats = df[df['PLAYER'].apply(matches_player)]
                if not player_stats.empty:
                    for _, row in player_stats.iterrows():
                        player_data.append({
                            "Play_Type": play_type,
                            "TEAM": row['TEAM'],
                            "PTS": row['PTS']
                        })
                        if player_team is None:
                            player_team = row['TEAM']
        
        if not player_data:
            print(f"No data found for {player} across any play type.")
            continue
        
        # Display player's offensive stats
        team_groups = {}
        for entry in player_data:
            team = entry['TEAM']
            if team not in team_groups:
                team_groups[team] = []
            team_groups[team].append(entry)
        
        print(f"\n{'='*60}")
        print(f"Play type points for {player} (grouped by team, sorted by points):")
        print(f"{'='*60}")
        for team in team_groups:
            print(f"\nTeam: {team}")
            sorted_team_data = sorted(team_groups[team], key=lambda x: x["PTS"], reverse=True)
            for entry in sorted_team_data:
                print(f"  {entry['Play_Type']}: {entry['PTS']} PTS")
        
        # Now ask for team name for defensive rankings
        team_search = input("\nEnter a team name to see their defensive rankings (or press Enter to skip): ").strip()
        if not team_search:
            continue
        
        # Get list of play types the player actually has stats for
        player_play_types = set(entry['Play_Type'] for entry in player_data)
        
        # Find defensive rankings for the team, filtered by player's play types
        print(f"\n{'='*60}")
        print(f"Defensive PPP Rankings for teams matching '{team_search}':")
        print(f"(Showing only play types where {player} has offensive stats)")
        print(f"{'='*60}")
        
        found_team = False
        team_defense_data = []
        
        for play_type, df in all_defensive_stats.items():
            # Only include play types that the player has stats for
            if play_type not in player_play_types:
                continue
                
            # Filter for team name (case insensitive, partial match)
            team_stats = df[df['TEAM'].str.contains(team_search, case=False, na=False)]
            if not team_stats.empty:
                found_team = True
                for _, row in team_stats.iterrows():
                    team_defense_data.append({
                        "Play_Type": play_type,
                        "TEAM": row['TEAM'],
                        "RANK": row['RANK'],
                        "PPP": row['PPP']
                    })
        
        if not found_team:
            print(f"No team found matching '{team_search}'")
        else:
            # Group by team name (in case multiple matches)
            defense_by_team = {}
            for entry in team_defense_data:
                team = entry['TEAM']
                if team not in defense_by_team:
                    defense_by_team[team] = []
                defense_by_team[team].append(entry)
            
            # Display defensive rankings for each matching team
            for team_name, rankings in defense_by_team.items():
                print(f"\n{team_name} Defensive Rankings (lower rank = better defense):")
                
                # Create a map of play_type to defensive data
                defense_map = {entry['Play_Type']: entry for entry in rankings}
                
                # Display in the same order as player's offensive stats
                for entry in sorted_team_data:
                    play_type = entry['Play_Type']
                    if play_type in defense_map:
                        defense_entry = defense_map[play_type]
                        print(f"  {play_type}: Rank #{defense_entry['RANK']} ({defense_entry['PPP']:.2f} PPP)")

if __name__ == "__main__":
    main()