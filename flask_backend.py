from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import time
import os
import google.generativeai as genai
from playerstyles1 import get_offensive_stats, get_defensive_stats, normalize_text

app = Flask(__name__)
CORS(app)

# Configure Gemini API
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'YOUR_API_KEY_HERE')
genai.configure(api_key=GEMINI_API_KEY)

# Cache storage
offensive_cache = {}
defensive_cache = {}
cache_timestamp = None
CACHE_DURATION = 3600  # 1 hour

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

def refresh_cache():
    """Background task to refresh data cache"""
    global offensive_cache, defensive_cache, cache_timestamp
    
    print("Starting cache refresh...")
    
    # Scrape offensive stats
    temp_offensive = {}
    for play_type, url in offensive_play_types.items():
        print(f"Caching {play_type}...")
        stats = get_offensive_stats(url, play_type)
        if stats is not None:
            temp_offensive[play_type] = stats
    
    # Scrape defensive stats
    temp_defensive = {}
    for play_type, url in defensive_play_types.items():
        print(f"Caching {play_type} defense...")
        stats = get_defensive_stats(url, play_type)
        if stats is not None:
            temp_defensive[play_type] = stats
    
    offensive_cache = temp_offensive
    defensive_cache = temp_defensive
    cache_timestamp = time.time()
    print("Cache refresh complete!")

def ensure_cache():
    """Ensure cache is fresh"""
    global cache_timestamp
    if not cache_timestamp or (time.time() - cache_timestamp) > CACHE_DURATION:
        refresh_cache()

@app.route('/api/player/<player_name>', methods=['GET'])
def get_player(player_name):
    """Get player offensive stats"""
    ensure_cache()
    
    search_normalized = normalize_text(player_name)
    player_data = []
    
    for play_type, df in offensive_cache.items():
        if 'PLAYER' in df.columns and 'PTS' in df.columns and 'TEAM' in df.columns:
            def matches_player(name):
                return search_normalized in normalize_text(name)
            
            player_stats = df[df['PLAYER'].apply(matches_player)]
            if not player_stats.empty:
                for _, row in player_stats.iterrows():
                    player_data.append({
                        "playType": play_type,
                        "team": row['TEAM'],
                        "pts": float(row['PTS']) if row['PTS'] else 0,
                        "player": row['PLAYER']
                    })
    
    if not player_data:
        return jsonify({"error": "Player not found"}), 404
    
    return jsonify({
        "player": player_data[0]["player"] if player_data else player_name,
        "data": player_data
    })

@app.route('/api/defense/<team_name>', methods=['GET'])
def get_defense(team_name):
    """Get team defensive stats"""
    ensure_cache()
    
    defense_data = []
    
    for play_type, df in defensive_cache.items():
        team_stats = df[df['TEAM'].str.contains(team_name, case=False, na=False)]
        if not team_stats.empty:
            for _, row in team_stats.iterrows():
                defense_data.append({
                    "playType": play_type,
                    "team": row['TEAM'],
                    "rank": int(row['RANK']),
                    "ppp": float(row['PPP']) if row['PPP'] else 0
                })
    
    if not defense_data:
        return jsonify({"error": "Team not found"}), 404
    
    return jsonify({
        "team": defense_data[0]["team"] if defense_data else team_name,
        "data": defense_data
    })

@app.route('/api/matchup/<player_name>/<team_name>', methods=['GET'])
def get_matchup(player_name, team_name):
    """Get player vs team matchup"""
    ensure_cache()
    
    # Get player data
    player_response = get_player(player_name)
    if player_response[1] == 404:
        return player_response
    
    # Get defense data
    defense_response = get_defense(team_name)
    if defense_response[1] == 404:
        return defense_response
    
    player_json = player_response[0].get_json()
    defense_json = defense_response[0].get_json()
    
    return jsonify({
        "player": player_json,
        "defense": defense_json
    })

@app.route('/api/ai-analysis', methods=['POST'])
def ai_analysis():
    """Gemini AI analysis with web search"""
    try:
        data = request.json
        player_name = data.get('playerName')
        team_name = data.get('teamName')
        player_stats = data.get('playerStats', [])
        defense_stats = data.get('defenseStats', [])
        
        if not player_name or not team_name:
            return jsonify({"error": "Missing player or team name"}), 400
        
        # Format stats for the prompt
        player_stats_text = ', '.join([
            f"{s['playType']}: {s['pts']:.1f} PTS" 
            for s in sorted(player_stats, key=lambda x: x['pts'], reverse=True)
        ])
        
        defense_stats_text = ', '.join([
            f"{s['playType']}: Rank #{s['rank']} ({s['ppp']:.2f} PPP)" 
            for s in sorted(defense_stats, key=lambda x: x['rank'])
        ])
        
        prompt = f"""You are an NBA analyst with access to current information. Analyze this matchup using both the provided stats AND current information from web searches about the player's recent performance and the team's defensive strategies.

Player: {player_name}
Offensive Stats by Play Type: {player_stats_text}

Opponent: {team_name}
Defensive Stats by Play Type: {defense_stats_text}

Search for recent information about:
1. {player_name}'s current playing style, recent performance, injuries, or hot streaks
2. {team_name}'s defensive strategies, recent defensive performance, and key defenders
3. Any recent head-to-head matchups between {player_name} and {team_name}

Based on the stats above AND your web research, provide:
1. Player's strongest offensive play types and how they've been performing recently
2. Team's defensive weaknesses and recent defensive trends
3. Specific matchup advantages where player's strengths align with defensive weaknesses
4. 2-3 tactical betting insights or predictions for this matchup
5. Any concerns or areas where the defense might contain the player

Keep analysis concise but insightful (250-350 words). Cite any recent stats or news you find."""

        # Try with retry logic for quota limits
        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Use Gemini 2.5 Flash model (updated from 2.0)
                model = genai.GenerativeModel('gemini-2.5-flash')
                
                response = model.generate_content(prompt)
                
                return jsonify({
                    "analysis": response.text,
                    "player": player_name,
                    "team": team_name
                })
            except Exception as quota_error:
                error_str = str(quota_error)
                if 'quota' in error_str.lower() or '429' in error_str:
                    if attempt < max_retries - 1:
                        print(f"Quota exceeded, waiting 5 seconds before retry {attempt + 1}...")
                        time.sleep(5)
                        continue
                    else:
                        return jsonify({
                            "error": "Rate limit exceeded. Please wait a minute and try again. The free tier has limited requests per minute."
                        }), 429
                raise
        
    except Exception as e:
        print(f"AI Analysis error: {e}")
        error_message = str(e)
        if 'quota' in error_message.lower() or '429' in error_message:
            return jsonify({
                "error": "Rate limit exceeded. Please wait a minute and try again."
            }), 429
        return jsonify({"error": error_message}), 500

@app.route('/api/refresh', methods=['POST'])
def manual_refresh():
    """Manually trigger cache refresh"""
    thread = threading.Thread(target=refresh_cache)
    thread.start()
    return jsonify({"message": "Cache refresh started"})

@app.route('/api/status', methods=['GET'])
def status():
    """Check cache status"""
    return jsonify({
        "cached": cache_timestamp is not None,
        "age_seconds": time.time() - cache_timestamp if cache_timestamp else None,
        "offensive_types": len(offensive_cache),
        "defensive_types": len(defensive_cache)
    })

if __name__ == '__main__':
    # Initial cache load
    refresh_cache()
    
    # Run server
    app.run(debug=True, host='0.0.0.0', port=5000)