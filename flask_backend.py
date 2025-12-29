from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import time
import os
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from playerstyles1 import get_offensive_stats, get_defensive_stats, normalize_text

app = Flask(__name__)
CORS(app)

# Configure Gemini API
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'YOUR_API_KEY_HERE')

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

# CrewAI Tools
@tool("Search Recent NBA Info")
def search_recent_nba_info(query: str) -> str:
    """Search for recent NBA information using Gemini's web search capabilities.
    Use this to find recent player performance, injuries, team strategies, etc."""
    try:
        # Using ChatGoogleGenerativeAI for web search (langchain integration)
        search_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=GEMINI_API_KEY,
            temperature=0.3
        )
        
        prompt = f"""Search the web for recent NBA information about: {query}
        
Provide a concise summary (2-3 sentences) of the most relevant and recent findings.
Focus on:
- Recent game performance (last 5-10 games)
- Current injuries or lineup changes
- Head-to-head matchup history
- Recent trends or momentum"""
        
        response = search_llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Search failed: {str(e)}"

@tool("Analyze Statistical Matchup")
def analyze_statistical_matchup(player_stats: str, defense_stats: str) -> str:
    """Analyze the statistical matchup between player offense and team defense.
    Returns key insights about favorable/unfavorable matchups."""
    # This tool helps agents reason about the raw stats
    return f"""Statistical Analysis:
Player Stats: {player_stats}
Defense Stats: {defense_stats}

Key points to consider:
- Compare player's strongest play types vs defense's weakest areas
- Look for rank discrepancies (player excels where defense struggles)
- Consider PPP efficiency vs defensive PPP allowed"""

@app.route('/api/ai-analysis', methods=['POST'])
def ai_analysis():
    """CrewAI multi-agent collaborative analysis"""
    try:
        data = request.json
        player_name = data.get('playerName')
        team_name = data.get('teamName')
        player_stats = data.get('playerStats', [])
        defense_stats = data.get('defenseStats', [])
        
        if not player_name or not team_name:
            return jsonify({"error": "Missing player or team name"}), 400
        
        # Format stats for agents
        player_stats_text = ', '.join([
            f"{s['playType']}: {s['pts']:.1f} PTS" 
            for s in sorted(player_stats, key=lambda x: x['pts'], reverse=True)
        ])
        
        defense_stats_text = ', '.join([
            f"{s['playType']}: Rank #{s['rank']} ({s['ppp']:.2f} PPP)" 
            for s in sorted(defense_stats, key=lambda x: x['rank'])
        ])
        
        print(f"🤖 Starting CrewAI analysis for {player_name} vs {team_name}...")
        
        # Configure Gemini LLM for CrewAI
        gemini_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=GEMINI_API_KEY,
            temperature=0.7
        )
        
        # Define Agents with debate-focused personalities
        # These agents will actively engage in discussion rather than just write reports
        offensive_specialist = Agent(
            role='Offensive Analytics Specialist',
            goal=f'Advocate for {player_name}\'s scoring potential and defend your position in debate',
            backstory=f"""You are an offensive-minded analyst in a live debate about {player_name} vs {team_name}. 
            You believe in the player's ability to score. You actively listen to counter-arguments and respond 
            with evidence. If the defensive analyst makes a good point, acknowledge it but find counter-evidence. 
            You're in a conversation, not writing a report. Be conversational, challenge claims, ask questions.""",
            verbose=True,
            allow_delegation=True,
            tools=[search_recent_nba_info, analyze_statistical_matchup],
            llm=gemini_llm
        )
        
        defensive_specialist = Agent(
            role='Defensive Strategy Analyst',
            goal=f'Advocate for {team_name}\'s defensive ability and challenge offensive claims',
            backstory=f"""You are a defensive strategist in a live debate about {player_name} vs {team_name}. 
            You believe the defense can contain scoring. You actively challenge the offensive analyst's claims. 
            When they cite stats, you find counter-stats. When they're optimistic, you provide reality checks. 
            You're in a conversation, not writing a report. Be conversational, push back on weak arguments.""",
            verbose=True,
            allow_delegation=True,
            tools=[search_recent_nba_info, analyze_statistical_matchup],
            llm=gemini_llm
        )
        
        neutral_analyst = Agent(
            role='Neutral Statistical Moderator',
            goal='Moderate the debate and provide objective statistical truth',
            backstory="""You are the moderator of this debate. When the offensive and defensive analysts make 
            claims, you fact-check them. You ask them to clarify vague statements. You point out when someone 
            is cherry-picking data. You're not writing a report - you're moderating a live discussion. 
            Ask questions, request evidence, call out bias.""",
            verbose=True,
            allow_delegation=True,
            tools=[search_recent_nba_info, analyze_statistical_matchup],
            llm=gemini_llm
        )
        
        betting_strategist = Agent(
            role='Sports Betting Decision Maker',
            goal='Listen to the full debate and make the final betting call',
            backstory="""You are listening to this debate and will make the final betting recommendation. 
            You can ask any analyst follow-up questions. You want to understand their reasoning fully 
            before making your call. You're decisive but thoughtful. You translate debate conclusions 
            into specific betting recommendations.""",
            verbose=True,
            allow_delegation=True,
            tools=[search_recent_nba_info],
            llm=gemini_llm
        )
        
        # Single collaborative debate task instead of rigid sequential tasks
        matchup_context = f"""
MATCHUP: {player_name} vs {team_name}

PLAYER OFFENSIVE STATS:
{player_stats_text}

TEAM DEFENSIVE STATS:
{defense_stats_text}
"""
        
        debate_task = Task(
            description=f"""{matchup_context}

DEBATE INSTRUCTIONS:
You are all participating in a collaborative analysis debate. This is a CONVERSATION, not separate reports.

Offensive Specialist:
- Start by making your case for why {player_name} will succeed
- Search for recent performance data
- Be prepared to defend your claims when challenged

Defensive Specialist:
- Listen to the offensive case and challenge weak points
- Search for counter-evidence
- Make your case for how {team_name} will contain {player_name}

Neutral Analyst (Moderator):
- Fact-check claims from both sides
- Ask clarifying questions when arguments are vague
- Request additional evidence when needed
- Keep the debate focused on data

Betting Strategist:
- Listen to the full debate
- Ask follow-up questions to any analyst
- Synthesize the discussion into betting recommendations
- Make your final call with confidence levels

The debate should flow naturally. Challenge each other. Respond to points made. 
Search for evidence as needed. Come to a collaborative conclusion.

FINAL OUTPUT REQUIREMENTS:
- Clear summary of the debate's key points
- 2-3 specific betting recommendations
- Confidence levels (HIGH/MEDIUM/LOW) with reasoning""",
            agent=betting_strategist,  # Betting strategist coordinates but all agents participate
            expected_output="""A comprehensive betting analysis that includes:
1. Summary of offensive arguments and evidence
2. Summary of defensive counter-arguments and evidence  
3. Key statistical insights from neutral analysis
4. 2-3 specific betting recommendations (player props, totals, etc.)
5. Confidence levels with clear reasoning
6. Explanation of how the debate informed the final decision"""
        )
        
        # Create the Crew with HIERARCHICAL process for true collaboration
        # The manager will coordinate back-and-forth discussion between agents
        analysis_crew = Crew(
            agents=[offensive_specialist, defensive_specialist, neutral_analyst, betting_strategist],
            tasks=[offensive_task, defensive_task, neutral_task, betting_task],
            process=Process.hierarchical,  # Changed from sequential to hierarchical
            manager_llm=gemini_llm,  # Manager coordinates the conversation
            verbose=True
        )
        
        # Execute the crew
        print("🚀 Crew kickoff - agents are collaborating...")
        result = analysis_crew.kickoff()
        
        # Format the output
        complete_analysis = f"""🔵 OFFENSIVE SPECIALIST PERSPECTIVE
{offensive_task.output.raw_output if hasattr(offensive_task.output, 'raw_output') else str(offensive_task.output)}

---

🔴 DEFENSIVE SPECIALIST PERSPECTIVE
{defensive_task.output.raw_output if hasattr(defensive_task.output, 'raw_output') else str(defensive_task.output)}

---

⚪ NEUTRAL ANALYST PERSPECTIVE
{neutral_task.output.raw_output if hasattr(neutral_task.output, 'raw_output') else str(neutral_task.output)}

---

🎯 FINAL BETTING RECOMMENDATION
{betting_task.output.raw_output if hasattr(betting_task.output, 'raw_output') else str(betting_task.output)}"""
        
        print("✅ CrewAI analysis complete!")
        
        return jsonify({
            "analysis": complete_analysis,
            "player": player_name,
            "team": team_name,
            "crew_result": str(result)
        })
        
    except Exception as e:
        print(f"❌ CrewAI Analysis error: {e}")
        import traceback
        traceback.print_exc()
        
        error_message = str(e)
        if 'quota' in error_message.lower() or '429' in error_message:
            return jsonify({
                "error": "Rate limit exceeded. Please wait a minute and try again."
            }), 429
        return jsonify({"error": f"Analysis failed: {error_message}"}), 500

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