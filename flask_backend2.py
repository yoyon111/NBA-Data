from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import time
import os
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
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

# CrewAI Tools
@tool("Search Recent NBA Info")
def search_recent_nba_info(query: str) -> str:
    """Search for recent NBA information using Gemini's web search capabilities.
    Use this to find recent player performance, injuries, team strategies, etc."""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content(
            f"Search the web for recent information about: {query}. Provide a concise summary of the most relevant and recent findings.",
            tools='google_search_retrieval'
        )
        return response.text
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
        
        # Define Agents with distinct personalities and roles - HIERARCHICAL MODE
        # These agents can now delegate to each other and have true back-and-forth discussions
        offensive_specialist = Agent(
            role='Offensive Analytics Specialist',
            goal=f'Identify and advocate for {player_name}\'s scoring opportunities and strengths in this matchup',
            backstory=f"""You are an offensive-minded analyst who deeply studies player strengths 
            and scoring patterns. You believe in {player_name}'s ability to exploit defensive weaknesses. 
            You use data and recent performance trends to support your optimistic view of offensive potential.
            You're enthusiastic but back up your claims with evidence. When the defensive specialist 
            challenges your points, you defend your position with additional evidence or concede when data proves you wrong.""",
            verbose=True,
            allow_delegation=True,  # Now agents can delegate questions to each other
            tools=[search_recent_nba_info, analyze_statistical_matchup],
            llm=gemini_llm
        )
        
        defensive_specialist = Agent(
            role='Defensive Strategy Analyst',
            goal=f'Identify how {team_name}\'s defense can contain and limit {player_name}',
            backstory=f"""You are a defensive strategist who focuses on team schemes, individual 
            defenders, and how defenses can neutralize offensive threats. You respect strong defense 
            and believe {team_name} has the tools to limit scoring. You analyze defensive rankings, 
            recent adjustments, and specific matchup advantages. You're skeptical but fair.
            When the offensive specialist makes claims, you challenge them with counter-evidence 
            and ask them to justify their optimistic projections.""",
            verbose=True,
            allow_delegation=True,  # Can challenge and question the offensive specialist
            tools=[search_recent_nba_info, analyze_statistical_matchup],
            llm=gemini_llm
        )
        
        neutral_analyst = Agent(
            role='Neutral Statistical Analyst',
            goal='Provide balanced, data-driven analysis without bias toward offense or defense',
            backstory="""You are a pure statistician who calls it as the data shows. You don't 
            favor offense or defense - you simply present probabilities, trends, and likely outcomes 
            based on numbers. You mediate between optimistic and pessimistic views with cold, hard facts. 
            You're the voice of reason and objectivity. You can ask both specialists to clarify their 
            arguments or provide additional data when their claims seem unsupported.""",
            verbose=True,
            allow_delegation=True,  # Can request clarification from both specialists
            tools=[search_recent_nba_info, analyze_statistical_matchup],
            llm=gemini_llm
        )
        
        betting_strategist = Agent(
            role='Sports Betting Strategist',
            goal='Synthesize all perspectives into actionable betting recommendations',
            backstory="""You are the final decision-maker who listens to all analysts and makes 
            the call. You weigh competing arguments, identify which analyst made the strongest case, 
            and provide concrete betting advice. You're decisive but acknowledge uncertainty. 
            You translate analysis into props, spreads, and confidence levels. You can ask any 
            analyst follow-up questions to ensure you fully understand their reasoning before 
            making your final recommendation.""",
            verbose=True,
            allow_delegation=True,  # Can question any analyst for clarification
            tools=[search_recent_nba_info],
            llm=gemini_llm
        )
        
        # Define Tasks for collaborative analysis
        matchup_context = f"""
MATCHUP: {player_name} vs {team_name}

PLAYER OFFENSIVE STATS:
{player_stats_text}

TEAM DEFENSIVE STATS:
{defense_stats_text}

Use the search tool to find:
1. Recent performance trends for {player_name} (last 5-10 games)
2. {team_name}'s recent defensive schemes and adjustments
3. Any recent head-to-head history
4. Injury reports or lineup changes
"""
        
        offensive_task = Task(
            description=f"""{matchup_context}
            
As the Offensive Specialist, build the case for why {player_name} will succeed:
- Identify his 2-3 strongest play types and why they'll work
- Find recent hot streaks or momentum
- Search for defensive vulnerabilities in {team_name}'s scheme
- Make your most compelling argument for the OVER on player props
- Be prepared to defend your position if challenged by other analysts

Keep your analysis focused (150-200 words). Be enthusiastic but evidence-based.""",
            agent=offensive_specialist,
            expected_output="Offensive analysis highlighting player strengths and scoring opportunities, with evidence to support claims"
        )
        
        defensive_task = Task(
            description=f"""{matchup_context}
            
As the Defensive Specialist, build the case for why {team_name} will contain {player_name}:
- Identify their 2-3 best defensive play types that match player's strengths
- Search for recent defensive improvements or schemes
- Highlight specific defenders who can match up
- Make your most compelling argument for the UNDER on player props
- Challenge the Offensive Specialist's claims if you find counter-evidence

Keep your analysis focused (150-200 words). Be realistic but defense-focused.""",
            agent=defensive_specialist,
            expected_output="Defensive analysis highlighting containment strategies and limitations, with rebuttals to offensive arguments"
        )
        
        neutral_task = Task(
            description=f"""{matchup_context}
            
As the Neutral Analyst, provide the data-driven perspective:
- Review both the offensive and defensive specialists' arguments
- If their claims seem unsupported, delegate questions back to them for clarification
- Present statistical probabilities and trends objectively
- Identify key factors that truly matter vs noise
- Call out any weak arguments from either specialist
- Synthesize the debate into objective insights

Keep your analysis focused (150-200 words). Be completely objective.""",
            agent=neutral_analyst,
            expected_output="Balanced statistical analysis with objective probabilities, acting as mediator between opposing views",
            context=[offensive_task, defensive_task]
        )
        
        betting_task = Task(
            description=f"""Review all three analysts' perspectives and any debates that occurred.

Your job:
1. Summarize which analyst made the strongest points and why
2. Identify where analysts agree/disagree and what that means
3. If you need clarification on any analyst's reasoning, ask them directly
4. Make your final verdict on the most likely outcome
5. Provide 2-3 SPECIFIC betting recommendations (player props, team totals, etc.)
6. Assign confidence level (HIGH/MEDIUM/LOW) with clear reasoning
7. Explain how the debate between analysts influenced your decision

Be decisive but honest about uncertainty. This is the final word (200-250 words).""",
            agent=betting_strategist,
            expected_output="Final betting recommendations with confidence levels, incorporating insights from the collaborative debate",
            context=[offensive_task, defensive_task, neutral_task]
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