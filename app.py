"""
Coachella Resource Bot
A LangChain-powered chatbot that helps users plan their Coachella experience.
Tools: Artist/Schedule Lookup, Artist Recommendations, Budget Estimator, Festival Tips RAG
"""

import os
import csv
import json
import datetime
import re
from pathlib import Path

import gradio as gr
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

LOG_FILE = LOGS_DIR / f"conversations_{datetime.date.today()}.jsonl"

# ── Load Data ────────────────────────────────────────────────────────────────
def load_artists() -> list[dict]:
    with open(DATA_DIR / "artists.csv", newline="") as f:
        return list(csv.DictReader(f))

def load_budget_items() -> list[dict]:
    with open(DATA_DIR / "budget_items.csv", newline="") as f:
        return list(csv.DictReader(f))

def load_festival_tips() -> str:
    return (DATA_DIR / "festival_tips.md").read_text()

ARTISTS = load_artists()
BUDGET_ITEMS = load_budget_items()
FESTIVAL_TIPS = load_festival_tips()

# ── Logging ──────────────────────────────────────────────────────────────────
def log_interaction(user_msg: str, bot_response: str):
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "user": user_msg,
        "assistant": bot_response,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

# ── Tools ────────────────────────────────────────────────────────────────────

@tool
def lookup_artist_schedule(query: str) -> str:
    """
    Look up Coachella 2025 artist schedules. Use this when the user asks about
    what time or day an artist performs, which stage they're on, or wants to
    see the full lineup for a specific day or stage.

    Args:
        query: The artist name, day (Friday/Saturday/Sunday), or stage name to search for.
    """
    query_lower = query.lower()
    results = []

    for artist in ARTISTS:
        artist_name = artist["artist"].lower()
        stage = artist["stage"].lower()
        day = artist["day"].lower()
        genre = artist["genre"].lower()

        if (query_lower in artist_name or
            query_lower in stage or
            query_lower in day or
            query_lower in genre):
            results.append(artist)

    if not results:
        return f"No artists found matching '{query}'. Try searching by artist name, day (Friday/Saturday/Sunday), stage name (Main Stage, Sahara, Outdoor Theatre, Mojave, Gobi, Yuma), or genre."

    lines = []
    for a in results:
        headliner_tag = " ⭐ HEADLINER" if a["headliner"] == "Yes" else ""
        lines.append(
            f"🎵 {a['artist']}{headliner_tag}\n"
            f"   📅 {a['day']} | ⏰ {a['start_time']} – {a['end_time']}\n"
            f"   🏟️  {a['stage']} | 🎶 {a['genre']}"
        )

    header = f"Found {len(results)} result(s) for '{query}':\n\n"
    return header + "\n\n".join(lines)


@tool
def recommend_artists(preferences: str) -> str:
    """
    Recommend Coachella 2025 artists based on the user's music taste or genre preferences.
    Use this when the user says things like 'I like hip-hop', 'recommend me some indie artists',
    'who should I see if I like electronic music', or asks for suggestions.

    Args:
        preferences: A description of music genres, vibes, or artists the user already likes.
    """
    pref_lower = preferences.lower()

    genre_keywords = {
        "hip-hop": ["hip-hop", "rap", "trap"],
        "electronic": ["electronic", "edm", "dance", "techno", "house", "rave", "dj"],
        "indie": ["indie", "alternative", "alt"],
        "pop": ["pop"],
        "rock": ["rock", "punk"],
        "r&b": ["r&b", "soul", "rnb"],
        "funk": ["funk", "disco"],
        "afrobeats": ["afrobeats", "afropop", "african"],
        "country": ["country", "americana"],
        "dream pop": ["dream", "shoegaze", "ambient", "chill"],
    }

    matched_genres = []
    for genre, keywords in genre_keywords.items():
        if any(kw in pref_lower for kw in keywords):
            matched_genres.append(genre)

    # also do a direct artist name match (they might name similar artists)
    artist_name_hints = {
        "doja cat": ["pop", "hip-hop"],
        "billie eilish": ["pop", "indie"],
        "tyler": ["hip-hop"],
        "kanye": ["hip-hop"],
        "kendrick": ["hip-hop"],
        "taylor": ["pop", "indie"],
        "beyonce": ["pop", "r&b"],
        "arctic monkeys": ["rock", "indie"],
        "tame impala": ["indie", "electronic"],
        "flume": ["electronic"],
        "diplo": ["electronic"],
        "bad bunny": ["latin", "pop"],
    }
    for hint_artist, hint_genres in artist_name_hints.items():
        if hint_artist in pref_lower:
            matched_genres.extend(hint_genres)

    if not matched_genres:
        # Return a general spread if no genre detected
        matched_genres = ["pop", "hip-hop", "indie", "electronic"]

    recs = []
    seen = set()
    for artist in ARTISTS:
        ag = artist["genre"].lower()
        for mg in matched_genres:
            if mg in ag and artist["artist"] not in seen:
                recs.append(artist)
                seen.add(artist["artist"])
                break

    if not recs:
        return "I couldn't find specific matches, but check out the full lineup at coachella.com!"

    # Sort headliners first
    recs.sort(key=lambda x: (x["headliner"] != "Yes", x["day"], x["start_time"]))

    lines = []
    for a in recs[:8]:  # cap at 8 recs
        headliner_tag = " ⭐" if a["headliner"] == "Yes" else ""
        lines.append(
            f"🎵 **{a['artist']}**{headliner_tag} — {a['genre']}\n"
            f"   {a['day']}, {a['start_time']} at {a['stage']}"
        )

    intro = f"Based on your taste for **{', '.join(set(matched_genres))}**, here are my picks:\n\n"
    outro = "\n\n💡 Tip: Use the schedule lookup to get full details on any of these artists!"
    return intro + "\n\n".join(lines) + outro


@tool
def estimate_budget(trip_details: str) -> str:
    """
    Estimate the total cost of attending Coachella based on the user's trip details.
    Use this when the user asks about cost, how much money they need, budget planning,
    or anything related to spending and expenses.

    Args:
        trip_details: Description of the user's planned trip, including ticket type
                     (GA/VIP), accommodation preference, travel method, and spending style
                     (budget/moderate/splurge).
    """
    details_lower = trip_details.lower()

    # Determine spending tier
    if any(w in details_lower for w in ["budget", "cheap", "save", "low", "frugal", "broke"]):
        tier = "low_cost"
        tier_label = "Budget"
    elif any(w in details_lower for w in ["splurge", "vip", "luxury", "high", "money is no"]):
        tier = "high_cost"
        tier_label = "Splurge"
    else:
        tier = "mid_cost"
        tier_label = "Moderate"

    # Determine ticket type
    if "vip" in details_lower:
        ticket_type = "VIP Weekend Pass"
    elif "ga+" in details_lower or "ga plus" in details_lower:
        ticket_type = "GA+ Weekend Pass"
    else:
        ticket_type = "GA Weekend Pass"

    # Determine accommodation
    if any(w in details_lower for w in ["airbnb", "house", "off-site", "offsite", "hotel"]):
        if "hotel" in details_lower:
            accom_type = "Hotel (per night)"
            nights = 3
        else:
            accom_type = "Airbnb (split 4 people, per person)"
            nights = 3
    elif any(w in details_lower for w in ["lake eldorado", "premium camp", "luxury camp"]):
        accom_type = "Lake Eldorado (premium camping)"
        nights = 1
    elif any(w in details_lower for w in ["car camp", "car camping"]):
        accom_type = "On-site Car Camping (full weekend)"
        nights = 1
    else:
        accom_type = "On-site Tent Camping (full weekend)"
        nights = 1

    # Determine travel
    if any(w in details_lower for w in ["fly", "flight", "plane", "airport"]):
        if "lax" in details_lower:
            travel_type = "Flight to LAX (then drive)"
        else:
            travel_type = "Flight to Palm Springs (PSP)"
    elif any(w in details_lower for w in ["shuttle"]):
        travel_type = "Official Shuttle (round trip)"
    elif any(w in details_lower for w in ["drive", "car", "road"]):
        travel_type = "Gas (from LA, per car)"
    else:
        travel_type = "Official Shuttle (round trip)"

    # Build cost breakdown
    cost_map = {item["item"]: item for item in BUDGET_ITEMS}
    breakdown = {}
    total = 0

    def get_cost(item_name):
        item = cost_map.get(item_name)
        if item:
            return int(item[tier])
        return 0

    breakdown["🎟️  Ticket"] = get_cost(ticket_type)
    accom_cost = get_cost(accom_type)
    if nights > 1:
        accom_cost *= nights
    breakdown["🏕️  Accommodation"] = accom_cost
    breakdown["✈️  Travel"] = get_cost(travel_type)
    breakdown["🍔 Food & Drink (3 days)"] = (
        get_cost("Meals per day (in festival)") +
        get_cost("Alcohol per day") +
        get_cost("Snacks and coffee")
    ) * 3
    breakdown["🧴 Extras & Supplies"] = (
        get_cost("Sunscreen and toiletries") +
        get_cost("Merch (on-site)")
    )

    total = sum(breakdown.values())

    lines = [f"**{tier_label} Budget Estimate for Coachella 2025**\n"]
    lines.append(f"{'Item':<30} {'Est. Cost':>10}")
    lines.append("─" * 42)
    for item, cost in breakdown.items():
        lines.append(f"{item:<30} ${cost:>8,}")
    lines.append("─" * 42)
    lines.append(f"{'💰 ESTIMATED TOTAL':<30} ${total:>8,}")
    lines.append(
        f"\n📌 This is a **{tier_label.lower()} estimate**. "
        "Actual costs vary. Book tickets and accommodation early for the best prices!"
    )

    return "\n".join(lines)


@tool
def get_festival_tips(topic: str) -> str:
    """
    Answer questions about Coachella logistics, tips, and advice using a knowledge base.
    Use this for questions about what to bring, parking, camping, health and safety,
    stages, food, the difference between Weekend 1 and Weekend 2, or any general
    festival planning questions.

    Args:
        topic: The topic or question the user wants tips about (e.g., "what to pack",
               "camping tips", "how to get there", "food and drinks", "staying safe").
    """
    topic_lower = topic.lower()

    # Map topics to relevant sections in the knowledge base
    section_keywords = {
        "Getting There & Parking": ["park", "drive", "shuttle", "uber", "lyft", "rideshare", "airport", "get there", "transportation", "travel"],
        "Tickets & Wristbands": ["ticket", "wristband", "ga", "vip", "price", "cost", "purchase", "buy"],
        "What to Bring": ["bring", "pack", "packing", "what to", "need", "sunscreen", "shoes", "outfit", "wear"],
        "Food & Drinks": ["food", "eat", "drink", "meal", "beer", "alcohol", "water", "snack", "vendor"],
        "Stages & Navigation": ["stage", "where", "map", "navigate", "yuma", "sahara", "gobi", "mojave", "main stage", "outdoor theatre"],
        "Health & Safety": ["safe", "health", "medical", "heat", "dehydrat", "sick", "emergency", "cell service", "phone"],
        "Camping Tips": ["camp", "tent", "sleep", "shower", "lake eldorado", "car camp"],
        "Weekend 1 vs Weekend 2": ["weekend 1", "weekend 2", "w1", "w2", "which weekend", "difference"],
        "The Art & Installations": ["art", "installation", "do lab", "flower", "photo", "explore"],
    }

    relevant_sections = []
    for section, keywords in section_keywords.items():
        if any(kw in topic_lower for kw in keywords):
            relevant_sections.append(section)

    # Extract relevant sections from the knowledge base
    lines = FESTIVAL_TIPS.split("\n")
    output_lines = []
    in_section = False
    current_section = None

    if not relevant_sections:
        # Return a general intro if nothing matched
        return (
            "Here are some quick Coachella tips!\n\n"
            "I can give detailed advice on: getting there, tickets & wristbands, "
            "what to pack, food & drinks, stages, health & safety, camping, "
            "Weekend 1 vs 2, and art installations.\n\n"
            "What would you like to know more about?"
        )

    for line in lines:
        if line.startswith("## "):
            section_name = line[3:].strip()
            in_section = section_name in relevant_sections
            current_section = section_name
        if in_section:
            output_lines.append(line)

    result = "\n".join(output_lines).strip()
    if not result:
        return f"I have general tips about '{topic}' but nothing super specific in my knowledge base. Try asking about packing, food, camping, stages, or getting there!"

    return f"📖 **Festival Tips: {', '.join(relevant_sections)}**\n\n{result}"


# ── Agent Setup ──────────────────────────────────────────────────────────────
TOOLS = [lookup_artist_schedule, recommend_artists, estimate_budget, get_festival_tips]

SYSTEM_PROMPT = """You are the Coachella Resource Bot 🌵🎵 — an enthusiastic, knowledgeable, 
and friendly festival planning assistant for Coachella 2025.

Your personality: warm, upbeat, and genuinely excited about music and festivals. 
You use emojis sparingly but effectively. You give practical, actionable advice.

You have access to 4 specialized tools:
1. **lookup_artist_schedule** — Find artist set times, stages, and days
2. **recommend_artists** — Recommend artists based on music taste/genre preferences  
3. **estimate_budget** — Estimate total trip costs based on trip style
4. **get_festival_tips** — Answer logistics questions (packing, camping, food, safety, etc.)

Always use the appropriate tool when the user's question matches a tool's purpose.
If a question spans multiple topics, use multiple tools in sequence.

When presenting information:
- Be concise but complete
- Highlight the most important details
- Offer a follow-up suggestion when relevant (e.g., "Want me to also estimate your budget?")
- If you don't know something specific, be honest and point them to coachella.com

Remember: You're helping someone plan what could be a once-in-a-lifetime experience. 
Make it feel exciting and achievable!"""

def create_agent():
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "No API key found. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable."
        )

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.3,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, TOOLS, prompt)
    return AgentExecutor(agent=agent, tools=TOOLS, verbose=True, max_iterations=5)

# ── Gradio Interface ─────────────────────────────────────────────────────────
def build_chat_history(history: list[list[str]]) -> list:
    """Convert Gradio history format to LangChain message format."""
    messages = []
    for human, ai in history:
        messages.append(HumanMessage(content=human))
        messages.append(AIMessage(content=ai))
    return messages

def respond(message: str, history: list[list[str]], agent_executor):
    chat_history = build_chat_history(history)
    result = agent_executor.invoke({
        "input": message,
        "chat_history": chat_history,
    })
    response = result["output"]
    log_interaction(message, response)
    return response

def create_demo():
    agent_executor = create_agent()

    EXAMPLE_QUESTIONS = [
        "What time does Lady Gaga perform and which stage?",
        "I love indie and electronic music — who should I see?",
        "I'm flying in from Chicago and doing tent camping. Estimate my budget.",
        "What should I pack for the weekend? First time attendee!",
        "What's the difference between Weekend 1 and Weekend 2?",
        "Show me the full Friday lineup",
        "How do I get from LAX to the festival?",
    ]

    with gr.Blocks(
        theme=gr.themes.Base(
            primary_hue="orange",
            secondary_hue="yellow",
            neutral_hue="stone",
        ),
        css="""
        .gradio-container {
            font-family: 'Georgia', serif;
            max-width: 900px !important;
        }
        #header {
            text-align: center;
            padding: 24px 0 8px 0;
        }
        #header h1 {
            font-size: 2.4em;
            font-weight: bold;
            color: #c2410c;
            margin-bottom: 4px;
        }
        #header p {
            color: #78716c;
            font-size: 1em;
        }
        .chatbot-wrap .message {
            border-radius: 12px !important;
        }
        #examples-label {
            color: #78716c;
            font-size: 0.85em;
            margin-top: 8px;
        }
        """,
        title="🌵 Coachella Resource Bot",
    ) as demo:

        gr.HTML("""
        <div id="header">
            <h1>🌵 Coachella Resource Bot 🎵</h1>
            <p>Your AI-powered guide to planning the perfect Coachella 2025 experience</p>
        </div>
        """)

        chatbot = gr.Chatbot(
            label="Chat",
            height=480,
            bubble_full_width=False,
            show_label=False,
        )

        with gr.Row():
            msg = gr.Textbox(
                placeholder="Ask me about artists, schedules, budget, packing tips...",
                show_label=False,
                scale=5,
                container=False,
            )
            send_btn = gr.Button("Send 🎪", variant="primary", scale=1)

        gr.HTML('<p id="examples-label">💡 Try one of these:</p>')
        examples = gr.Examples(
            examples=EXAMPLE_QUESTIONS,
            inputs=msg,
            label="",
        )

        gr.HTML("""
        <div style="text-align:center; color:#a8a29e; font-size:0.78em; margin-top:16px;">
            Powered by LangChain + Gemini | Data: Coachella 2025 | 
            <a href="https://github.com/ldamuni-cyber/coachella-resource-bot" 
               target="_blank" style="color:#c2410c;">GitHub Repo</a>
        </div>
        """)

        def user_submit(user_message, history):
            return "", history + [[user_message, None]]

        def bot_reply(history):
            user_message = history[-1][0]
            history[-1][1] = respond(user_message, history[:-1], agent_executor)
            return history

        msg.submit(user_submit, [msg, chatbot], [msg, chatbot]).then(
            bot_reply, chatbot, chatbot
        )
        send_btn.click(user_submit, [msg, chatbot], [msg, chatbot]).then(
            bot_reply, chatbot, chatbot
        )

    return demo

# ── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
