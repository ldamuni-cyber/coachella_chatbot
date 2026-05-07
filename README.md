# 🌵 Coachella Resource Bot

**An AI-powered festival planning assistant built with LangChain and Google Gemini.**

> 🚀 **[Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/ldamuni/coachella_chatbot)**

---

## Overview

The Coachella Resource Bot is a LangChain-powered chatbot that helps festival-goers plan their Coachella 2025 experience. It can look up artist schedules, recommend artists based on your music taste, estimate your trip budget, and answer logistics questions — all in a single conversational interface.

**Who it's for:** First-time Coachella attendees and returning festivalgoers who want a faster, smarter way to plan — instead of cross-referencing four different tabs.

---

## The Problem

Festivals like Coachella are overwhelming to navigate. The lineup has 150+ artists, the schedule spans three days across seven stages, and the logistics — tickets, travel, camping, food, packing — require hours of research across Reddit threads, the official app, and spreadsheets. I've seen people share weekend spending breakdowns that were clearly unplanned and unrealistic, and I've experienced the same anxiety at smaller festivals. This bot brings all of that research into one place.

---

## How It Works

The bot uses LangChain's `ChatGoogleGenerativeAI` with Gemini 2.0 Flash as the LLM. Each user message is routed to one of four specialized tools based on keyword matching, and the tool result is passed back to the LLM to generate a friendly, conversational response.
User Message
│
▼
Keyword Router
│
├──► 🎵 lookup_artist_schedule   → CSV database of Coachella 2025 artists
├──► 🎯 recommend_artists        → Genre-matching logic across the lineup
├──► 💰 estimate_budget          → Budget CSV with low/mid/high cost tiers
└──► 📖 get_festival_tips        → Markdown knowledge base (RAG)
│
▼
LangChain + Gemini 2.0 Flash
│
▼
Response to User

### Tools

| Tool | Trigger | Data Source |
|------|---------|-------------|
| `lookup_artist_schedule` | "What time does X perform?", "Show me Friday lineup" | `data/artists.csv` |
| `recommend_artists` | "I like hip-hop, who should I see?" | `data/artists.csv` + genre mapping |
| `estimate_budget` | "How much will this cost?" | `data/budget_items.csv` |
| `get_festival_tips` | "What should I pack?", "Camping tips?" | `data/festival_tips.md` |

---

## Sample Conversations

### 1. Artist Schedule Lookup
User: What time does Lady Gaga perform?
Bot: Lady Gaga is headlining the Main Stage on Friday night,
performing from 10:00 PM to 11:30 PM. She's closing out
the night so plan your evening around her set!

### 2. Artist Recommendations
User: I love hip-hop and electronic music — who should I see?
Bot: Based on your taste, here are my top picks:
- Post Malone (HEADLINER) — Hip-Hop/Pop | Sunday 10PM at Main Stage
- Travis Scott — Hip-Hop | Friday 7:30PM at Main Stage
- Charli XCX — Pop/Electronic | Friday 8:30PM at Sahara
- Four Tet — Electronic | Friday 11PM at Sahara

### 3. Budget Estimation
User: I'm flying in and doing tent camping, budget trip. How much?
Bot: Budget Estimate for Coachella 2025
Ticket: $549
Accommodation: $125
Travel: $150
Food & Drink (3 days): $165
Extras: $15
TOTAL: $1,004

### 4. Festival Tips
User: What should I pack? First time attendee!
Bot: Great question! Must-haves include a reusable water bottle
(refill stations are free!), SPF 50+ sunscreen, comfortable
broken-in shoes, light layers for cold nights, and a portable
phone charger. The desert is brutal during the day but drops
to 55°F at night so pack accordingly!

---

## Key Findings / What I Learned

Building this bot taught me how much dependency management matters in real deployments. Getting LangChain, Gradio, and the Gemini API to all work together on Hugging Face Spaces required careful version pinning and a lot of reading error logs. This was harder than writing the actual Python code.

The routing logic was also trickier than expected. Early versions would misroute questions or not trigger the right tool. Writing specific, detailed keyword lists for each tool and testing edge cases was one of the most important parts of making the bot work well. If I were to rebuild this, I'd use a proper vector database like Chroma for the festival tips RAG instead of keyword matching — it would handle more natural phrasing much better. I'd also set up a stricter local dev environment before deploying to avoid burning through API quota during debugging.

---

## Who Would Care

Festival attendees planning Coachella — especially first-timers who don't know where to start. Event promoters could adapt this architecture for any festival (Lollapalooza, CMA Fest, Bonnaroo) by swapping in a different lineup CSV and knowledge base. The budget estimation feature is particularly useful for younger attendees trying to make the math work on a real salary.

---

## How to Run

### Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/ldamuni-cyber/coachella-resource-bot.git
cd coachella-resource-bot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key
export GOOGLE_API_KEY=your_gemini_key_here
# Get a free key at: https://aistudio.google.com

# 4. Launch the app
python app.py
```

Open your browser to `http://localhost:7860`

### Deploy to Hugging Face Spaces

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select **Gradio** as the SDK
3. Push this repo to the Space
4. Add `GOOGLE_API_KEY` as a Secret in Space Settings

---

## Project Structure
coachella-resource-bot/
├── app.py                  # Main application (LangChain + Gradio UI)
├── requirements.txt        # Python dependencies
├── README.md
├── data/
│   ├── artists.csv         # Coachella 2025 lineup with schedule data
│   ├── budget_items.csv    # Cost estimates (low/mid/high tiers)
│   └── festival_tips.md    # Knowledge base for festival tips
└── logs/                   # Auto-generated conversation logs

---

## Tech Stack

- **LangChain** — LLM orchestration and conversation history
- **Google Gemini 2.0 Flash** — LLM (free tier available)
- **Gradio** — Chat interface
- **Python** — Core language
- **CSV + Markdown** — Lightweight data sources

---

## Reflection

For this project I set out to build a LangChain-powered chatbot that helps people plan their Coachella 2025 experience. The hardest part wasn't writing the Python code — it was getting all the dependencies to work together on Hugging Face. Gradio, LangChain, and the Gemini API all had version conflicts that took significant time to debug. I learned more about dependency management and reading error logs than I expected to.

If I were to do this differently, I would set up a local development environment first and test everything locally before deploying. I would also use a proper vector database like Chroma for the festival tips instead of keyword matching. Overall I'm proud of what I built — it's a real deployed app that solves a real problem.

---

*Built as a portfolio project for CS 444.*
