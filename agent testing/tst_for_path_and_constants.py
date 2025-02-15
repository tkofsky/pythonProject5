import os, csv, json, random
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

from openai import OpenAI

# ========= CONFIG =========
DATA_PATH = "agent_dataset_50.json"
PROMPTS_PATH = "prompt_variants.json"
#CSV_LOG = "bandit_fewshot_agent_log_two_pass.csv"
CSV_LOG = "bandit_fewshot_agent_log"

ITERATIONS = 30
EPSILON = 0.25
FEW_SHOT_LEVELS = [0, 1, 3]
TEMPS = [0.2, 0.3]
TWO_PASS_OPTIONS = [False, True]
GEN_MODEL = "gpt-4o-mini"

# Few-shot demos
FEW_SHOT = [
    {"input": "Book a train from Boston to New York tomorrow morning under $120.",
     "output": {"intent":"book_train","entities":{"from":"Boston","to":"New York","date":"tomorrow morning","budget":"120"},"constraints":["budget<=120"],"urgency":"normal","steps":["search_trains","filter_by_price_and_time","propose_top_options"]}},
    {"input": "Schedule a 30 minute Zoom check-in with Maya next Tuesday after 3pm.",
     "output": {"intent":"schedule_meeting","entities":{"participants":["Maya"],"duration":"30 minutes","time_window":"next Tuesday after 3pm","location":"Zoom"},"constraints":["include_zoom_link"],"urgency":"normal","steps":["find_common_slot","create_zoom","send_invites"]}},
    {"input": "Order 4 vegan lunches for pickup at 1pm at 9 King St.",
     "output": {"intent":"order_food","entities":{"headcount":4,"diet":"vegan","pickup_time":"1pm","address":"9 King St"},"constraints":["vegan_only"],"urgency":"time_sensitive","steps":["choose_restaurants","filter_menu","place_order"]}},
    {"input": "Write a brief thank-you email to the interviewer and ask for feedback.",
     "output": {"intent":"draft_email","entities":{"recipient":"interviewer","topic":"thank_you","extra_request":"feedback"},"constraints":["polite_tone","brief"],"urgency":"normal","steps":["draft_email","review_tone","send_or_copy"]}}
]

# ========= CLIENT =========
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("OPENAI_API_KEY is not set")
client = OpenAI(api_key=api_key)
