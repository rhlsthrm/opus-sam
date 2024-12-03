import datetime
import json
import os
from requests_oauthlib import OAuth1Session
import yaml
from supabase import create_client, Client
from dotenv import load_dotenv
import logging
from pathlib import Path
from anthropic import Anthropic

load_dotenv()

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger()

def load_prompts(prompt_path):
    """Load prompts from YAML file."""
    try:
        with open(prompt_path, "r") as f:
            config = yaml.safe_load(f)
            mention_prompt = config["mention_prompt"]
            thoughts_prompt = config["thoughts_prompt"]
            image_prompt = config["image_prompt"]
            logger.info("Loaded prompts from config file")
            return mention_prompt, thoughts_prompt, image_prompt
    except Exception as e:
        logger.error(f"Error loading prompt: {e}")

def get_claude_response(claude, message, prompt):
    """Generate a response using Claude API."""

    # let the app error, instead of public fallback error message
    response = claude.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=4096,
        temperature=0.7,
        messages=[{"role": "user", "content": message}],
        system=prompt,
    )
    return response.content[0].text


def main():
    logger.info("START self_tweet")
    # Supabase connection
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    supabase: Client = create_client(url, key)
    
    prompt_path = Path(__file__).parent / "prompt.yaml"
    mention_prompt, thoughts_prompt, image_prompt = load_prompts(prompt_path)
    
    oauth = OAuth1Session(
        os.getenv("TWITTER_API_KEY"),
        client_secret=os.getenv("TWITTER_API_SECRET"),
        resource_owner_key=os.getenv("TWITTER_ACCESS_TOKEN"),
        resource_owner_secret=os.getenv("TWITTER_ACCESS_TOKEN_SECRET"),
    )
    
    claude = Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))

    latest_ten_mentions = supabase.table("mentions").select("*").order("created_at", desc=True).limit(10).execute()

    formatted_mentions = {"mentions": []}

    for mention in latest_ten_mentions.data:
        logger.info(f"Mention: {mention}")
        formatted_mentions["mentions"].append([mention["text"], mention["author_username"]])

    if len(formatted_mentions["mentions"]) > 0:
        logger.info(f"Single message: {formatted_mentions}")
        response = get_claude_response(claude, json.dumps(formatted_mentions), thoughts_prompt)
        logger.info(f"Claude response: {response}")
        if os.getenv("POST_TO_TWITTER"):
            responseX = oauth.post(
                "https://api.twitter.com/2/tweets",
                json={
                    "text": response,
                },
            )
            logger.info(f"Posted thought: {response}")
            logger.info(f"Status Code: {responseX.status_code}")
            logger.info(f"headers: {dict(responseX.headers)}")
            logger.info(f"Response content: {responseX.json()}")
            supabase.table("thoughts").insert({
                "text": response,
                "tweet_id": responseX.json()["data"]["id"],
                "prompt": thoughts_prompt,
                "input_message": json.dumps(formatted_mentions),
                "created_at": datetime.now().isoformat(),
            }).execute()
        else:
            logger.info(
                f"POST_TO_TWITTER is `False`, so logging thought instead: {response}"
            )


if __name__ == "__main__":
    main()

def lambda_handler(event, context):
    main()
    return {"statusCode": 200, "body": "Success"}