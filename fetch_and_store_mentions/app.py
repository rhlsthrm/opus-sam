import os
import yaml
from supabase import create_client, Client
from requests_oauthlib import OAuth1Session
from dotenv import load_dotenv
import logging
from pathlib import Path

load_dotenv()

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger()


def load_whitelisted_ids():
    """Load whitelisted user IDs from a YAML file."""
    try:
        filepath = Path(__file__).parent / "whitelisted_ids_for_mentions.yaml"
        with open(filepath, "r") as file:
            data = yaml.safe_load(file)
            return set(data.get("whitelisted_ids_for_mentions", []))
    except Exception as e:
        logger.error(f"Error loading whitelisted IDs: {e}")
        return set()


def main():
    # Load whitelisted IDs
    whitelisted_ids = load_whitelisted_ids()

    # Supabase connection
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    supabase: Client = create_client(url, key)

    # Twitter API setup
    oauth = OAuth1Session(
        os.getenv("TWITTER_API_KEY"),
        client_secret=os.getenv("TWITTER_API_SECRET"),
        resource_owner_key=os.getenv("TWITTER_ACCESS_TOKEN"),
        resource_owner_secret=os.getenv("TWITTER_ACCESS_TOKEN_SECRET"),
    )

    # Fetch recent mentions
    bot_twitter_handle = os.getenv("BOT_TWITTER_HANDLE")
    latest_mention = (
        supabase.table("mentions")
        .select("*")
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    logger.info(f"Latest mention: {latest_mention}")

    params = {
        "query": f"{bot_twitter_handle} -is:retweet",
        "expansions": "author_id",
        "tweet.fields": "author_id,created_at,text,conversation_id,note_tweet",
        "user.fields": "username",
        "max_results": 10,
    }
    # If there is a latest tweet, we need to use it as the since_id
    if latest_mention.data is not None and len(latest_mention.data) > 0:
        params["since_id"] = latest_mention.data[0]["tweet_id"]

    logger.info(f"Querying Twitter for mentions with params: {params}")
    response = oauth.get(
        "https://api.twitter.com/2/tweets/search/recent", params=params
    )
    logger.info(f"Twitter API response: {response.json()}")
    if response.status_code != 200:
        logger.error(
            f"Error fetching mentions. Error code {response.status_code}, response: {response.json()}"
        )
        return
    tweets = response.json().get("data", [])
    includes = response.json().get("includes", [])
    logger.info(f"Fetched {len(tweets)} tweets.")
    for tweet in tweets:
        if tweet["author_id"] in whitelisted_ids:
            user = next(
                (
                    user
                    for user in includes["users"]
                    if user["id"] == tweet["author_id"]
                ),
                None,
            )
            try:
                # Use note_tweet text if available, otherwise use regular text
                tweet_text = (
                    tweet["note_tweet"]["text"]
                    if "note_tweet" in tweet and tweet["note_tweet"].get("text")
                    else tweet["text"]
                )

                supabase.table("mentions").insert(
                    {
                        "author_id": tweet["author_id"],
                        "tweet_id": tweet["id"],
                        "text": tweet_text,
                        "created_at": tweet["created_at"],
                        "conversation_id": tweet.get("conversation_id"),
                        "author_username": user["username"],
                    }
                ).execute()
                logger.info(
                    f"Inserted tweet with tweet_id {tweet['id']} from {tweet['author_id']} into mentions."
                )
            except Exception as e:
                logger.error(
                    f"Failed to insert tweet {tweet['id']} from {tweet['author_id']}: {str(e)}"
                )


if __name__ == "__main__":
    main()

def lambda_handler(event, context):
    main()
    return {"statusCode": 200, "body": "Mentions fetched and stored successfully"}
