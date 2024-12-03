import os
import json
import logging
import base64
import requests
from datetime import datetime
from requests_oauthlib import OAuth1Session
from supabase import create_client, Client
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger()


def get_current_date_string():
    """Get current date in the format used in prompts."""
    return datetime.now().strftime("%dth of %B %Y")


def get_thread_format_instructions():
    """Get format instructions for thread context."""
    return """
You are receiving messages in the following format:
{
    "mention": {
        "author": "@username",
        "text": "The message text"
    },
    "thread_context": [
        {
            "author": "@user3",
            "text": "Most recent tweet in thread"
        },
        {
            "author": "@user2",
            "text": "Earlier tweet in thread"
        },
        {
            "author": "@user1",
            "text": "Earliest tweet we have in thread"
        }
    ]
}

Note: The thread_context array contains the most recent N tweets in the conversation thread, 
ordered from most recent to oldest.
"""


def get_mentions_format_instructions():
    """Get format instructions for mentions without thread context."""
    return """
You are receiving messages in the following format:
{
    "mentions": [
        [
            "Text1",
            "@User1"
        ],
        [
            "Text2",
            "@User2"
        ]
    ]
}
"""


def build_prompt(base_prompt: str, thread_context=None):
    """Build a complete prompt by combining base prompt with dynamic elements."""
    current_date = get_current_date_string()

    # Choose format instructions based on whether we have thread context
    if thread_context:
        format_instructions = get_thread_format_instructions()
    else:
        format_instructions = get_mentions_format_instructions()

    dynamic_context = f"""
///
Today is {current_date}. You are now online on the global hypernetwork and interacting with both people and AIs.
{format_instructions}
"""

    return base_prompt + dynamic_context


def download_and_encode_image(url: str):
    """Download an image from a URL and encode it as base64."""
    try:
        logger.info(f"Downloading image from {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "image/jpeg")
        if not content_type.startswith("image/"):
            logger.error(f"Invalid content type for image: {content_type}")
            return None

        image_data = base64.b64encode(response.content).decode("utf-8")
        logger.info(f"Successfully encoded image from {url}")

        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": content_type,
                "data": image_data,
            },
        }
    except Exception as e:
        logger.error(f"Error processing image from {url}: {e}")
        return None


def collect_media_from_threads(threads):
    """Collect all media items from a list of thread contexts."""
    all_media = []
    try:
        for thread in threads:
            if not thread:  # Skip null threads
                continue
            for tweet in thread:
                if "media" in tweet:
                    logger.info(f"Found media in tweet from @{tweet['author']}")
                    all_media.extend(tweet["media"])
    except Exception as e:
        logger.error(f"Error collecting media from threads: {e}")

    return all_media


def format_thread_for_context(thread_tweets):
    """Format thread tweets into a readable context string."""
    if not thread_tweets:
        return None

    formatted_tweets = []
    for tweet in thread_tweets:
        tweet_data = tweet["data"]
        author = next(
            (
                user
                for user in tweet["includes"]["users"]
                if user["id"] == tweet_data["author_id"]
            ),
            None,
        )

        if not author:
            continue

        # Format basic tweet info
        formatted_tweet = {
            "author": f"@{author['username']}",
        }

        # Use note_tweet text if available, otherwise use regular text
        if "note_tweet" in tweet_data and tweet_data["note_tweet"].get("text"):
            formatted_tweet["text"] = tweet_data["note_tweet"]["text"]
        else:
            formatted_tweet["text"] = tweet_data["text"]

        formatted_tweets.append(formatted_tweet)
    logger.info(f"Formatted tweets: {formatted_tweets}")
    return formatted_tweets


def format_message_with_media(text, media_items=None):
    """Format a message with optional media into Claude's content block format."""
    content_blocks = []
    successful_media = 0
    failed_media = 0

    if media_items:
        logger.info(f"Processing {len(media_items)} media items")
        for media in media_items:
            if media["type"] == "photo" and "url" in media:
                image_block = download_and_encode_image(media["url"])
                if image_block:
                    content_blocks.append(image_block)
                    successful_media += 1
                else:
                    failed_media += 1

    content_blocks.append({"type": "text", "text": text})

    if media_items:
        logger.info(
            f"Media processing complete: {successful_media} successful, {failed_media} failed"
        )

    return content_blocks


def get_tweet(oauth: OAuth1Session, tweet_id: str):
    """Fetch a single tweet by its ID."""
    url = f"https://api.twitter.com/2/tweets/{tweet_id}"

    params = {
        "tweet.fields": "created_at,conversation_id,in_reply_to_user_id,author_id,referenced_tweets,attachments,note_tweet",
        "expansions": "author_id,referenced_tweets.id,attachments.media_keys",
        "user.fields": "username",
        "media.fields": "type,url,preview_image_url",
    }

    try:
        response = oauth.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error fetching tweet {tweet_id}: {e}")
        return None


def get_thread_to_root(oauth: OAuth1Session, start_tweet_id: str, max_depth: int = 10):
    """Fetch all tweets in the path from a specific tweet to the root of its thread."""
    thread_tweets = []
    current_tweet_id = start_tweet_id
    depth = 0

    while current_tweet_id and depth < max_depth:
        tweet_data = get_tweet(oauth, current_tweet_id)
        if not tweet_data or "data" not in tweet_data:
            break

        thread_tweets.append(tweet_data)
        depth += 1

        # Look for parent tweet
        parent_tweet_id = None
        if "referenced_tweets" in tweet_data["data"]:
            for ref in tweet_data["data"]["referenced_tweets"]:
                if ref["type"] == "replied_to":
                    parent_tweet_id = ref["id"]
                    break

        current_tweet_id = parent_tweet_id

    return thread_tweets


def get_latest_prompt(supabase, prompt_table):
    """Fetch the most recent prompt from the specified table."""
    try:
        response = (
            supabase.table(prompt_table)
            .select("prompt_text")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if response.data:
            logger.info(f"Loaded prompt from {prompt_table}")
            return response.data[0]["prompt_text"]
        logger.error(f"No prompts found in {prompt_table}")
        return None
    except Exception as e:
        logger.error(f"Error loading prompt from {prompt_table}: {e}")
        return None


def decode_unicode_surrogates(text):
    """
    Decode Unicode surrogate pairs in text.

    Args:
        text (str): Input text potentially containing surrogate pairs

    Returns:
        str: Decoded text with proper Unicode characters
    """
    try:
        # Encode surrogate pairs as UTF-16 and decode back to UTF-8
        return text.encode("utf-16", "surrogatepass").decode("utf-16")
    except Exception as e:
        logger.error(f"Decoding error: {e}")
        return text


def get_claude_response(claude, message, prompt):
    """Generate a response using Claude API."""
    try:
        response = claude.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=4096,
            temperature=0.7,
            messages=[{"role": "user", "content": message}],
            system=prompt,
        )
        logger.info(f"Raw Claude response: {response}")

        if not response:
            logger.error("Empty response from Claude API")
            return None

        # Decode Unicode surrogates in the response text
        return decode_unicode_surrogates(response.content[0].text)
    except Exception as e:
        logger.error(f"Error calling Claude API: {e}")
        return


def main():
    """Main function for responding to mentions."""
    logger.info("Starting respond_to_mention function")

    try:
        # Initialize Supabase client
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        logger.info("Initializing Supabase client")
        supabase: Client = create_client(url, key)

        # Load the latest mentions prompt
        prompt_response = (
            supabase.table("mentions_prompts")
            .select("id,prompt_text")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )

        if not prompt_response.data:
            logger.error("No prompts found in mentions_prompts table")
            return {"status": "error", "reason": "no_prompt_found"}

        prompt_id = prompt_response.data[0]["id"]
        base_prompt = prompt_response.data[0]["prompt_text"]
        logger.info(f"Loaded prompt with ID: {prompt_id}")

        # Initialize Claude client
        claude = Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))

        # Initialize Twitter client
        oauth = OAuth1Session(
            os.getenv("TWITTER_API_KEY"),
            client_secret=os.getenv("TWITTER_API_SECRET"),
            resource_owner_key=os.getenv("TWITTER_ACCESS_TOKEN"),
            resource_owner_secret=os.getenv("TWITTER_ACCESS_TOKEN_SECRET"),
        )

        # Get the latest unprocessed mention
        logger.info("Fetching latest unprocessed mention from Supabase")
        try:
            mentions = (
                supabase.table("mentions")
                .select("*")
                .eq("processing_status", "unprocessed")
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )
        except Exception as e:
            logger.error(f"Failed to fetch from Supabase: {str(e)}")
            return

        if not mentions.data:
            logger.info("No unprocessed mentions found")
            return {"status": "skipped", "reason": "no_unprocessed_mentions"}

        mention = mentions.data[0]

        # Get thread contextr
        thread_tweets = get_thread_to_root(oauth, mention["tweet_id"])
        if thread_tweets:
            logger.info(f"Thread context: {thread_tweets}")
            thread_context = format_thread_for_context(thread_tweets)
            logger.info(f"Thread context: {thread_context}")

        # Format message for Claude
        message_to_claude = {
            "mention": {
                "text": mention["text"],
                "author": f"@{mention['author_username']}",
            }
        }

        if thread_context:
            message_to_claude["thread_context"] = thread_context

        # Build complete prompt
        prompt_text = build_prompt(
            base_prompt,
            thread_context=thread_context,
        )

        # Collect media from mention
        media_items = []
        if "media" in mention:
            logger.info(f"Found media in mention from @{mention['author_username']}")
            media_items.extend(mention["media"])

        # Format content for Claude
        logger.info("Formatting message with media for Claude")
        content_blocks = format_message_with_media(
            json.dumps(message_to_claude), media_items=media_items
        )

        # Mark mention as processing
        try:
            supabase.table("mentions").update({"processing_status": "processing"}).eq(
                "tweet_id", mention["tweet_id"]
            ).execute()
            logger.info(f"Marked mention {mention['tweet_id']} as processing")
        except Exception as e:
            logger.error(f"Failed to mark mention as processing: {e}")
            return {
                "status": "error",
                "reason": "status_update_failed",
                "error": str(e),
            }

        logger.info(f"Content blocks for Claude: {content_blocks}")
        logger.info(f"Prompt text: {prompt_text}")

        # Generate response using Claude
        logger.info("Generating response using Claude")
        response = claude.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=4096,
            temperature=0.7,
            messages=[{"role": "user", "content": content_blocks}],
            system=prompt_text,
        )

        if not response:
            logger.error("Empty response from Claude")
            return {"status": "error", "reason": "empty_claude_response"}

        tweet_text = response.content[0].text
        logger.info("Successfully generated tweet text")

        # Post tweet if enabled
        if os.getenv("POST_TO_TWITTER", "false").lower() == "true":
            logger.info("Posting tweet to Twitter")
            try:
                response = oauth.post(
                    "https://api.twitter.com/2/tweets",
                    json={
                        "text": tweet_text,
                        "reply": {"in_reply_to_tweet_id": mention["tweet_id"]},
                    },
                )
                response.raise_for_status()

                tweet_id = response.json()["data"]["id"]
                logger.info(f"Successfully posted tweet with ID: {tweet_id}")
                tweet_id_for_storage = tweet_id
            except Exception as e:
                logger.error(f"Failed to post tweet: {e}")
                if hasattr(e, "response"):
                    logger.error(f"Twitter API response: {e.response.text}")
                tweet_id_for_storage = "failed_to_post"
                return {
                    "status": "error",
                    "reason": "twitter_post_failed",
                    "error": str(e),
                }
        else:
            logger.info("Twitter posting disabled, using mock tweet ID")
            tweet_id_for_storage = "mock_tweet_id"

        # Store response in Supabase
        logger.info("Storing response in Supabase")
        try:
            supabase.table("responses").insert(
                {
                    "text": tweet_text,
                    "tweet_id": tweet_id_for_storage,
                    "mention_id": mention["tweet_id"],
                    "prompt_id": prompt_id,
                    "in_reply_to_tweet_id": mention["tweet_id"],
                    "input_message": json.dumps(message_to_claude),
                    "created_at": datetime.now().isoformat(),
                }
            ).execute()
            logger.info("Successfully stored response in Supabase")
        except Exception as e:
            logger.error(f"Failed to store response: {e}")
            return {
                "status": "error",
                "reason": "supabase_store_failed",
                "error": str(e),
            }

        # Mark mention as complete
        logger.info("Marking mention as complete")
        try:
            supabase.table("mentions").update(
                {
                    "processed_at": datetime.now().isoformat(),
                    "processing_status": "complete",
                }
            ).eq("tweet_id", mention["tweet_id"]).execute()
            logger.info(f"Marked mention {mention['tweet_id']} as complete")
        except Exception as e:
            logger.error(f"Failed to mark mention as complete: {e}")
            # Continue as this is not a critical error

        return {
            "status": "success",
            "tweet": tweet_text,
            "mention_id": mention["tweet_id"],
            "tweet_id": tweet_id_for_storage,
            "model": "claude-3-opus-20240229",
        }

    except Exception as e:
        logger.error(f"Error in respond_to_mention: {str(e)}")
        return {"status": "error", "reason": "unexpected_error", "error": str(e)}


if __name__ == "__main__":
    main()

def lambda_handler(event, context):
    main()
    return {"statusCode": 200, "body": "Success"}