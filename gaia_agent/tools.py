import os
import tempfile
from typing import Callable
from urllib.parse import parse_qs, quote_plus, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from openai.types import FunctionDefinition
from youtube_transcript_api import NoTranscriptFound, TranscriptsDisabled, YouTubeTranscriptApi

from .config import CONFIG
from .llm import get_llm


def download_file_from_url(url: str, filename: str | None = None) -> str:
    """Download a file from a URL and save it to a temporary location."""
    try:
        # Parse URL to get filename if not provided
        if not filename:
            path = urlparse(url).path
            filename = os.path.basename(path)
            if not filename:
                # Generate a random name if we couldn't extract one
                import uuid

                filename = f"downloaded_{uuid.uuid4().hex[:8]}"

        # Create temporary file
        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, filename)

        # Download the file
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        # Save the file
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    except Exception as e:  # noqa: BLE001
        return f"Error downloading file: {e!s}"
    else:
        return f"File downloaded to {filepath}. You can now process this file."


def read_content_from_webpage(url: str) -> str:
    """Read content from a webpage and return it as text."""
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        content = soup.get_text()
        return content[:2000]  # Limit to first 2000 characters
    except requests.RequestException as e:
        print(f"Error reading webpage: {e}")
        return ""


def analyze_excel_file(file_path: str) -> str:
    """Analyze the Excel file and return a summary."""
    df = pd.read_excel(file_path, engine="openpyxl")
    result = f"Excel file loaded with {len(df)} rows and {len(df.columns)} columns.\n"
    result += f"Columns: {', '.join(df.columns)}\n\n"

    # Add summary statistics
    result += "Summary statistics:\n"
    result += str(df.describe())

    return result


def analyze_csv_file(file_path: str) -> str:
    """Analyze the CSV file and return a summary."""
    df = pd.read_excel(file_path)
    result = f"CSV file loaded with {len(df)} rows and {len(df.columns)} columns.\n"
    result += f"Columns: {', '.join(df.columns)}\n\n"

    # Add summary statistics
    result += "Summary statistics:\n"
    result += str(df.describe())

    return result


def analyze_image(base64_image: str) -> str:
    """Analyze the image and return a description."""
    # Placeholder for image analysis logic
    prompt = (
        "Analyze the image and provide a detailed description of its content.\n"
        "The important pieces of information to extract from the image are:\n"
        "1. Visual elements and objects\n"
        "2. Colors, Patterns, Composition and Style\n"
        "3. Text, Numbers and Symbols if present\n"
        "4. Contextual information\n"
        "5. Overall context and meaning\n"
        "6. Any other relevant details\n"
    )
    llm = get_llm()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            ],
        }
    ]

    completion = llm.chat.completions.create(model=CONFIG.AGENT_MODEL_NAME, messages=messages)
    return completion.choices[0].message.content


def search_web(search_term: str, num_results: int = 5) -> list[dict[str, str]]:
    """Websearch using DuckDuckGo."""
    encoded_search_term = quote_plus(search_term)
    headers = {
        # "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",  # noqa: E501
    }

    search_url = f"https://html.duckduckgo.com/html/?q={encoded_search_term}"

    try:
        response = requests.get(search_url, headers=headers, timeout=60)
        response.raise_for_status()  # Fehler bei HTTP-Statuscode != 200 auslösen
        soup = BeautifulSoup(response.text, "html.parser")

        results = []
        result_elements = soup.select(".result")

        for element in result_elements[:num_results]:
            title_element = element.select_one(".result__title")
            link_element = element.select_one(".result__url")
            snippet_element = element.select_one(".result__snippet")

            if title_element and link_element:
                title = title_element.get_text(strip=True)
                link = link_element.get("href") if link_element.get("href") else link_element.get_text(strip=True)
                snippet = snippet_element.get_text(strip=True) if snippet_element else ""

                results.append({"title": title, "link": link, "snippet": snippet})

        return results  # noqa: TRY300

    except requests.RequestException as e:
        print(f"Fehler bei der Websuche: {e}")
        return []


def check_available_wikipedia_articles(possible_title: str) -> list[str]:
    """Check if a Wikipedia article with the given title exists."""
    try:
        search_url = f"https://de.wikipedia.org/w/index.php?search={possible_title.replace(' ', '_')}"
        response = requests.get(search_url, timeout=60)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # check if redirected to page
        if "Suchergebnisse" not in soup.title.string:
            page_title = soup.find("h1", id="firstHeading").text.strip()
            return [page_title]
        search_results = []

        # main results
        results = soup.find_all("div", class_="mw-search-result-heading")
        if results:
            for result in results:
                link = result.find("a")
                if link and link.get("title"):
                    search_results.append(link.get("title"))

        # Check for "Did you mean" suggestions
        did_you_mean = soup.find("div", class_="searchdidyoumean")
        if did_you_mean:
            link = did_you_mean.find("a")
            if link and link.get("title") and link.get("title") not in search_results:
                search_results.append(link.get("title"))

        # Check for exact matches
        exact_match = soup.find("p", class_="mw-search-exists")
        if exact_match:
            link = exact_match.find("a")
            if link and link.get("title") and link.get("title") not in search_results:
                search_results.insert(
                    0,
                    link.get("title"),
                )

        return search_results  # noqa: TRY300

    except requests.RequestException as e:
        print(f"Fehler bei der Wikipedia-Suche: {e}")
        return []


def get_wikipedia_article(title: str) -> str:
    """Return the content of a Wikipedia article."""
    try:
        url = f"https://de.wikipedia.org/wiki/{title.replace(' ', '_')}"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        content = soup.find("div", class_="mw-parser-output").get_text()
        return content[:2000]
    except requests.RequestException as e:
        print(f"Fehler beim Abrufen des Wikipedia-Artikels: {e}")
        return ""


def download_youtube_transcript(video_url: str) -> str:
    """Download the transcript of a YouTube video."""
    try:
        # Extract video ID from URL
        parsed_url = urlparse(video_url)
        if parsed_url.hostname == "youtu.be":
            video_id = parsed_url.path[1:]
        elif parsed_url.hostname in ("www.youtube.com", "youtube.com"):
            if parsed_url.path == "/watch":
                query_params = parse_qs(parsed_url.query)
                video_id = query_params.get("v", [None])[0]
            elif parsed_url.path.startswith("/embed/") or parsed_url.path.startswith("/v/"):
                video_id = parsed_url.path.split("/")[2]
            else:
                return "Error: Could not extract video ID from URL."
        else:
            return "Error: Invalid YouTube URL."

        if not video_id:
            return "Error: Could not extract video ID from URL."

        # Get transcript
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Try fetching German or English transcript first
        try:
            transcript = transcript_list.find_generated_transcript(["de", "en"])
        except NoTranscriptFound:
            # If not found, fetch any available transcript
            transcript = transcript_list.find_generated_transcript(transcript_list._generated_transcripts.keys())  # noqa: SLF001

        transcript_data = transcript.fetch()

        # Format transcript
        full_transcript = " ".join([item["text"] for item in transcript_data])
        return full_transcript[:4000]  # Limit length if necessary

    except (NoTranscriptFound, TranscriptsDisabled):
        return "Error: No transcript found for this video or transcripts are disabled."
    except Exception as e:  # noqa: BLE001
        return f"An unexpected error occurred: {e!s}"


def get_tool_list() -> dict[str, tuple[Callable, FunctionDefinition]]:
    """Return a dictionary of available tools, keyed by their string name."""
    # Define the original mapping of callable -> FunctionDefinition
    tools_mapping = {
        download_file_from_url: FunctionDefinition(
            name="download_file_from_url",
            description="Download a file from a URL and save it to a temporary location. Returns the path to the downloaded file or an error message.",  # noqa: E501
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL of the file to download."},
                    "filename": {
                        "type": "string",
                        "description": "Optional filename to save the file as. If not provided, it will be inferred from the URL or generated randomly.",  # noqa: E501
                    },
                },
                "required": ["url"],
            },
        ),
        read_content_from_webpage: FunctionDefinition(
            name="read_content_from_webpage",
            description="Read the textual content from a webpage URL and return the first 2000 characters.",
            parameters={
                "type": "object",
                "properties": {"url": {"type": "string", "description": "The URL of the webpage to read."}},
                "required": ["url"],
            },
        ),
        analyze_excel_file: FunctionDefinition(
            name="analyze_excel_file",
            description="Analyze an Excel file (.xlsx) located at the given file path. Returns a summary including row/column count, column names, and basic descriptive statistics.",  # noqa: E501
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The local file path to the Excel file to analyze. This file should typically be downloaded first using 'download_file_from_url'.",  # noqa: E501
                    }
                },
                "required": ["file_path"],
            },
        ),
        analyze_csv_file: FunctionDefinition(
            name="analyze_csv_file",
            description="Analyze a CSV file located at the given file path. Returns a summary including row/column count, column names, and basic descriptive statistics.",  # noqa: E501
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The local file path to the CSV file to analyze. This file should typically be downloaded first using 'download_file_from_url'.",  # noqa: E501
                    }
                },
                "required": ["file_path"],
            },
        ),
        analyze_image: FunctionDefinition(
            name="analyze_image",
            description="Analyze an image provided as a base64 encoded string and return a detailed textual description of its content.",  # noqa: E501
            parameters={
                "type": "object",
                "properties": {
                    "base64_image": {
                        "type": "string",
                        "description": "The base64 encoded string representation of the image to analyze.",
                    }
                },
                "required": ["base64_image"],
            },
        ),
        search_web: FunctionDefinition(
            name="search_web",
            description="Perform a web search using DuckDuckGo with the given search term and return a list of search results, each containing a title, link, and snippet.",  # noqa: E501
            parameters={
                "type": "object",
                "properties": {
                    "search_term": {"type": "string", "description": "The term or query to search for on the web."},
                    "num_results": {
                        "type": "integer",
                        "description": "The maximum number of search results to return. Defaults to 5.",
                        "default": 5,
                    },
                },
                "required": ["search_term"],
            },
        ),
        check_available_wikipedia_articles: FunctionDefinition(
            name="check_available_wikipedia_articles",
            description="Check German Wikipedia for articles matching the given search term. Returns a list of potential exact article titles.",  # noqa: E501
            parameters={
                "type": "object",
                "properties": {
                    "possible_title": {
                        "type": "string",
                        "description": "The search term or potential title to look up on German Wikipedia.",
                    }
                },
                "required": ["possible_title"],
            },
        ),
        get_wikipedia_article: FunctionDefinition(
            name="get_wikipedia_article",
            description="Retrieve the content of a specific German Wikipedia article using its exact title. Returns the first 2000 characters of the article's main content.",  # noqa: E501
            parameters={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The exact title of the German Wikipedia article to retrieve. Use 'check_available_wikipedia_articles' first to find potential titles.",  # noqa: E501
                    }
                },
                "required": ["title"],
            },
        ),
        download_youtube_transcript: FunctionDefinition(
            name="download_youtube_transcript",
            description="Download the available transcript (preferring German or English) for a given YouTube video URL. Returns the first 4000 characters of the transcript text or an error message.",  # noqa: E501
            parameters={
                "type": "object",
                "properties": {"video_url": {"type": "string", "description": "The URL of the YouTube video."}},
                "required": ["video_url"],
            },
        ),
    }

    # Transform the dictionary using a dictionary comprehension
    return {
            func_def.name: (callable_obj, func_def)
            for callable_obj, func_def in tools_mapping.items()
            }
