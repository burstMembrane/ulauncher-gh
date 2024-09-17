""" Main Module """

from enum import Enum
import logging
import os
from typing import List
from dotenv import load_dotenv
from duckdb import description
import gi

gi.require_version("Gtk", "3.0")
# pylint: disable=import-error
from gi.repository import Gio, Gtk
from ulauncher.api.client.EventListener import EventListener
from ulauncher.api.client.Extension import Extension
from ulauncher.api.shared.action.DoNothingAction import DoNothingAction
from ulauncher.api.shared.action.HideWindowAction import HideWindowAction
from ulauncher.api.shared.action.OpenAction import OpenAction
from ulauncher.api.shared.action.RenderResultListAction import RenderResultListAction
from ulauncher.api.shared.action.RunScriptAction import RunScriptAction
from ulauncher.api.shared.event import KeywordQueryEvent
from ulauncher.api.shared.item.ExtensionResultItem import ExtensionResultItem
from ulauncher.api.shared.item.ExtensionSmallResultItem import ExtensionSmallResultItem
from ulauncher.utils.fuzzy_search import get_score
import requests
from rapidfuzz.distance import Levenshtein


from dataclasses import dataclass
from textwrap import wrap
from cachier import cachier
import re

logger = logging.getLogger("ulauncher-gh")


repos = {}

# load the dotenv if it is available for dev
load_dotenv()


class SearchDirection(Enum):
    FORWARD = "forward"
    BACKWARD = "backward"


@dataclass
class GithubRepo:
    short_name: str
    name: str
    description: str
    icon: str
    url: str
    stars: int
    topics: list
    creation_date: str = ""
    language: str = ""
    forks: int = 0
    last_updated: str = ""
    home_page: str = ""


# Command score function
def command_score(string, abbreviation, aliases=None):
    if aliases is None:
        aliases = []

    # Score constants
    SCORE_CONTINUE_MATCH = 1
    SCORE_SPACE_WORD_JUMP = 0.9
    SCORE_NON_SPACE_WORD_JUMP = 0.8
    SCORE_CHARACTER_JUMP = 0.17

    PENALTY_SKIPPED = 0.999
    PENALTY_CASE_MISMATCH = 0.9999

    # Regular expressions for matching gaps and spaces
    IS_GAP_REGEXP = re.compile(r'[\\\/_+.#"@\[\(\{&]')
    IS_SPACE_REGEXP = re.compile(r"[\s-]")

    # Convert string and aliases to lowercase
    lower_string = (string + " " + " ".join(aliases)).lower()
    lower_abbreviation = abbreviation.lower()

    # Recursive function to calculate score
    def score(string_index, abbr_index, memo=None):
        if memo is None:
            memo = {}

        # Memoization key
        memo_key = (string_index, abbr_index)
        if memo_key in memo:
            return memo[memo_key]

        # Base case: if we have matched all abbreviation characters
        if abbr_index == len(abbreviation):
            return SCORE_CONTINUE_MATCH if string_index == len(string) else 0.99

        # Find the next matching character in the string
        abbreviation_char = lower_abbreviation[abbr_index]
        high_score = 0
        index = lower_string.find(abbreviation_char, string_index)

        # Loop through possible matches
        while index != -1:
            temp_score = score(index + 1, abbr_index + 1, memo)

            # Continuous match
            if index == string_index:
                temp_score *= SCORE_CONTINUE_MATCH
            # Word boundary match
            elif IS_SPACE_REGEXP.match(lower_string[index - 1]):
                temp_score *= SCORE_SPACE_WORD_JUMP
            elif IS_GAP_REGEXP.match(lower_string[index - 1]):
                temp_score *= SCORE_NON_SPACE_WORD_JUMP
            # Character jump
            else:
                temp_score *= SCORE_CHARACTER_JUMP
                if string_index > 0:
                    temp_score *= PENALTY_SKIPPED ** (index - string_index)

            # handle indexerror
            if index < len(string):
                temp_score *= PENALTY_SKIPPED ** (len(string) - index)
            if abbr_index < len(abbreviation):
                temp_score *= PENALTY_SKIPPED ** (len(abbreviation) - abbr_index)
            elif string[index] != abbreviation[abbr_index]:
                # Case mismatch penalty

                temp_score *= PENALTY_CASE_MISMATCH

            # Update the best score
            if temp_score > high_score:
                high_score = temp_score

            # Look for the next match in the string
            index = lower_string.find(abbreviation_char, index + 1)

        # Memoize the result
        memo[memo_key] = high_score
        return high_score

    # Start the scoring from the first character
    return score(0, 0)


class GithubExtension(Extension):
    """Main Extension Class"""

    def __init__(self):
        """Initializes the extension"""
        super(GithubExtension, self).__init__()
        self.subscribe(KeywordQueryEvent, KeywordQueryEventListener())
        # using an access token

        if not self.preferences.get("access_token"):
            logger.warning("No access token found in preferences. Please add one.")

        auth_token = self.preferences.get("access_token") or os.getenv(
            "GITHUB_AUTH_TOKEN"
        )

        # cloak the access token in logs but log the first 4 characters then xxxxx
        logger.info(
            "Using access token: %s", auth_token[:4] + "x" * len(auth_token[4:])
        )
        self.auth_token = auth_token

        self.prev_query = ""
        self.prev_results = []

        self.queries = []

        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"token {auth_token}"})

        self.distance_threshold = 5

    @cachier()
    def search_github(
        self, query, per_page=10, max_repos=10, sort="stars", order="desc"
    ):
        repos = []

        page = 1
        headers = {"Authorization": f"token {self.auth_token}"}
        while len(repos) < max_repos:
            url = f"https://api.github.com/search/repositories?q={query}&per_page={per_page}&page={page}"

            if sort:
                url += f"&sort={sort}"
            if order:
                url += f"&order={order}"

            response = self.session.get(url, headers=headers).json()

            items = response.get("items", [])

            if not items:
                break

            repos.extend(items)
            page += 1

        repo_objs = [
            {
                "short_name": repo.get("name", ""),
                "name": repo.get("full_name", ""),
                "url": repo.get("html_url", ""),
                "creation_date": repo.get("created_at", ""),
                "stars": repo.get("stargazers_count", 0),
                "forks": repo.get("forks_count", 0),
                "language": repo.get("language", ""),
                "last_updated": repo.get("updated_at", ""),
                "description": repo.get("description", ""),
                "topics": repo.get("topics", []),
                "icon": "images/repo.png",
                "home_page": repo.get("homepage", ""),
            }
            for repo in repos
        ]

        # reorder the repos by their distance form the input query
        repos = [GithubRepo(**repo) for repo in repo_objs]
        return repos

    def sort_repos(self, repos: List[GithubRepo], query):
        """
        Sort the repos based on their similarity to the query using Levenshtein distance and their stars.
        """
        return sorted(
            repos, key=lambda x: x.stars * command_score(x.name, query), reverse=True
        )

    def search(self, query):
        """Search for GitHub repos using the API with fuzzy search for cached repos."""

        # Check if the current query is a forward continuation of the previous one

        # if there's a cached query -- return it

        if (
            self.queries
            and len(self.queries[-1]) < len(query)
            and query.startswith(self.queries[-1])
        ):
            logger.info("Going forward")
            self.direction = "forward"
        elif (
            self.queries
            and len(self.queries[-1]) > len(query)
            and self.queries[-1].startswith(query)
            and Levenshtein.distance(self.queries[-1], query) < self.distance_threshold
        ):
            logger.info("Going backward")
            self.direction = "backward"
            return self.prev_results

        if (
            query.startswith(self.prev_query)
            and Levenshtein.distance(self.prev_query, query) < self.distance_threshold
        ):
            results = self.sort_repos(self.prev_results, query)
            return results

        # If it's a completely new query, perform a fresh search
        logger.info("Performing a new search for %s", query)
        repos = self.search_github(query, per_page=10, max_repos=10)

        logger.info("Found %s results", len(repos))

        if repos:
            # Cache the current query and results for future use
            self.prev_query = query

            self.prev_results = repos

            self.queries.append(query)
            logger.debug("Queries: %s", self.queries)

        repos = self.sort_repos(repos, query)
        return repos


class KeywordQueryEventListener(EventListener):
    """Listener that handles the user input"""

    # pylint: disable=unused-argument,no-self-use
    def on_event(self, event, extension):
        """Handles the event"""
        items = []

        query = event.get_argument()

        keyword = event.get_keyword()
        logger.info("Received keyword: %s", keyword)
        logger.info("Received query: %s", query)

        if not extension.auth_token:
            return RenderResultListAction(
                [
                    ExtensionResultItem(
                        icon="images/github-mark-white.png",
                        name="Auth Token Not Set. Please set your Github access token in the extension preferences",
                        on_enter=OpenAction("preferences://extensions/"),
                    )
                ]
            )

        if not query or len(query) < 2:
            return RenderResultListAction(
                [
                    ExtensionResultItem(
                        icon="images/github-mark-white.png",
                        name="Keep typing to search Github repositories",
                        on_enter=DoNothingAction(),
                    ),
                ]
            )

        # Find the keyword id using the keyword (since the keyword can be changed by users)
        repos = extension.search(query) or extension.prev_results
        if not repos:
            return RenderResultListAction(
                [
                    ExtensionResultItem(
                        icon="images/github-mark-white.png",
                        name="No Results found matching %s" % query,
                        on_enter=HideWindowAction(),
                    )
                ]
            )

        items = []
        for repo in repos:

            description = repo.description[:180] if repo.description else ""
            description = (
                description + "..." if len(description) >= 180 else description
            )
            description = "\n".join(wrap(description, 60))

            items.append(
                ExtensionResultItem(
                    name=repo.name or "",
                    on_enter=OpenAction(repo.url),
                    on_alt_enter=OpenAction(repo.home_page),
                    icon=repo.icon or "",
                    description=f"{description} \n  ⭐ {humanfriendly_numbers(repo.stars)} ",
                )
            )

        return RenderResultListAction(items)


def humanfriendly_numbers(number):
    """Convert a number to a human-friendly format"""
    if number < 1000:
        return number
    if number < 1000000:
        return f"{number/1000:.1f}k"
    return f"{number/1000000:.1f}M"


if __name__ == "__main__":
    GithubExtension().run()
