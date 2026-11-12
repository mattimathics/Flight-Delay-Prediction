"""Browser test for the live demo (docs/index.html).

It opens the page the way a visitor would, does the four steps and checks a result appears,
so a later edit can't quietly break the demo. Skipped when Playwright isn't installed, so the
data tests still run anywhere; CI installs it in its own job.
"""
import functools
import http.server
import os
import threading
from pathlib import Path

import pytest

sync_api = pytest.importorskip("playwright.sync_api")

DOCS = Path(__file__).resolve().parent.parent / "docs"


@pytest.fixture(scope="module")
def page_url():
    # Serve docs/ on a free port, as GitHub Pages would, because the page fetches flights.json
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(DOCS))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/index.html"
    server.shutdown()


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as p:
        # PLAYWRIGHT_CHROMIUM lets a machine with its own Chromium use it instead of a download
        path = os.environ.get("PLAYWRIGHT_CHROMIUM")
        b = p.chromium.launch(executable_path=path) if path else p.chromium.launch()
        yield b
        b.close()


@pytest.fixture
def page(browser, page_url):
    page = browser.new_page(viewport={"width": 375, "height": 812})  # a phone, the tightest layout
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(page_url)
    page.wait_for_selector("#airport option[value='SFO']", state="attached")
    yield page
    assert errors == [], f"the page threw errors: {errors}"
    page.close()


def test_nothing_is_shown_until_all_four_steps_are_done(page):
    check = page.locator("#check")
    assert page.locator("#result").is_hidden()
    assert check.is_disabled()

    page.select_option("#airport", "SFO")
    page.select_option("#destination", "LAX")
    page.locator("#cal-grid button.has").first.click()
    page.locator("#departures button").first.click()
    # Three steps done: still nothing to check
    assert check.is_disabled()
    assert page.locator("#result").is_hidden()

    page.locator("#when button[data-field='risk_2h']").click()
    assert check.is_enabled()
    check.click()

    result = page.locator("#result")
    assert result.is_visible()
    assert "SFO" in result.locator(".route").inner_text()
    assert "%" in result.locator(".big").inner_text()
    # The verdict box says whether the model was right
    assert result.locator(".judgement").get_attribute("class").split()[1] in {"right", "false-alarm", "missed"}


def test_show_a_mistake_only_shows_confident_mistakes(page):
    for _ in range(10):
        page.click("#mistake")
        verdict = page.locator("#result .judgement")
        kind = verdict.get_attribute("class").split()[1]
        chance = page.locator("#result .big").inner_text()
        value = 99.5 if chance.startswith("Over") else 0.5 if chance.startswith("Under") else float(chance.rstrip("%"))
        # Confident mistakes only: false alarms at 70%+, misses at 15% or less
        assert kind in {"false-alarm", "missed"}
        assert value >= 70 if kind == "false-alarm" else value <= 15


def test_the_header_shows_the_published_score(page):
    stats = page.locator("#hero-stats").inner_text()
    assert "74 / 100" in stats
    assert "695,298" in stats
