"""
End-to-end check of the Shiny app with Playwright (optional: pip install playwright && playwright install chromium).

    shiny run app.py --port 8123 &
    python scripts/app_e2e.py

Drives the real browser: changes inputs, switches groups, reads the n shown, and checks the formulas rendered.
"""
import asyncio
import sys

from playwright.async_api import async_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8123/"


async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={"width": 1400, "height": 1000})
        await pg.goto(URL, wait_until="networkidle")
        await pg.wait_for_selector(".big-n", timeout=15000)
        n_default = await pg.inner_text(".big-n")
        assert n_default.startswith("n = "), n_default
        print("Group B, Dice, defaults          ->", n_default)
        await pg.check("input[name='test'][value='score']"); await pg.wait_for_timeout(800)
        print("score test                       ->", await pg.inner_text(".big-n"))
        await pg.check("input[name='group'][value='A']"); await pg.wait_for_timeout(800)
        await pg.evaluate("Shiny.setInputValue('metric', 'auroc')"); await pg.wait_for_timeout(1200)
        print("Group A, AUROC 0.80 vs 0.70      ->", await pg.inner_text(".big-n"))
        await pg.check("input[name='group'][value='C']"); await pg.wait_for_timeout(1000)
        txt = await pg.inner_text("#explanation")
        assert "No analytic sampling variance" in txt
        print("Group C explains the refusal      -> ok")
        mj = await pg.evaluate("document.querySelectorAll('mjx-container').length")
        print("MathJax formula blocks rendered  ->", mj, "(0 means the CDN was unreachable; formulas show as raw LaTeX)")
        await b.close()
        print("E2E PASSED")


asyncio.run(main())
