"""
Parser Module — HTML → Structured Sections

Extracts structured, section-level text from Groww mutual fund pages.
Uses real CSS selectors derived from actual Groww HTML structure.

Each Groww page is a React (Next.js) SSR app. The key data points are
rendered in the initial HTML inside specific div containers with
hashed CSS module class names. We use partial class-name matching
to handle class hash changes across builds.

See: Docs/ChunkingEmbeddingArchitecture.md §2 for full spec.
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from bs4 import BeautifulSoup, Tag

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import get_scheme_info, get_scheme_slug

logger = logging.getLogger(__name__)


@dataclass
class ParsedSection:
    """A single extracted section from a Groww fund page."""

    section_name: str           # e.g., "fund_details", "holdings"
    raw_text: str               # Cleaned text content
    data_points: list[str]      # Specific data fields found
    source_url: str
    scheme_name: str
    scheme_slug: str


@dataclass
class ParsedPage:
    """Complete parse result for a single Groww fund page."""

    url: str
    scheme_name: str
    scheme_slug: str
    category: str
    sections: list[ParsedSection]
    fund_facts: dict = field(default_factory=dict)  # Structured key-value facts
    parse_warnings: list[str] = field(default_factory=list)


def _find_by_partial_class(soup: BeautifulSoup, tag: str, class_fragment: str) -> Optional[Tag]:
    """Find a tag whose class attribute contains a partial match."""
    return soup.find(tag, class_=lambda c: c and class_fragment in c if c else False)


def _find_all_by_partial_class(soup: BeautifulSoup, tag: str, class_fragment: str) -> list[Tag]:
    """Find all tags whose class attribute contains a partial match."""
    return soup.find_all(tag, class_=lambda c: c and class_fragment in c if c else False)


def _clean_text(text: str) -> str:
    """Clean extracted text: normalize whitespace, remove noise."""
    if not text:
        return ""
    # Remove HTML comment artifacts from React (<!-- -->)
    text = re.sub(r'<!--\s*-->', '', text)
    # Normalize whitespace
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def _extract_fund_details(soup: BeautifulSoup, url: str, scheme_info: dict) -> tuple[Optional[ParsedSection], dict]:
    """
    Extract the Fund Details section (NAV, Min SIP, Fund Size, Expense Ratio, Rating).
    
    Groww HTML structure:
    <div class="fundDetails_fundDetailsContainer__Lj8nM">
      <div class="flex flex-column fundDetails_gap4__E4x8C">
        <div class="...contentTertiary...">NAV: 10 Apr '26</div>
        <div class="...contentPrimary...">₹215.55</div>
      </div>
      ...repeated for Min SIP, Fund Size, Expense Ratio, Rating...
    </div>
    """
    fund_facts = {}
    container = _find_by_partial_class(soup, "div", "fundDetails_fundDetailsContainer")

    if not container:
        return None, fund_facts

    # Each data pair is in a flex-column div with a label + value
    data_pairs = _find_all_by_partial_class(container, "div", "fundDetails_gap4")

    labels_values = []
    for pair in data_pairs:
        label_div = _find_by_partial_class(pair, "div", "contentTertiary")
        value_div = _find_by_partial_class(pair, "div", "contentPrimary")

        if label_div and value_div:
            label = _clean_text(label_div.get_text(strip=True))
            value = _clean_text(value_div.get_text(strip=True))
            labels_values.append((label, value))

    # Parse into structured facts
    data_points = []
    text_parts = [f"{scheme_info['scheme_name']} — Fund Details"]

    for label, value in labels_values:
        text_parts.append(f"{label}: {value}")

        label_lower = label.lower()
        if "nav" in label_lower:
            fund_facts["nav"] = value
            # Extract NAV date from label text
            date_match = re.search(r"NAV:\s*(.+)", label)
            if date_match:
                fund_facts["nav_date"] = date_match.group(1).strip()
            data_points.append("nav")
        elif "sip" in label_lower:
            fund_facts["min_sip"] = value
            data_points.append("min_sip")
        elif "fund size" in label_lower or "aum" in label_lower:
            fund_facts["fund_size"] = value
            data_points.append("fund_size")
        elif "expense" in label_lower:
            fund_facts["expense_ratio"] = value
            data_points.append("expense_ratio")
        elif "rating" in label_lower:
            fund_facts["rating"] = value
            data_points.append("rating")

    if not text_parts:
        return None, fund_facts

    section = ParsedSection(
        section_name="fund_details",
        raw_text="\n".join(text_parts),
        data_points=data_points,
        source_url=url,
        scheme_name=scheme_info["scheme_name"],
        scheme_slug=scheme_info["scheme_slug"],
    )
    return section, fund_facts


def _extract_header(soup: BeautifulSoup, url: str, scheme_info: dict) -> Optional[ParsedSection]:
    """
    Extract the header section (fund name, category tags, 3Y return).
    
    Groww HTML:
    <h1 class="...header_schemeName__...">HDFC Mid Cap Fund Direct Growth</h1>
    <div class="pills_container__...">Equity | Mid Cap | Very High Risk</div>
    <div class="returnStats_...">+25.00% | 3Y annualised</div>
    """
    text_parts = []

    # Fund name from <h1>
    h1 = soup.find("h1")
    if h1:
        text_parts.append(f"Fund Name: {_clean_text(h1.get_text(strip=True))}")

    # Category pills
    pills_container = _find_by_partial_class(soup, "div", "pills_container")
    if pills_container:
        pills = pills_container.find_all("span")
        pill_texts = [p.get_text(strip=True) for p in pills if p.get_text(strip=True)]
        if pill_texts:
            text_parts.append(f"Category: {' | '.join(pill_texts)}")

    # Return stats
    return_container = _find_by_partial_class(soup, "section", "returnStats_returnStatsContainer")
    if return_container:
        return_text = _clean_text(return_container.get_text(separator=" ", strip=True))
        if return_text:
            text_parts.append(f"Returns: {return_text}")

    if not text_parts:
        return None

    return ParsedSection(
        section_name="header",
        raw_text="\n".join(text_parts),
        data_points=["fund_name", "category", "risk_level", "returns"],
        source_url=url,
        scheme_name=scheme_info["scheme_name"],
        scheme_slug=scheme_info["scheme_slug"],
    )


def _extract_holdings(soup: BeautifulSoup, url: str, scheme_info: dict) -> Optional[ParsedSection]:
    """
    Extract Top Holdings section.
    
    Groww HTML:
    <section id="holdingsContainer">
      <h2>Holdings (78)</h2>
      <table> ... rows with Name | Sector | Instruments | Assets% ... </table>
    </section>
    """
    holdings_section = soup.find("section", id="holdingsContainer")
    if not holdings_section:
        return None

    text_parts = [f"{scheme_info['scheme_name']} — Top Holdings"]

    # Get all visible holding rows (skip hidden ones)
    rows = _find_all_by_partial_class(holdings_section, "tr", "holdings_row")
    holdings_count = 0
    for row in rows:
        # Check if row is hidden
        row_classes = " ".join(row.get("class", []))
        if "hidden" in row_classes.lower():
            continue

        cells = row.find_all("td")
        if len(cells) >= 4:
            name = _clean_text(cells[0].get_text(strip=True))
            sector = _clean_text(cells[1].get_text(strip=True))
            instrument = _clean_text(cells[2].get_text(strip=True))
            assets_pct = _clean_text(cells[3].get_text(strip=True))
            text_parts.append(f"{name} | {sector} | {instrument} | {assets_pct}")
            holdings_count += 1

    if holdings_count == 0:
        return None

    return ParsedSection(
        section_name="holdings",
        raw_text="\n".join(text_parts),
        data_points=["top_holdings", f"holdings_count:{holdings_count}"],
        source_url=url,
        scheme_name=scheme_info["scheme_name"],
        scheme_slug=scheme_info["scheme_slug"],
    )


def _extract_returns_rankings(soup: BeautifulSoup, url: str, scheme_info: dict) -> Optional[ParsedSection]:
    """
    Extract Returns and Rankings table.
    
    Groww HTML:
    <div class="returnsAndRankings_container__...">
      <table> Fund returns | Category average | Rank across 3Y, 5Y, 10Y, All </table>
    </div>
    """
    container = _find_by_partial_class(soup, "div", "returnsAndRankings_container")
    if not container:
        return None

    text_parts = [f"{scheme_info['scheme_name']} — Returns and Rankings"]

    table = container.find("table")
    if table:
        # Header row
        headers = table.find("thead")
        if headers:
            header_cells = headers.find_all("th")
            header_text = [_clean_text(h.get_text(strip=True)) for h in header_cells]
            text_parts.append(" | ".join(header_text))

        # Data rows
        tbody = table.find("tbody")
        if tbody:
            for row in tbody.find_all("tr"):
                cells = row.find_all("td")
                row_text = [_clean_text(c.get_text(strip=True)) for c in cells]
                text_parts.append(" | ".join(row_text))

    if len(text_parts) <= 1:
        return None

    return ParsedSection(
        section_name="returns_rankings",
        raw_text="\n".join(text_parts),
        data_points=["fund_returns", "category_average", "rank"],
        source_url=url,
        scheme_name=scheme_info["scheme_name"],
        scheme_slug=scheme_info["scheme_slug"],
    )


def _extract_exit_load_tax(soup: BeautifulSoup, url: str, scheme_info: dict) -> Optional[ParsedSection]:
    """
    Extract Exit Load, Stamp Duty, and Tax Implications section.
    
    Groww HTML:
    <div class="exitLoadStampDutyTax_container__...">
      <h3>Exit load, stamp duty and tax</h3>
      <div> exit load text, stamp duty, STCG/LTCG rates </div>
    </div>
    """
    container = _find_by_partial_class(soup, "div", "exitLoadStampDutyTax_container")
    if not container:
        return None

    text_parts = [f"{scheme_info['scheme_name']} — Exit Load, Stamp Duty & Tax"]

    # Extract each sub-section (exit load, stamp duty, tax)
    sections = _find_all_by_partial_class(container, "div", "exitLoadStampDutyTax_section")
    for sec in sections:
        heading = sec.find(["h4", "h3"])
        content = _find_by_partial_class(sec, "div", "contentSecondary")

        heading_text = _clean_text(heading.get_text(strip=True)) if heading else ""
        content_text = _clean_text(content.get_text(strip=True)) if content else ""

        if heading_text:
            text_parts.append(f"{heading_text}")
        if content_text:
            text_parts.append(content_text)

    if len(text_parts) <= 1:
        return None

    return ParsedSection(
        section_name="exit_load_tax",
        raw_text="\n".join(text_parts),
        data_points=["exit_load", "stamp_duty", "stcg", "ltcg"],
        source_url=url,
        scheme_name=scheme_info["scheme_name"],
        scheme_slug=scheme_info["scheme_slug"],
    )


def _extract_min_investments(soup: BeautifulSoup, url: str, scheme_info: dict) -> Optional[ParsedSection]:
    """
    Extract Minimum Investments section.
    
    Groww HTML:
    <div class="minInvestments_...">
      Min. for 1st investment: ₹100
      Min. for 2nd investment: ₹100
      Min. for SIP: ₹100
    </div>
    """
    container = _find_by_partial_class(soup, "div", "minInvestments_tableContainer")
    if not container:
        return None

    text_parts = [f"{scheme_info['scheme_name']} — Minimum Investments"]

    tables = _find_all_by_partial_class(container, "div", "minInvestments_table")
    for table in tables:
        rows = table.find_all("div", class_=lambda c: c and "vspace-between" in (c if isinstance(c, str) else " ".join(c)))
        for row in rows:
            children = row.find_all("div", recursive=False)
            if len(children) >= 2:
                label = _clean_text(children[0].get_text(strip=True))
                value = _clean_text(children[1].get_text(strip=True))
                text_parts.append(f"{label}: {value}")

    if len(text_parts) <= 1:
        return None

    return ParsedSection(
        section_name="min_investments",
        raw_text="\n".join(text_parts),
        data_points=["min_first_investment", "min_second_investment", "min_sip"],
        source_url=url,
        scheme_name=scheme_info["scheme_name"],
        scheme_slug=scheme_info["scheme_slug"],
    )


def _extract_fund_manager(soup: BeautifulSoup, url: str, scheme_info: dict) -> Optional[ParsedSection]:
    """
    Extract Fund Management section (fund manager name, tenure, education, experience).
    
    Groww HTML:
    <div class="fundManagement_container__...">
      Manager accordions with name, initials, education, experience
    </div>
    """
    container = _find_by_partial_class(soup, "div", "fundManagement_container")
    if not container:
        return None

    text_parts = [f"{scheme_info['scheme_name']} — Fund Management"]

    # Each fund manager is in an accordion
    accordions = _find_all_by_partial_class(container, "div", "fundManagement_accordion")
    for accordion in accordions:
        # Manager name
        name_div = _find_by_partial_class(accordion, "div", "fundManagement_personName")
        if name_div:
            text_parts.append(f"Manager: {_clean_text(name_div.get_text(strip=True))}")

        # Tenure
        tenure_div = accordion.find("div", class_=lambda c: c and "contentSecondary" in c and "bodyLarge" in c if c else False)
        if tenure_div:
            text_parts.append(f"Tenure: {_clean_text(tenure_div.get_text(strip=True))}")

        # Education & Experience (inside expanded content)
        expanded = _find_by_partial_class(accordion, "div", "fundManagement_expandedContent")
        if expanded:
            details = expanded.find_all("div", recursive=False)
            for detail in details:
                title_div = _find_by_partial_class(detail, "div", "fundManagement_detailTitle")
                if title_div:
                    title = _clean_text(title_div.get_text(strip=True))
                    # Get the next sibling div with the content
                    content_div = title_div.find_next_sibling("div")
                    if content_div:
                        content = _clean_text(content_div.get_text(strip=True))
                        text_parts.append(f"{title}: {content}")

    if len(text_parts) <= 1:
        return None

    return ParsedSection(
        section_name="fund_manager",
        raw_text="\n".join(text_parts),
        data_points=["fund_manager_name", "fund_manager_tenure", "fund_manager_education"],
        source_url=url,
        scheme_name=scheme_info["scheme_name"],
        scheme_slug=scheme_info["scheme_slug"],
    )


def _extract_return_calculator(soup: BeautifulSoup, url: str, scheme_info: dict) -> Optional[ParsedSection]:
    """
    Extract Return Calculator section (SIP returns over 1Y, 3Y, 5Y, 10Y).
    
    Groww HTML:
    <div class="returnCalculator_container__...">
      <table> period | total investment | would've become | returns% </table>
    </div>
    """
    container = _find_by_partial_class(soup, "div", "returnCalculator_container")
    if not container:
        return None

    text_parts = [f"{scheme_info['scheme_name']} — Return Calculator (SIP)"]

    table = container.find("table")
    if table:
        tbody = table.find("tbody")
        if tbody:
            for row in tbody.find_all("tr"):
                cells = row.find_all("td")
                if len(cells) >= 3:
                    period = _clean_text(cells[0].get_text(strip=True))
                    invested = _clean_text(cells[1].get_text(strip=True))
                    became = _clean_text(cells[2].get_text(strip=True))
                    # Returns % is in the last cell
                    returns_pct = _clean_text(cells[-1].get_text(strip=True)) if len(cells) > 3 else ""
                    text_parts.append(f"{period}: Invested {invested} → Became {became} ({returns_pct})")

    if len(text_parts) <= 1:
        return None

    return ParsedSection(
        section_name="return_calculator",
        raw_text="\n".join(text_parts),
        data_points=["sip_returns_1y", "sip_returns_3y", "sip_returns_5y", "sip_returns_10y"],
        source_url=url,
        scheme_name=scheme_info["scheme_name"],
        scheme_slug=scheme_info["scheme_slug"],
    )


def _extract_compare_similar(soup: BeautifulSoup, url: str, scheme_info: dict) -> Optional[ParsedSection]:
    """
    Extract Compare Similar Funds section.
    
    Groww HTML:
    <div class="compareSimilarFunds_container__...">
      <table> Name | 1Y | 3Y | Fund Size </table>
    </div>
    """
    container = _find_by_partial_class(soup, "div", "compareSimilarFunds_container")
    if not container:
        return None

    text_parts = [f"{scheme_info['scheme_name']} — Compare Similar Funds"]

    table = container.find("table")
    if table:
        tbody = table.find("tbody")
        if tbody:
            for row in tbody.find_all("tr"):
                cells = row.find_all("td")
                if len(cells) >= 4:
                    # Name is in the second cell (first is checkbox)
                    name_span = _find_by_partial_class(cells[1], "span", "compareSimilarFunds_fundName")
                    name = _clean_text(name_span.get_text(strip=True)) if name_span else _clean_text(cells[1].get_text(strip=True))
                    ret_1y = _clean_text(cells[2].get_text(strip=True))
                    ret_3y = _clean_text(cells[3].get_text(strip=True))
                    fund_size = _clean_text(cells[4].get_text(strip=True)) if len(cells) > 4 else ""
                    text_parts.append(f"{name} | 1Y: {ret_1y} | 3Y: {ret_3y} | Size: {fund_size} Cr")

    if len(text_parts) <= 1:
        return None

    return ParsedSection(
        section_name="compare_similar",
        raw_text="\n".join(text_parts),
        data_points=["peer_comparison"],
        source_url=url,
        scheme_name=scheme_info["scheme_name"],
        scheme_slug=scheme_info["scheme_slug"],
    )


# =============================================================================
# Main Parser Entry Point
# =============================================================================

def parse_groww_page(html: str, url: str) -> ParsedPage:
    """
    Parse a Groww mutual fund HTML page into structured sections.

    Returns a ParsedPage with all extracted sections and fund facts.
    """
    scheme_info = get_scheme_info(url)
    soup = BeautifulSoup(html, "html.parser")

    sections: list[ParsedSection] = []
    fund_facts: dict = {}
    warnings: list[str] = []

    # --- Extract each section ---
    extractors = [
        ("header", _extract_header),
        ("fund_details", lambda s, u, si: _extract_fund_details(s, u, si)),
        ("holdings", _extract_holdings),
        ("returns_rankings", _extract_returns_rankings),
        ("exit_load_tax", _extract_exit_load_tax),
        ("min_investments", _extract_min_investments),
        ("fund_manager", _extract_fund_manager),
        ("return_calculator", _extract_return_calculator),
        ("compare_similar", _extract_compare_similar),
    ]

    for section_name, extractor in extractors:
        try:
            if section_name == "fund_details":
                result, facts = extractor(soup, url, scheme_info)
                fund_facts.update(facts)
            else:
                result = extractor(soup, url, scheme_info)

            if result:
                sections.append(result)
                logger.info(
                    f"  ✓ Extracted '{section_name}': {len(result.raw_text)} chars, "
                    f"data_points={result.data_points}"
                )
            else:
                warnings.append(f"Section '{section_name}' not found")
                logger.warning(f"  ✗ Section '{section_name}' not found for {url}")

        except Exception as e:
            warnings.append(f"Error extracting '{section_name}': {str(e)}")
            logger.error(f"  ✗ Error extracting '{section_name}' for {url}: {e}")

    logger.info(
        f"Parsed {scheme_info['scheme_name']}: "
        f"{len(sections)} sections extracted, {len(warnings)} warnings"
    )

    return ParsedPage(
        url=url,
        scheme_name=scheme_info["scheme_name"],
        scheme_slug=scheme_info["scheme_slug"],
        category=scheme_info["category"],
        sections=sections,
        fund_facts=fund_facts,
        parse_warnings=warnings,
    )


def parse_raw_html_file(filepath: str, url: str) -> ParsedPage:
    """
    Parse a saved raw HTML file from data/raw/.
    Convenience function for testing and the chunking pipeline.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        html = f.read()
    return parse_groww_page(html, url)
