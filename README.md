# MCP Web Tools

This package provides a powerful MCP server to equip LLMs with web access, going beyond naive methods of searching, fetching and extracting content.

## Introduction

I created this package out of the frustration that most MCP servers enabling web access to LLMs, didn't perform as well as I hoped. Some of these shortcomings I wanted fix, include:

- [x] Good search results without requiring an API key
- [x] Sophisticated fetching for more complex JavaScript sites
- [x] Extracting content in nicely formatted Markdown
- [x] Support for extracting content from PDFs
- [x] Support for loading and displaying images
- [x] Usage options for advanced cases like loading raw HTML

## Installation

### Claude Desktop

```json
```

### Claude Code

```bash
claude mcp add web-tools uvx mcp-web-tools
```

Or to also set the Brave Search API key:

```bash
claude mcp add web-tools uvx mcp-web-tools -e BRAVE_SEARCH_API_KEY=<key>
```


## Internals

The package is written in Python using powerful libraries and services under the hood to improve results.

### Searching

We use the [Brave Search API](https://brave.com/search/api) if an API key is provided with fallbacks a scraped Google workaround and finally DuckDuckGO if everything else fails. While we'd recommend getting a free Brave API key, the fallbacks should work more than well enough for most use cases and work loads.

### Fetching

The fetching of web content is based on [Zendriver](https://github.com/stephanlensky/zendriver), a fork of [nodriver](https://github.com/ultrafunkamsterdam/nodriver/) for next level webscraping and performance. It should stay undetected for most anti-bot solutions and fetch content even from complex JS-based sites.

### Extracting

For web extraction, we use [Trafilatura](https://trafilatura.readthedocs.io/en/latest/index.html) which consistently outperforms other alternatives for extracting content from HTML pages. For PDFs, we use [PyMuPDF4LLM](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/) which similarly extracts content in an easy-to-read format for LLMs, with advanced layout support.

## Contributing

While it's impossible to support all pages and layouts, we thrive to make this package better over time. For unsupported sites, problems, or feature requests open an issue.