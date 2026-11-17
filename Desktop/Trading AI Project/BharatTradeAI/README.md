# BharatTradeAI

BharatTradeAI is an Indian market intelligence and alert platform focused on options flow, technical structure, market regime, risk controls, and AI-assisted signal generation.

## Iteration 1

This iteration sets up the base environment:
- Python 3.11+
- Virtual environment
- project dependencies
- environment variables
- base folder structure

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create a local `.env` file based on `.env` in the project root.

## News intelligence

`GET /api/v1/news?query=Reliance&limit=25` searches recent article metadata. By default it uses GDELT (past seven days); GDELT permits commercial use with attribution, but may rate-limit requests. For a more reliable local feed, register for NewsData.io's free key and add it to the project `.env`:

```dotenv
NEWS_DATA_API_KEY=your_private_key
```

The NewsData.io free tier provides up to 200 credits/day, returns at most 10 articles per request, and delays coverage by 12 hours. Keep the key private and use NewsData only for personal or internal purposes under its [terms](https://newsdata.io/terms); do not expose this keyed API through a public service.

Headline sentiment describes wording and coverage only; it is not a price forecast, a buy/sell recommendation, or a substitute for market-price analysis. GDELT's terms permit commercial use and require attribution when redistributing its data; see [GDELT terms](https://www.gdeltproject.org/about.html#termsofuse).

## IPO market data

The IPO Market section of the frontend uses Xflot's free public API; no API key or signup is required. The backend proxies and caches these endpoints for 60 seconds. Responses include Xflot provider, last-successful-fetch, provider update time (when supplied), cache age, and fresh/stale status; a cached response is explicitly marked stale when the upstream request fails. The directory prioritizes open and upcoming issues. GMP is marked unofficial and may be disputed:

- `GET /api/v1/market/ipos` — issue calendar with status, board, price band, dates, GMP, and subscription summary.
- `GET /api/v1/market/ipos/{slug}` — issue details, GMP history, and category-wise subscription.
- `GET /api/v1/market/deals` — disclosed bulk and block deals.
- `GET /api/v1/market/scoreboard` — GMP-implied versus actual listing performance.

The dashboard attributes the data to [Xflot](https://www.xflot.com/api-docs) and displays the provider's disclaimer. GMP is unofficial and may be disputed; it is not investment advice. Keep the provider attribution and disclaimer when reusing the data.

## Stock live quotes

The Stock → Live page fetches NIFTY 50, Bank Nifty, and a starter NSE large-cap watchlist from Yahoo Finance's public chart endpoint. No API key, signup, or broker session is required. The backend proxies and caches the 1-minute chart snapshots for at least 60 seconds; the page displays price, daily change, OHLC, volume, and the provider's quote timestamp.

Yahoo Finance does not document this endpoint as a public API, and it may change, rate-limit requests, or return delayed/stale quotes. This is not a streaming or guaranteed real-time exchange feed. The UI identifies the source, last successful refresh, provider timestamp, approximate quote age, market-open state, and per-quote freshness. Future provider timestamps more than two minutes ahead are marked unverified; the snapshot is still displayed with a warning but is not eligible for AI opportunity screening. Small clock-skew differences are flagged and displayed using the server timestamp.

The Stock page also supports a five-day, five-minute intraday chart from the same Yahoo chart endpoint (`GET /api/v1/stocks/intraday/{symbol}?interval=5m`) and a stored daily-history chart. Intraday intervals are limited to 1m, 5m, and 15m; retrieval is bounded and cached for 60 seconds. It does not scrape provider pages or introduce another market-data source.

The Stock → Live page retains the small starter watchlist. AI Stock Picks uses a separately configured 50-symbol NSE large-cap universe, with Yahoo Finance quote snapshots cached for five minutes to limit requests to the public provider. Its cards are based on the 50-symbol model-status response, so a stock remains visible when its quote or prediction is unavailable; unavailable prices are not fabricated. Model probabilities are displayed only when that symbol has sufficient daily history and its own model has passed calibration and chronological out-of-sample evaluation. `GET /api/v1/ai-stock-picks/quotes` returns quotes for the expanded universe, and `GET /api/v1/ai-stock-picks/predictions` returns per-symbol validation states. The tracked constituent list is configured in `app/services/market/live_quotes.py` and should be reviewed when NIFTY 50 membership or Yahoo Finance ticker symbols change.

`GET /api/v1/stocks/history/{symbol}` reads persisted daily OHLCV bars (five-year range) from `market_candles`, including previous closes and the latest market timestamp. `POST /api/v1/stocks/history/sync` refreshes the configured history universe and is rate-limited to one manual sync per 15 minutes. A background job refreshes history on weekdays at 16:15 Asia/Kolkata time. Sync status is stored per symbol in `market_data_syncs`; the history response marks data stale when its newest market bar is over four days old. The public provider may rate-limit a wide sync; partial results and per-symbol failures are surfaced, and no model probability is shown for missing or stale data.

The Next-session opportunities screen defaults to a strict calibrated upward-probability threshold above 80%; users can choose above 65%, 70%, or 80% (`GET /api/v1/ai-stock-picks/opportunities?threshold_percent=65|70|80`). The selected threshold only filters actual calibrated model outputs; it does not change or generate probabilities. Out-of-sample validation, fresh quote and daily-history checks, complete technical inputs, descending probability ranking, and the requirement that probability be strictly greater than the chosen threshold remain in effect. No qualifying prediction is shown as an empty result rather than replaced with a forced pick.

The history request uses Yahoo's existing chart data endpoint, not HTML-page scraping or a second market-data provider. Yahoo's endpoint is unofficial and subject to its terms, availability, delays, and rate limits. AI Stock Picks builds its stock cards from the 50-symbol model-status response and merges in Yahoo quote rows; all tracked stocks remain visible even when Yahoo cannot return a quote. The quote endpoint fetches the expanded universe concurrently and reports unavailable symbols rather than dropping them. It does not display hard-coded sample picks or sample probabilities. A direction probability is displayed only if a saved model is trained, calibrated, out-of-sample evaluated, and has fresh daily history. The screen refreshes after quote polling and distinguishes current quote context from the daily-bar input cutoff.

News Intelligence shows publication and retrieval times separately, supports company/source/recency filters, and labels provider sentiment, provider tone, and local keyword heuristics separately from FinBERT. The News Intelligence page does not claim that its headline score is FinBERT output.

AI Stock Picks company news is collected only through the configured NewsData.io integration or its existing GDELT fallback. `POST /api/v1/ai-stock-picks/news/sync` queries the existing provider adapter for tracked NSE companies, matches company names/tickers, deduplicates by URL/headline, labels sentiment and event type, and stores the publication timestamp and publisher in `ai_stock_news`. `GET /api/v1/ai-stock-picks/news` returns company-specific sentiment averages and article context from the preceding seven days; an optional timezone-aware ISO 8601 `cutoff` excludes any item published at or after the cutoff. Sync requests are limited to one every 15 minutes. News source failures and stale-cache warnings are surfaced in the AI Stock Picks view. The existing Market Intelligence API and UI behavior are unchanged. FinBERT sentiment is used as a model feature only when analyzed article publication times precede the historical cutoff; sentiment itself is not a stock-return prediction.

### FinBERT financial sentiment

AI Stock Picks analyzes stored company headlines and descriptions with the pretrained `ProsusAI/finbert` model. Install the pinned PyTorch and Transformers dependencies with `pip install -r requirements.txt`; the tokenizer and model weights are loaded lazily on the first analysis and cached in process. The initial model load requires Hugging Face model access and local disk space. Inference defaults to CPU. If dependencies or model access are unavailable, the analyze endpoint returns `503` rather than substituting heuristic sentiment.

- `POST /api/v1/ai-stock-picks/sentiment/analyze` — run FinBERT on up to 10 latest stored news items per stock by default and persist per-article positive, negative, and neutral probabilities. The `limit` query parameter can raise this to 500 items per stock.
- `GET /api/v1/ai-stock-picks/sentiment` — return stored article-level probabilities and stock-level aggregates. Supports optional `symbol`, timezone-aware ISO 8601 `cutoff`, and `limit` query parameters (10 articles per stock by default). For example: `GET /api/v1/ai-stock-picks/sentiment?symbol=TCS&cutoff=2026-11-03T10:00:00%2B00:00`.

The service excludes articles outside the preceding seven days and all articles published at or after the requested cutoff. Company aliases and provider ticker matches associate news with tracked stocks; title matches receive greater relevance than description-only or ticker-only matches. Stock-level probabilities are weighted by this relevance and by an exponential freshness decay with a 48-hour half-life, then normalized to sum to one. The overall sentiment score is `positive_probability - negative_probability` (range -1 to 1). Article inference is cached by symbol, article, and model ID in `ai_stock_article_sentiment`; the model registry and `FinancialSentimentModel` protocol allow another classifier to be added without changing the aggregation API. The AI Stock Picks UI exposes class probabilities, relevance/freshness weights, the cutoff, and the FinBERT model ID. For historical model features, both the article publication and retrieval and its FinBERT analysis must precede the daily-bar cutoff. Sentiment is news tone, not the probability of a stock return.

### LangChain + point-in-time news RAG

The News Intelligence page archives company-matched news snapshots (headline, available description, publisher, provider, URL, publication time, first retrieval time, and symbol). Content revisions are stored as separate immutable snapshots. Historical retrieval requires both publication and retrieval timestamps to be strictly earlier than the requested cutoff, so an old article first collected later cannot appear as evidence for an earlier date. Legacy company-news rows are backfilled with their stored retrieval timestamp; earlier collection history that was never recorded cannot be reconstructed.

The grounded-news question panel uses LangChain `ChatPromptTemplate` and `ChatOllama`, LangChain Ollama embeddings, `RecursiveCharacterTextSplitter`, and a persistent Chroma vector store at `.data/rag/chroma`. Install the configured local embedding model once:

```powershell
ollama pull nomic-embed-text
```

The defaults are `RAG_EMBEDDING_MODEL=nomic-embed-text` and a vector store under the project `.data` directory; both can be overridden in `.env` using `RAG_EMBEDDING_MODEL` and `RAG_VECTOR_STORE_PATH`. Ask a question from News Intelligence or call `POST /api/v1/ai-stock-picks/news/grounded-analysis` with a tracked `symbol`, a question, and optionally a timezone-aware ISO 8601 cutoff. The service indexes archived snapshots on demand, retrieves a bounded number of matching chunks for that symbol and cutoff, and returns source URLs, dates, excerpts, and an evidence-status indicator. The LLM must cite retrieved article IDs; unknown citations are rejected. No retrieved evidence returns an explicit no-evidence answer, and mixed positive/negative provider sentiment labels are flagged as potentially conflicting rather than reconciled away.

RAG improves evidence discovery and explanation, not numerical forecasting by itself. Gemma/LangChain never supplies or changes a stock price, technical indicator, or upward probability. Existing numerical predictions remain outputs of the calibrated, out-of-sample-tested XGBoost/LSTM service. FinBERT/news-derived features are eligible only when their archived news was both published and retrieved before each historical candle cutoff. Any future model change using RAG-derived features must first pass chronological walk-forward evaluation against the current baseline; no accuracy improvement is assumed.

`GET /api/v1/ai-stock-picks/models/news-impact/{symbol}` performs a paired expanding-window XGBoost ablation with and without the time-aware FinBERT news inputs. It reports out-of-sample Brier score, expected calibration error, classification metrics, and metric deltas on matched folds; it returns an explicit insufficient-news-history status when the archived news does not cover at least 30 evaluation observations. This audit does not change the production model or add LLM-generated text to prediction features.

### XGBoost + LSTM next-session direction

The AI Stock Picks model uses the stored daily OHLCV history from the existing Yahoo Finance adapter; sync now requests five years to cover the 200-session SMA warm-up and chronological holdouts. It predicts whether the next regular trading session's close will be strictly higher than the preceding regular session's close. End-of-day features include OHLCV returns and volume, RSI(14), MACD, EMA(20/50), SMA(200), Bollinger Bands, momentum, ATR, realized volatility, and timestamp-bounded FinBERT/market/sector context. Missing news, sector, or market context is represented as unavailable rather than silently fabricated.

- `POST /api/v1/ai-stock-picks/models/train/{symbol}?lstm_epochs=30` trains and versions XGBoost and LSTM models; `lstm_epochs` accepts 1–100.
- `POST /api/v1/ai-stock-picks/models/train` queues background training for stocks in the live quote watchlist without a usable validated model. Poll `GET /api/v1/ai-stock-picks/models/train/jobs/{job_id}` for progress and per-stock results; existing validated models are reused, and data/model failures do not produce probabilities.
- `GET /api/v1/ai-stock-picks/predictions` returns the tracked quote-watchlist symbols with a model validation status for each; `GET /api/v1/ai-stock-picks/predictions/{symbol}` returns one stock's latest prediction.
- `GET /api/v1/ai-stock-picks/predictions/{symbol}` returns the latest saved prediction and its out-of-sample metrics. It returns `not_trained`, `insufficient_data`, or `stale_data` rather than a probability when the model cannot be used safely.
- Training uses chronological 60% training, 15% probability calibration, 10% ensemble-selection, and 15% untouched test windows. The LSTM scaler is fitted only on its training sequences. A sigmoid calibrator is fitted on the separate calibration window.
- The XGBoost/LSTM ensemble weight is selected on the dedicated selection window using Brier score and is used only if it improves over the better individual model. Otherwise the better validation model is kept. The response compares both model test results, reports precision, recall, F1 and Brier score, and includes a calibrated expanding-window XGBoost backtest.
- Probability at or above 80% is only a screening threshold (`probability_threshold_met`); it is not an 80% accuracy claim or a guarantee. Reported quality metrics are historical out-of-sample measurements, not future performance guarantees.
- Versioned artifacts are saved under `model_artifacts/ai_stock_picks/{SYMBOL}` by default; `AI_STOCK_MODEL_DIR` can override the directory. Install the pinned XGBoost, scikit-learn, and PyTorch dependencies using the Quick start instructions.

The model API supports the configured 50-symbol NSE AI universe and active NSE/BSE stock instruments already present in `market_instruments` with daily candles stored in `market_candles`; it does not invent BSE symbols or bypass the configured market-data pipeline. The UI lists every tracked symbol, but “not trained,” “insufficient data,” or “unavailable” is a real status—not a prediction. Use “Sync daily history” before training newly added symbols; the NIFTY 50 training job evaluates each stock separately and displays only models that pass the existing validation gates. Sector returns are derived only when sector metadata and peer history exist; NIFTY 50 history is used for market trend context. Stored FinBERT sentiment contributes only when its article publication timestamp precedes the corresponding Indian-market close cutoff. Historical news may be sparse; the feature's availability flag and API context status expose this limitation.

### Validated opportunities and Gemma 3 explanations

`GET /api/v1/ai-stock-picks/opportunities` screens the existing Yahoo Finance quote watchlist. It requires a fresh quote (15-minute limit), a daily bar no older than four days, an out-of-sample-evaluated model with the chronological sigmoid calibrator, complete technical history, and an upward probability **strictly greater than 80%**. Results are ordered by calibrated probability; risk and FinBERT sentiment are included as analysis context, not used to fabricate or adjust the model probability. Risk is `HIGH` at ATR(14) of at least 4% or 20-session daily volatility of at least 3%, `MEDIUM` at ATR(14) of at least 2% or volatility of at least 1.5%, and `LOW` otherwise. If no stock qualifies, the API returns `No high-confidence opportunities found for the next trading day.` Expected price ranges remain `null` because the current classifier predicts direction only and has no validated price-range model.

The endpoint sends the validated prediction, Yahoo quote, historical technical summary, and stored FinBERT aggregates to the configured local Ollama `/api/generate` endpoint using `OLLAMA_BASE_URL` and `OLLAMA_MODEL` (or the legacy `MODEL_NAME`, default `gemma3:latest`). The API and model run locally; no Ollama cloud API key is needed. Gemma explanation generation runs as a background task so slow CPU inference cannot hold the opportunity-screen response open. The UI shows pending/ready/unavailable explanation state; validated ML probabilities and selection are independent of Gemma. The model is constrained to selecting IDs from server-generated bullish, bearish, and risk evidence. The API renders all displayed labels, probabilities, and numerical facts from trusted application data; LLM prose and new numerical claims are not accepted. If Ollama is down or returns invalid evidence IDs, the opportunity remains available with an explicit unavailable explanation status; its ML probability is unaffected. The next-session date uses a weekday estimate because an NSE/BSE holiday calendar is not currently configured. The quote integration is the existing Yahoo public endpoint; it may be delayed or rate-limited, and no new provider or unauthorized scraping is introduced.

## Iteration 11

Signal scoring combines the six analysis components into a 0-100 score:

- Technical: 20%
- Options: 20%
- ML: 25%
- Market regime: 15%
- News: 10%
- Fundamental: 10%

Scores are classified as VERY WEAK (0-30), WEAK (31-45), NEUTRAL (46-59), WATCH (60-74), STRONG (75-84), or VERY STRONG (85-100). Missing components are excluded and the remaining weights are renormalized. The defaults are provisional until backtesting supplies calibrated weights; custom weights can be passed as `signal_weights` in the analysis state.

## Iteration 14

Paper trading is simulation-only and must precede live trading. `PaperTradingService` records the AI signal as a paper order, tracks it as OPEN or CLOSED, calculates simulated P&L, and compares the predicted direction with the actual price direction. It does not import or call any broker client and never submits an order.
