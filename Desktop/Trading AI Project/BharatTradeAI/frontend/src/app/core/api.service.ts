import { HttpClient } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { firstValueFrom } from 'rxjs';

export interface HealthResponse {
  status: string;
  service?: string;
  environment?: string;
}

export interface ReadinessResponse {
  status: string;
  checks: Record<string, { status: string; detail?: string }>;
}

export interface BrokerResponse {
  status: string;
  provider?: string;
  message?: string;
  data?: unknown;
}

export interface NewsArticle {
  id: string;
  title: string;
  url: string;
  source: string;
  published_at: string | null;
  retrieved_at?: string | null;
  companies: string[];
  listings: { symbol: string; name: string; exchanges: string[]; match_type: 'provider' | 'headline' }[];
  unmatched_symbols: string[];
  event: string;
  tone: number | null;
  symbols: string[];
  market_ids: string[];
  sentiment: { label: 'POSITIVE' | 'NEGATIVE' | 'NEUTRAL'; score: number };
  sentiment_source?: 'provider' | 'provider_tone' | 'keyword_heuristic';
}

export interface NewsResponse {
  source: string;
  source_url: string;
  status: string;
  query: string;
  count: number;
  sentiment_counts: { positive: number; negative: number; neutral: number };
  headline_bias: 'POSITIVE' | 'NEGATIVE' | 'MIXED' | 'NEUTRAL';
  articles: NewsArticle[];
  notice: string;
  cache_status?: 'fresh' | 'stale';
  cache_age_seconds?: number;
  retrieved_at?: string | null;
  cache_warning?: string;
  archived_article_count?: number;
  archived_symbols?: string[];
}

export interface GroundedNewsCitation {
  article_id: string;
  title: string;
  source: string;
  provider: string;
  published_at: string;
  retrieved_at: string;
  url: string | null;
  excerpt: string;
  event_category: string;
  provider_sentiment: string;
}

export interface GroundedNewsResponse {
  status: 'ready';
  provider: string;
  embedding_model: string;
  article_count: number;
  indexed_article_count: number;
  indexed_chunk_count: number;
  skipped_article_count: number;
  indexed_at: string;
  symbol: string;
  query: string;
  cutoff: string;
  retrieved_at: string;
  evidence_status: 'no_evidence' | 'limited' | 'mixed_provider_sentiment' | 'multiple_sources';
  citations: GroundedNewsCitation[];
  answer: string;
  citation_ids: string[];
}

export type IpoStatus = 'UPCOMING' | 'OPEN' | 'CLOSED' | 'LISTED';
export type IpoBoard = 'MAINBOARD' | 'SME';

export interface IpoSummary {
  slug: string;
  name: string;
  shortName?: string | null;
  board: IpoBoard;
  status: IpoStatus;
  exchange?: string | null;
  symbol?: string | null;
  sector?: string | null;
  registrar?: string | null;
  priceMin: number | null;
  priceMax: number | null;
  lotSize: number | null;
  issueSizeCr: number | null;
  openDate: string | null;
  closeDate: string | null;
  allotmentDate?: string | null;
  listingDate?: string | null;
  lastGmp: number | null;
  lastGmpPct: number | null;
  gmpSources: number | null;
  gmpSpread: number | null;
  gmpDisputed: boolean;
  lastSubX: number | null;
  issuePrice?: number | null;
  listingPrice?: number | null;
  listingGainPct?: number | null;
  updatedAt?: string | null;
}

export interface IpoGmpEntry {
  observedAt: string;
  gmp: number | null;
  gmpPct: number | null;
  source?: string | null;
  unofficial?: boolean;
}

export interface IpoSubscription {
  category: string;
  timesSub: number | null;
}

export interface IpoDetail extends IpoSummary {
  gmp: IpoGmpEntry[];
  subs: IpoSubscription[];
}

export interface IpoListResponse {
  ok: boolean;
  count: number;
  ipos: IpoSummary[];
  disclaimer: string;
  dataFreshness?: IpoDataFreshness;
}

export interface IpoDataFreshness {
  status: 'fresh' | 'stale';
  provider: string;
  lastSuccessfulFetch: string;
  cacheAgeSeconds: number;
  providerUpdatedAt: string | null;
  warning: string | null;
}

export interface IpoDetailResponse {
  ok: boolean;
  ipo: IpoDetail;
  disclaimer: string;
  dataFreshness?: IpoDataFreshness;
}

export interface MarketDeal {
  symbol: string;
  clientName: string;
  dealType: string;
  kind: string;
  quantity: number | null;
  price: number | null;
  valueCr: number | null;
  exchange: string;
  tradeDate: string;
}

export interface MarketDealsResponse {
  ok: boolean;
  deals: MarketDeal[];
  disclaimer: string;
  dataFreshness?: IpoDataFreshness;
}

export interface IpoScoreboardRow {
  slug: string;
  name: string;
  issuePrice: number | null;
  gmpAtClose: number | null;
  impliedPct: number | null;
  listingPrice: number | null;
  actualPct: number | null;
  errorPp: number | null;
  directionHit: boolean | null;
  listedOn: string;
}

export interface IpoScoreboardResponse {
  ok: boolean;
  summary: { n: number; avgAbsErrorPp: number | null; directionHitPct: number | null };
  rows: IpoScoreboardRow[];
  disclaimer: string;
  dataFreshness?: IpoDataFreshness;
}

export interface LiveStockQuote {
  symbol: string;
  name: string;
  exchange: string;
  type: 'INDEX' | 'EQUITY';
  lastPrice: number | null;
  change: number | null;
  changePct: number | null;
  open: number | null;
  high: number | null;
  low: number | null;
  previousClose: number | null;
  volume: number | null;
  timestamp: string | null;
  providerTimestamp?: string | null;
  timestampStatus?: 'fresh' | 'delayed' | 'stale' | 'unavailable' | 'clock_skew' | 'future_timestamp';
  timestampWarning?: string | null;
}

export interface LiveStockQuotesResponse {
  status: 'ok';
  provider: 'Yahoo Finance';
  refreshedAt: string;
  lastSuccessfulRefresh?: string | null;
  cacheStatus?: 'fresh' | 'stale';
  cacheAgeSeconds?: number;
  providerUrl: string;
  quotes: LiveStockQuote[];
  warnings: string[];
}

export interface StockHistoryCandle {
  timestamp: string | null;
  open: number;
  high: number;
  low: number;
  close: number;
  previousClose: number | null;
  volume: number | null;
}

export interface StockHistoryResponse {
  status: 'ok' | 'no_data';
  provider: 'Yahoo Finance';
  symbol: string;
  exchange: string;
  timeframe: '1d';
  range: '5y';
  count: number;
  lastSyncedAt: string | null;
  latestMarketTime: string | null;
  freshness: 'fresh' | 'stale' | 'unknown';
  syncStatus: 'ok' | 'error' | 'never';
  syncError: string | null;
  candles: StockHistoryCandle[];
}

export interface StockIntradayCandle {
  timestamp: string | null;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number | null;
}

export interface StockIntradayResponse {
  status: 'ok';
  provider: 'Yahoo Finance';
  symbol: string;
  exchange: string;
  interval: '1m' | '5m' | '15m';
  range: '1d' | '5d' | '1mo';
  retrievedAt: string;
  providerTimestamp: string | null;
  freshness: 'fresh' | 'stale' | 'future_timestamp';
  warning: string | null;
  candles: StockIntradayCandle[];
}

export interface StockHistorySyncResponse {
  status: 'ok' | 'partial';
  provider: 'Yahoo Finance';
  timeframe: '1d';
  range: '5y';
  syncedAt: string;
  syncedSymbols: string[];
  warnings: string[];
}

export interface AiStockModelMetrics {
  sample_count: number;
  precision: number | null;
  recall: number | null;
  f1: number | null;
  brier_score: number | null;
  expected_calibration_error?: number | null;
  method?: string;
  model?: string;
  step?: number;
}

export interface AiStockNewsFeatureImpactResponse {
  symbol: string;
  exchange: string;
  provider: string;
  evaluation: string;
  status: string;
  message?: string | null;
  sample_count?: number;
  news_available_sample_count?: number;
  required_news_sample_count?: number;
  market_data?: { status: string; message: string | null; sample_count: number; latest_market_time: string | null };
  method?: string;
  step?: number;
  without_news?: AiStockModelMetrics;
  with_news?: AiStockModelMetrics;
  metric_delta_with_news_minus_without_news?: {
    brier_score: number | null;
    expected_calibration_error: number | null;
    precision: number | null;
    recall: number | null;
    f1: number | null;
  };
  out_of_sample_window?: { input_start: string; input_end: string; predicted_sessions: number };
}

export interface AiStockDirectionResponse {
  symbol: string;
  company_name?: string;
  exchange: string;
  sector?: string | null;
  status: string;
  validation_status: string;
  message?: string;
  model_version?: string;
  prediction_for?: string;
  prediction_date?: string;
  last_close?: number;
  last_market_time?: string;
  upward_probability?: number;
  downward_probability?: number;
  direction?: 'UP' | 'DOWN';
  probability_threshold_met?: boolean;
  threshold?: number;
  probability_note?: string;
  model_probabilities?: { xgboost_upward: number; lstm_upward: number };
  ensemble?: { used: boolean; xgboost_weight: number; lstm_weight: number };
  metrics?: {
    xgboost?: AiStockModelMetrics;
    lstm?: AiStockModelMetrics;
    selected?: AiStockModelMetrics;
    walk_forward_xgboost?: AiStockModelMetrics;
  };
  selection?: {
    xgboost_validation_brier: number;
    lstm_validation_brier: number;
    selected_validation_brier: number;
    ensemble_used: boolean;
    xgboost_weight: number;
    lstm_weight: number;
  };
  split_sizes?: { training: number; calibration: number; selection: number; test: number };
  sample_count?: number;
  required_sample_count?: number;
  market_data?: { status: string; message: string | null; sample_count: number; latest_market_time: string | null };
  calibration_status?: string;
  news_context?: string;
  sector_context?: string;
  market_context?: string;
  prediction?: AiStockDirectionResponse;
}

export interface AiStockPredictionsResponse {
  status: 'ready' | 'no_data';
  provider: string;
  updated_at: string;
  predictions: AiStockDirectionResponse[];
  notice: string;
}

export interface AiStockOpportunity {
  company_name: string;
  exchange: string;
  symbol: string;
  current_price: number;
  daily_change_pct: number;
  quote_source: string;
  quote_timestamp: string;
  upward_probability: number;
  predicted_direction: 'UP' | 'DOWN';
  expected_price_range: { low: number; high: number } | null;
  expected_price_range_status: string;
  news_sentiment: {
    label: string;
    score: number | null;
    article_count: number;
    analyzed_article_count: number;
    probabilities: FinBertProbabilities | null;
  };
  technical_analysis: {
    indicators: Record<string, number>;
    summary: Record<string, string>;
    historical_pattern: Record<string, number>;
    history_candle_count: number | null;
    last_market_time: string;
  };
  risk_classification: { level: 'LOW' | 'MEDIUM' | 'HIGH'; summary: string };
  prediction_timestamp: string;
  next_trading_day: string;
  next_trading_day_is_estimate: boolean;
  calendar_note: string;
  model_version: string;
  validation_status: string;
  historical_test_metrics: AiStockModelMetrics | null;
  gemma_explanation: {
    status: 'ready' | 'pending' | 'unavailable';
    provider: string;
    model: string;
    summary: string;
    bullish_signals?: string[];
    bearish_signals?: string[];
    risks?: string[];
    analysis?: string;
  };
}

export interface AiStockOpportunitiesResponse {
  status: 'ready' | 'no_opportunities';
  provider: string;
  threshold: number;
  threshold_comparison: string;
  generated_at: string;
  evaluated_count: number;
  excluded_count: number;
  exclusion_reasons: { symbol: string; reason: string }[];
  model_statuses: AiStockDirectionResponse[];
  opportunities: AiStockOpportunity[];
  empty_message: string | null;
  notice: string;
}

export interface AiStockModelTrainingJobResponse {
  job_id: string;
  status: 'queued' | 'running' | 'ready' | 'partial' | 'failed';
  completed_count: number;
  total_count: number;
  trained_count: number;
  reused_count: number;
  failed_count: number;
  updated_at: string;
  message: string;
  predictions: (AiStockDirectionResponse & {
    training_action: 'trained' | 'existing_model_reused' | 'not_trained' | 'failed';
    prediction?: AiStockDirectionResponse;
  })[];
}

export interface AiPickNewsArticle {
  id: string;
  title: string;
  description: string | null;
  url: string | null;
  source: string;
  provider: string;
  publishedAt: string;
  retrievedAt: string;
  sentiment: 'POSITIVE' | 'NEGATIVE' | 'NEUTRAL' | 'NO_DATA';
  sentimentScore: number | null;
  event: string;
}

export interface AiPickCompanyNews {
  symbol: string;
  companyName: string;
  sentiment: 'POSITIVE' | 'NEGATIVE' | 'NEUTRAL' | 'NO_DATA';
  sentimentScore: number | null;
  articleCount: number;
  eventCounts: Record<string, number>;
  articles: AiPickNewsArticle[];
}

export interface AiPickNewsResponse {
  status: 'ready' | 'no_data';
  cutoff: string;
  lookbackDays: number;
  provider: string;
  lastSyncedAt: string | null;
  syncStatus: 'ok' | 'partial' | 'error' | 'never';
  syncWarning: string | null;
  companies: AiPickCompanyNews[];
  notice: string;
}

export interface AiPickNewsSyncResponse {
  status: 'ok' | 'partial';
  provider: string;
  syncedAt: string;
  syncedSymbols: string[];
  warnings: string[];
}

export interface FinBertProbabilities {
  positive: number;
  negative: number;
  neutral: number;
}

export interface AiPickFinBertArticle {
  id: string;
  title: string;
  description: string | null;
  url: string | null;
  source: string;
  provider: string;
  publishedAt: string;
  retrievedAt: string;
  event: string;
  relevanceScore: number;
  freshnessWeight: number;
  sentiment: 'POSITIVE' | 'NEGATIVE' | 'NEUTRAL' | null;
  probabilities: FinBertProbabilities | null;
  sentimentScore: number | null;
  analyzedAt: string | null;
}

export interface AiPickFinBertCompany {
  symbol: string;
  companyName: string;
  sentiment: 'POSITIVE' | 'NEGATIVE' | 'NEUTRAL' | 'PENDING' | 'NO_DATA';
  sentimentScore: number | null;
  probabilities: FinBertProbabilities | null;
  articleCount: number;
  analyzedArticleCount: number;
  eventCounts: Record<string, number>;
  articles: AiPickFinBertArticle[];
}

export interface AiPickSentimentResponse {
  status: 'ready' | 'pending' | 'no_data';
  model: string;
  cutoff: string;
  lookbackDays: number;
  freshnessHalfLifeHours: number;
  aggregation: string;
  articleCount: number;
  analyzedArticleCount: number;
  companies: AiPickFinBertCompany[];
  notice: string;
}

export type BrokerResource = 'profile' | 'funds' | 'holdings' | 'positions' | 'orders' | 'instruments';

@Injectable({ providedIn: 'root' })
export class ApiService {
  private readonly http = inject(HttpClient);

  getHealth(): Promise<HealthResponse> {
    return firstValueFrom(this.http.get<HealthResponse>('/api/v1/health'));
  }

  getReadiness(): Promise<ReadinessResponse> {
    return firstValueFrom(this.http.get<ReadinessResponse>('/api/v1/ready'));
  }

  getBrokerResource(resource: BrokerResource): Promise<BrokerResponse> {
    return firstValueFrom(this.http.get<BrokerResponse>(`/api/v1/broker/${resource}`));
  }

  getNews(query: string): Promise<NewsResponse> {
    return firstValueFrom(this.http.get<NewsResponse>('/api/v1/news', {
      params: { query, limit: 25 }
    }));
  }

  askGroundedNewsQuestion(request: {
    symbol: string;
    question: string;
    cutoff?: string;
  }): Promise<GroundedNewsResponse> {
    return firstValueFrom(this.http.post<GroundedNewsResponse>(
      '/api/v1/ai-stock-picks/news/grounded-analysis',
      request
    ));
  }

  getIpos(): Promise<IpoListResponse> {
    return firstValueFrom(this.http.get<IpoListResponse>('/api/v1/market/ipos'));
  }

  getIpo(slug: string): Promise<IpoDetailResponse> {
    return firstValueFrom(this.http.get<IpoDetailResponse>(`/api/v1/market/ipos/${encodeURIComponent(slug)}`));
  }

  getMarketDeals(): Promise<MarketDealsResponse> {
    return firstValueFrom(this.http.get<MarketDealsResponse>('/api/v1/market/deals'));
  }

  getIpoScoreboard(): Promise<IpoScoreboardResponse> {
    return firstValueFrom(this.http.get<IpoScoreboardResponse>('/api/v1/market/scoreboard'));
  }

  getLiveStockQuotes(): Promise<LiveStockQuotesResponse> {
    return firstValueFrom(this.http.get<LiveStockQuotesResponse>('/api/v1/stocks/live'));
  }

  getAiStockQuotes(): Promise<LiveStockQuotesResponse> {
    return firstValueFrom(this.http.get<LiveStockQuotesResponse>('/api/v1/ai-stock-picks/quotes'));
  }

  getStockHistory(symbol: string): Promise<StockHistoryResponse> {
    return firstValueFrom(this.http.get<StockHistoryResponse>(
      `/api/v1/stocks/history/${encodeURIComponent(symbol)}`,
      { params: { limit: 120 } }
    ));
  }

  getStockIntraday(symbol: string, interval: '1m' | '5m' | '15m' = '5m'): Promise<StockIntradayResponse> {
    return firstValueFrom(this.http.get<StockIntradayResponse>(
      `/api/v1/stocks/intraday/${encodeURIComponent(symbol)}`,
      { params: { interval } }
    ));
  }

  syncStockHistory(): Promise<StockHistorySyncResponse> {
    return firstValueFrom(this.http.post<StockHistorySyncResponse>('/api/v1/stocks/history/sync', {}));
  }

  getAiStockPrediction(symbol: string): Promise<AiStockDirectionResponse> {
    return firstValueFrom(this.http.get<AiStockDirectionResponse>(
      `/api/v1/ai-stock-picks/predictions/${encodeURIComponent(symbol)}`
    ));
  }

  getAiStockPredictions(): Promise<AiStockPredictionsResponse> {
    return firstValueFrom(this.http.get<AiStockPredictionsResponse>(
      '/api/v1/ai-stock-picks/predictions'
    ));
  }

  getAiStockOpportunities(thresholdPercent: 65 | 70 | 80 = 80): Promise<AiStockOpportunitiesResponse> {
    return firstValueFrom(this.http.get<AiStockOpportunitiesResponse>(
      '/api/v1/ai-stock-picks/opportunities',
      { params: { threshold_percent: thresholdPercent } }
    ));
  }

  trainAiStockModel(symbol: string): Promise<AiStockDirectionResponse> {
    return firstValueFrom(this.http.post<AiStockDirectionResponse>(
      `/api/v1/ai-stock-picks/models/train/${encodeURIComponent(symbol)}`,
      {}
    ));
  }

  getAiStockNewsFeatureImpact(symbol: string, step = 60): Promise<AiStockNewsFeatureImpactResponse> {
    return firstValueFrom(this.http.get<AiStockNewsFeatureImpactResponse>(
      `/api/v1/ai-stock-picks/models/news-impact/${encodeURIComponent(symbol)}`,
      { params: { step } }
    ));
  }

  trainAiStockModels(): Promise<AiStockModelTrainingJobResponse> {
    return firstValueFrom(this.http.post<AiStockModelTrainingJobResponse>(
      '/api/v1/ai-stock-picks/models/train',
      {}
    ));
  }

  getAiStockModelTrainingJob(jobId: string): Promise<AiStockModelTrainingJobResponse> {
    return firstValueFrom(this.http.get<AiStockModelTrainingJobResponse>(
      `/api/v1/ai-stock-picks/models/train/jobs/${encodeURIComponent(jobId)}`
    ));
  }

  getAiPickNews(): Promise<AiPickNewsResponse> {
    return firstValueFrom(this.http.get<AiPickNewsResponse>('/api/v1/ai-stock-picks/news'));
  }

  syncAiPickNews(): Promise<AiPickNewsSyncResponse> {
    return firstValueFrom(this.http.post<AiPickNewsSyncResponse>('/api/v1/ai-stock-picks/news/sync', {}));
  }

  getAiPickSentiment(cutoff?: string): Promise<AiPickSentimentResponse> {
    return firstValueFrom(this.http.get<AiPickSentimentResponse>('/api/v1/ai-stock-picks/sentiment', {
      params: cutoff ? { cutoff } : {}
    }));
  }

  analyzeAiPickSentiment(cutoff?: string): Promise<AiPickSentimentResponse> {
    return firstValueFrom(this.http.post<AiPickSentimentResponse>(
      '/api/v1/ai-stock-picks/sentiment/analyze',
      {},
      { params: cutoff ? { cutoff } : {} }
    ));
  }
}