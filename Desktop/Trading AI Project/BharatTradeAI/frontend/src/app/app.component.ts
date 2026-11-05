import { Component, OnDestroy, OnInit, inject, signal } from '@angular/core';
import { CurrencyPipe, DatePipe, DecimalPipe, KeyValuePipe } from '@angular/common';
import { HttpErrorResponse } from '@angular/common/http';
import {
  ApiService,
  AiStockDirectionResponse,
  AiStockModelTrainingJobResponse,
  AiStockOpportunitiesResponse,
  AiStockPredictionsResponse,
  AiPickFinBertCompany,
  AiPickSentimentResponse,
  BrokerResource,
  BrokerResponse,
  HealthResponse,
  IpoBoard,
  IpoDetail,
  IpoDetailResponse,
  IpoListResponse,
  IpoScoreboardResponse,
  IpoStatus,
  LiveStockQuote,
  LiveStockQuotesResponse,
  MarketDealsResponse,
  NewsResponse,
  ReadinessResponse,
  StockHistoryCandle,
  StockHistoryResponse
} from './core/api.service';

type Section = 'overview' | 'news' | 'stock' | 'ai-picks' | 'ipos' | 'account' | 'system';
type BrokerRow = Record<string, unknown>;
type IpoStatusFilter = 'ALL' | IpoStatus;
type IpoBoardFilter = 'ALL' | IpoBoard;
type AiPickExchange = 'ALL' | 'NSE' | 'BSE';

interface AiStockPick {
  id: string;
  companyName: string;
  symbol: string;
  exchange: Exclude<AiPickExchange, 'ALL'>;
  sector: string | null;
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CurrencyPipe, DatePipe, DecimalPipe, KeyValuePipe],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss'
})
export class AppComponent implements OnInit, OnDestroy {
  private readonly api = inject(ApiService);

  readonly section = signal<Section>('overview');
  readonly health = signal<HealthResponse | null>(null);
  readonly readiness = signal<ReadinessResponse | null>(null);
  readonly statusLoading = signal(true);
  readonly statusError = signal('');
  readonly lastUpdated = signal<Date | null>(null);
  readonly activeResource = signal<BrokerResource>('funds');
  readonly brokerResponse = signal<BrokerResponse | null>(null);
  readonly brokerLoading = signal(false);
  readonly brokerError = signal('');
  readonly newsQuery = signal('India business stocks');
  readonly newsResponse = signal<NewsResponse | null>(null);
  readonly newsLoading = signal(false);
  readonly newsError = signal('');
  readonly ipoLoading = signal(false);
  readonly ipoError = signal('');
  readonly ipoList = signal<IpoListResponse | null>(null);
  readonly ipoDeals = signal<MarketDealsResponse | null>(null);
  readonly ipoScoreboard = signal<IpoScoreboardResponse | null>(null);
  readonly ipoDisclaimer = signal('');
  readonly selectedIpoSlug = signal('');
  readonly ipoDetailResponse = signal<IpoDetailResponse | null>(null);
  readonly ipoDetailLoading = signal(false);
  readonly ipoDetailError = signal('');
  readonly ipoDealsError = signal('');
  readonly ipoScoreboardError = signal('');
  readonly ipoStatusFilter = signal<IpoStatusFilter>('ALL');
  readonly ipoBoardFilter = signal<IpoBoardFilter>('ALL');
  readonly ipoSearchQuery = signal('');
  private ipoLoaded = false;
  readonly stockQuotes = signal<LiveStockQuotesResponse | null>(null);
  readonly stockLoading = signal(false);
  readonly stockError = signal('');
  private stockRefreshTimer: ReturnType<typeof setInterval> | null = null;
  readonly aiPickExchange = signal<AiPickExchange>('ALL');
  readonly aiPickSector = signal('ALL');
  readonly aiPickMinProbability = signal(0);
  readonly selectedAiPickId = signal('');
  readonly aiStockHistory = signal<StockHistoryResponse | null>(null);
  readonly aiHistoryLoading = signal(false);
  readonly aiHistorySyncing = signal(false);
  readonly aiHistoryError = signal('');
  readonly aiHistoryNotice = signal('');
  private aiHistoryRequestId = 0;
  readonly aiStockPredictions = signal<Record<string, AiStockDirectionResponse>>({});
  readonly aiPredictionLoading = signal(false);
  readonly aiPredictionList = signal<AiStockPredictionsResponse | null>(null);
  readonly aiStockOpportunities = signal<AiStockOpportunitiesResponse | null>(null);
  readonly aiOpportunitiesLoading = signal(false);
  readonly aiOpportunitiesError = signal('');
  readonly aiModelTraining = signal(false);
  readonly aiModelTrainingNotice = signal('');
  readonly aiPredictionError = signal('');
  private aiPredictionRequestId = 0;
  readonly aiPickSentiment = signal<AiPickSentimentResponse | null>(null);
  readonly aiPickSentimentLoading = signal(false);
  readonly aiPickSentimentAnalyzing = signal(false);
  readonly aiPickSentimentError = signal('');
  readonly aiPickNewsSyncing = signal(false);
  readonly aiPickNewsError = signal('');
  readonly aiPickNewsNotice = signal('');

  readonly resources: { id: BrokerResource; label: string }[] = [
    { id: 'funds', label: 'Funds' },
    { id: 'holdings', label: 'Holdings' },
    { id: 'positions', label: 'Positions' },
    { id: 'orders', label: 'Orders' },
    { id: 'profile', label: 'Profile' },
    { id: 'instruments', label: 'Instruments' }
  ];

  ngOnInit(): void {
    void this.refreshStatus();
  }

  ngOnDestroy(): void {
    this.stopStockRefresh();
  }

  async refreshStatus(): Promise<void> {
    this.statusLoading.set(true);
    this.statusError.set('');

    const [healthResult, readinessResult] = await Promise.allSettled([
      this.api.getHealth(),
      this.api.getReadiness()
    ]);

    if (healthResult.status === 'fulfilled') {
      this.health.set(healthResult.value);
    } else {
      this.statusError.set('Could not reach the API. Confirm that the FastAPI server is running.');
    }

    if (readinessResult.status === 'fulfilled') {
      this.readiness.set(readinessResult.value);
    } else if (!this.statusError()) {
      this.statusError.set('Readiness check failed. The API may still be starting.');
    }

    this.lastUpdated.set(new Date());
    this.statusLoading.set(false);
  }

  setSection(section: Section): void {
    this.section.set(section);
    if (section === 'stock' || section === 'ai-picks') {
      if (!this.stockRefreshTimer) {
        void this.loadStockQuotes();
        this.stockRefreshTimer = setInterval(() => void this.loadStockQuotes(), 60_000);
      }
    } else {
      this.stopStockRefresh();
    }
    if (section === 'ai-picks') {
      void this.loadAiStockOpportunities();
      this.refreshSelectedAiPickHistory();
      if (!this.aiPickSentiment() && !this.aiPickSentimentLoading()) {
        void this.loadAiPickSentiment();
      }
    }
    if (section === 'account' && !this.brokerResponse() && !this.brokerLoading()) {
      void this.loadResource(this.activeResource());
    }
    if (section === 'news' && !this.newsResponse() && !this.newsLoading()) {
      void this.searchNews();
    }
    if (section === 'ipos' && !this.ipoLoaded && !this.ipoLoading()) {
      void this.loadIpoDashboard();
    }
  }

  setAiPickExchange(value: string): void {
    if (value === 'ALL' || value === 'NSE' || value === 'BSE') {
      this.aiPickExchange.set(value);
      this.refreshSelectedAiPickHistory();
    }
  }

  setAiPickSector(value: string): void {
    if (value === 'ALL' || this.aiPickSectors.includes(value)) {
      this.aiPickSector.set(value);
      this.refreshSelectedAiPickHistory();
    }
  }

  setAiPickMinProbability(value: string): void {
    const probability = Number(value);
    if ([0, 60, 70, 80].includes(probability)) {
      this.aiPickMinProbability.set(probability);
      this.refreshSelectedAiPickHistory();
    }
  }

  resetAiPickFilters(): void {
    this.aiPickExchange.set('ALL');
    this.aiPickSector.set('ALL');
    this.aiPickMinProbability.set(0);
    this.refreshSelectedAiPickHistory();
  }

  get filteredAiStockPicks(): AiStockPick[] {
    return this.aiStockPicks.filter(pick =>
      (this.aiPickExchange() === 'ALL' || pick.exchange === this.aiPickExchange()) &&
      (this.aiPickSector() === 'ALL' || pick.sector === this.aiPickSector()) &&
      (this.aiPickMinProbability() === 0 ||
        (this.aiPickPrediction(pick.symbol)?.status === 'ready' &&
          (this.aiPickPrediction(pick.symbol)?.upward_probability ?? 0) * 100 >= this.aiPickMinProbability()))
    );
  }

  get aiStockPicks(): AiStockPick[] {
    return this.stockRows.map(quote => ({
      id: quote.symbol,
      companyName: quote.name,
      symbol: quote.symbol,
      exchange: quote.exchange === 'BSE' ? 'BSE' : 'NSE',
      sector: this.aiPickPrediction(quote.symbol)?.sector ?? null
    }));
  }

  get aiPickSectors(): string[] {
    return [...new Set(this.aiStockPicks.map(pick => pick.sector).filter((sector): sector is string => !!sector))].sort();
  }

  get selectedAiStockPick(): AiStockPick | null {
    const picks = this.filteredAiStockPicks;
    return picks.find(pick => pick.id === this.selectedAiPickId()) ?? picks[0] ?? null;
  }

  selectAiStockPick(id: string): void {
    if (this.filteredAiStockPicks.some(pick => pick.id === id)) {
      this.selectedAiPickId.set(id);
      void this.loadAiStockHistory(id);
    }
  }

  aiPickPrediction(symbol: string): AiStockDirectionResponse | null {
    return this.aiStockPredictions()[symbol] ?? null;
  }

  async loadAiStockPrediction(symbol: string): Promise<void> {
    const requestId = ++this.aiPredictionRequestId;
    this.aiPredictionLoading.set(true);
    this.aiPredictionError.set('');
    try {
      const result = await this.api.getAiStockPrediction(symbol);
      if (requestId === this.aiPredictionRequestId) {
        this.aiStockPredictions.update(current => ({ ...current, [symbol]: result }));
      }
    } catch (error: unknown) {
      if (requestId === this.aiPredictionRequestId) {
        this.aiPredictionError.set(
          error instanceof HttpErrorResponse && typeof error.error?.detail === 'string'
            ? error.error.detail
            : 'The validated stock prediction could not be loaded. Check the API connection.'
        );
      }
    } finally {
      if (requestId === this.aiPredictionRequestId) this.aiPredictionLoading.set(false);
    }
  }

  async loadAiStockPredictions(): Promise<void> {
    if (this.aiPredictionLoading()) return;
    this.aiPredictionLoading.set(true);
    this.aiPredictionError.set('');
    try {
      const response = await this.api.getAiStockPredictions();
      this.aiPredictionList.set(response);
      this.aiStockPredictions.set(
        Object.fromEntries(response.predictions.map(prediction => [prediction.symbol, prediction]))
      );
    } catch (error: unknown) {
      this.aiPredictionError.set(
        error instanceof HttpErrorResponse && typeof error.error?.detail === 'string'
          ? error.error.detail
          : 'Validated stock-model statuses could not be loaded. Check the API connection.'
      );
    } finally {
      this.aiPredictionLoading.set(false);
    }
  }

  async refreshAiStockAnalysis(): Promise<void> {
    await Promise.all([
      this.loadStockQuotes(),
      this.loadAiStockOpportunities(),
      this.loadAiPickSentiment()
    ]);
    this.refreshSelectedAiPickHistory();
  }

  async loadAiStockOpportunities(): Promise<void> {
    if (this.aiOpportunitiesLoading()) return;
    this.aiOpportunitiesLoading.set(true);
    this.aiOpportunitiesError.set('');
    try {
      const response = await this.api.getAiStockOpportunities();
      this.aiStockOpportunities.set(response);
      this.aiPredictionList.set({
        status: response.model_statuses.length ? 'ready' : 'no_data',
        provider: response.provider,
        updated_at: response.generated_at,
        predictions: response.model_statuses,
        notice: response.notice
      });
      this.aiStockPredictions.set(
        Object.fromEntries(response.model_statuses.map(prediction => [prediction.symbol, prediction]))
      );
    } catch (error: unknown) {
      this.aiStockOpportunities.set(null);
      this.aiOpportunitiesError.set(
        error instanceof HttpErrorResponse && typeof error.error?.detail === 'string'
          ? error.error.detail
          : 'Validated stock opportunities could not be calculated. Check the market-data and prediction services.'
      );
    } finally {
      this.aiOpportunitiesLoading.set(false);
    }
  }

  async trainAiStockModel(): Promise<void> {
    const pick = this.selectedAiStockPick;
    if (!pick || this.aiModelTraining()) return;
    this.aiPredictionRequestId += 1;
    this.aiPredictionLoading.set(false);
    this.aiModelTraining.set(true);
    this.aiPredictionError.set('');
    this.aiModelTrainingNotice.set('');
    try {
      const result = await this.api.trainAiStockModel(pick.symbol);
      const prediction = result.prediction ?? result;
      this.aiStockPredictions.update(current => ({ ...current, [pick.symbol]: prediction }));
      await this.loadAiStockOpportunities();
    } catch (error: unknown) {
      this.aiPredictionError.set(
        error instanceof HttpErrorResponse && typeof error.error?.detail === 'string'
          ? error.error.detail
          : 'Model training failed. Confirm the model dependencies and stored history, then retry.'
      );
    } finally {
      this.aiModelTraining.set(false);
    }
  }

  async trainAiStockModels(): Promise<void> {
    if (this.aiModelTraining()) return;
    this.aiPredictionRequestId += 1;
    this.aiPredictionLoading.set(false);
    this.aiModelTraining.set(true);
    this.aiPredictionError.set('');
    this.aiModelTrainingNotice.set('');
    try {
      let job: AiStockModelTrainingJobResponse = await this.api.trainAiStockModels();
      while (job.status === 'queued' || job.status === 'running') {
        this.updateAiStockTrainingResults(job);
        this.aiModelTrainingNotice.set(
          `Training and validating stock models: ${job.completed_count} / ${job.total_count} complete.`
        );
        await new Promise(resolve => setTimeout(resolve, 2000));
        job = await this.api.getAiStockModelTrainingJob(job.job_id);
      }
      this.updateAiStockTrainingResults(job);
      if (job.status === 'failed') {
        this.aiPredictionError.set(job.message || 'Watchlist model training failed.');
      } else {
        const blocked = job.predictions.filter(item => item.status !== 'ready');
        const summary = `${job.trained_count} model(s) trained; ${job.reused_count} validated model(s) reused.`;
        this.aiModelTrainingNotice.set(blocked.length
          ? `${summary} No probability is shown for: ${blocked.map(item => `${item.symbol} (${item.status})`).join(', ')}.`
          : summary);
        await this.loadAiStockOpportunities();
      }
    } catch (error: unknown) {
      this.aiPredictionError.set(
        error instanceof HttpErrorResponse && typeof error.error?.detail === 'string'
          ? error.error.detail
          : 'Watchlist training failed. Confirm the model dependencies and stored history, then retry.'
      );
    } finally {
      this.aiModelTraining.set(false);
    }
  }

  private updateAiStockTrainingResults(job: AiStockModelTrainingJobResponse): void {
    if (job.predictions.length) {
      const updates = Object.fromEntries(
          job.predictions.map(item => [item.symbol, item.prediction ?? item])
      );
      this.aiStockPredictions.update(current => ({ ...current, ...updates }));
    }
  }

  aiPickCompanyNews(symbol: string): AiPickFinBertCompany | null {
    return this.aiPickSentiment()?.companies.find(company => company.symbol === symbol) ?? null;
  }

  async loadAiPickSentiment(): Promise<void> {
    if (this.aiPickSentimentLoading()) return;
    this.aiPickSentimentLoading.set(true);
    this.aiPickSentimentError.set('');
    try {
      this.aiPickSentiment.set(await this.api.getAiPickSentiment());
    } catch (error: unknown) {
      if (error instanceof HttpErrorResponse && typeof error.error?.detail === 'string') {
        this.aiPickSentimentError.set(error.error.detail);
      } else {
        this.aiPickSentimentError.set('Stored FinBERT sentiment could not be loaded. Check the API connection.');
      }
    } finally {
      this.aiPickSentimentLoading.set(false);
    }
  }

  async analyzeAiPickSentiment(): Promise<void> {
    if (this.aiPickSentimentAnalyzing()) return;
    this.aiPickSentimentAnalyzing.set(true);
    this.aiPickSentimentError.set('');
    try {
      this.aiPickSentiment.set(await this.api.analyzeAiPickSentiment());
    } catch (error: unknown) {
      if (error instanceof HttpErrorResponse && typeof error.error?.detail === 'string') {
        this.aiPickSentimentError.set(error.error.detail);
      } else {
        this.aiPickSentimentError.set('FinBERT analysis could not be completed. Check model dependencies and try again.');
      }
    } finally {
      this.aiPickSentimentAnalyzing.set(false);
    }
    await this.loadAiStockOpportunities();
  }

  async syncAiPickNews(): Promise<void> {
    if (this.aiPickNewsSyncing()) return;
    this.aiPickNewsSyncing.set(true);
    this.aiPickNewsError.set('');
    this.aiPickNewsNotice.set('');
    try {
      const result = await this.api.syncAiPickNews();
      this.aiPickNewsNotice.set(result.status === 'ok'
        ? `Company news refreshed for ${result.syncedSymbols.length} symbols from ${result.provider}.`
        : `Company news refreshed with warnings: ${result.warnings.join(' · ')}`);
    } catch (error: unknown) {
      if (error instanceof HttpErrorResponse && typeof error.error?.detail === 'string') {
        this.aiPickNewsError.set(error.error.detail);
      } else {
        this.aiPickNewsError.set('Configured news sources could not be reached. Check the API connection and retry later.');
      }
    } finally {
      this.aiPickNewsSyncing.set(false);
    }
    await this.analyzeAiPickSentiment();
  }

  aiPickLiveQuote(pick: AiStockPick): LiveStockQuote | null {
    return this.stockQuotes()?.quotes.find(quote => quote.symbol === pick.symbol) ?? null;
  }

  get aiPickHistoryRows(): StockHistoryCandle[] {
    return [...(this.aiStockHistory()?.candles ?? [])].reverse().slice(0, 30);
  }

  async syncAiStockHistory(): Promise<void> {
    if (this.aiHistorySyncing()) return;
    this.aiHistorySyncing.set(true);
    this.aiHistoryError.set('');
    this.aiHistoryNotice.set('');
    try {
      const result = await this.api.syncStockHistory();
      this.aiHistoryNotice.set(result.status === 'ok'
        ? `Daily history updated for ${result.syncedSymbols.length} symbols.`
        : `History updated with warnings: ${result.warnings.join(' · ')}`);
      const selectedSymbol = this.selectedAiStockPick?.symbol;
      const refreshes: Promise<unknown>[] = [
        this.loadStockQuotes(),
        this.loadAiStockOpportunities()
      ];
      if (selectedSymbol) refreshes.push(this.loadAiStockHistory(selectedSymbol));
      await Promise.all(refreshes);
    } catch (error: unknown) {
      if (error instanceof HttpErrorResponse && typeof error.error?.detail === 'string') {
        this.aiHistoryError.set(error.error.detail);
      } else {
        this.aiHistoryError.set('Yahoo Finance history sync could not be completed. Check the API connection and retry later.');
      }
    } finally {
      this.aiHistorySyncing.set(false);
    }
  }

  private refreshSelectedAiPickHistory(): void {
    const selected = this.selectedAiStockPick;
    if (selected && this.selectedAiPickId() !== selected.id) {
      this.selectedAiPickId.set(selected.id);
    }
    if (selected) {
      void this.loadAiStockHistory(selected.id);
    } else {
      this.aiStockHistory.set(null);
    }
  }

  private async loadAiStockHistory(id: string): Promise<void> {
    const pick = this.aiStockPicks.find(item => item.id === id);
    if (!pick) return;
    const requestId = ++this.aiHistoryRequestId;
    this.aiHistoryLoading.set(true);
    this.aiHistoryError.set('');
    try {
      const history = await this.api.getStockHistory(pick.symbol);
      if (this.selectedAiPickId() === id && requestId === this.aiHistoryRequestId) {
        this.aiStockHistory.set(history);
      }
    } catch (error: unknown) {
      if (this.selectedAiPickId() === id && requestId === this.aiHistoryRequestId) {
        if (error instanceof HttpErrorResponse && typeof error.error?.detail === 'string') {
          this.aiHistoryError.set(error.error.detail);
        } else {
          this.aiHistoryError.set('Stored daily market history could not be loaded. Check the API connection.');
        }
        this.aiStockHistory.set(null);
      }
    } finally {
      if (requestId === this.aiHistoryRequestId) {
        this.aiHistoryLoading.set(false);
      }
    }
  }

  async loadStockQuotes(): Promise<void> {
    if (this.stockLoading()) return;

    this.stockLoading.set(true);
    this.stockError.set('');
    try {
      this.stockQuotes.set(await this.api.getLiveStockQuotes());
      if (this.section() === 'ai-picks' && !this.selectedAiPickId() && this.aiStockPicks.length) {
        this.selectedAiPickId.set(this.aiStockPicks[0].id);
        this.refreshSelectedAiPickHistory();
      }
    } catch (error: unknown) {
      if (error instanceof HttpErrorResponse && typeof error.error?.detail === 'string') {
        this.stockError.set(error.error.detail);
      } else {
        this.stockError.set('Public Yahoo Finance quotes could not be loaded. Check the API connection and retry.');
      }
    } finally {
      this.stockLoading.set(false);
    }
  }

  get stockIndices(): LiveStockQuote[] {
    return (this.stockQuotes()?.quotes ?? []).filter(quote => quote.type === 'INDEX');
  }

  get stockRows(): LiveStockQuote[] {
    return (this.stockQuotes()?.quotes ?? []).filter(quote => quote.type === 'EQUITY');
  }

  private stopStockRefresh(): void {
    if (this.stockRefreshTimer) {
      clearInterval(this.stockRefreshTimer);
      this.stockRefreshTimer = null;
    }
  }

  async loadIpoDashboard(): Promise<void> {
    this.ipoLoading.set(true);
    this.ipoError.set('');
    this.ipoDealsError.set('');
    this.ipoScoreboardError.set('');

    const [iposResult, dealsResult, scoreboardResult] = await Promise.allSettled([
      this.api.getIpos(),
      this.api.getMarketDeals(),
      this.api.getIpoScoreboard()
    ]);

    if (iposResult.status === 'fulfilled') {
      this.ipoList.set(iposResult.value);
      this.ipoDisclaimer.set(iposResult.value.disclaimer);
      const selectedSlug = this.selectedIpoSlug();
      if (!selectedSlug || !iposResult.value.ipos.some(ipo => ipo.slug === selectedSlug)) {
        this.selectedIpoSlug.set(iposResult.value.ipos[0]?.slug ?? '');
      }
    } else {
      this.ipoError.set('IPO listings could not be loaded. Check the API connection and try again.');
    }

    if (dealsResult.status === 'fulfilled') {
      this.ipoDeals.set(dealsResult.value);
      if (!this.ipoDisclaimer()) this.ipoDisclaimer.set(dealsResult.value.disclaimer);
    } else {
      this.ipoDealsError.set('Bulk and block deals could not be loaded.');
    }

    if (scoreboardResult.status === 'fulfilled') {
      this.ipoScoreboard.set(scoreboardResult.value);
      if (!this.ipoDisclaimer()) this.ipoDisclaimer.set(scoreboardResult.value.disclaimer);
    } else {
      this.ipoScoreboardError.set('The IPO listing-performance scoreboard could not be loaded.');
    }

    this.ipoLoaded = iposResult.status === 'fulfilled';
    this.ipoLoading.set(false);

    const selectedSlug = this.selectedIpoSlug();
    if (selectedSlug) {
      await this.loadIpoDetail(selectedSlug);
    }
  }

  async selectIpo(slug: string): Promise<void> {
    this.selectedIpoSlug.set(slug);
    await this.loadIpoDetail(slug);
  }

  async loadIpoDetail(slug: string): Promise<void> {
    if (!slug) {
      this.ipoDetailResponse.set(null);
      return;
    }

    this.ipoDetailLoading.set(true);
    this.ipoDetailError.set('');
    this.ipoDetailResponse.set(null);
    try {
      const response = await this.api.getIpo(slug);
      if (this.selectedIpoSlug() !== slug) return;
      this.ipoDetailResponse.set(response);
      if (!this.ipoDisclaimer()) this.ipoDisclaimer.set(response.disclaimer);
    } catch {
      if (this.selectedIpoSlug() === slug) {
        this.ipoDetailError.set('IPO details could not be loaded. Try again in a minute.');
      }
    } finally {
      if (this.selectedIpoSlug() === slug) this.ipoDetailLoading.set(false);
    }
  }

  setIpoStatusFilter(value: string): void {
    switch (value) {
      case 'ALL':
      case 'UPCOMING':
      case 'OPEN':
      case 'CLOSED':
      case 'LISTED':
        this.ipoStatusFilter.set(value);
        break;
    }
  }

  setIpoBoardFilter(value: string): void {
    if (value === 'ALL' || value === 'MAINBOARD' || value === 'SME') {
      this.ipoBoardFilter.set(value);
    }
  }

  updateIpoSearchQuery(value: string): void {
    this.ipoSearchQuery.set(value);
  }

  get filteredIpos() {
    return (this.ipoList()?.ipos ?? []).filter(ipo =>
      (this.ipoStatusFilter() === 'ALL' || ipo.status === this.ipoStatusFilter()) &&
      (this.ipoBoardFilter() === 'ALL' || ipo.board === this.ipoBoardFilter()) &&
      [ipo.name, ipo.slug, ipo.symbol ?? '', ipo.sector ?? '']
        .some(value => value.toLowerCase().includes(this.ipoSearchQuery().trim().toLowerCase()))
    );
  }

  get selectedIpo(): IpoDetail | null {
    const detail = this.ipoDetailResponse()?.ipo;
    return detail?.slug === this.selectedIpoSlug() ? detail : null;
  }

  get ipoStatusCounts(): Record<IpoStatus, number> {
    const ipos = this.ipoList()?.ipos ?? [];
    return {
      UPCOMING: ipos.filter(ipo => ipo.status === 'UPCOMING').length,
      OPEN: ipos.filter(ipo => ipo.status === 'OPEN').length,
      CLOSED: ipos.filter(ipo => ipo.status === 'CLOSED').length,
      LISTED: ipos.filter(ipo => ipo.status === 'LISTED').length
    };
  }

  formatMarketDate(value: string | null | undefined, includeTime = false): string {
    if (!value) return '—';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return '—';
    return new Intl.DateTimeFormat('en-IN', {
      timeZone: 'Asia/Kolkata',
      day: '2-digit',
      month: 'short',
      year: 'numeric',
      ...(includeTime ? { hour: '2-digit', minute: '2-digit' } : {})
    }).format(date);
  }

  async searchNews(): Promise<void> {
    this.newsLoading.set(true);
    this.newsError.set('');

    try {
      this.newsResponse.set(await this.api.getNews(this.newsQuery().trim()));
    } catch (error: unknown) {
      this.newsResponse.set(null);
      if (error instanceof HttpErrorResponse && error.status === 503) {
        this.newsError.set(typeof error.error?.detail === 'string' ? error.error.detail : 'The free news provider is rate-limiting searches. Wait before trying again.');
      } else if (error instanceof HttpErrorResponse && error.status === 502) {
        this.newsError.set(typeof error.error?.detail === 'string' ? error.error.detail : 'The current news provider did not respond. Try again later.');
      } else {
        this.newsError.set('The free news service could not be reached. Check your internet connection and try again.');
      }
    } finally {
      this.newsLoading.set(false);
    }
  }

  updateNewsQuery(value: string): void {
    this.newsQuery.set(value);
  }

  async loadResource(resource: BrokerResource): Promise<void> {
    this.activeResource.set(resource);
    this.brokerLoading.set(true);
    this.brokerError.set('');
    this.brokerResponse.set(null);

    try {
      this.brokerResponse.set(await this.api.getBrokerResource(resource));
    } catch {
      this.brokerError.set('Unable to load broker data. Check the API and broker configuration.');
    } finally {
      this.brokerLoading.set(false);
    }
  }

  get readinessChecks(): { name: string; status: string; detail: string }[] {
    return Object.entries(this.readiness()?.checks ?? {}).map(([name, check]) => ({
      name: name.replaceAll('_', ' '),
      status: check.status,
      detail: check.detail ?? ''
    }));
  }

  get brokerRows(): BrokerRow[] {
    let value: unknown = this.brokerResponse()?.data;

    while (this.isRecord(value) && Object.hasOwn(value, 'data')) {
      value = value['data'];
    }

    if (Array.isArray(value)) {
      return value.filter((item): item is BrokerRow => this.isRecord(item));
    }

    if (this.isRecord(value)) {
      return Object.entries(value).map(([field, fieldValue]) => ({ field, value: fieldValue }));
    }

    return value === undefined || value === null ? [] : [{ value }];
  }

  get brokerColumns(): string[] {
    if (this.brokerRows.length === 0) {
      return [];
    }

    return Object.hasOwn(this.brokerRows[0], 'field')
      ? ['field', 'value']
      : Object.keys(this.brokerRows[0]).slice(0, 6);
  }

  formatValue(value: unknown): string {
    if (value === null || value === undefined || value === '') {
      return '—';
    }

    return typeof value === 'object' ? JSON.stringify(value) : String(value);
  }

  private isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
  }
}