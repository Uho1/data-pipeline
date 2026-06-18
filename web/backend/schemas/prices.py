from pydantic import BaseModel, Field


class PriceBar(BaseModel):
    time: str        # "YYYY-MM-DD"
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None


class PriceResponse(BaseModel):
    ticker: str
    market: str
    bars: list[PriceBar]
    is_mock: bool = False  # True when real data unavailable


class ValuationPoint(BaseModel):
    time: str  # "YYYY-MM-DD"
    eps: float | None = None
    bps: float | None = None
    per: float | None = None
    pbr: float | None = None


class ValuationResponse(BaseModel):
    ticker: str
    market: str
    points: list[ValuationPoint]
    default_per: float | None = None
    default_pbr: float | None = None
    is_mock: bool = False
    warnings: list[str] = Field(default_factory=list)


class RealtimeQuote(BaseModel):
    """Real-time quote from iTick API."""
    ticker: str
    market: str
    price: float | None = None
    open: float | None = None
    prev_close: float | None = None
    high: float | None = None
    low: float | None = None
    change: float | None = None
    change_pct: float | None = None
    volume: int | None = None
    timestamp: int | None = None  # epoch ms


class TickerItem(BaseModel):
    ticker: str
    company_name: str = ""
    name_kr: str = ""


class TickerListResponse(BaseModel):
    market: str
    tickers: list[str]
    items: list[TickerItem] = Field(default_factory=list)
    is_mock: bool = False
