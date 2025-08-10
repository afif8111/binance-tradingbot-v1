# ================= REAL TRADING BOT - BINANCE LIVE =================


import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import sqlite3
from contextlib import contextmanager
from functools import wraps
import signal
import sys
import warnings
warnings.filterwarnings('ignore')

# Additional imports for Binance API
import hmac
import hashlib
from urllib.parse import urlencode
from scipy.signal import argrelextrema
from decimal import Decimal, ROUND_DOWN
from typing import Tuple

# === Utility Functions ===
def round_to_tick(value: float, tick_size: float) -> float:
    return float(Decimal(value).quantize(Decimal(str(tick_size)), rounding=ROUND_DOWN))


# ================== CONFIGURATION ==================
class Config:
    BOT_TOKEN = "your bot token"
    TELEGRAM_RECIPIENTS = [
        {"chat_id": "chat_id", "name": "Main User"},
    ]
    BINANCE_API_KEY = "your api key"
    BINANCE_SECRET_KEY = "your_api_secret"

    CSV_TRADES = "real_trades.csv"
    CSV_SIGNALS = "real_signals.csv"
    JSON_POSITIONS = "real_positions.json"
    JSON_EXCHANGE_RULES = "exchange_rules.json"
    LOG_FILE = "real_trading_bot.log"
    DATABASE_FILE = "real_trading_data.db"

    CHECK_INTERVAL_SECONDS = 60  # Faster for real trading
    POSITION_SYNC_INTERVAL = 30
    EXCHANGE_RULES_UPDATE = 3600
    MAX_WORKERS = 8
    API_TIMEOUT = 30
    RETRY_ATTEMPTS = 3
    RETRY_DELAY = 2

    BINANCE_BASE_URL = "https://api.binance.com/api/v3"
    BINANCE_KLINES_URL = f"{BINANCE_BASE_URL}/klines"
    BINANCE_24HR_URL = f"{BINANCE_BASE_URL}/ticker/24hr"
    BINANCE_EXCHANGE_INFO_URL = f"{BINANCE_BASE_URL}/exchangeInfo"
    BINANCE_ACCOUNT_URL = f"{BINANCE_BASE_URL}/account"
    BINANCE_ORDER_URL = f"{BINANCE_BASE_URL}/order"
    BINANCE_PRICE_URL = f"{BINANCE_BASE_URL}/ticker/price"
    BINANCE_OPEN_ORDERS_URL = f"{BINANCE_BASE_URL}/openOrders"

    # REAL TRADING PARAMETERS  
    POSITION_SIZE_USDT = 70  # 70 usdt per position
    MAX_CONCURRENT_POSITIONS = 5  # Conservative
    MAX_POSITIONS_PER_SYMBOL = 1
    MIN_POSITION_SIZE = 11  # Minimum Binance requirement
    MIN_USDT_BALANCE = 5  # Reduced from 30 to 5 (more aggressive)

    STOP_LOSS_PCT = 0.02  # 2% stop loss
    TAKE_PROFIT_PCT = 0.03  # 3% take profit
    TRAILING_STOP_PCT = 0.015
    MAX_PORTFOLIO_RISK = 0.05  # 5% max risk
    MAX_DAILY_LOSS = 5  # $5 max daily loss

    # Strategy parameters
    VOLUME_LOOKBACK = 12
    HIGH_VOLUME_THRESHOLD = 1.3
    ACCUMULATION_MIN_COUNT = 3
    DISTRIBUTION_MAX_COUNT = 0
    DISTRIBUTION_WARNING_THRESHOLD = 1
    VOLUME_MOMENTUM_LOOKBACK = 5

    SWING_LOOKBACK = 20
    MIN_ZONE_TOUCHES = 2
    ZONE_PROXIMITY_PCT = 1.5
    ZONE_STRENGTH_MULTIPLIER = 1.2

    MIN_ZONE_STRENGTH = 3
    MIN_CONFIDENCE_SCORE = 0.65  # Higher threshold for real trading
    MAX_DISTANCE_FROM_ZONE = 0.015
    VOLUME_CONFIDENCE_WEIGHT = 0.4
    ZONE_CONFIDENCE_WEIGHT = 0.6

    MIN_DATA_POINTS = 50
    HISTORICAL_DAYS = 15

    FOCUS_PAIRS = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT',
        'SOLUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT', 'DOTUSDT'
    ] #add more coin pairs to focus on if needed

# ================== SETUP LOGGING ==================
def setup_logging():
    os.makedirs("logs", exist_ok=True)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(f"logs/{Config.LOG_FILE}", encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    error_handler = logging.FileHandler(f"logs/errors.log", encoding='utf-8')
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.ERROR)

    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[file_handler, console_handler, error_handler]
    )

    return logging.getLogger(__name__)

logger = setup_logging()

# ================== ENUM DAN DATA MODELS ==================
class ActionType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class OrderStatus(Enum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    PENDING_CANCEL = "PENDING_CANCEL"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

class OrderType(Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    STOP_LOSS_LIMIT = "STOP_LOSS_LIMIT"

class ZoneType(Enum):
    DEMAND = "DEMAND"
    SUPPLY = "SUPPLY"

class VolumePattern(Enum):
    ACCUMULATION = "ACCUMULATION"
    DISTRIBUTION = "DISTRIBUTION"
    ABSORPTION = "ABSORPTION"
    NORMAL = "NORMAL"
    MIXED = "MIXED"

@dataclass
class Zone:
    level: float
    zone_type: ZoneType
    strength: int
    touches: int
    last_touch_age: int
    confidence: float

@dataclass
class EnhancedVolumeAnalysis:
    pattern: VolumePattern
    accumulation_count: int
    distribution_count: int
    confidence: float
    warning: bool
    distribution_warning: bool
    momentum_score: float
    volume_trend: str

@dataclass
class BinanceBalance:
    asset: str
    free: float
    locked: float
    total: float

@dataclass
class RealPosition:
    symbol: str
    side: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    percentage: float
    entry_order_id: str
    stop_order_id: Optional[str]
    take_profit_order_id: Optional[str]
    opened_at: datetime
    status: str

@dataclass
class OrderRequest:
    symbol: str
    side: str
    type: str
    quantity: float
    price: Optional[float] = None
    stopPrice: Optional[float] = None
    timeInForce: str = "GTC"

@dataclass
class TradingSignal:
    symbol: str
    action: ActionType
    current_price: float
    entry_price: float
    stop_loss: float
    take_profit: float
    zone_level: float
    zone_type: ZoneType
    zone_strength: int
    volume_pattern: VolumePattern
    volume_analysis: EnhancedVolumeAnalysis
    confidence_score: float
    reasoning: str
    position_size_usdt: float
    quantity: float
    timestamp: datetime

# ================== RETRY DECORATOR ==================
def retry_on_failure(max_attempts: int = 3, delay: float = 2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        logger.error(f"❌ Failed after {max_attempts} attempts: {e}")
                        raise
                    logger.warning(f"⚠️ Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

# ================== BINANCE API SIGNATURE ==================
def create_signature(query_string: str, secret_key: str) -> str:
    """Create HMAC SHA256 signature for Binance API"""
    return hmac.new(
        secret_key.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

def get_timestamp() -> int:
    """Get current timestamp in milliseconds"""
    return int(time.time() * 1000)

# ================== REAL TRADING DATABASE ==================
class RealTradingDatabase:
    def __init__(self, db_file: str = Config.DATABASE_FILE):
        self.db_file = f"data/{db_file}"
        os.makedirs("data", exist_ok=True)
        self.init_database()

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_file)
        try:
            yield conn
        finally:
            conn.close()

    def init_database(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Real orders table
            cursor.execute('''CREATE TABLE IF NOT EXISTS real_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT UNIQUE NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                type TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL,
                stop_price REAL,
                status TEXT NOT NULL,
                filled_qty REAL DEFAULT 0,
                avg_price REAL DEFAULT 0,
                commission REAL DEFAULT 0,
                created_at DATETIME NOT NULL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')

            # Real positions table
            cursor.execute('''CREATE TABLE IF NOT EXISTS real_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                size REAL NOT NULL,
                entry_price REAL NOT NULL,
                current_price REAL NOT NULL,
                unrealized_pnl REAL NOT NULL,
                entry_order_id TEXT NOT NULL,
                stop_order_id TEXT,
                tp_order_id TEXT,
                opened_at DATETIME NOT NULL,
                closed_at DATETIME,
                status TEXT DEFAULT 'OPEN',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')

            # Account balance history
            cursor.execute('''CREATE TABLE IF NOT EXISTS balance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                usdt_balance REAL NOT NULL,
                total_positions_value REAL NOT NULL,
                unrealized_pnl REAL NOT NULL,
                daily_pnl REAL NOT NULL,
                active_positions INTEGER NOT NULL,
                timestamp DATETIME NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')

            # Real trading signals
            cursor.execute('''CREATE TABLE IF NOT EXISTS trading_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                entry_price REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                confidence REAL NOT NULL,
                zone_type TEXT NOT NULL,
                volume_pattern TEXT NOT NULL,
                executed INTEGER DEFAULT 0,
                order_id TEXT,
                timestamp DATETIME NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')

            conn.commit()
            logger.info("✅ Real trading database initialized")

    def save_order(self, order_data: Dict[str, Any]):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO real_orders 
                (order_id, symbol, side, type, quantity, price, stop_price, status, 
                 filled_qty, avg_price, commission, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                order_data['orderId'], order_data['symbol'], order_data['side'],
                order_data['type'], order_data['origQty'], order_data.get('price'),
                order_data.get('stopPrice'), order_data['status'],
                order_data.get('executedQty', 0), order_data.get('avgPrice', 0),
                order_data.get('commission', 0), datetime.utcnow()
            ))
            conn.commit()

    def save_position(self, position: RealPosition):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO real_positions 
                (symbol, side, size, entry_price, current_price, unrealized_pnl,
                 entry_order_id, stop_order_id, tp_order_id, opened_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                position.symbol, position.side, position.size, position.entry_price,
                position.current_price, position.unrealized_pnl, position.entry_order_id,
                position.stop_order_id, position.take_profit_order_id, position.opened_at,
                position.status
            ))
            conn.commit()

    def save_trading_signal(self, signal_data: Dict[str, Any]):
        """Save trading signal to database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO trading_signals 
                (symbol, action, entry_price, stop_loss, take_profit, confidence,
                 zone_type, volume_pattern, executed, order_id, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal_data['symbol'], signal_data['action'], signal_data['entry_price'],
                signal_data['stop_loss'], signal_data['take_profit'], signal_data['confidence'],
                signal_data['zone_type'], signal_data['volume_pattern'], 
                signal_data['executed'], signal_data.get('order_id'), signal_data['timestamp']
            ))
            conn.commit()

# ================== BINANCE REAL TRADING API ==================
class BinanceRealTrading:
    def __init__(self):
        self.api_key = Config.BINANCE_API_KEY
        self.secret_key = Config.BINANCE_SECRET_KEY
        self.base_url = Config.BINANCE_BASE_URL
        self.session = requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": self.api_key})

    def _make_signed_request(self, method: str, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Fixed signature method for Binance API"""
        if params is None:
            params = {}
        
        # Add timestamp - use server time for better sync
        try:
            server_time_response = self.session.get(f"{self.base_url}/time", timeout=5)
            if server_time_response.status_code == 200:
                server_time = server_time_response.json()['serverTime']
                params['timestamp'] = server_time
            else:
                params['timestamp'] = get_timestamp()
        except:
            params['timestamp'] = get_timestamp()
        
        # Convert all values to strings and sort parameters (CRITICAL for signature)
        str_params = {}
        for key, value in params.items():
            str_params[key] = str(value)
        
        # Sort parameters alphabetically by key (Binance requirement)
        sorted_params = sorted(str_params.items())
        query_string = '&'.join([f"{key}={value}" for key, value in sorted_params])
        
        # Generate signature
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        # Build final URL with signature
        url = f"{self.base_url}/{endpoint}?{query_string}&signature={signature}"
        
        try:
            if method.upper() == 'GET':
                response = self.session.get(url, timeout=Config.API_TIMEOUT)
                
            elif method.upper() == 'POST':
                # For POST requests, send as URL with query string (Binance style)
                response = self.session.post(url, timeout=Config.API_TIMEOUT)
                
            elif method.upper() == 'DELETE':
                response = self.session.delete(url, timeout=Config.API_TIMEOUT)
                
            else:
                logger.error(f"Unsupported HTTP method: {method}")
                return None
                
            # Enhanced error logging
            if response.status_code != 200:
                logger.error(f"❌ API Error {response.status_code}: {response.text}")
                logger.error(f"🔍 Method: {method}")
                logger.error(f"🔍 Endpoint: {endpoint}")
                logger.error(f"🔍 Query string: {query_string}")
                logger.error(f"🔍 Signature (first 20 chars): {signature[:20]}...")
                logger.error(f"🔍 Timestamp: {params.get('timestamp')}")
                return None
                
            return response.json()
            
        except Exception as e:
            logger.error(f"❌ API request failed: {e}")
            logger.error(f"🔍 URL: {url[:100]}...")
            return None

    def test_api_credentials(self) -> bool:
        """Test API credentials with simple account call"""
        logger.info("🔐 Testing API credentials...")
        
        try:
            # Simple test with minimal parameters
            timestamp = get_timestamp()
            query_string = f"timestamp={timestamp}"
            signature = hmac.new(
                self.secret_key.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            url = f"{self.base_url}/account?{query_string}&signature={signature}"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                account_data = response.json()
                logger.info("✅ API credentials are valid")
                logger.info(f"📊 Account type: {account_data.get('accountType', 'Unknown')}")
                logger.info(f"🔄 Can trade: {account_data.get('canTrade', False)}")
                logger.info(f"💰 Total balances: {len(account_data.get('balances', []))}")
                return True
            else:
                logger.error(f"❌ API test failed: {response.status_code}")
                logger.error(f"❌ Error: {response.text}")
                logger.error(f"🔍 Test query: {query_string}")
                logger.error(f"🔍 Test signature: {signature[:20]}...")
                return False
                
        except Exception as e:
            logger.error(f"❌ Credential test error: {e}")
            return False

    @retry_on_failure(max_attempts=3, delay=2)
    def get_account_info(self) -> Optional[Dict[str, Any]]:
        return self._make_signed_request('GET', 'account')

    @retry_on_failure(max_attempts=3, delay=2)
    def get_open_orders(self, symbol: str = None) -> Optional[List[Dict[str, Any]]]:
        params = {}
        if symbol:
            params['symbol'] = symbol
        return self._make_signed_request('GET', 'openOrders', params)

    @retry_on_failure(max_attempts=3, delay=2)
    def place_order(self, order: OrderRequest, skip_quantity_adjustment: bool = False) -> Optional[Dict[str, Any]]:
        # Only adjust quantity if not already validated
        if not skip_quantity_adjustment:
            # Get current price for quantity adjustment
            current_price = self.get_current_price(order.symbol)
            if not current_price and not order.price:
                logger.error(f"❌ Could not get price for {order.symbol}")
                return None
                
            price_for_calc = order.price if order.price else current_price
            
            # Adjust quantity to meet exchange filters
            adjusted_quantity = self.adjust_quantity_to_filters(order.symbol, order.quantity, price_for_calc)
            if not adjusted_quantity:
                logger.error(f"❌ Could not adjust quantity for {order.symbol}")
                return None
        else:
            # Use the quantity as-is (already validated)
            adjusted_quantity = order.quantity
            
        # Format quantity with proper precision
        if adjusted_quantity < 0.001:
            quantity_str = f"{adjusted_quantity:.8f}".rstrip('0').rstrip('.')
        else:
            quantity_str = f"{adjusted_quantity:.6f}".rstrip('0').rstrip('.')
        
        params = {
            'symbol': order.symbol,
            'side': order.side,
            'type': order.type,
            'quantity': quantity_str
        }
        
        # Only add timeInForce for non-MARKET orders
        if order.type != "MARKET":
            params['timeInForce'] = order.timeInForce
            
        # Adjust prices to meet PRICE_FILTER requirements (only if not skipping adjustment)
        if order.price:
            if not skip_quantity_adjustment:
                adjusted_price = self.adjust_price_to_filters(order.symbol, order.price)
                if adjusted_price:
                    params['price'] = f"{adjusted_price:.8f}".rstrip('0').rstrip('.')
                else:
                    logger.error(f"❌ Could not adjust price {order.price} for {order.symbol}")
                    return None
            else:
                # Use price as-is
                params['price'] = f"{order.price:.8f}".rstrip('0').rstrip('.')
                
        if order.stopPrice:
            if not skip_quantity_adjustment:
                adjusted_stop_price = self.adjust_price_to_filters(order.symbol, order.stopPrice)
                if adjusted_stop_price:
                    params['stopPrice'] = f"{adjusted_stop_price:.8f}".rstrip('0').rstrip('.')
                else:
                    logger.error(f"❌ Could not adjust stop price {order.stopPrice} for {order.symbol}")
                    return None
            else:
                # Use stop price as-is
                params['stopPrice'] = f"{order.stopPrice:.8f}".rstrip('0').rstrip('.')
            
        logger.info(f"📤 Placing {order.type} order: {order.side} {quantity_str} {order.symbol}")
        if 'price' in params:
            logger.info(f"💲 Price: {params['price']}")
        if 'stopPrice' in params:
            logger.info(f"🛑 Stop Price: {params['stopPrice']}")
        
        result = self._make_signed_request('POST', 'order', params)
        
        if result:
            logger.info(f"✅ Order placed: {result.get('orderId', 'N/A')} | Status: {result.get('status', 'N/A')}")
        else:
            logger.error("❌ Failed to place order")
            
        return result

    def place_test_order(self, order: OrderRequest, skip_quantity_adjustment: bool = False) -> Optional[Dict[str, Any]]:
        """Place test order (doesn't execute) with proper quantity adjustment"""
        # Only adjust quantity if not already validated
        if not skip_quantity_adjustment:
            # Get current price for quantity adjustment
            current_price = self.get_current_price(order.symbol)
            if not current_price and not order.price:
                logger.error(f"❌ Could not get price for {order.symbol}")
                return None
                
            price_for_calc = order.price if order.price else current_price
            
            # Adjust quantity to meet exchange filters
            adjusted_quantity = self.adjust_quantity_to_filters(order.symbol, order.quantity, price_for_calc)
            if not adjusted_quantity:
                logger.error(f"❌ Could not adjust quantity for {order.symbol}")
                return None
        else:
            # Use the quantity as-is (already validated)
            adjusted_quantity = order.quantity
            
        quantity_str = f"{adjusted_quantity:.8f}".rstrip('0').rstrip('.')
        
        params = {
            'symbol': order.symbol,
            'side': order.side,
            'type': order.type,
            'quantity': quantity_str
        }
        
        if order.type != "MARKET":
            params['timeInForce'] = order.timeInForce
            
        # Adjust prices to meet PRICE_FILTER requirements (only if not skipping adjustment)
        if order.price:
            if not skip_quantity_adjustment:
                adjusted_price = self.adjust_price_to_filters(order.symbol, order.price)
                if adjusted_price:
                    params['price'] = f"{adjusted_price:.8f}".rstrip('0').rstrip('.')
                else:
                    logger.error(f"❌ Could not adjust price for test order")
                    return None
            else:
                params['price'] = f"{order.price:.8f}".rstrip('0').rstrip('.')
                
        if order.stopPrice:
            if not skip_quantity_adjustment:
                adjusted_stop_price = self.adjust_price_to_filters(order.symbol, order.stopPrice)
                if adjusted_stop_price:
                    params['stopPrice'] = f"{adjusted_stop_price:.8f}".rstrip('0').rstrip('.')
                else:
                    logger.error(f"❌ Could not adjust stop price for test order")
                    return None
            else:
                params['stopPrice'] = f"{order.stopPrice:.8f}".rstrip('0').rstrip('.')
            
        logger.info(f"🧪 Testing order: {order.side} {quantity_str} {order.symbol}")
        
        result = self._make_signed_request('POST', 'order/test', params)
        
        if result == {}:  # Test endpoint returns empty dict on success
            logger.info("✅ Test order validation passed")
            return {"status": "TEST_SUCCESS"}
        else:
            logger.error("❌ Test order failed")
            return None

    @retry_on_failure(max_attempts=3, delay=2)
    def cancel_order(self, symbol: str, order_id: str) -> Optional[Dict[str, Any]]:
        params = {
            'symbol': symbol,
            'orderId': order_id
        }
        
        logger.info(f"🚫 Cancelling order: {order_id}")
        result = self._make_signed_request('DELETE', 'order', params)
        
        if result:
            logger.info(f"✅ Order cancelled: {order_id}")
        else:
            logger.error(f"❌ Failed to cancel order: {order_id}")
            
        return result

    @retry_on_failure(max_attempts=2, delay=1)
    def get_current_price(self, symbol: str) -> Optional[float]:
        try:
            response = self.session.get(f"{Config.BINANCE_PRICE_URL}?symbol={symbol}")
            response.raise_for_status()
            return float(response.json()['price'])
        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            return None

    @retry_on_failure(max_attempts=2, delay=1)
    def fetch_klines(self, symbol: str, interval: str = '5m', limit: int = 100) -> Optional[pd.DataFrame]:
        try:
            url = f"{Config.BINANCE_KLINES_URL}?symbol={symbol}&interval={interval}&limit={limit}"
            response = self.session.get(url, timeout=Config.API_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            
            klines_data = []
            for candle in data:
                klines_data.append({
                    "timestamp": datetime.utcfromtimestamp(candle[0] / 1000),
                    "open_price": float(candle[1]),
                    "high_price": float(candle[2]),
                    "low_price": float(candle[3]),
                    "close_price": float(candle[4]),
                    "volume": float(candle[5])
                })
            
            return pd.DataFrame(klines_data)
        except Exception as e:
            logger.error(f"Failed to fetch klines for {symbol}: {e}")
            return None

    def get_asset_balance(self, asset: str) -> Dict[str, float]:
        """Get actual balance for a specific asset from account"""
        account = self.get_account_info()
        if not account:
            return {"free": 0.0, "locked": 0.0, "total": 0.0}
            
        for balance in account['balances']:
            if balance['asset'] == asset:
                free_balance = float(balance['free'])
                locked_balance = float(balance['locked'])
                total_balance = free_balance + locked_balance
                
                return {
                    "free": free_balance,
                    "locked": locked_balance, 
                    "total": total_balance
                }
        
        return {"free": 0.0, "locked": 0.0, "total": 0.0}

    def get_sellable_quantity(self, symbol: str) -> float:
        """Get actual sellable quantity for a symbol (removes USDT from symbol)"""
        # Extract base asset from symbol (e.g., BTCUSDT -> BTC)
        base_asset = symbol.replace('USDT', '').replace('BUSD', '').replace('USDC', '')
        
        balance = self.get_asset_balance(base_asset)
        sellable_qty = balance["free"]  # Only free balance is sellable
        
        logger.info(f"💰 {base_asset} balance: Free={sellable_qty:.8f}, Locked={balance['locked']:.8f}")
        return sellable_qty

    def validate_sell_quantity(self, symbol: str, intended_qty: float) -> Optional[float]:
        """Validate and adjust sell quantity based on actual balance"""
        actual_sellable = self.get_sellable_quantity(symbol)
        
        if actual_sellable <= 0:
            logger.warning(f"❌ No sellable balance for {symbol}")
            return None
            
        # If we have less than intended, use what we have (CRITICAL FIX)
        if actual_sellable < intended_qty:
            logger.info(f"📉 Adjusting sell quantity: {intended_qty:.8f} → {actual_sellable:.8f} (using available balance)")
            final_qty = actual_sellable
        else:
            final_qty = intended_qty
            
        # Get current price for filter adjustment
        current_price = self.get_current_price(symbol)
        if not current_price:
            logger.warning(f"❌ Could not get price for {symbol}")
            return None
            
        # Apply exchange filters but NEVER exceed available balance
        adjusted_qty = self.adjust_quantity_to_filters(symbol, final_qty, current_price)
        
        # CRITICAL: Ensure we never try to sell more than we have
        if adjusted_qty and adjusted_qty > actual_sellable:
            logger.warning(f"⚠️ Filter adjustment {adjusted_qty:.8f} > available {actual_sellable:.8f}")
            
            # Try using slightly less to meet filters
            reduced_qty = actual_sellable * 0.999  # Use 99.9% of available
            adjusted_qty = self.adjust_quantity_to_filters(symbol, reduced_qty, current_price)
            
            # If still too much, use raw available balance
            if adjusted_qty and adjusted_qty > actual_sellable:
                logger.warning(f"📉 Using raw balance: {actual_sellable:.8f}")
                adjusted_qty = actual_sellable
                
        if not adjusted_qty or adjusted_qty <= 0:
            logger.warning(f"❌ Cannot create valid sell order for {symbol}")
            return None
            
        # Final validation
        notional_value = adjusted_qty * current_price
        if notional_value < 10.0:  # Minimum notional check
            logger.warning(f"❌ Order too small: ${notional_value:.2f} < $10.00 for {symbol}")
            return None
            
        logger.info(f"✅ Valid sell quantity: {adjusted_qty:.8f} {symbol} = ${notional_value:.2f}")
        return adjusted_qty

    def get_usdt_balance(self) -> float:
        """Get USDT balance specifically"""
        balance = self.get_asset_balance('USDT')
        return balance["free"]

    def get_exchange_info(self, symbol: str = None) -> Optional[Dict]:
        """Get exchange trading rules and filters"""
        try:
            url = f"{self.base_url}/exchangeInfo"
            if symbol:
                url += f"?symbol={symbol}"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"❌ Failed to get exchange info: {e}")
            return None

    def get_symbol_filters(self, symbol: str) -> Dict[str, Any]:
        """Get trading filters for a specific symbol"""
        exchange_info = self.get_exchange_info(symbol)
        if not exchange_info:
            return {}
            
        for symbol_info in exchange_info.get('symbols', []):
            if symbol_info['symbol'] == symbol:
                filters = {}
                for filter_info in symbol_info.get('filters', []):
                    filter_type = filter_info['filterType']
                    filters[filter_type] = filter_info
                return filters
        return {}

    def adjust_quantity_to_filters(self, symbol: str, quantity: float, price: float) -> Optional[float]:
        """Adjust quantity to meet Binance lot size and notional filters"""
        filters = self.get_symbol_filters(symbol)
        
        if not filters:
            logger.warning(f"⚠️ Could not get filters for {symbol}, using original quantity")
            return quantity
            
        # Apply LOT_SIZE filter
        if 'LOT_SIZE' in filters:
            lot_filter = filters['LOT_SIZE']
            min_qty = float(lot_filter['minQty'])
            max_qty = float(lot_filter['maxQty'])
            step_size = float(lot_filter['stepSize'])
            
            # Adjust to minimum quantity
            if quantity < min_qty:
                logger.info(f"📏 Adjusting quantity from {quantity} to minimum {min_qty}")
                quantity = min_qty
                
            # Adjust to step size
            if step_size > 0:
                quantity = round(quantity / step_size) * step_size
                quantity = round(quantity, 8)  # Round to 8 decimals
                
            # Check maximum
            if quantity > max_qty:
                logger.warning(f"❌ Quantity {quantity} exceeds maximum {max_qty}")
                return None
                
        # Apply MIN_NOTIONAL filter
        if 'MIN_NOTIONAL' in filters:
            notional_filter = filters['MIN_NOTIONAL']
            min_notional = float(notional_filter['minNotional'])
            current_notional = quantity * price
            
            if current_notional < min_notional:
                required_qty = min_notional / price
                logger.info(f"💰 Adjusting quantity from {quantity} to meet min notional: {required_qty}")
                quantity = required_qty
                
                # Re-apply LOT_SIZE after notional adjustment
                if 'LOT_SIZE' in filters:
                    step_size = float(filters['LOT_SIZE']['stepSize'])
                    if step_size > 0:
                        quantity = round(quantity / step_size) * step_size
                        quantity = round(quantity, 8)
        
        logger.info(f"✅ Adjusted quantity: {quantity} {symbol} = ${quantity * price:.2f}")
        return quantity

    def adjust_price_to_filters(self, symbol: str, price: float) -> Optional[float]:
        """Adjust price to meet Binance price filter requirements"""
        filters = self.get_symbol_filters(symbol)
        
        if not filters:
            logger.warning(f"⚠️ Could not get price filters for {symbol}, using original price")
            return price
            
        # Apply PRICE_FILTER
        if 'PRICE_FILTER' in filters:
            price_filter = filters['PRICE_FILTER']
            min_price = float(price_filter['minPrice'])
            max_price = float(price_filter['maxPrice'])
            tick_size = float(price_filter['tickSize'])
            
            # Check minimum price
            if price < min_price:
                logger.warning(f"💲 Price {price} below minimum {min_price}")
                return None
                
            # Check maximum price
            if price > max_price:
                logger.warning(f"💲 Price {price} above maximum {max_price}")
                return None
                
            # Adjust to tick size
            if tick_size > 0:
                adjusted_price = round(price / tick_size) * tick_size
                
                # Determine precision based on tick size
                if tick_size >= 1:
                    precision = 0
                elif tick_size >= 0.1:
                    precision = 1
                elif tick_size >= 0.01:
                    precision = 2
                else:
                    precision = 8
                    
                adjusted_price = round(adjusted_price, precision)
                
                if abs(adjusted_price - price) > price * 0.001:  # Log if adjustment > 0.1%
                    logger.info(f"💲 Adjusting price from {price:.8f} to {adjusted_price:.{precision}f}")
                    
                return adjusted_price
                
        return price

    def test_trading_connection(self) -> bool:
        """Test if we can place orders with proper lot size validation"""
        try:
            symbol = 'BTCUSDT'
            
            # Get current price
            current_price = self.get_current_price(symbol)
            if not current_price:
                logger.error("❌ Could not get BTCUSDT price")
                return False
                
            # Get exchange filters first
            logger.info("📋 Getting exchange trading rules...")
            filters = self.get_symbol_filters(symbol)
            
            if filters:
                logger.info("✅ Retrieved trading filters:")
                if 'LOT_SIZE' in filters:
                    lot_filter = filters['LOT_SIZE']
                    logger.info(f"📏 LOT_SIZE: min={lot_filter['minQty']}, max={lot_filter['maxQty']}, step={lot_filter['stepSize']}")
                if 'MIN_NOTIONAL' in filters:
                    notional_filter = filters['MIN_NOTIONAL']  
                    logger.info(f"💰 MIN_NOTIONAL: {notional_filter['minNotional']}")
                if 'PRICE_FILTER' in filters:
                    price_filter = filters['PRICE_FILTER']
                    logger.info(f"💲 PRICE_FILTER: min={price_filter['minPrice']}, max={price_filter['maxPrice']}, tick={price_filter['tickSize']}")
            
            # Start with desired notional value
            desired_notional = 15.0  # $15 to be safe above minimum
            initial_quantity = desired_notional / current_price
            
            # Adjust quantity to meet all filters
            adjusted_quantity = self.adjust_quantity_to_filters(symbol, initial_quantity, current_price)
            if not adjusted_quantity:
                logger.error("❌ Could not adjust quantity to meet filters")
                return False
                
            final_notional = adjusted_quantity * current_price
            logger.info(f"🧪 Testing with {adjusted_quantity:.8f} BTCUSDT (${final_notional:.2f})")
            
            # Create test order with adjusted quantity
            test_order = OrderRequest(
                symbol=symbol,
                side='BUY',
                type='MARKET',
                quantity=adjusted_quantity
            )
            
            # Use test endpoint first
            result = self.place_test_order(test_order)
            
            if result and result.get('status') == 'TEST_SUCCESS':
                logger.info("✅ Trading connection test passed")
                return True
            else:
                logger.error("❌ Trading connection test failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Trading connection test failed: {e}")
            return False

    def validate_order_size(self, symbol: str, quantity: float, price: float) -> bool:
        # Calculate notional value
        notional = quantity * price
        
        # Check minimum notional (usually $10-12 for most pairs)
        if notional < 10.0:
            logger.warning(f"❌ Order too small: ${notional:.2f} < $10.00")
            return False
            
        # Check minimum quantity (basic validation)
        if quantity <= 0:
            logger.warning(f"❌ Invalid quantity: {quantity}")
            return False
            
        logger.info(f"✅ Order size valid: {quantity:.6f} {symbol} = ${notional:.2f}")
        return True

# ================== VOLUME ANALYZER ==================
class VolumeAnalyzer:
    def __init__(self, lookback: int = Config.VOLUME_LOOKBACK):
        self.lookback = lookback

    def analyze(self, df: pd.DataFrame) -> Optional[EnhancedVolumeAnalysis]:
        if df is None or df.empty or len(df) < self.lookback:
            return None

        try:
            volumes = df['volume'].values[-self.lookback:]
            closes = df['close_price'].values[-self.lookback:]
            changes = np.diff(closes)

            avg_volume = np.mean(volumes)
            high_volume_threshold = avg_volume * Config.HIGH_VOLUME_THRESHOLD

            accumulation_count = 0
            distribution_count = 0

            for i in range(1, len(volumes)):
                if volumes[i] > high_volume_threshold:
                    if changes[i - 1] > 0:
                        accumulation_count += 1
                    elif changes[i - 1] < 0:
                        distribution_count += 1

            volume_diff = volumes[-1] - volumes[-2] if len(volumes) > 1 else 0
            volume_trend = "UP" if volume_diff > 0 else "DOWN" if volume_diff < 0 else "FLAT"

            momentum_score = np.mean(np.abs(changes[-Config.VOLUME_MOMENTUM_LOOKBACK:])) * (volume_diff / avg_volume)
            warning = accumulation_count < Config.ACCUMULATION_MIN_COUNT
            distribution_warning = distribution_count > Config.DISTRIBUTION_WARNING_THRESHOLD

            pattern = VolumePattern.MIXED
            if accumulation_count >= Config.ACCUMULATION_MIN_COUNT and distribution_count == 0:
                pattern = VolumePattern.ACCUMULATION
            elif distribution_count > 2 and accumulation_count < 2:
                pattern = VolumePattern.DISTRIBUTION
            elif warning:
                pattern = VolumePattern.NORMAL

            confidence = max(0.1, 1 - (distribution_count / self.lookback))

            return EnhancedVolumeAnalysis(
                pattern=pattern,
                accumulation_count=accumulation_count,
                distribution_count=distribution_count,
                confidence=confidence,
                warning=warning,
                distribution_warning=distribution_warning,
                momentum_score=momentum_score,
                volume_trend=volume_trend
            )
        except Exception as e:
            logger.error(f"Error analyzing volume: {e}")
            return None

# ================== ZONE DETECTOR ==================
class ZoneDetector:
    def __init__(self, lookback: int = Config.SWING_LOOKBACK):
        self.lookback = lookback

    def detect_zones(self, df: pd.DataFrame) -> List[Zone]:
        if df is None or df.empty or len(df) < self.lookback:
            return []

        try:
            highs = df['high_price'].values[-self.lookback:]
            lows = df['low_price'].values[-self.lookback:]

            local_max_idx = argrelextrema(highs, np.greater_equal, order=3)[0]
            local_min_idx = argrelextrema(lows, np.less_equal, order=3)[0]

            zones = []

            for idx in local_min_idx:
                level = lows[idx]
                touches = sum(np.abs(lows - level)/level * 100 < Config.ZONE_PROXIMITY_PCT)
                if touches >= Config.MIN_ZONE_TOUCHES:
                    strength = int(touches * Config.ZONE_STRENGTH_MULTIPLIER)
                    confidence = min(1.0, strength / 10)
                    zones.append(Zone(
                        level=level,
                        zone_type=ZoneType.DEMAND,
                        strength=strength,
                        touches=touches,
                        last_touch_age=self.lookback - idx,
                        confidence=confidence
                    ))

            for idx in local_max_idx:
                level = highs[idx]
                touches = sum(np.abs(highs - level)/level * 100 < Config.ZONE_PROXIMITY_PCT)
                if touches >= Config.MIN_ZONE_TOUCHES:
                    strength = int(touches * Config.ZONE_STRENGTH_MULTIPLIER)
                    confidence = min(1.0, strength / 10)
                    zones.append(Zone(
                        level=level,
                        zone_type=ZoneType.SUPPLY,
                        strength=strength,
                        touches=touches,
                        last_touch_age=self.lookback - idx,
                        confidence=confidence
                    ))

            return [z for z in zones if z.strength >= Config.MIN_ZONE_STRENGTH]

        except Exception as e:
            logger.error(f"Error detecting zones: {e}")
            return []

# ================== RISK GUARD ==================
class RiskGuard:
    def __init__(self, binance_api: BinanceRealTrading):
        self.binance_api = binance_api
        self.failed_trades: List[datetime] = []
        self.daily_loss_usdt: float = 0
        self.max_positions = Config.MAX_CONCURRENT_POSITIONS
        
    def can_open_position(self, position_size_usdt: float) -> bool:
        # Check USDT balance with more realistic buffer
        balance = self.binance_api.get_usdt_balance()
        required_balance = Config.MIN_USDT_BALANCE + position_size_usdt
        
        if balance < required_balance:
            # Log more detailed info
            available_for_trading = balance - Config.MIN_USDT_BALANCE
            logger.warning(f"❌ Insufficient balance: ${balance:.2f} available")
            logger.warning(f"💰 Available for trading: ${available_for_trading:.2f} (need ${position_size_usdt:.2f})")
            return False
            
        # Check daily loss limit
        if self.daily_loss_usdt >= Config.MAX_DAILY_LOSS:
            logger.warning(f"❌ Daily loss limit reached: ${self.daily_loss_usdt:.2f}")
            return False
        
        # Check recent failures
        self._trim_old_entries()
        if len(self.failed_trades) >= 2:  # More conservative
            logger.warning(f"❌ Too many recent failures: {len(self.failed_trades)}")
            return False
            
        return True

    def register_loss(self, loss_amount: float):
        self.failed_trades.append(datetime.utcnow())
        self.daily_loss_usdt += abs(loss_amount)
        self._trim_old_entries()
        
    def should_halt_trading(self) -> bool:
        self._trim_old_entries()
        recent_fails = len(self.failed_trades)
        return recent_fails >= 3 or self.daily_loss_usdt >= Config.MAX_DAILY_LOSS
        
    def _trim_old_entries(self):
        one_day_ago = datetime.utcnow() - timedelta(days=1)
        self.failed_trades = [t for t in self.failed_trades if t > one_day_ago]

# ================== SIGNAL GENERATOR ==================
class RealSignalGenerator:
    def __init__(self, volume_analyzer: VolumeAnalyzer, zone_detector: ZoneDetector, risk_guard: RiskGuard, binance_api: BinanceRealTrading):
        self.volume_analyzer = volume_analyzer
        self.zone_detector = zone_detector
        self.risk_guard = risk_guard
        self.binance_api = binance_api

    def generate_signal(self, symbol: str, df: pd.DataFrame, current_price: float) -> Optional[TradingSignal]:
        if df is None or len(df) < Config.MIN_DATA_POINTS:
            return None

        # Risk check first
        if self.risk_guard.should_halt_trading():
            logger.warning("🛑 Trading halted by risk guard")
            return None

        # Volume analysis
        volume_analysis = self.volume_analyzer.analyze(df)
        if not volume_analysis or volume_analysis.pattern == VolumePattern.DISTRIBUTION:
            return None

        # Zone detection
        zones = self.zone_detector.detect_zones(df)
        selected_zone = self._select_relevant_zone(current_price, zones)
        if not selected_zone:
            return None

        # Higher confidence threshold for real trading
        distance = abs(current_price - selected_zone.level) / current_price
        if distance > Config.MAX_DISTANCE_FROM_ZONE:
            return None

        volume_score = volume_analysis.confidence * Config.VOLUME_CONFIDENCE_WEIGHT
        zone_score = selected_zone.confidence * Config.ZONE_CONFIDENCE_WEIGHT
        confidence = volume_score + zone_score

        if confidence < Config.MIN_CONFIDENCE_SCORE:
            return None

        # Only BUY signals for safety (can be extended)
        if selected_zone.zone_type != ZoneType.DEMAND:
            return None

        action = ActionType.BUY
        entry = current_price
        
        # Calculate quantity with proper filter adjustment
        initial_quantity = Config.POSITION_SIZE_USDT / current_price
        adjusted_quantity = self.binance_api.adjust_quantity_to_filters(symbol, initial_quantity, current_price)
        
        if not adjusted_quantity:
            logger.warning(f"❌ Could not calculate valid quantity for {symbol}")
            return None
            
        # Recalculate actual position size based on adjusted quantity
        actual_position_size = adjusted_quantity * current_price
        
        # Calculate stops
        sl_price = entry * (1 - Config.STOP_LOSS_PCT)
        tp_price = entry * (1 + Config.TAKE_PROFIT_PCT)
        

    
        # Final risk check with actual position size
        if not self.risk_guard.can_open_position(actual_position_size):
            return None

        return TradingSignal(
            symbol=symbol,
            action=action,
            current_price=current_price,
            entry_price=entry,
            stop_loss=sl_price,
            take_profit=tp_price,
            zone_level=selected_zone.level,
            zone_type=selected_zone.zone_type,
            zone_strength=selected_zone.strength,
            volume_pattern=volume_analysis.pattern,
            volume_analysis=volume_analysis,
            confidence_score=confidence,
            reasoning=f"{volume_analysis.pattern.value} + {selected_zone.zone_type.value} zone",
            position_size_usdt=actual_position_size,
            quantity=adjusted_quantity,
            timestamp=datetime.utcnow()
        )

    def _select_relevant_zone(self, current_price: float, zones: List[Zone]) -> Optional[Zone]:
        nearby_zones = [z for z in zones if abs(current_price - z.level) / current_price <= Config.MAX_DISTANCE_FROM_ZONE]
        if not nearby_zones:
            return None
        return max(nearby_zones, key=lambda z: z.confidence)

# ================== POSITION MANAGER ==================
class RealPositionManager:
    def __init__(self, binance_api: BinanceRealTrading, db: RealTradingDatabase):
        self.binance_api = binance_api
        self.db = db
        self.positions: Dict[str, RealPosition] = {}
        
        # Load existing positions on startup
        self.load_positions_from_database()
        self.sync_positions_with_binance()

    def load_positions_from_database(self):
        """Load existing open positions from database"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT symbol, side, size, entry_price, current_price, unrealized_pnl,
                           entry_order_id, stop_order_id, tp_order_id, opened_at, status
                    FROM real_positions 
                    WHERE status IN ('OPEN', 'OPEN_NO_STOPS')
                    ORDER BY opened_at DESC
                """)
                
                rows = cursor.fetchall()
                loaded_count = 0
                
                for row in rows:
                    symbol = row[0]
                    position = RealPosition(
                        symbol=symbol,
                        side=row[1],
                        size=float(row[2]),
                        entry_price=float(row[3]),
                        current_price=float(row[4]),
                        unrealized_pnl=float(row[5]),
                        percentage=0,  # Will be recalculated
                        entry_order_id=row[6],
                        stop_order_id=row[7],
                        take_profit_order_id=row[8],
                        opened_at=datetime.fromisoformat(row[9].replace(' ', 'T')) if isinstance(row[9], str) else row[9],
                        status=row[10]
                    )
                    
                    self.positions[symbol] = position
                    loaded_count += 1
                    logger.info(f"📂 Loaded position: {symbol} | Size: {position.size:.8f} | Entry: ${position.entry_price:.4f}")
                
                if loaded_count > 0:
                    logger.info(f"✅ Loaded {loaded_count} existing positions from database")
                else:
                    logger.info("📭 No existing positions found in database")
                    
        except Exception as e:
            logger.error(f"❌ Error loading positions from database: {e}")

    def sync_positions_with_binance(self):
        """Sync positions with actual Binance balances"""
        logger.info("🔄 Syncing positions with Binance account...")
        
        try:
            # Get all non-zero balances
            account = self.binance_api.get_account_info()
            if not account:
                logger.warning("⚠️ Could not get account info for position sync")
                return
                
            detected_positions = 0
            
            for balance_info in account['balances']:
                asset = balance_info['asset']
                free_balance = float(balance_info['free'])
                locked_balance = float(balance_info['locked'])
                total_balance = free_balance + locked_balance
                
                # Skip USDT and assets with very small balances
                if asset == 'USDT' or total_balance < 0.001:
                    continue
                
                # Construct symbol (assume USDT pairs for now)
                symbol = f"{asset}USDT"
                
                # Check if we have this position tracked
                if symbol in self.positions:
                    # Update existing position with actual balance
                    position = self.positions[symbol]
                    old_size = position.size
                    position.size = free_balance  # Use free balance as sellable
                    
                    if abs(old_size - free_balance) > 0.00001:
                        logger.info(f"📊 Updated {symbol}: {old_size:.8f} → {free_balance:.8f}")
                        
                else:
                    # Detect untracked position (from manual trades or previous sessions)
                    if free_balance > 0.001:  # Significant balance
                        logger.warning(f"🔍 Detected untracked position: {symbol} = {free_balance:.8f}")
                        
                        # Try to get current price
                        current_price = self.binance_api.get_current_price(symbol)
                        if current_price:
                            # Create position record (without entry price info)
                            position = RealPosition(
                                symbol=symbol,
                                side="BUY",  # Assume buy position
                                size=free_balance,
                                entry_price=current_price,  # Use current price as entry (not accurate but best we have)
                                current_price=current_price,
                                unrealized_pnl=0,
                                percentage=0,
                                entry_order_id="MANUAL_OR_PREVIOUS",
                                stop_order_id=None,
                                take_profit_order_id=None,
                                opened_at=datetime.utcnow(),
                                status="OPEN_NO_STOPS"
                            )
                            
                            self.positions[symbol] = position
                            self.db.save_position(position)
                            detected_positions += 1
                            
                            logger.warning(f"⚠️ Added untracked position: {symbol} | Size: {free_balance:.8f}")
                            logger.warning(f"⚠️ Using current price ${current_price:.4f} as estimated entry")
            
            if detected_positions > 0:
                logger.warning(f"🔍 Found {detected_positions} untracked positions")
            
            logger.info(f"✅ Position sync complete. Total positions: {len(self.positions)}")
            
        except Exception as e:
            logger.error(f"❌ Error syncing positions with Binance: {e}")

    def get_position_summary(self) -> str:
        """Get a summary of all positions"""
        if not self.positions:
            return "📭 No active positions"
            
        summary = []
        total_value = 0
        total_pnl = 0
        
        for symbol, position in self.positions.items():
            if position.status in ["OPEN", "OPEN_NO_STOPS", "PROTECTED_BY_STOP"]:
                current_value = position.size * position.current_price if position.size > 0 else 0
                total_value += current_value
                total_pnl += position.unrealized_pnl
                
                if position.status == "OPEN":
                    status_emoji = "✅"
                elif position.status == "OPEN_NO_STOPS":
                    status_emoji = "⚠️"  
                elif position.status == "PROTECTED_BY_STOP":
                    status_emoji = "🛡️"  # Protected by stop loss
                else:
                    status_emoji = "❓"
                
                if position.size > 0:
                    summary.append(f"{status_emoji} {symbol}: {position.size:.6f} | ${position.current_price:.2f} | {position.percentage:+.1f}%")
                else:
                    summary.append(f"{status_emoji} {symbol}: Protected by stop loss | ${position.current_price:.2f}")
        
        if summary:
            return f"📊 Active Positions ({len(summary)}):\n" + "\n".join(summary) + f"\n💰 Total Value: ${total_value:.2f} | PnL: ${total_pnl:+.2f}"
        else:
            return "📭 No active positions"

    def execute_signal(self, signal: TradingSignal) -> bool:
        if signal.action != ActionType.BUY:
            return False

        try:
            # Validate order
            if not self.binance_api.validate_order_size(signal.symbol, signal.quantity, signal.entry_price):
                logger.error(f"❌ Invalid order size for {signal.symbol}")
                return False

            # Place market buy order
            buy_order = OrderRequest(
                symbol=signal.symbol,
                side="BUY",
                type="MARKET",
                quantity=signal.quantity
            )
            
            logger.info(f"📤 Placing BUY order: {signal.quantity:.8f} {signal.symbol}")
            order_result = self.binance_api.place_order(buy_order)
            
            if not order_result:
                logger.error(f"❌ Failed to place buy order for {signal.symbol}")
                return False
                
            if order_result.get('status') != 'FILLED':
                logger.error(f"❌ Buy order not filled for {signal.symbol}: {order_result.get('status')}")
                return False

            # Save order to database
            self.db.save_order(order_result)

            # Extract actual entry price and quantity
            actual_entry = float(order_result.get('avgPrice', signal.entry_price))
            actual_quantity = float(order_result.get('executedQty', signal.quantity))
            order_id = order_result['orderId']
            
            logger.info(f"✅ BUY executed: {actual_quantity:.8f} {signal.symbol} at ${actual_entry:.4f}")

            # Wait longer for balance update (increased from 2 to 5 seconds)
            logger.info("⏳ Waiting for balance update...")
            time.sleep(5)

            # Validate sellable quantity with retries
            sellable_quantity = None
            for attempt in range(3):
                sellable_quantity = self.binance_api.validate_sell_quantity(signal.symbol, actual_quantity)
                if sellable_quantity:
                    break
                logger.warning(f"⚠️ Attempt {attempt + 1}: No sellable balance yet, waiting...")
                time.sleep(2)

            # Determine position status and size
            if sellable_quantity:
                position_size = sellable_quantity
                position_status = "OPEN"
                logger.info(f"✅ Sellable quantity confirmed: {sellable_quantity:.8f}")
            else:
                position_size = actual_quantity
                position_status = "OPEN_NO_STOPS"
                logger.warning(f"⚠️ Using executed quantity as position size: {actual_quantity:.8f}")

            # Create position record
            position = RealPosition(
                symbol=signal.symbol,
                side="BUY",
                size=position_size,
                entry_price=actual_entry,
                current_price=actual_entry,
                unrealized_pnl=0,
                percentage=0,
                entry_order_id=order_id,
                stop_order_id=None,
                take_profit_order_id=None,
                opened_at=datetime.utcnow(),
                status=position_status
            )
            
            # Save position
            self.positions[signal.symbol] = position
            self.db.save_position(position)

            # Log summary
            logger.info(f"📊 Position created: {position.size:.8f} {signal.symbol} | Status: {position_status}")
            logger.info(
                f"📊 Position summary: {position.size:.8f} {signal.symbol} | Entry: ${actual_entry:.4f} | "
                f"SL: ${signal.stop_loss:.4f} | TP: ${signal.take_profit:.4f}"
            )

            return True
            
        except Exception as e:
            logger.exception(f"❌ Critical error executing signal for {signal.symbol}: {e}")
            return False

    def update_positions(self):
        """FIXED: Single unified position update method"""
        positions_to_remove = []
        
        for symbol, position in self.positions.items():
            if position.status not in ["OPEN", "OPEN_NO_STOPS", "PROTECTED_BY_STOP"]:
                continue

            try:
                # Sync with actual balance first
                self.sync_position_with_balance(symbol, position)

                # Get current price with retry logic
                current_price = self.get_current_price_with_retry(symbol)
                if not current_price:
                    logger.warning(f"⚠️ Could not get price for {symbol}, skipping update")
                    continue

                # Update position price and PnL
                position.current_price = current_price
                if position.size > 0:
                    position.unrealized_pnl = (current_price - position.entry_price) * position.size
                    position.percentage = position.unrealized_pnl / (position.entry_price * position.size) * 100
                else:
                    position.unrealized_pnl = 0
                    position.percentage = 0

                # === Dynamic Take Profit ===
                tp_trigger_price = position.entry_price * (1 + Config.TAKE_PROFIT_PCT)
                if current_price >= tp_trigger_price and position.size > 0:
                    if self.execute_take_profit(symbol, position):
                        positions_to_remove.append(symbol)
                        continue

                # === Dynamic Stop Loss ===
                sl_trigger_price = position.entry_price * (1 - Config.STOP_LOSS_PCT)
                if current_price <= sl_trigger_price and position.size > 0:
                    if self.execute_stop_loss(symbol, position):
                        positions_to_remove.append(symbol)
                        continue

                # Save updated position
                self.db.save_position(position)
                
            except Exception as e:
                logger.error(f"❌ Error updating position {symbol}: {e}")
                continue
        
        # Remove closed positions
        for symbol in positions_to_remove:
            if symbol in self.positions:
                del self.positions[symbol]
                logger.info(f"🗑️ Removed closed position: {symbol}")

    def get_current_price_with_retry(self, symbol: str, max_retries: int = 3) -> Optional[float]:
        """Get current price with retry logic"""
        for attempt in range(max_retries):
            try:
                price = self.binance_api.get_current_price(symbol)
                if price:
                    return price
            except Exception as e:
                logger.warning(f"⚠️ Price fetch attempt {attempt + 1} failed for {symbol}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
        return None

    def execute_take_profit(self, symbol: str, position: RealPosition) -> bool:
        """Execute take profit order"""
        try:
            sell_qty = self.binance_api.validate_sell_quantity(symbol, position.size)
            if not sell_qty:
                logger.warning(f"❌ No sellable quantity for TP on {symbol}")
                return False

            tp_order = OrderRequest(
                symbol=symbol,
                side="SELL",
                type="MARKET",
                quantity=sell_qty
            )
            
            result = self.binance_api.place_order(tp_order, skip_quantity_adjustment=True)
            if result and result.get('status') == 'FILLED':
                logger.info(f"✅ TP executed for {symbol} at ${position.current_price:.4f}")
                position.status = "CLOSED_TP"
                self.db.save_position(position)
                return True
            else:
                logger.error(f"❌ TP order failed for {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error executing TP for {symbol}: {e}")
            return False

    def execute_stop_loss(self, symbol: str, position: RealPosition) -> bool:
        """Execute stop loss order"""
        try:
            sell_qty = self.binance_api.validate_sell_quantity(symbol, position.size)
            if not sell_qty:
                logger.warning(f"❌ No sellable quantity for SL on {symbol}")
                return False

            sl_order = OrderRequest(
                symbol=symbol,
                side="SELL",
                type="MARKET",
                quantity=sell_qty
            )
            
            result = self.binance_api.place_order(sl_order, skip_quantity_adjustment=True)
            if result and result.get('status') == 'FILLED':
                logger.info(f"🛑 SL executed for {symbol} at ${position.current_price:.4f}")
                position.status = "CLOSED_SL"
                self.db.save_position(position)
                return True
            else:
                logger.error(f"❌ SL order failed for {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error executing SL for {symbol}: {e}")
            return False

    def sync_position_with_balance(self, symbol: str, position: RealPosition):
        """Sync position quantity with actual account balance"""
        try:
            actual_sellable = self.binance_api.get_sellable_quantity(symbol)
            
            # If actual balance differs significantly from position size
            if abs(actual_sellable - position.size) > 0.00001:  # More than 1 satoshi difference
                logger.info(f"🔄 Syncing {symbol}: Position={position.size:.8f} → Actual={actual_sellable:.8f}")
                
                # Update position size to match actual balance
                old_size = position.size
                position.size = actual_sellable
                
                # Recalculate PnL with actual quantity
                if position.current_price:
                    position.unrealized_pnl = (position.current_price - position.entry_price) * position.size
                    if position.size > 0:
                        position.percentage = position.unrealized_pnl / (position.entry_price * position.size) * 100
                    else:
                        position.percentage = 0
                
                # If balance is zero or very small, mark position as potentially closed
                if actual_sellable < 0.00001:
                    logger.warning(f"⚠️ {symbol} balance very low ({actual_sellable:.8f}), position may be closed")
                    position.status = "BALANCE_ZERO"
                
                self.db.save_position(position)
                
        except Exception as e:
            logger.error(f"❌ Error syncing {symbol} position balance: {e}")

    def fix_missing_stop_orders(self):
        """Try to place missing stop orders for positions without them"""
        for symbol, position in self.positions.items():
            try:
                # Skip if position already has stop orders or is protected/closed
                if position.status not in ["OPEN_NO_STOPS"]:
                    continue
                    
                logger.info(f"🔧 Attempting to fix missing stop orders for {symbol}")
                
                # Get current sellable balance
                sellable_qty = self.binance_api.validate_sell_quantity(symbol, position.size)
                if not sellable_qty:
                    logger.warning(f"❌ Still no sellable balance for {symbol}")
                    continue
                
                # Calculate stops based on current entry price  
                current_price = self.binance_api.get_current_price(symbol)
                if not current_price:
                    continue
                    
                # Use position entry price for stop calculation
                stop_loss_price = position.entry_price * (1 - Config.STOP_LOSS_PCT)
                take_profit_price = position.entry_price * (1 + Config.TAKE_PROFIT_PCT)
                
                # Only place stop loss if current price is still above it
                if current_price > stop_loss_price:
                    stop_order = OrderRequest(
                        symbol=symbol,
                        side="SELL",
                        type="STOP_LOSS_LIMIT", 
                        quantity=sellable_qty,
                        price=stop_loss_price * 0.999,
                        stopPrice=stop_loss_price
                    )
                    
                    # Use skip_quantity_adjustment=True since sellable_qty is already validated
                    stop_result = self.binance_api.place_order(stop_order, skip_quantity_adjustment=True)
                    if stop_result:
                        position.stop_order_id = stop_result['orderId']
                        logger.info(f"✅ Stop loss fixed for {symbol}: {stop_result['orderId']}")
                
                # NOTE: Skipping take profit to avoid balance conflicts
                # Only place take profit if we implement OCO orders
                logger.info(f"💡 Take profit target: ${take_profit_price:.4f} (manual exit recommended)")
                
                # Update status if we successfully placed stop order
                if position.stop_order_id:
                    position.status = "OPEN"
                    position.size = sellable_qty  # Update to actual sellable quantity
                    self.db.save_position(position)
                    logger.info(f"✅ Position {symbol} status updated to OPEN with stop loss protection")
                    
            except Exception as e:
                logger.error(f"❌ Error fixing stop orders for {symbol}: {e}")

# ================== TELEGRAM NOTIFIER ==================
class TelegramNotifier:
    def __init__(self, bot_token: str, recipients: List[Dict[str, str]]):
        self.bot_token = bot_token
        self.recipients = recipients
        self.base_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    def send_message(self, message: str):
        for recipient in self.recipients:
            payload = {
                "chat_id": recipient["chat_id"],
                "text": message,
                "parse_mode": "HTML"
            }
            try:
                response = requests.post(self.base_url, data=payload, timeout=10)
                if response.status_code != 200:
                    logger.error(f"❌ Telegram send failed: {response.text}")
            except Exception as e:
                logger.error(f"📵 Telegram error: {e}")

    def send_trade_notification(self, signal: TradingSignal, success: bool):
        status = "✅ EXECUTED" if success else "❌ FAILED"
        message = f"""
🤖 <b>REAL TRADE {status}</b>

💰 <b>{signal.symbol}</b>
📊 Action: <b>{signal.action.value}</b>
💵 Entry: <b>${signal.entry_price:.4f}</b>
📉 Stop Loss: <b>${signal.stop_loss:.4f}</b>
📈 Take Profit: <b>${signal.take_profit:.4f}</b>
💎 Size: <b>{signal.quantity:.4f}</b>
💸 Value: <b>${signal.position_size_usdt:.2f}</b>
📊 Confidence: <b>{signal.confidence_score:.1%}</b>
🎯 Pattern: <b>{signal.volume_pattern.value}</b>
📍 Zone: <b>{signal.zone_type.value}</b>

⏰ {signal.timestamp.strftime('%H:%M:%S')}
"""
        self.send_message(message)

# ================== MAIN TRADING BOT ==================
def main():
    logger.info("🚀 REAL TRADING BOT STARTING...")
    
    # Initialize components
    binance_api = BinanceRealTrading()
    db = RealTradingDatabase()
    risk_guard = RiskGuard(binance_api)
    volume_analyzer = VolumeAnalyzer()
    zone_detector = ZoneDetector()
    signal_generator = RealSignalGenerator(volume_analyzer, zone_detector, risk_guard, binance_api)
    position_manager = RealPositionManager(binance_api, db)
    telegram = TelegramNotifier(Config.BOT_TOKEN, Config.TELEGRAM_RECIPIENTS)

    # Check and log IP address
    logger.info("🌐 Checking IP address for whitelist...")
    try:
        ip_response = requests.get("https://api.ipify.org", timeout=5)
        if ip_response.status_code == 200:
            current_ip = ip_response.text.strip()
            logger.info(f"📍 Current IP: {current_ip}")
            logger.info(f"⚠️ Make sure this IP is whitelisted in Binance API settings!")
        else:
            logger.warning("⚠️ Could not determine current IP address")
    except Exception as e:
        logger.warning(f"⚠️ IP check failed: {e}")

    # Verify connection and permissions
    logger.info("🔍 Checking API connection...")
    
    # Test basic connectivity first (no auth needed)
    try:
        test_response = requests.get(f"{Config.BINANCE_BASE_URL}/time", timeout=10)
        if test_response.status_code == 200:
            server_time = test_response.json()['serverTime']
            local_time = get_timestamp()
            time_diff = abs(server_time - local_time)
            logger.info(f"⏰ Server time: {server_time}, Local time: {local_time}")
            logger.info(f"⏰ Time difference: {time_diff}ms")
            if time_diff > 5000:  # More than 5 seconds
                logger.warning(f"⚠️ Large time difference: {time_diff}ms - may cause signature issues")
                logger.warning("💡 Try syncing your system clock!")
        else:
            logger.error("❌ Cannot reach Binance servers")
            telegram.send_message("❌ <b>CONNECTION ERROR</b>\nCannot reach Binance servers")
            return
    except Exception as e:
        logger.error(f"❌ Network error: {e}")
        telegram.send_message(f"❌ <b>NETWORK ERROR</b>\n{str(e)}")
        return

    # Test API credentials with improved method
    if not binance_api.test_api_credentials():
        logger.error("❌ API Authentication failed")
        telegram.send_message("""❌ <b>API AUTHENTICATION FAILED</b>

Possible issues:
• Invalid API Key/Secret
• IP not whitelisted in Binance
• API permissions insufficient  
• System clock not synchronized

Steps to fix:
1. Check API credentials in code
2. Whitelist your IP in Binance
3. Enable Spot Trading permission
4. Sync system clock""")
        return
    
    # Get account info
    account = binance_api.get_account_info()
    if not account:
        logger.error("❌ Could not fetch account info")
        return
        
    # Check if account can trade
    if not account.get('canTrade', False):
        logger.error("❌ Account cannot trade - check API permissions")
        telegram.send_message("❌ <b>TRADING DISABLED</b>\nAPI Key does not have SPOT TRADING permissions")
        return

    balance = binance_api.get_usdt_balance()
    logger.info(f"💰 USDT Balance: ${balance:.2f}")
    
    if balance < Config.MIN_USDT_BALANCE:
        logger.error(f"❌ Insufficient balance: ${balance:.2f} < ${Config.MIN_USDT_BALANCE}")
        telegram.send_message(f"❌ <b>LOW BALANCE WARNING</b>\nBalance: ${balance:.2f}\nMinimum required: ${Config.MIN_USDT_BALANCE}")
        return

    # Test trading connection
    logger.info("🧪 Testing trading capabilities...")
    if not binance_api.test_trading_connection():
        logger.error("❌ Trading test failed")
        telegram.send_message("❌ <b>TRADING TEST FAILED</b>\nCannot place orders - check API permissions")
        return
    
    logger.info("✅ All systems ready - starting live trading!")

    # Show initial position summary
    logger.info("📊 Current Portfolio Status:")
    logger.info(position_manager.get_position_summary())

    active_positions_count = len([p for p in position_manager.positions.values() if p.status in ["OPEN", "OPEN_NO_STOPS", "PROTECTED_BY_STOP"]])
    
    startup_message = f"""
🚀 <b>REAL TRADING BOT STARTED</b>

💰 Balance: <b>${balance:.2f} USDT</b>
📊 Monitoring: <b>{len(Config.FOCUS_PAIRS)} pairs</b>
📈 Active Positions: <b>{active_positions_count}</b>
⚙️ Position Size: <b>${Config.POSITION_SIZE_USDT}</b>
🛡️ Max Daily Loss: <b>${Config.MAX_DAILY_LOSS}</b>
⏰ Check Interval: <b>{Config.CHECK_INTERVAL_SECONDS}s</b>

{position_manager.get_position_summary()}

⚠️ <b>LIVE TRADING ACTIVE</b>
"""
    telegram.send_message(startup_message)

    try:
        while True:
            try:
                logger.info("🔄 Starting analysis cycle...")
                
                # Update existing positions first
                position_manager.update_positions()
                
                # Try to fix any positions missing stop orders (every 5 cycles to reduce spam)
                import random
                if random.randint(1, 5) == 1:  # ~20% chance each cycle
                    position_manager.fix_missing_stop_orders()
                
                # Check for new signals (limit to 1 signal per cycle for safety)
                signals_generated = 0
                max_signals_per_cycle = 1
                
                for symbol in Config.FOCUS_PAIRS:
                    try:
                        # Skip if already have position
                        if symbol in position_manager.positions:
                            current_position = position_manager.positions[symbol]
                            if current_position.status in ["OPEN", "OPEN_NO_STOPS", "PROTECTED_BY_STOP"]:
                                if current_position.status == "PROTECTED_BY_STOP":
                                    logger.info(f"⏭️ Skipping {symbol} - protected by stop loss")
                                else:
                                    logger.info(f"⏭️ Skipping {symbol} - already have position: {current_position.size:.6f}")
                                continue
                            
                        # Skip if already generated enough signals this cycle
                        if signals_generated >= max_signals_per_cycle:
                            break

                        # Fetch market data
                        df = binance_api.fetch_klines(symbol)
                        if df is None or df.empty:
                            continue

                        current_price = binance_api.get_current_price(symbol)
                        if not current_price:
                            continue

                        # Generate signal
                        signal = signal_generator.generate_signal(symbol, df, current_price)
                        if not signal:
                            continue

                        logger.info(f"📈 Signal: {symbol} | {signal.action.value} | Confidence: {signal.confidence_score:.1%}")
                        signals_generated += 1

                        # Execute trade
                        success = position_manager.execute_signal(signal)
                        
                        # Save signal to database
                        try:
                            db.save_trading_signal({
                                'symbol': signal.symbol,
                                'action': signal.action.value,
                                'entry_price': signal.entry_price,
                                'stop_loss': signal.stop_loss,
                                'take_profit': signal.take_profit,
                                'confidence': signal.confidence_score,
                                'zone_type': signal.zone_type.value,
                                'volume_pattern': signal.volume_pattern.value,
                                'executed': 1 if success else 0,
                                'order_id': None,
                                'timestamp': signal.timestamp
                            })
                        except Exception as e:
                            logger.error(f"❌ Failed to save signal to DB: {e}")
                        
                        # Send notification
                        telegram.send_trade_notification(signal, success)
                        
                        if success:
                            # Add delay after successful trade
                            time.sleep(10)
                            break  # Only 1 trade per cycle for safety

                    except Exception as e:
                        logger.error(f"❌ Error processing {symbol}: {e}")
                        continue

                # Log current status with detailed summary
                logger.info("📊 Current Portfolio Status:")
                logger.info(position_manager.get_position_summary())
                for symbol, position in position_manager.positions.items():
                    if position.status in ["OPEN", "OPEN_NO_STOPS", "PROTECTED_BY_STOP"]:
                        tp_trigger_price = position.entry_price * (1 + Config.TAKE_PROFIT_PCT)
                        logger.info(f"🎯 TP target for {symbol}: ${tp_trigger_price:.4f}")
                current_balance = binance_api.get_usdt_balance()
                logger.info(f"💰 USDT Balance: ${current_balance:.2f}")
                
            except Exception as e:
                logger.exception(f"❌ Error in main loop: {e}")
                telegram.send_message(f"⚠️ <b>Bot Error</b>\n{str(e)[:200]}...")

            # Sleep before next cycle
            time.sleep(Config.CHECK_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")
        telegram.send_message("🛑 <b>TRADING BOT STOPPED</b>\nBot terminated by user")
    except Exception as e:
        logger.exception(f"💥 Fatal error: {e}")
        telegram.send_message(f"💥 <b>FATAL BOT ERROR</b>\n{str(e)[:200]}...")

if __name__ == "__main__":
    main()
