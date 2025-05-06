#!/usr/bin/env python3
# test_api.py - Enhanced version with more detailed logging
import os
import sys
import logging
import alpaca_trade_api as tradeapi

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Determine if we're in paper or live mode
is_paper = os.getenv('ALPACA_PAPER', 'true').lower() == 'true'

# Print the mode for clarity
logger.info(f"Running in {'PAPER' if is_paper else 'LIVE'} trading mode")

# Get API credentials based on mode
if is_paper:
    API_KEY = os.getenv('ALPACA_PAPER_API_KEY')
    API_SECRET = os.getenv('ALPACA_PAPER_API_SECRET')
    BASE_URL = os.getenv('ALPACA_PAPER_URL', 'https://paper-api.alpaca.markets')
else:
    API_KEY = os.getenv('ALPACA_LIVE_API_KEY')
    API_SECRET = os.getenv('ALPACA_LIVE_API_SECRET')
    BASE_URL = os.getenv('ALPACA_LIVE_URL', 'https://api.alpaca.markets')

# Check if API credentials are set
if not API_KEY or not API_SECRET:
    logger.error("API credentials are missing. Please check your environment variables.")
    logger.error(f"API_KEY: {'Set' if API_KEY else 'Missing'}")
    logger.error(f"API_SECRET: {'Set' if API_SECRET else 'Missing'}")
    sys.exit(1)

logger.info(f"Using API KEY: {API_KEY[:5]}...")
logger.info(f"BASE URL: {BASE_URL}")

# Initialize the API client
try:
    api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
    logger.info("API client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize API client: {str(e)}")
    sys.exit(1)

# Test the API connection
try:
    account = api.get_account()
    logger.info(f"Connection successful! Account ID: {account.id}")
    logger.info(f"Account status: {account.status}")
    logger.info(f"Account value: ${account.portfolio_value}")
    logger.info(f"Cash balance: ${account.cash}")
    logger.info(f"Buying power: ${account.buying_power}")
    
    # Test getting market data
    logger.info("Testing market data retrieval...")
    try:
        aapl = api.get_latest_trade('AAPL')
        logger.info(f"AAPL latest price: ${aapl.price}")
    except Exception as e:
        logger.warning(f"Failed to get market data: {str(e)}")
    
    # Test listing orders
    logger.info("Testing order listing...")
    try:
        orders = api.list_orders(limit=5)
        logger.info(f"Recent orders count: {len(orders)}")
        
        # Show some details about recent orders if available
        if orders:
            for i, order in enumerate(orders):
                logger.info(f"Order {i+1}: {order.symbol} {order.side} {order.qty} @ {order.type} - Status: {order.status}")
    except Exception as e:
        logger.warning(f"Failed to list orders: {str(e)}")
    
    # Test listing positions
    logger.info("Testing positions listing...")
    try:
        positions = api.list_positions()
        logger.info(f"Current positions count: {len(positions)}")
        
        # Show some details about current positions if available
        if positions:
            for i, position in enumerate(positions):
                logger.info(f"Position {i+1}: {position.qty} shares of {position.symbol} @ ${position.avg_entry_price}")
    except Exception as e:
        logger.warning(f"Failed to list positions: {str(e)}")
    
    # Check market status
    logger.info("Checking market status...")
    try:
        clock = api.get_clock()
        if clock.is_open:
            logger.info(f"Market is OPEN. Next close at: {clock.next_close}")
        else:
            logger.info(f"Market is CLOSED. Next open at: {clock.next_open}")
    except Exception as e:
        logger.warning(f"Failed to check market status: {str(e)}")
    
    logger.info("API test completed successfully!")
    
except Exception as e:
    logger.error(f"API connection or operation failed: {str(e)}")
    sys.exit(1)