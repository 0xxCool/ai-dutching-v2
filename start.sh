#!/bin/bash
# ==========================================================
# AI DUTCHING SYSTEM - START SCRIPT
# ==========================================================

set -e  # Exit on error

echo "🚀 AI DUTCHING SYSTEM - STARTUP"
echo "========================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found!${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Python 3 found${NC}"

# Check .env
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠️  .env file not found!${NC}"
    echo "Creating .env from template..."

    cat > .env << EOF
# Sportmonks API Token
SPORTMONKS_API_TOKEN=your_token_here

# Optional: Alert Settings
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
DISCORD_WEBHOOK_URL=
EOF

    echo -e "${YELLOW}📝 Please edit .env with your API credentials${NC}"
fi

# Check config.yaml
if [ ! -f config.yaml ]; then
    if [ -f config.yaml.template ]; then
        echo -e "${YELLOW}⚠️  config.yaml not found. Copying from template...${NC}"
        cp config.yaml.template config.yaml
        echo -e "${GREEN}✅ Created config.yaml${NC}"
    fi
fi

# Check dependencies
echo ""
echo "Checking dependencies..."

if ! python3 -c "import pandas" &> /dev/null; then
    echo -e "${YELLOW}⚠️  Dependencies not installed${NC}"
    echo "Install with: pip install -r requirements.txt"
    read -p "Install now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pip install -r requirements.txt
    fi
fi

echo -e "${GREEN}✅ Dependencies OK${NC}"

# Check database
if [ ! -f game_database_sportmonks.csv ]; then
    echo -e "${YELLOW}⚠️  Database not found${NC}"
    echo "Run scraper first: python sportmonks_xg_scraper.py"

    read -p "Run scraper now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python3 sportmonks_xg_scraper.py
    fi
fi

# Menu
echo ""
echo "========================================"
echo "What would you like to do?"
echo "========================================"
echo "1) Start Dashboard (Recommended)"
echo "2) Run Scraper (xG Data)"
echo "3) Run Dutching System (1X2, O/U, BTTS)"
echo "4) Run Correct Score System"
echo "5) Run Backtest"
echo "6) Train ML Models"
echo "7) Exit"
echo ""
read -p "Enter choice [1-7]: " choice

case $choice in
    1)
        echo -e "${GREEN}🎯 Starting Dashboard...${NC}"
        echo "Dashboard will open at: http://localhost:8501"
        streamlit run dashboard.py
        ;;
    2)
        echo -e "${GREEN}📊 Running Scraper...${NC}"
        python3 sportmonks_xg_scraper.py
        ;;
    3)
        echo -e "${GREEN}💰 Running Dutching System...${NC}"
        python3 sportmonks_dutching_system.py
        ;;
    4)
        echo -e "${GREEN}⚽ Running Correct Score System...${NC}"
        python3 sportmonks_correct_score_system.py
        ;;
    5)
        echo -e "${GREEN}🧪 Running Backtest...${NC}"
        python3 -c "
from backtesting_framework import Backtester, BacktestConfig
import pandas as pd

print('Loading historical data...')
data = pd.read_csv('game_database_sportmonks.csv')

# Simple prediction function
def predict(row):
    return {
        'market': '3Way Result',
        'selection': 'Home',
        'probability': 0.45,
        'confidence': 0.7,
        'odds': 2.0
    }

config = BacktestConfig(initial_bankroll=1000.0)
backtester = Backtester(config)

print('Running backtest...')
result = backtester.run_backtest(data.tail(100), predict)
backtester.print_results(result)
"
        ;;
    6)
        echo -e "${GREEN}🤖 Training ML Models...${NC}"
        python3 -c "
from ml_prediction_models import XGBoostMatchPredictor, FeatureEngineer
import pandas as pd
import numpy as np

print('Loading data...')
data = pd.read_csv('game_database_sportmonks.csv')

print('Feature engineering...')
engineer = FeatureEngineer(data)

X = []
y = []

for idx, row in data.iterrows():
    try:
        features = engineer.create_match_features(
            row['home_team'],
            row['away_team'],
            row['date']
        )

        if row['home_score'] > row['away_score']:
            label = 0
        elif row['home_score'] == row['away_score']:
            label = 1
        else:
            label = 2

        X.append(features)
        y.append(label)
    except:
        continue

X = np.array(X)
y = np.array(y)

print(f'Training on {len(X)} samples...')
model = XGBoostMatchPredictor()
model.train(X, y)

print('✅ Training complete!')
"
        ;;
    7)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}✅ Done!${NC}"
