#!/bin/bash
# Yoga AI Trainer - Development Startup Script

echo "🧘‍♀️ Yoga AI Trainer - Starting Development Environment"
echo "========================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if virtual environment exists
if [ ! -d "major_project-venv" ]; then
    echo -e "${RED}❌ Virtual environment not found!${NC}"
    echo "Please create it with: python -m venv major_project-venv"
    exit 1
fi

echo -e "${BLUE}🔧 Activating virtual environment...${NC}"
source major_project-venv/bin/activate

echo -e "${BLUE}📦 Installing/updating dependencies...${NC}"
pip install -q -r yoga_ai_trainer/requirements.txt

echo -e "${BLUE}🧪 Running system tests...${NC}"
python test_pose_detection.py

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed!${NC}"
else
    echo -e "${RED}❌ Some tests failed. Please check the output above.${NC}"
    echo "You can still start the server, but some features might not work correctly."
fi

echo -e "${BLUE}🚀 Starting backend server...${NC}"
cd yoga_ai_trainer/backend

# Start server in background
source ../../major_project-venv/bin/activate
python main.py &
SERVER_PID=$!

# Wait a moment for server to start
sleep 3

# Check if server is running
if curl -s http://localhost:8000/ > /dev/null; then
    echo -e "${GREEN}✅ Backend server started successfully at http://localhost:8000${NC}"
    echo -e "${GREEN}📱 Frontend available at: file://$(pwd)/../frontend/index.html${NC}"
    echo ""
    echo -e "${YELLOW}📋 Quick Commands:${NC}"
    echo -e "   • Test backend: ${BLUE}curl http://localhost:8000/${NC}"
    echo -e "   • Check poses: ${BLUE}curl http://localhost:8000/poses${NC}"
    echo -e "   • Open frontend: ${BLUE}firefox ../frontend/index.html${NC}"
    echo -e "   • Stop server: ${BLUE}kill $SERVER_PID${NC}"
    echo ""
    echo -e "${GREEN}🎉 Development environment ready!${NC}"
    echo "Press Ctrl+C to stop the server"
    
    # Wait for interrupt
    trap "echo -e '\n${YELLOW}🛑 Stopping server...${NC}'; kill $SERVER_PID; exit 0" INT
    
    # Keep the script running
    wait $SERVER_PID
    
else
    echo -e "${RED}❌ Failed to start backend server${NC}"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi
