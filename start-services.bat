@echo off
setlocal

rem Start MCP servers
start "MCP Salesforce" cmd /k "rca-app mcp-salesforce --host 0.0.0.0 --port 8600"
start "MCP SAP" cmd /k "rca-app mcp-sap --host 0.0.0.0 --port 8700"
start "MCP Supply Chain" cmd /k "rca-app mcp-supply-chain --host 0.0.0.0 --port 8800"

rem Start FastAPI service
start "FastAPI" cmd /k "uvicorn rca_app.api:app --host 0.0.0.0 --port 8000"

rem Start React UI
start "UI" cmd /k "cd frontend && npm install && npm run dev"

endlocal
