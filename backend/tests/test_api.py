import pytest
from httpx import AsyncClient, ASGITransport

@pytest.mark.asyncio
async def test_health_endpoint():
    from main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/api/health")
    
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
