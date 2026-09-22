from fastapi.testclient import TestClient
from app.main import app

def test_health_and_prediction():
    with TestClient(app) as client:
        assert client.get('/api/health').status_code == 200
        response=client.post('/api/feedback/analyze',json={'text':'The library wifi keeps disconnecting during lectures','category':'Library','rating':2})
        assert response.status_code == 200
        assert set(response.json()['predictions']) == {'sentiment','topic','priority'}

def test_feedback_persistence_and_analytics():
    with TestClient(app) as client:
        saved=client.post('/api/feedback',json={'text':'New feedback about broken laboratory computers','category':'Laboratory','rating':2,
        })
        assert saved.status_code == 200
        item=saved.json()
        assert client.put(f"/api/feedback/{item['id']}/label",json={'actual_sentiment':'negative','actual_topic':'laboratory','actual_priority':'high'}).status_code == 200
        assert client.get('/api/analytics/overview').json()['total_feedback'] >= 1
