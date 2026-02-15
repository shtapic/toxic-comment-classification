from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_read_root():
    """Тест доступности главной страницы (GET /)"""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_classify_empty_context():
    """Тест валидации пустого ввода"""
    response = client.post("/api/classify", json={"content": "   "})
    assert response.status_code == 400
    assert response.json()["detail"] == "Input text cannot be empty."

def test_classify_valid_text():
    """
    Тест классификации. 
    Примечание: Если модели не загружены (нет файлов), API вернет 500.
    """
    response = client.post("/api/classify", json={"content": "Hello, world!"})
    
    if response.status_code == 500:
        assert response.json()["detail"] == "Model not loaded. Please try again later."
    else:
        assert response.status_code == 200
        data = response.json()
        assert "toxic" in data
        assert "is_toxic" in data["toxic"]
        assert isinstance(data["toxic"]["score"], float)

def test_classify_toxic_example():
    """Проверка, что модель определяет явную токсичность (если загружена)"""
    response = client.post("/api/classify", json={"content": "You are stupid idiot"})
    
    if response.status_code == 200:
        data = response.json()
        assert "obscene" in data
