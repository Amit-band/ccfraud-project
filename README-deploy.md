Deploy bundle for Credit Card Fraud Webapp

Quick steps:
1. Place your webapp folder as 'webapp/' next to this docker-compose.yml
2. Place cc_fraud.sql next to docker-compose.yml
3. Run: docker compose up --build
4. Open http://localhost:8080
5. The ML worker runs once and writes fraud_predictions table in DB
