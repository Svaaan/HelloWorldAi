# HelloWorldAi

## Innehållsförteckning

- [Översikt](#översikt)
- [Projektstruktur](#projektstruktur)
- [Backend](#backend)
- [Frontend](#frontend)
- [Docker & Deployment](#docker--deployment)
- [Installation & Kom igång](#installation--kom-igång)
- [Utvecklingsflöde](#utvecklingsflöde)
- [Kontakt](#kontakt)

## Översikt

HelloWorldAi är ett projekt som kombinerar backend och frontend för att hantera och visualisera AI-relaterade uppgifter och noder. Systemet är uppdelat i flera moduler för att möjliggöra skalbarhet och enkel vidareutveckling.

## Projektstruktur

├── src/ │ ├── app.py # Huvudfil för att starta applikationen │ ├── backend/ # Backend-logik och tjänster │ │ ├── coordinator.py # Hanterar koordinering mellan noder │ │ ├── dashboard.py # Dashboard-funktionalitet │ │ ├── executeTask.py # Utför uppgifter på noder │ │ ├── node.py # Nodhantering │ │ ├── proxypage.py # Proxy för sidor │ │ ├── terminate_port.py # Hanterar portavslut │ │ ├── database/ # Databaslogik │ │ ├── service/ # Backend-tjänster (t.ex. GPU, systeminfo) │ │ ├── shared/ # Delad logik och tillstånd │ │ ├── utils/ # Hjälpfunktioner och konfiguration │ ├── frontend/ # Frontend-resurser │ │ ├── static/ # CSS och JS för UI │ │ ├── template/ # HTML-mallar7

## Backend

Backend består av flera moduler:

- **coordinator.py**: Koordinerar kommunikation och arbetsfördelning mellan noder.
- **dashboard.py**: Hanterar dashboard-funktioner och statistik.
- **executeTask.py**: Utför och hanterar uppgifter på noder.
- **node.py**: Representerar och hanterar enskilda noder.
- **service/**: Innehåller tjänster för autentisering, systeminfo, GPU-hantering m.m.
- **database/**: Databaslogik för att lagra nodinformation.
- **shared/**: Delade tillstånd och typer mellan moduler.
- **utils/**: Hjälpfunktioner och konfiguration.

## Frontend

Frontend består av:

- **static/**: CSS och JavaScript för användargränssnittet.
- **template/**: HTML-mallar för olika sidor (t.ex. connect, distribution, node, setup).

## Docker

Projektet innehåller Docker-konfigurationer för att underlätta deployment och testning:

- **docker-compose.yml**: Standardkonfiguration.
- **docker-compose.local.yml**: Lokal utvecklingsmiljö.
- **docker-compose.test.yml**: Testmiljö.
- **Dockerfile**: Bygger backend-miljön.

## Kom igång

1. ### Installera beroenden:

   - pip install -r requirements.txt

2. ### Starta backend:

   - python app.py

3. ### 3. Starta med Docker:
   - docker-compose up

## Utvecklingsflöde

- Backendutveckling sker främst i `src/backend/`.
- Frontendutveckling sker i `src/frontend/static/` och `src/frontend/template/`.
- För att lägga till nya tjänster, skapa en ny fil i `service/` och importera den i backend.
- Testning kan ske via Docker eller direkt med Python.

## Kontakt

Vid frågor, kontakta projektägaren eller skapa ett issue på GitHub.
