# Projektübersicht

**Naeem Zain Uddin** · M.Sc. Automatisierung und Robotik, TU Dortmund

Eine kurze Übersicht relevanter Robotik- und Softwareprojekte. Eine vollständige Projektliste ist auf [nz5253.github.io](https://nz5253.github.io) verfügbar.

---

## 1. Masterarbeit — Reinforcement Learning für autonomes Einparken auf einem physischen Roboter

**Zeitraum:** Okt 2025 – Mai 2026
**Institut:** Lehrstuhl für Regelungstechnik und Systemdynamik (RST), TU Dortmund
**Stack:** Python, C++, ROS 2, PyTorch, PyBullet, CasADi, IPOPT

Vollständige Entwicklung eines RL-basierten Regelungssystems für einen 1:28-Maßstab-Roboter (Chronos Car) im autonomen Einparkproblem:

- **PPO-Agent** mit eigenem Belohnungsfunktions-Design (14 Komponenten) und mehrstufigem Trainingscurriculum (7 Stufen)
- **NMPC-Baseline** mit CasADi + IPOPT für direkten Vergleich
- **ROS-2-Steuerungssystem in C++** auf embedded Linux mit 1000 Hz Regelkreis
- **MOCAP-basierte Zustandsschätzung**, Sensor-Pipeline-Debugging und Hardware-Iteration für stabilen Realbetrieb
- **Ergebnisse:** 90 % Erfolgsrate in randomisierten Szenarien (RL), 100 % in festen Szenarien (NMPC), Sub-Zentimeter-Positioniergenauigkeit (2,28 cm)
- **Policy-Inference auf <1 ms optimiert** für Echtzeitbetrieb

Eine direkte Vorarbeit für das IL-Pipeline-Konzept im technischen Konzept-Dokument.

---

## 2. Bachelorarbeit — Autonomes Fahrzeugsystem (AGV)

**Zeitraum:** Jun 2021 – Sep 2022
**Institut:** SZABIST, Karachi
**Stack:** Arduino, Python, multi-sensor (Kamera, Liniensensoren, IR), Motorsteuerung

Vollständiger Aufbau eines autonomen mobilen Roboters von Grund auf:

- Multi-Sensor-Integration (Kamera, Liniensensoren, IR)
- Python-basierte Pfadplanung
- Motorsteuerung und Hardware-Integration
- Realer autonomer Feldeinsatz
- **Peer-Review-Publikation** im International Journal of Aquatic Science (2021)

Hands-on-Erfahrung mit dem vollständigen Lebenszyklus eines Robotikprojekts von der Spezifikation über die Hardware-Inbetriebnahme bis zum Feldeinsatz.

---

## 3. Digital Twin für Stromnetze — Programming Team Lead

**Zeitraum:** Sep 2024 – Jul 2025
**Institut:** TU Dortmund
**Stack:** Docker, RabbitMQ, Python, CIM-CGMES, PSDM

Leitung der Entwicklung einer containerisierten Simulationsplattform für Stromnetz-Modellierung:

- Multi-Source-Datenintegration: CIM-CGMES → PSDM-Konvertierung
- Synchronisierte Zustandsupdates über RabbitMQ
- Integration erneuerbarer Energiequellen (PV / WEC)
- UI für Echtzeit-Spannungsanalyse über verteilte Grid-Zustände

Relevant für die jetzige Aufgabe wegen der verteilten Architektur und der Datensynchronisation über mehrere Quellen.

---

## 4. Aktuelle Arbeit (Mai 2026) — IL-Pipeline-Konzept für MyBotShop

**Stack:** ROS 2, Python, PyTorch, LeRobot, FastAPI

Entwicklung des im Bewerbungsprozess vorgeschlagenen Konzepts:

- Vollständige Architekturdokumentation, ROS-2-Knotenstruktur, Dataset-Schema, REST + WebSocket API
- ROS-2-Knoten-Skelette (data_logger, inference) mit Service-Verträgen
- LeRobotDataset-Writer und Frame-Validator
- BC-Referenz-Policy und Training-Pipeline
- FastAPI-Webschicht mit Bridge zur ROS-2-Schicht
- **23 Unit-Tests, alle bestanden**
- **End-to-End-Demo ohne ROS / GPU funktioniert** (Pipeline-Korrektheits-Nachweis)
- Working Prototype mit Gazebo + Franka/UR5e wird nach Verteidigung am Lab-Rechner umgesetzt

Vollständige Materialien werden mit der ausgearbeiteten Lösung übergeben.

---

## 5. Werkstudent — Software & Test-Automatisierung (parallel zur Masterarbeit)

### Kandou Bus GmbH (Apr 2024 – Mär 2025)
- Webbasierte Testausführungsplattform für Intel/AMD-Interoperabilitäts-Benchmarks
- SQL-gestütztes Logging- und Reporting-System
- CI/CD-Pipelines und Remote-Testausführung über SSH und MQTT

### TÜV Rheinland (Apr 2025 – Mär 2026)
- 300+ automatisierte Zertifizierungstestfälle für CSA Matter und Zigbee R23
- Eigenes Python-Test-Framework mit serieller Geräteansteuerung und Regex-Log-Parsing
- Produktiver Einsatz in den Zertifizierungs-Workflows der TÜV-Ingenieure

### Wissenschaftliche Hilfskraft, Lehrstuhl für Zuverlässigkeitstechnik (Jun 2025 – Mai 2026)
- FEM-Benchmark-Katalog mit 87 Testfällen über 8 Physikdomänen
- Python-Tokenizer für Fortran-Quellcode-Parsing
- Python-basierte Regressionstestsuite

---

## Schwerpunkte

- **Robotik:** ROS 2, RL/IL, MPC, Sensor-Integration, Echtzeit-Regelkreise
- **Programmierung:** Python, C++ (täglich)
- **Inbetriebnahme:** Vom Simulationsmodell zum stabilen Feldeinsatz, Hardware-Debugging
- **Testautomatisierung:** Frameworks, CI/CD, produktive Workflows

---

**Kontakt**
E-Mail: naeemzainuddin5253@gmail.com
Tel.: +49 176 43277891
LinkedIn: linkedin.com/in/nz-515253han/
Portfolio: nz5253.github.io
